"""Build the FAISS indices the sampler's vecdb retriever reads.

Reads the per-table feature blobs written by the featurize scripts and writes
one FAISS index plus L2-normalized vectors per table, in the layout rustler's
``vector_db_path`` knob consumes:

    <vector_db_root>/<db>/<table>.index         (FAISS, METRIC_INNER_PRODUCT)
    <vector_db_root>/<db>/<table>_vectors.bin   (row-major f32, normalized)

Vectors are L2-normalized so inner-product search is cosine similarity.
Tables above ``ivf_threshold`` rows get an IVF index with ``nprobe`` baked in
(0 = auto ``max(8, sqrt(nlist))``); smaller tables get a Flat index. Pass a
separate ``vector_db_root`` per feature set (rdblearn / rt).
"""

import json
import time
from pathlib import Path


def build_all(
    *,
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    features_subdir: str,
    vector_db_root: str,
    ivf_threshold: int,
    nprobe: int,
) -> None:
    import faiss
    import numpy as np

    from rt.data import resolve_db_task_list

    from expts.repaper_baselines.rel2tab.featurizer import table_offset_and_len

    # One index per unique (db, table); a db's tasks can share a table.
    pairs = sorted(set(resolve_db_task_list(db_task_list)))
    for db, table in pairs:
        out_dir = Path(vector_db_root).expanduser() / db
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / f"{table}.index"
        vectors_path = out_dir / f"{table}_vectors.bin"
        if index_path.exists() and vectors_path.exists():
            print(f"{db}/{table}: index exists, skipping", flush=True)
            continue

        feat_dir = Path(features_root).expanduser() / db / features_subdir
        with open(feat_dir / f"{table}_meta.json") as f:
            meta = json.load(f)
        min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
        assert (
            meta["min_offset"] == min_offset and meta["total_nodes"] == total_nodes
        ), (
            f"{db}/{table}: feature meta {meta} disagrees with table_info "
            f"(offset {min_offset}, nodes {total_nodes}); features are stale"
        )
        vectors = np.fromfile(
            feat_dir / f"{table}_vectors.bin", dtype=np.float32
        ).reshape(total_nodes, meta["n_features"])

        # Zero rows (no features) get unit length before the divide so they
        # land at cosine zero rather than NaN.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        vectors = vectors / norms

        num_nodes, dim = vectors.shape
        t0 = time.perf_counter()
        if num_nodes > ivf_threshold:
            nlist = min(max(int(4 * np.sqrt(num_nodes)), 16), 65536)
            chosen_nprobe = nprobe if nprobe > 0 else max(8, int(np.sqrt(nlist)))
            index = faiss.index_factory(
                dim, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT
            )
            index.train(vectors[: min(nlist * 40, num_nodes)])
            index.add(vectors)
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", chosen_nprobe)
            # On-disk inverted lists: each consumer process mmaps the postings
            # instead of copying them into its own heap -- the eval dataloader
            # runs several worker processes, each of which loads the index.
            index.own_invlists = False
            old_invlists = index.invlists
            new_invlists = faiss.OnDiskInvertedLists(
                index.nlist, index.code_size, str(out_dir / f"{table}.ivfdata")
            )
            new_invlists.merge_from(old_invlists, 0)
            index.replace_invlists(new_invlists, True)
            new_invlists.this.disown()
            del old_invlists
            kind = f"IVF{nlist},Flat (ondisk, nprobe={chosen_nprobe})"
        else:
            index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
            index.add(vectors)
            kind = "Flat"

        faiss.write_index(index, str(index_path))
        vectors.astype(np.float32).tofile(vectors_path)
        print(
            f"{db}/{table}: {num_nodes:,} x {dim} -> {kind} "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
