import json
import time
from pathlib import Path


def featurize_db(
    *,
    db: str,
    db_task_list: str,
    pre_dir: str,
    raw_dir: str,
    features_root: str,
    graph_cache_dir: str,
    text_embedder_dir: str,
) -> None:
    """The entity's own row, and nothing relational.

    One feature vector per task row: the entity table's encoded columns for that
    entity, plus the entity id and the task row's cutoff time. Zero hops, so this
    is the floor the relational featurizers are measured against -- rdblearn
    aggregates over 2 joins, rt-j attends over a sampled neighbourhood, the gnn
    passes messages over 2 hops, and this reads one row.

    The encoding is `make_pkey_fkey_graph`'s, taken straight out of the
    materialized TensorFrame and flattened rather than re-derived: numeric
    columns as floats, categoricals as their codes, text as its GloVe embedding,
    timestamps as torch-frame's decomposition. That shares the graph cache the
    gnn featurizer already built, so this arm's inputs are exactly the gnn arm's
    inputs with the network removed.
    """
    import numpy as np
    import torch
    from relbench import load_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal, to_unix_time
    from torch_frame.config.text_embedder import TextEmbedderConfig

    from expts.repaper.baselines.gnn_text_embedder import GloveTextEmbedding
    from expts.repaper.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        table_offset_and_len,
    )

    out_dir = Path(features_root).expanduser() / db / "entity_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = sorted(
        t for d, t in json.loads(Path(db_task_list).expanduser().read_text()) if d == db
    )
    assert tables, f"no tasks for {db} in {db_task_list}"

    dataset = load_dataset(str(Path(raw_dir).expanduser() / db))
    rdb = dataset.get_db()
    proposal = get_stype_proposal(rdb)
    cache_dir = str(Path(graph_cache_dir).expanduser() / db)
    embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(
            model_path=Path(text_embedder_dir).expanduser(), device="cpu"
        ),
        batch_size=256,
    )

    for table in tables:
        vectors_path = out_dir / f"{table}_vectors.bin"
        meta_path = out_dir / f"{table}_meta.json"
        if vectors_path.exists() and meta_path.exists():
            print(f"[{db}] {table}: already featurized, skipping", flush=True)
            continue
        tic = time.time()

        task = dataset.load_task(table)
        data, _col_stats = make_pkey_fkey_graph(
            rdb,
            col_to_stype_dict=proposal,
            text_embedder_cfg=embedder_cfg,
            cache_dir=cache_dir,
            remove_columns=task.hidden_columns(),
        )

        min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
        splits_info = get_table_splits(load_table_info(pre_dir, db), table)
        ordered = sorted(splits_info.items(), key=lambda kv: kv[1]["node_idx_offset"])
        frames = [task.get_table(s).df.reset_index(drop=True) for s, _ in ordered]
        rows = sum(len(f) for f in frames)
        assert rows == total_nodes, (
            f"{db}/{table}: {rows} task rows vs {total_nodes} nodes in "
            f"table_info.json -- the data the features are for is not the data "
            f"that was preprocessed"
        )

        entity_idx = np.concatenate(
            [f[task.entity_col].to_numpy().astype("int64") for f in frames]
        )
        # A torch index, not the numpy array: MultiEmbeddingTensor's __getitem__
        # dispatches on Tensor or int and falls through to an
        # "AssertionError: Should not reach here." on anything else.
        entity_sel = torch.from_numpy(entity_idx)
        cutoff = np.concatenate([to_unix_time(f[task.time_col]) for f in frames])

        tf = data[task.entity_table].tf
        blocks, names, skipped = [], [], []
        for st, cols in tf.col_names_dict.items():
            feat = tf.feat_dict[st]
            if isinstance(feat, torch.Tensor):
                # (rows, cols) for numeric and categorical, (rows, cols, d) for
                # timestamps, which torch-frame decomposes into d parts.
                block = feat[entity_sel].float()
                if block.dim() == 2:
                    block = block.unsqueeze(-1)
                per_col = block.shape[-1]
                blocks.append(block.reshape(len(entity_idx), -1))
                names += [
                    f"{c}[{i}]" if per_col > 1 else c
                    for c in cols
                    for i in range(per_col)
                ]
            elif hasattr(feat, "values") and hasattr(feat, "offset"):
                # MultiEmbeddingTensor: the text columns after the embedder, so
                # each column is a fixed width and they sit concatenated in
                # .values with .offset giving the boundaries. It is not a
                # torch.Tensor and reports shape (rows, cols, -1), so it has to
                # be unpacked rather than indexed like one -- and it must not be
                # skipped: on rel-f1's drivers these five columns are 1500 of
                # the row's 1510 features.
                bounds = feat.offset.tolist()
                blocks.append(feat[entity_sel].values.float())
                names += [
                    f"{cols[j]}[{i}]"
                    for j in range(len(cols))
                    for i in range(bounds[j + 1] - bounds[j])
                ]
            else:
                # MultiNestedTensor: one cell holds a variable-length list, so
                # there is no fixed width to flatten into. Skipped rather than
                # summarised, because any summary (count, first k) would be a
                # modelling choice smuggled into a featurizer whose point is
                # "the row as stored". Recorded in the meta as skipped_columns.
                print(
                    f"[{db}] {table}: skipping {len(cols)} {st} column(s) "
                    f"{cols} -- {type(feat).__name__} is variable-width",
                    flush=True,
                )
                skipped += list(cols)
                continue

        # The task row's own non-target columns. The entity id is deliberately
        # included: it is what the task table carries, and a predictor is free
        # to ignore it.
        blocks.append(torch.from_numpy(entity_idx).float().unsqueeze(1))
        names.append(f"__entity__{task.entity_col}")
        blocks.append(torch.from_numpy(cutoff).float().unsqueeze(1))
        names.append(f"__cutoff__{task.time_col}")

        if "time" in data[task.entity_table]:
            ent_time = data[task.entity_table].time[entity_sel].float()
            blocks.append((torch.from_numpy(cutoff).float() - ent_time).unsqueeze(1))
            names.append("__entity_age_seconds__")

        feats = torch.cat(blocks, dim=1).numpy().astype(np.float32)
        assert feats.shape[0] == total_nodes, f"{feats.shape[0]} vs {total_nodes}"

        # Same standardization as the other blobs: one column scale across every
        # context. Done in float64 over row blocks so a wide blob never needs a
        # float64 copy of itself.
        block_rows = 1 << 16
        channels = feats.shape[1]
        count = np.zeros(channels, dtype=np.int64)
        total = np.zeros(channels, dtype=np.float64)
        for i in range(0, total_nodes, block_rows):
            sl = feats[i : i + block_rows]
            finite = np.isfinite(sl)
            count += finite.sum(axis=0)
            total += np.where(finite, sl, 0.0).sum(axis=0, dtype=np.float64)
        mean = total / np.maximum(count, 1)
        sq = np.zeros(channels, dtype=np.float64)
        for i in range(0, total_nodes, block_rows):
            sl = feats[i : i + block_rows]
            d = np.where(np.isfinite(sl), sl - mean, 0.0)
            sq += np.einsum("ij,ij->j", d, d)
        std = np.sqrt(sq / np.maximum(count, 1))
        std = np.where(std < 1e-8, 1.0, std)
        mean32, std32 = mean.astype(np.float32), std.astype(np.float32)
        for i in range(0, total_nodes, block_rows):
            sl = feats[i : i + block_rows]
            sl -= mean32
            sl /= std32
            np.nan_to_num(sl, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        assert np.isfinite(feats).all()

        feats.tofile(vectors_path)
        meta_path.write_text(
            json.dumps(
                {
                    "n_features": feats.shape[1],
                    "min_offset": min_offset,
                    "total_nodes": total_nodes,
                    "reproducible": True,
                    "config": {
                        "featurizer": "entity_row",
                        "entity_table": task.entity_table,
                        "entity_col": task.entity_col,
                        "time_col": task.time_col,
                        "hops": 0,
                        "columns": names,
                        "skipped_columns": skipped,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(
            f"[{db}] {table}: {total_nodes} rows x {feats.shape[1]} features "
            f"from {task.entity_table} in {time.time() - tic:.0f}s",
            flush=True,
        )
        del feats, data
