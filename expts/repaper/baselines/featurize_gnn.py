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
    channels: int,
    num_layers: int,
    num_neighbors: int,
    aggr: str,
    temporal_strategy: str,
    sampler_threads: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> None:
    import numpy as np
    import torch
    from relbench import load_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.nn import (
        HeteroEncoder,
        HeteroGraphSAGE,
        HeteroTemporalEncoder,
    )
    from relbench.modeling.utils import get_stype_proposal, to_unix_time
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.seed import seed_everything

    from expts.repaper.baselines.gnn_text_embedder import GloveTextEmbedding
    from expts.repaper.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        table_offset_and_len,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(features_root).expanduser() / db / "gnn_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = sorted(
        t for d, t in json.loads(Path(db_task_list).expanduser().read_text()) if d == db
    )
    assert tables, f"no tasks for {db} in {db_task_list}"

    # load_dataset takes a local path as readily as a Hub id, so the graph is
    # built from the same RAW_DIR files featurize_rdblearn reads rather than from
    # a fresh download: same vintage, no network on the compute node. (RAW_DIR
    # holds rel-event/user-repeat and rel-stack/post-votes at revision
    # 2718db98b3c3 to match relbench-preprocessed; a re-download would break the
    # row-count assert below.)
    dataset = load_dataset(str(Path(raw_dir).expanduser() / db))
    rdb = dataset.get_db()
    proposal = get_stype_proposal(rdb)
    # Per db, not per task: materializing every table into TensorFrames -- text
    # columns through GloVe included -- is the expensive half of this job, and
    # make_pkey_fkey_graph's cache is keyed on the dataset-level db precisely so
    # every task on that db can share it. One job per db also means no two jobs
    # race to write the same cache, which is how seven rdblearn jobs died on the
    # relbench cache (184096-184110).
    cache_dir = str(Path(graph_cache_dir).expanduser() / db)
    # cpu, not `device`, exactly as examples/gnn_entity.py hardcodes it: the
    # embeddings go into the TensorFrames that make_pkey_fkey_graph writes to
    # cache_dir, so a cuda embedder makes the cache a file full of cuda tensors
    # that torch_frame.load refuses to read on a cpu host ("Attempting to
    # deserialize object on a CUDA device"). The GNN itself still runs on
    # `device`; only this one-off encode is pinned.
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
        # Never the example's include_task_tables="all"/"current_only": those add
        # the label tables to the graph as autoregressive features, which is the
        # target leaking into its own input. featurize_rdblearn sets
        # enable_target_augmentation=False for the same reason, and here the
        # labels are never even read -- the task table supplies only the seed
        # node and its cutoff.
        data, col_stats_dict = make_pkey_fkey_graph(
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

        # get_node_train_table_input's two tensors, without going through it: it
        # also builds an AttachTargetTransform, and a featurizer has no business
        # reading the target column at all. Concatenated in node_idx_offset order
        # so row r of the blob is node min_offset + r, the contract
        # PrecomputedFeaturizer indexes by.
        entity_table = task.entity_table
        nodes = torch.from_numpy(
            np.concatenate(
                [f[task.entity_col].to_numpy().astype("int64") for f in frames]
            )
        )
        input_time = torch.from_numpy(
            np.concatenate([to_unix_time(f[task.time_col]) for f in frames])
        )

        # Seeded here, not at entry: make_pkey_fkey_graph is cached, so seeding
        # before it would make the initialization depend on whether the cache was
        # warm. Everything random is constructed below this line.
        seed_everything(seed)
        encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        ).to(device)
        temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        ).to(device)
        gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        ).to(device)
        for module in (encoder, temporal_encoder, gnn):
            module.eval()

        # pyg-lib samples in parallel with one RandintEngine per thread, and
        # each engine is seeded deterministically (vslNewStream(.., MT19937, 1)),
        # so nothing here draws from entropy -- what varies run to run is which
        # thread draws for which seed node. seed_everything cannot reach any of
        # it. Measured on rel-f1/driver-top3, two separate processes with a warm
        # graph cache: default threads DIFFER, one thread IDENTICAL
        # (scripts/probe_sampler_seed.py, scripts/featurize_once.py). Set after
        # make_pkey_fkey_graph so a cold cache still materializes and embeds in
        # parallel; the GNN forward is on `device`, so this only costs sampling.
        torch.set_num_threads(sampler_threads)
        loader = NeighborLoader(
            data,
            num_neighbors=[num_neighbors // 2**i for i in range(num_layers)],
            time_attr="time",
            input_nodes=(entity_table, nodes),
            input_time=input_time,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        # examples/model.py's Model.forward minus the MLP head: the head maps to
        # out_channels=1, and what a featurizer wants is the channels-wide seed
        # representation the head would have consumed. shallow_list and
        # id_awareness are the example's defaults (off), so nothing else of it is
        # skipped.
        chunks = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                seed_time = batch[entity_table].seed_time
                x_dict = encoder(batch.tf_dict)
                rel_time_dict = temporal_encoder(
                    seed_time, batch.time_dict, batch.batch_dict
                )
                for node_type, rel_time in rel_time_dict.items():
                    x_dict[node_type] = x_dict[node_type] + rel_time
                x_dict = gnn(
                    x_dict,
                    batch.edge_index_dict,
                    batch.num_sampled_nodes_dict,
                    batch.num_sampled_edges_dict,
                )
                chunks.append(x_dict[entity_table][: seed_time.size(0)].float().cpu())
                if i % 200 == 0:
                    done = sum(c.shape[0] for c in chunks)
                    print(
                        f"[{db}] {table}: {done}/{total_nodes} rows "
                        f"({time.time() - tic:.0f}s)",
                        flush=True,
                    )

        arr = torch.cat(chunks, dim=0).numpy().astype(np.float64)
        assert arr.shape[0] == total_nodes, f"{arr.shape[0]} vs {total_nodes}"

        # Same standardization as the rdblearn and sql blobs: the predictors see
        # one column scale across every context, and TabICL's float32
        # per-context standardization cannot be trusted with raw magnitudes.
        arr = np.where(np.isfinite(arr), arr, np.nan)
        mean = np.nanmean(arr, axis=0, keepdims=True)
        std = np.nanstd(arr, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        feats = np.nan_to_num((arr - mean) / std, nan=0.0).astype(np.float32)
        assert np.isfinite(feats).all()

        feats.tofile(vectors_path)
        meta_path.write_text(
            json.dumps(
                {
                    "n_features": feats.shape[1],
                    "min_offset": min_offset,
                    "total_nodes": total_nodes,
                }
            )
        )
        print(
            f"[{db}] {table}: {total_nodes} rows x {feats.shape[1]} features "
            f"in {time.time() - tic:.0f}s",
            flush=True,
        )
