"""Precompute RT-J row embeddings for one database's task tables.

For every row of every task table: build a small walk-free local context
around the row (single BFS, the row's target cell masked), run the RT-J
checkpoint matching the task's type with ``return_embeddings=True``, and keep
the embedding at the masked target cell as the row's vector. Written as:

    <features_root>/<db>/rt_features/<table>_vectors.bin   (row-major f32)
    <features_root>/<db>/rt_features/<table>_meta.json

These vectors feed ``build_vector_db.py`` for the RT-similarity retriever
ablation. Runs in the default environment on one GPU.
"""

import json
from pathlib import Path


def featurize_db(
    *,
    db: str,
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    ckpt_clf: str,
    ckpt_reg: str,
    local_ctx_size: int,
    bfs_width: int,
    shuffle_seed: int,
    context_seed: int,
    db_cutoff: str | int | None,
    batch_size: int,
) -> None:
    import numpy as np
    import torch

    from rt.data import RustlerDataset, get_tasks, process_batch
    from rt.model import load_rt_model

    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len

    device = "cuda"
    tasks = [t for t in get_tasks(pre_dir, db_task_list, ("test",)) if t.db_name == db]
    assert tasks, f"no tasks for {db} in {db_task_list}"
    # One embedding file per table; the first task of a table names its target.
    by_table = {}
    for t in tasks:
        by_table.setdefault(t.table_name, t)

    nets = {}  # task_type -> (net, config)
    for task in sorted(by_table.values(), key=lambda t: t.table_name):
        if task.task_type not in nets:
            ckpt = ckpt_clf if task.task_type == "clf" else ckpt_reg
            net, config = load_rt_model(ckpt, device=device, compile=False)
            nets[task.task_type] = (net.to(torch.bfloat16).eval(), config)
        net, config = nets[task.task_type]

        ds = RustlerDataset(
            tasks=[task],
            pre_dir=pre_dir,
            global_rank=0,
            local_rank=0,
            world_size=1,
            local_ctx_size_list=[local_ctx_size],
            bfs_width_list=[bfs_width],
            num_walks=0,
            walk_length=0,
            prefer_latest_list=[False],
            mask_prob_max=0.0,
            embedder=config["embedder"],
            d_text=config["d_text"],
            shuffle_seed=shuffle_seed,
            context_seed=context_seed,
            items_per_task=10_000_000,
            quiet=True,
            ignore_data_errors=False,
            mmap_populate=True,
            timeout_per_item=3600.0,
            vector_db_path=None,
            db_cutoff=db_cutoff,
        )

        min_offset, total_nodes = table_offset_and_len(pre_dir, db, task.table_name)
        out_dir = Path(features_root).expanduser() / db / "rt_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        vectors_path = out_dir / f"{task.table_name}_vectors.bin"
        meta_path = out_dir / f"{task.table_name}_meta.json"
        if vectors_path.exists() and meta_path.exists():
            print(f"[{db}] {task.table_name}: already featurized, skipping", flush=True)
            continue

        chunks = []
        with torch.inference_mode():
            for start in range(0, total_nodes, batch_size):
                node_idxs = list(
                    range(
                        min_offset + start,
                        min_offset + min(start + batch_size, total_nodes),
                    )
                )
                tup = ds.sampler.batch_for_nodes_py(node_idxs, 0, local_ctx_size)
                batch = process_batch(tup, ds.d_text)
                batch.pop("batch_mask", None)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                x = net(batch, return_embeddings=True)  # (B, S, d_model)
                # forward sorts cells by column index; pick the target cell
                # through the same sort or the gather lands on the wrong cell.
                sort_keys = batch["col_name_idxs"].masked_fill(
                    batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
                )
                sort_idxs = sort_keys.argsort(dim=-1, stable=True)
                sorted_is_targets = batch["is_targets"].gather(1, sort_idxs)
                per_row = sorted_is_targets.sum(dim=1)
                assert (per_row == 1).all(), (
                    f"{db}/{task.table_name}: expected one target cell per row, "
                    f"got counts {per_row.unique().tolist()}"
                )
                chunks.append(x[sorted_is_targets].float().cpu().numpy())
                if (start // batch_size) % 50 == 0:
                    print(
                        f"[{db}] {task.table_name}: {start + len(node_idxs)}"
                        f"/{total_nodes}",
                        flush=True,
                    )

        feats = np.concatenate(chunks, axis=0).astype(np.float32)
        assert feats.shape[0] == total_nodes
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
            f"[{db}] {task.table_name}: {total_nodes} rows x {feats.shape[1]} dims",
            flush=True,
        )
