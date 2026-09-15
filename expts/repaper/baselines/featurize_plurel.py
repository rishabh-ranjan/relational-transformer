import json
from pathlib import Path


def featurize_db(
    *,
    db: str,
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    ckpt: str,
    local_ctx_size: int,
    bfs_width: int,
    shuffle_seed: int,
    context_seed: int,
    db_cutoff: str | int | None,
    batch_size: int,
) -> None:
    import numpy as np
    import torch

    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len
    from expts.repaper.baselines.rel2tabv2.plurel_embed import (
        LEGACY_EMBEDDER,
        LEGACY_MODEL_DIMS,
        PluRelEmbedder,
    )
    from rt.data import RustlerDataset, get_tasks, process_batch

    device = "cuda"
    tasks = [t for t in get_tasks(pre_dir, db_task_list, ("test",)) if t.db_name == db]
    assert tasks, f"no tasks for {db} in {db_task_list}"
    by_table = {}
    for t in tasks:
        by_table.setdefault(t.table_name, t)

    net = PluRelEmbedder.from_pretrained(ckpt, device=device)
    net = net.to(torch.bfloat16).eval()
    d_text = LEGACY_MODEL_DIMS["d_text"]

    for task in sorted(by_table.values(), key=lambda t: t.table_name):
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
            embedder=LEGACY_EMBEDDER,
            d_text=d_text,
            shuffle_seed=shuffle_seed,
            context_seed=context_seed,
            items_per_task=10_000_000,
            quiet=True,
            ignore_data_errors=False,
            mmap_populate=True,
            timeout_per_item=3600.0,
            vector_db_path=None,
            db_cutoff=db_cutoff,
            legacy_boolean=True,
        )

        min_offset, total_nodes = table_offset_and_len(pre_dir, db, task.table_name)
        out_dir = Path(features_root).expanduser() / db / "plurel_features"
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
                x = net.embed(batch)
                # PluRel never reorders the sequence (RT sorts its cells by
                # col_name_idx and returns x in that order, which is why
                # featurize_rt has to gather is_targets through the same
                # permutation), so is_targets indexes x as it stands.
                per_row = batch["is_targets"].sum(dim=1)
                assert (per_row == 1).all(), (
                    f"{db}/{task.table_name}: expected one target cell per row, "
                    f"got counts {per_row.unique().tolist()}"
                )
                chunks.append(x[batch["is_targets"]].float().cpu().numpy())
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
