import json
from pathlib import Path


def main(
    *,
    db_task_list: str,
    pre_dir: str,
    labels_root: str,
    context_rows: int,
    context_seed: int,
    batch_size: int,
) -> None:
    import time

    import numpy as np

    from rt.data import RustlerDataset, get_tasks, process_batch

    from expts.repaper.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
    )
    from expts.repaper.baselines.rel2tabv2.context_sampler import UniformRandomSampler

    # features_rt-j holds vectors and no labels, so an in-loop val metric has
    # nowhere to read the target from. This writes it once: every val row, plus
    # the exact draw GlobalContextRetriever makes for the context, so both the
    # 8192-row monitor and a later 2**18 run read the same file. No test row is
    # touched.
    out_root = Path(labels_root).expanduser()
    t0 = time.time()
    for db in sorted({d for d, _ in json.loads(Path(db_task_list).expanduser().read_text())}):
        tasks = [t for t in get_tasks(pre_dir, db_task_list, ("train",)) if t.db_name == db]
        assert tasks, f"{db}: no task in {db_task_list}"
        out_dir = out_root / db
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [t for t in tasks if not (out_dir / f"{t.table_name}.npz").is_file()]
        if not todo:
            print(f"[{db}] all {len(tasks)} tasks done, skipping", flush=True)
            continue

        # ctx_size 1 reads the target cell and nothing else, so there is no
        # graph to walk and no reason to pull 51 GiB of relbench into the page
        # cache first.
        ds = RustlerDataset(
            tasks=todo,
            pre_dir=pre_dir,
            global_rank=0,
            local_rank=0,
            world_size=1,
            local_ctx_size_list=[1],
            bfs_width_list=[0],
            num_walks=0,
            walk_length=0,
            prefer_latest_list=[False],
            mask_prob_max=0.0,
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            shuffle_seed=0,
            context_seed=context_seed,
            items_per_task=10_000_000,
            quiet=True,
            ignore_data_errors=False,
            mmap_populate=False,
            timeout_per_item=3600.0,
            vector_db_path=None,
            db_cutoff=None,
            legacy_boolean=False,
        )

        for dataset_idx, task in enumerate(todo):
            splits = get_table_splits(load_table_info(pre_dir, db), task.table_name)
            val, train = splits["val"], splits["train"]
            val_rows = np.arange(
                val["node_idx_offset"], val["node_idx_offset"] + val["num_nodes"]
            )
            candidates = np.arange(
                train["node_idx_offset"], train["node_idx_offset"] + train["num_nodes"]
            )
            drawn = UniformRandomSampler(seed=context_seed).sample(
                candidates, min(context_rows, len(candidates))
            )
            rows = np.union1d(val_rows, drawn)
            labels = np.empty(len(rows), dtype=np.float32)
            for start in range(0, len(rows), batch_size):
                chunk = [int(v) for v in rows[start : start + batch_size]]
                tup = ds.sampler.batch_for_nodes_py(chunk, dataset_idx, 1)
                batch = process_batch(tup, ds.d_text)
                batch.pop("batch_mask", None)
                is_targets = batch["is_targets"]
                per_row = is_targets.sum(dim=1)
                assert (per_row == 1).all(), (
                    f"{db}/{task.table_name}: expected one target cell per row, "
                    f"got counts {per_row.unique().tolist()}"
                )
                vals = batch["number_values"].squeeze(-1)
                labels[start : start + len(chunk)] = (
                    (vals * is_targets.to(vals.dtype)).sum(dim=1).float().numpy()
                )
            np.savez(
                out_dir / f"{task.table_name}.npz",
                node_idxs=rows.astype(np.int64),
                labels=labels,
                n_val=np.int64(len(val_rows)),
                val_lo=np.int64(val["node_idx_offset"]),
                val_hi=np.int64(val["node_idx_offset"] + val["num_nodes"]),
            )
            finite = labels[np.isfinite(labels)]
            print(
                f"[{db}] {task.table_name}: {len(rows)} rows "
                f"({len(val_rows)} val + {len(drawn)} context), "
                f"{task.task_type}, frac>0 {float((finite > 0).mean()):.4f}, "
                f"{(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
