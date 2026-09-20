import json
import shutil
from pathlib import Path


def listed_tasks(pre_dir: str, db: str, db_task_list: str) -> list[tuple[str, object]]:
    from rt.data import get_tasks, resolve_db_task_list

    meta = json.loads((Path(pre_dir).expanduser() / db / "meta.json").read_text())
    known = {t["name"] for t in meta.get("tasks", [])}
    names = sorted(
        n for d, n in resolve_db_task_list(db_task_list) if d == db and n in known
    )
    # One call per name, not one call for all of them: get_tasks keys an
    # autocomplete task by its entity table, so two of them on the same table
    # come back indistinguishable and the task name -- which is what a dump
    # file is named after -- is lost.
    out = []
    for name in names:
        got = get_tasks(pre_dir, [(db, name)], ("train",))
        if got:
            out.append((name, got[0]))
    return out


def pool_rows(np, pre_dir, db, name, table, shuffle_seed, max_rows, **_):
    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len

    lo, n = table_offset_and_len(pre_dir, db, table)
    if n <= max_rows:
        return np.arange(lo, lo + n), n
    # A uniform draw, not a prefix: node idx order is time order, so the first
    # max_rows rows of a big task table are its earliest horizons rather than a
    # sample of it.
    rng = np.random.default_rng(abs(hash((shuffle_seed, db, name))) % 2**63)
    return np.sort(rng.choice(n, size=max_rows, replace=False) + lo), n


def val_and_context_rows(np, pre_dir, db, name, table, context_seed, context_rows, **_):
    from expts.repaper.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
    )
    from expts.repaper.baselines.rel2tabv2.context_sampler import UniformRandomSampler

    splits = get_table_splits(load_table_info(pre_dir, db), table)
    val, train = splits["val"], splits["train"]
    val_rows = np.arange(
        val["node_idx_offset"], val["node_idx_offset"] + val["num_nodes"]
    )
    candidates = np.arange(
        train["node_idx_offset"], train["node_idx_offset"] + train["num_nodes"]
    )
    # The identical draw GlobalContextRetriever makes, so the rows this dump
    # holds are exactly the rows the evaluator will ask for. Test is never
    # touched: the dump cannot score a split it does not contain.
    drawn = UniformRandomSampler(seed=context_seed).sample(
        candidates, min(context_rows, len(candidates))
    )
    return np.union1d(val_rows, drawn), train["num_nodes"] + val["num_nodes"]


def featurize_dbs(
    *,
    dbs: list[str],
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    ckpt: str,
    row_rule: str,
    tap_unit: int,
    local_ctx_size: int,
    bfs_width: int,
    shuffle_seed: int,
    context_seed: int,
    batch_size: int,
    min_rows: int,
    max_rows: int,
    context_rows: int,
    expected_gib: float,
) -> None:
    import time

    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch

    from rt.data import RustlerDataset, process_batch
    from rt.model import load_rt_model

    select = {"pool": pool_rows, "val+context": val_and_context_rows}[row_rule]

    device = "cuda"
    net, config = load_rt_model(ckpt, device=device, compile=False)
    net = net.to(torch.bfloat16).eval()
    d_model = net.d_model

    # The target cell's raw residual stream at the end of relational unit
    # tap_unit -- no norm_out, the same quantity the relbench rt_taps dump holds
    # at that tap. norm_out is an RMSNorm, whose per-row rescaling a linear
    # adapter cannot undo, so raw is what both corpora have to store.
    tap = {}
    handle = net.blocks[tap_unit - 1].register_forward_hook(
        lambda _m, _i, out: tap.__setitem__("x", out)
    )

    total_written = 0
    t_start = time.time()
    for db in dbs:
        named = listed_tasks(pre_dir, db, db_task_list)
        if not named:
            print(f"[{db}] no listed task in this build, skipping", flush=True)
            continue
        out_dir = Path(features_root).expanduser() / db
        out_dir.mkdir(parents=True, exist_ok=True)

        # One sampler for the whole db: batch_for_nodes_py takes a dataset_idx
        # into the task list, so the mmap populate is paid once per database
        # rather than once per task, and a db carries up to 276 of them.
        ds = RustlerDataset(
            tasks=[t for _n, t in named],
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
            db_cutoff=None,
            legacy_boolean=False,
        )

        db_written = 0
        n_done = n_skipped = 0
        for dataset_idx, (name, task) in enumerate(named):
            bin_path = out_dir / f"{name}_target.bin"
            meta_path = out_dir / f"{name}_meta.json"
            if bin_path.exists() and meta_path.exists():
                n_done += 1
                continue

            rows, total_nodes = select(
                np,
                pre_dir=pre_dir,
                db=db,
                name=name,
                table=task.table_name,
                shuffle_seed=shuffle_seed,
                max_rows=max_rows,
                context_seed=context_seed,
                context_rows=context_rows,
            )
            if len(rows) < min_rows:
                n_skipped += 1
                continue
            n_rows = len(rows)

            need = n_rows * d_model * 2
            free = shutil.disk_usage(out_dir).free
            assert need * 1.1 < free, (
                f"[{db}] {name} needs {need / 2**20:.0f} MiB and only "
                f"{free / 2**30:.1f} GiB is free"
            )

            labels = np.empty(n_rows, dtype=np.float32)
            tmp = bin_path.with_suffix(".bin.partial")
            with open(tmp, "wb") as fh, torch.inference_mode():
                for start in range(0, n_rows, batch_size):
                    node_idxs = [int(v) for v in rows[start : start + batch_size]]
                    B = len(node_idxs)
                    tup = ds.sampler.batch_for_nodes_py(
                        node_idxs, dataset_idx, local_ctx_size
                    )
                    batch = process_batch(tup, ds.d_text)
                    batch.pop("batch_mask", None)
                    batch = {
                        k: v.to(device, non_blocking=True) for k, v in batch.items()
                    }
                    tap.clear()
                    net(batch, return_embeddings=True)

                    # forward() sorts the sequence before the blocks run, so the
                    # hook output is already in the canonical frame; only
                    # is_targets, which comes off the unsorted batch, has to be
                    # permuted to meet it.
                    keys = batch["col_name_idxs"].masked_fill(
                        batch["is_padding"],
                        torch.iinfo(batch["col_name_idxs"].dtype).max,
                    )
                    si = keys.argsort(dim=-1, stable=True)
                    is_targets = batch["is_targets"].gather(1, si)
                    per_row = is_targets.sum(dim=1)
                    assert (per_row == 1).all(), (
                        f"{db}/{name}: expected one masked target cell per row, "
                        f"got counts {per_row.unique().tolist()}"
                    )
                    vals = (
                        batch["number_values"]
                        .gather(1, si.unsqueeze(-1).expand_as(batch["number_values"]))
                        .squeeze(-1)
                    )
                    labels[start : start + B] = (
                        (vals * is_targets.to(vals.dtype))
                        .sum(dim=1)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    emb = tap["x"][is_targets].bfloat16()
                    assert emb.shape == (B, d_model), emb.shape
                    fh.write(emb.view(torch.uint16).cpu().numpy().tobytes())

            tmp.replace(bin_path)
            np.savez(
                out_dir / f"{name}_rows.npz",
                node_idxs=rows.astype(np.int64),
                labels=labels,
            )
            written = bin_path.stat().st_size
            db_written += written
            total_written += written
            finite = labels[np.isfinite(labels)]
            meta_path.write_text(
                json.dumps(
                    {
                        "dtype": "bfloat16",
                        "shape": [n_rows, d_model],
                        "tap": tap_unit,
                        "task": name,
                        "table": task.table_name,
                        "task_type": task.task_type,
                        "target_column": task.target_column,
                        "row_rule": row_rule,
                        # The mixture weight: pretraining draws a row uniformly
                        # from a pool holding min(rows, 100_000) of every task,
                        # so P(task) is set by this, not by what we dumped.
                        "total_nodes": total_nodes,
                        "sampled_rows": n_rows,
                        "label_mean": float(finite.mean()) if finite.size else None,
                        "label_std": float(finite.std()) if finite.size else None,
                        "label_frac_positive": float((finite > 0).mean())
                        if finite.size
                        else None,
                        "n_distinct_labels": int(np.unique(finite).size),
                        "n_label_nonfinite": int(labels.size - finite.size),
                        "local_ctx_size": local_ctx_size,
                        "bfs_width": bfs_width,
                        "context_seed": context_seed,
                        "shuffle_seed": shuffle_seed,
                        "normalisation": "raw residual stream, no norm_out",
                    },
                    indent=2,
                )
            )
            n_done += 1

        gib = total_written / 2**30
        print(
            f"[{db}] {n_done} tasks, {n_skipped} under {min_rows} rows, "
            f"{db_written / 2**20:.0f} MiB | cumulative {gib:.2f} of an expected "
            f"{expected_gib:.2f} GiB, {(time.time() - t_start) / 60:.1f} min",
            flush=True,
        )
        assert gib <= expected_gib * 1.5 + 0.5, (
            f"wrote {gib:.2f} GiB against an expected {expected_gib:.2f} GiB; "
            f"stopping before the remaining dbs run the disk down"
        )

    handle.remove()
    print(
        f"done: {total_written / 2**30:.2f} GiB over {len(dbs)} dbs in "
        f"{(time.time() - t_start) / 60:.1f} min",
        flush=True,
    )
