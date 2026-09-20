import json
import shutil
from pathlib import Path

from expts.repaper.baselines.featurize_rt_taps import union_slots


def forecast_tasks(pre_dir: str, db: str, db_task_list: str) -> list:
    from rt.data import get_tasks, resolve_db_task_list

    meta = json.loads((Path(pre_dir).expanduser() / db / "meta.json").read_text())
    wanted = {name for d, name in resolve_db_task_list(db_task_list) if d == db}
    names = {
        t["name"]
        for t in meta["tasks"]
        if t.get("kind") == "forecast" and t["name"] in wanted
    }
    # get_tasks hands back a forecast task keyed by its task table and an
    # autocomplete task keyed by its *entity* table, and union_slots only knows
    # how to read the first kind. A name that is both would make the filter
    # below silently keep the wrong one.
    entities = {
        t["entity_table"] for t in meta["tasks"] if t.get("kind") == "autocomplete"
    }
    clash = names & entities
    assert not clash, f"{db}: {sorted(clash)} is both a forecast task and an entity table"
    return [
        t for t in get_tasks(pre_dir, [(db, n) for n in sorted(names)], ("train",))
        if t.table_name in names
    ]


def featurize_db_join(
    *,
    db: str,
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    ckpt: str,
    tap_unit: int,
    local_ctx_size: int,
    bfs_width: int,
    shuffle_seed: int,
    context_seed: int,
    batch_size: int,
    min_rows: int,
    max_rows: int,
    max_slots: int,
    expected_gib: float,
) -> None:
    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch

    from rt.data import RustlerDataset, process_batch
    from rt.model import load_rt_model

    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len

    device = "cuda"
    tasks = forecast_tasks(pre_dir, db, db_task_list)
    assert tasks, f"{db}: no forecast task in {db_task_list}"
    print(f"[{db}] {len(tasks)} forecast tasks", flush=True)

    net, config = load_rt_model(ckpt, device=device, compile=False)
    net = net.to(torch.bfloat16).eval()
    d_model = net.d_model

    # One hook, one unit: the Join dump is the final relational unit only, and
    # the residual stream is stored RAW -- no norm_out -- so it is the same
    # quantity the relbench rt_taps dump holds at tap 12 and one indexer reads
    # both.
    tap = {}
    handle = net.blocks[tap_unit - 1].register_forward_hook(
        lambda _m, _i, out: tap.__setitem__("x", out)
    )

    out_dir = Path(features_root).expanduser() / db / "rt_taps"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    skipped = []

    for task in sorted(tasks, key=lambda t: t.table_name):
        table = task.table_name
        bin_path = out_dir / f"{table}_taps.bin"
        meta_path = out_dir / f"{table}_meta.json"
        if bin_path.exists() and meta_path.exists():
            print(f"[{db}] {table}: already dumped, skipping", flush=True)
            continue

        min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
        if total_nodes < min_rows:
            skipped.append((table, f"{total_nodes} rows < {min_rows}"))
            continue

        names, slot_of, target_name = union_slots(pre_dir, db, table)
        n_slots = len(names)
        if n_slots > max_slots:
            skipped.append((table, f"{n_slots} union columns > {max_slots}"))
            continue

        # A uniform draw, not a prefix: node idx order is time order, so the
        # first max_rows rows of a big task table are its earliest horizons
        # rather than a sample of it. Seeded by the table so the same rows come
        # back if the dump is rerun.
        rng = np.random.default_rng(abs(hash((shuffle_seed, db, table))) % 2**63)
        if total_nodes <= max_rows:
            rows = np.arange(min_offset, min_offset + total_nodes)
        else:
            rows = np.sort(
                rng.choice(total_nodes, size=max_rows, replace=False) + min_offset
            )
        n_rows = len(rows)

        need = n_rows * n_slots * d_model * 2
        free = shutil.disk_usage(out_dir).free
        assert need * 1.1 < free, (
            f"[{db}] {table} needs {need / 2**30:.2f} GiB and only "
            f"{free / 2**30:.1f} GiB is free"
        )

        lut = torch.full((max(slot_of) + 1,), -1, dtype=torch.long, device=device)
        for idx, s in slot_of.items():
            lut[idx] = s

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
            db_cutoff=None,
            legacy_boolean=False,
        )

        filled = np.zeros(n_slots, dtype=np.int64)
        labels = np.empty(n_rows, dtype=np.float32)
        n_no_parent = 0
        tmp = bin_path.with_suffix(".bin.partial")
        with open(tmp, "wb") as fh, torch.inference_mode():
            for start in range(0, n_rows, batch_size):
                node_idxs = [int(v) for v in rows[start : start + batch_size]]
                B = len(node_idxs)
                tup = ds.sampler.batch_for_nodes_py(node_idxs, 0, local_ctx_size)
                batch = process_batch(tup, ds.d_text)
                batch.pop("batch_mask", None)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                tap.clear()
                net(batch, return_embeddings=True)

                keys = batch["col_name_idxs"].masked_fill(
                    batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
                )
                si = keys.argsort(dim=-1, stable=True)
                nidx = batch["node_idxs"].gather(1, si)
                pad = batch["is_padding"].gather(1, si)
                cols = batch["col_name_idxs"].gather(1, si)
                f2p = batch["f2p_nbr_idxs"].gather(
                    1, si.unsqueeze(-1).expand_as(batch["f2p_nbr_idxs"])
                )
                is_targets = batch["is_targets"].gather(1, si)
                per_row = is_targets.sum(dim=1)
                assert (per_row == 1).all(), (
                    f"{db}/{table}: expected one masked target cell per row, "
                    f"got counts {per_row.unique().tolist()}"
                )
                vals = batch["number_values"].gather(
                    1, si.unsqueeze(-1).expand_as(batch["number_values"])
                ).squeeze(-1)
                labels[start : start + B] = (
                    (vals * is_targets.to(vals.dtype)).sum(dim=1).float().cpu().numpy()
                )

                seeds = torch.tensor(node_idxs, device=device).view(B, 1)
                is_seed = (nidx == seeds) & ~pad
                seed_f2p = f2p.masked_fill(~is_seed.unsqueeze(-1), -1)
                is_parent = (
                    (nidx.unsqueeze(-1).unsqueeze(-1) == seed_f2p.unsqueeze(1))
                    .any(-1)
                    .any(-1)
                    & ~pad
                    & ~is_seed
                )
                n_no_parent += int((~is_parent.any(dim=1)).sum())

                safe = cols.long().clamp(0, lut.numel() - 1)
                slot_ids = torch.where(
                    cols.long() < lut.numel(), lut[safe], torch.full_like(safe, -1)
                )
                take = (is_seed | is_parent) & (slot_ids >= 0)

                buf = torch.full(
                    (B, n_slots, d_model),
                    float("nan"),
                    dtype=torch.bfloat16,
                    device=device,
                )
                b_idx, s_idx = take.nonzero(as_tuple=True)
                k_slot = slot_ids[b_idx, s_idx]
                buf[b_idx, k_slot, :] = tap["x"][b_idx, s_idx, :].bfloat16()
                filled += np.bincount(
                    k_slot.cpu().numpy(), minlength=n_slots
                ).astype(np.int64)

                fh.write(buf.view(torch.uint16).cpu().numpy().tobytes())

        tmp.replace(bin_path)
        np.savez(
            out_dir / f"{table}_rows.npz",
            node_idxs=rows.astype(np.int64),
            labels=labels,
        )
        written = bin_path.stat().st_size
        total_written += written
        finite = labels[np.isfinite(labels)]
        meta_path.write_text(
            json.dumps(
                {
                    "dtype": "bfloat16",
                    "shape": [n_rows, 1, n_slots, d_model],
                    "taps": [tap_unit],
                    "slots": names,
                    "target_slot": names.index(target_name),
                    "min_offset": min_offset,
                    "total_nodes": total_nodes,
                    "sampled_rows": n_rows,
                    "task_type": task.task_type,
                    "target_column": task.target_column,
                    "label_mean": float(finite.mean()) if finite.size else None,
                    "label_std": float(finite.std()) if finite.size else None,
                    "label_frac_positive": float((finite > 0).mean())
                    if finite.size
                    else None,
                    "n_label_nonfinite": int(labels.size - finite.size),
                    "slot_filled": filled.tolist(),
                    "dead_slots": [i for i, c in enumerate(filled) if c == 0],
                    "rows_without_parent": n_no_parent,
                    "local_ctx_size": local_ctx_size,
                    "bfs_width": bfs_width,
                    "context_seed": context_seed,
                    "shuffle_seed": shuffle_seed,
                    "normalisation": "raw residual stream, no norm_out",
                },
                indent=2,
            )
        )
        print(
            f"[{db}] {table}: {n_rows}/{total_nodes} rows x {n_slots} slots "
            f"-> {written / 2**30:.3f} GiB, {task.task_type}, "
            f"frac>0 {float((finite > 0).mean()) if finite.size else float('nan'):.4f}, "
            f"{n_no_parent} rows without a parent",
            flush=True,
        )

        gib = total_written / 2**30
        assert gib <= expected_gib * 1.5 + 0.5, (
            f"[{db}] wrote {gib:.2f} GiB against an expected {expected_gib:.2f} "
            f"GiB; stopping before the remaining tasks run the disk down"
        )

    handle.remove()
    print(
        f"[{db}] done: {total_written / 2**30:.2f} GiB, "
        f"{len(skipped)} tasks skipped",
        flush=True,
    )
    for table, why in skipped:
        print(f"[{db}]   skipped {table}: {why}", flush=True)
