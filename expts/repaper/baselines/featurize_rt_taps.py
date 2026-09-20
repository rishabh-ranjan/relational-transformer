import json
import shutil
from collections import defaultdict
from pathlib import Path


def union_slots(pre_dir: str, db: str, table: str) -> tuple[list[str], dict[int, int], str]:
    pre = Path(pre_dir).expanduser()
    by_table: dict[str, dict[str, int]] = defaultdict(dict)
    for key, idx in json.loads((pre / db / "column_index.json").read_text()).items():
        col, _, tbl = key.rpartition(" of ")
        by_table[tbl][col] = idx
    spec = {
        t["name"]: t
        for t in json.loads((pre / db / "meta.json").read_text())["tasks"]
    }[table]
    entity = spec["entity_table"]

    union = {f"{c} of {table}": i for c, i in by_table[table].items()}
    union |= {f"{c} of {entity}": i for c, i in by_table[entity].items()}
    names = sorted(union)
    slot_of = {union[n]: s for s, n in enumerate(names)}
    assert len(slot_of) == len(names), f"{db}/{table}: two union columns share an idx"
    return names, slot_of, f"{spec['target_col']} of {table}"


def featurize_db_taps(
    *,
    db: str,
    tables: list[str],
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    ckpt: str,
    tap_units: list[int],
    local_ctx_size: int,
    bfs_width: int,
    shuffle_seed: int,
    context_seed: int,
    db_cutoff: str | int | None,
    batch_size: int,
    expected_gib: float,
) -> None:
    import ml_dtypes
    import numpy as np
    import torch

    from rt.data import RustlerDataset, get_tasks, process_batch
    from rt.model import load_rt_model

    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len

    device = "cuda"
    tasks = {
        t.table_name: t
        for t in get_tasks(pre_dir, db_task_list, ("test",))
        if t.db_name == db and t.table_name in tables
    }
    missing = sorted(set(tables) - set(tasks))
    assert not missing, f"{db}: no task for {missing}"

    net, config = load_rt_model(ckpt, device=device, compile=False)
    net = net.to(torch.bfloat16).eval()

    # One forward pass yields every tap: a hook per block stashes that block's
    # output, so depth costs nothing beyond the i/o of writing it. Unit u is
    # blocks[u - 1]; the residual stream is stored RAW, with no norm_out, so a
    # tap is comparable to itself across depths and the choice of
    # normalisation stays with the consumer.
    taps: dict[int, "torch.Tensor"] = {}

    def make_hook(b):
        def hook(_mod, _inp, out):
            taps[b] = out

        return hook

    tap_blocks = [u - 1 for u in tap_units]
    handles = [net.blocks[b].register_forward_hook(make_hook(b)) for b in tap_blocks]

    out_dir = Path(features_root).expanduser() / db / "rt_taps"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for table in sorted(tasks):
        task = tasks[table]
        bin_path = out_dir / f"{table}_taps.bin"
        meta_path = out_dir / f"{table}_meta.json"
        if bin_path.exists() and meta_path.exists():
            print(f"[{db}] {table}: already dumped, skipping", flush=True)
            continue

        names, slot_of, target_name = union_slots(pre_dir, db, table)
        n_slots, n_taps = len(names), len(tap_units)
        min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
        d_model = net.d_model

        # Check the space before writing rather than after: the point of
        # tracking disk is to stop a table that would fill the filesystem, and
        # a check that runs once the bytes are down cannot do that.
        need = total_nodes * n_taps * n_slots * d_model * 2
        free = shutil.disk_usage(out_dir).free
        print(
            f"[{db}] {table}: {need / 2**30:.2f} GiB to write, "
            f"{free / 2**30:.1f} GiB free",
            flush=True,
        )
        assert need * 1.1 < free, (
            f"[{db}] {table} needs {need / 2**30:.2f} GiB and only "
            f"{free / 2**30:.1f} GiB is free"
        )

        # col_name_idx -> slot, as a lookup tensor so the mapping is one gather
        # rather than a python loop over cells.
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
            db_cutoff=db_cutoff,
            legacy_boolean=False,
        )

        filled = np.zeros(n_slots, dtype=np.int64)
        n_dup = 0
        n_no_parent = 0
        tmp = bin_path.with_suffix(".bin.partial")
        # Streamed, not accumulated: the old featurizer kept every chunk in
        # host ram and concatenated at the end, which at n_taps x n_slots is
        # hundreds of GiB on a large table.
        with open(tmp, "wb") as fh, torch.inference_mode():
            for start in range(0, total_nodes, batch_size):
                node_idxs = list(
                    range(
                        min_offset + start,
                        min_offset + min(start + batch_size, total_nodes),
                    )
                )
                B = len(node_idxs)
                tup = ds.sampler.batch_for_nodes_py(node_idxs, 0, local_ctx_size)
                batch = process_batch(tup, ds.d_text)
                batch.pop("batch_mask", None)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                taps.clear()
                net(batch, return_embeddings=True)

                # Everything below lives in the sorted frame, because that is
                # the frame forward() puts the blocks in and therefore the
                # frame the hook outputs are indexed by. Selecting unsorted and
                # indexing a sorted tensor is silently wrong, not a crash.
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

                seeds = torch.tensor(node_idxs, device=device).view(B, 1)
                is_seed = (nidx == seeds) & ~pad
                # The parent rows are the f2p targets of the seed row's own
                # cells -- the same `same_node | kv_in_f2p` the feat attention
                # mask uses, restricted to the seed.
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

                # Missing cells stay NaN: a column absent here is null in the
                # database (or its parent row does not exist), and NaN is what
                # tabpfn reads as missing.
                buf = torch.full(
                    (B, n_taps, n_slots, d_model),
                    float("nan"),
                    dtype=torch.bfloat16,
                    device=device,
                )
                b_idx, s_idx = take.nonzero(as_tuple=True)
                k_slot = slot_ids[b_idx, s_idx]
                for ti, blk in enumerate(tap_blocks):
                    buf[b_idx, ti, k_slot, :] = taps[blk][b_idx, s_idx, :].bfloat16()

                pairs = b_idx * n_slots + k_slot
                n_dup += int(pairs.numel() - torch.unique(pairs).numel())
                filled += np.bincount(
                    k_slot.cpu().numpy(), minlength=n_slots
                ).astype(np.int64)

                fh.write(buf.view(torch.uint16).cpu().numpy().tobytes())
                if (start // batch_size) % 20 == 0:
                    print(
                        f"[{db}] {table}: {start + B}/{total_nodes}", flush=True
                    )

        tmp.replace(bin_path)
        written = bin_path.stat().st_size
        total_written += written
        meta_path.write_text(
            json.dumps(
                {
                    "dtype": "bfloat16",
                    "shape": [total_nodes, n_taps, n_slots, d_model],
                    "taps": tap_units,
                    "slots": names,
                    "target_slot": names.index(target_name),
                    "min_offset": min_offset,
                    "total_nodes": total_nodes,
                    # Rows in which each slot was present. A slot at 0 is a
                    # key column: rt carries keys as edges, never as value
                    # cells, so those slots are all NaN by construction.
                    "slot_filled": filled.tolist(),
                    "dead_slots": [i for i, c in enumerate(filled) if c == 0],
                    "rows_without_parent": n_no_parent,
                    "duplicate_cell_writes": n_dup,
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
            f"[{db}] {table}: {total_nodes} rows x {n_taps} taps x {n_slots} slots "
            f"x {d_model} -> {written / 2**30:.2f} GiB, "
            f"{len(json.loads(meta_path.read_text())['dead_slots'])} dead slots, "
            f"{n_no_parent} rows without a parent, {n_dup} duplicate writes",
            flush=True,
        )

        # Cumulative, checked between tables so an overrun stops the next one
        # rather than being reported once everything is already on disk.
        gib = total_written / 2**30
        print(
            f"[{db}] cumulative {gib:.2f} GiB of an expected "
            f"{expected_gib:.2f} GiB",
            flush=True,
        )
        assert gib <= expected_gib * 1.5 + 0.5, (
            f"[{db}] wrote {gib:.2f} GiB against an expected {expected_gib:.2f} "
            f"GiB; stopping before the remaining tables run the disk down"
        )

    for h in handles:
        h.remove()
