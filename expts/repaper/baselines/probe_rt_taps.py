import json
from collections import defaultdict
from pathlib import Path


def probe(
    *,
    db: str,
    table: str,
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
    max_rows: int,
) -> None:
    import numpy as np
    import torch

    from rt.data import RustlerDataset, get_tasks, process_batch
    from rt.model import load_rt_model

    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len

    device = "cuda"
    (task,) = [
        t
        for t in get_tasks(pre_dir, db_task_list, ("test",))
        if t.db_name == db and t.table_name == table
    ]

    pre = Path(pre_dir).expanduser()
    col_index = json.loads((pre / db / "column_index.json").read_text())
    meta_tasks = {
        t["name"]: t for t in json.loads((pre / db / "meta.json").read_text())["tasks"]
    }
    spec = meta_tasks[table]
    entity = spec["entity_table"]

    by_table = defaultdict(dict)
    for key, idx in col_index.items():
        col, _, tbl = key.rpartition(" of ")
        by_table[tbl][col] = idx

    # The union: every column of the task table plus every column of the entity
    # table its f2p link reaches. The target column is kept -- its cell carries
    # the mask embedding, and its depth-12 representation is exactly the feature
    # the existing blob holds.
    union = {f"{c} of {table}": i for c, i in by_table[table].items()}
    union |= {f"{c} of {entity}": i for c, i in by_table[entity].items()}
    slots = {idx: s for s, (_, idx) in enumerate(sorted(union.items()))}
    names = [n for n, _ in sorted(union.items())]
    assert len(slots) == len(union), "two union columns share a col_name_idx"
    print(f"\n=== {db}/{table}  entity={entity}  n_union={len(union)}", flush=True)
    for n in names:
        print(f"    slot {slots[union[n]]:3d}  idx {union[n]:6d}  {n}", flush=True)

    net, config = load_rt_model(ckpt, device=device, compile=False)
    net = net.to(torch.bfloat16).eval()

    taps = {}
    tap_blocks = [0, 3, 7, 11]

    def make_hook(i):
        def hook(_mod, _inp, out):
            taps[i] = out

        return hook

    handles = [net.blocks[i].register_forward_hook(make_hook(i)) for i in tap_blocks]

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

    min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
    n_probe = min(max_rows, total_nodes)

    feat_dir = Path(features_root).expanduser() / db / "rt_features"
    blob_meta = json.loads((feat_dir / f"{table}_meta.json").read_text())
    blob = np.fromfile(feat_dir / f"{table}_vectors.bin", dtype=np.float32).reshape(
        blob_meta["total_nodes"], blob_meta["n_features"]
    )

    present = np.zeros(len(union), dtype=np.int64)
    n_seen = 0
    n_parent_rows, n_dup_cells, n_seed_mismatch = [], 0, 0
    max_gap = 0.0

    with torch.inference_mode():
        for start in range(0, n_probe, batch_size):
            node_idxs = list(
                range(min_offset + start, min_offset + min(start + batch_size, n_probe))
            )
            tup = ds.sampler.batch_for_nodes_py(node_idxs, 0, local_ctx_size)
            batch = process_batch(tup, ds.d_text)
            batch.pop("batch_mask", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            taps.clear()
            x_final = net(batch, return_embeddings=True)

            B = len(node_idxs)
            seeds = torch.tensor(node_idxs, device=device).view(B, 1)
            nidx = batch["node_idxs"]
            pad = batch["is_padding"]
            is_seed_cell = (nidx == seeds) & ~pad
            # Cross-check the two other ways the batch marks the seed row.
            if "bfs_depths" in batch:
                by_depth = (batch["bfs_depths"] == 0) & ~pad
                n_seed_mismatch += int((by_depth != is_seed_cell).any(dim=1).sum())

            # Parent rows: the f2p targets of the seed row's own cells.
            f2p = batch["f2p_nbr_idxs"]
            seed_f2p = f2p.masked_fill(~is_seed_cell.unsqueeze(-1), -1)
            is_parent_cell = (
                (nidx.unsqueeze(-1).unsqueeze(-1) == seed_f2p.unsqueeze(1))
                .any(-1)
                .any(-1)
                & ~pad
                & ~is_seed_cell
            )
            sel = is_seed_cell | is_parent_cell

            for b in range(B):
                cols = batch["col_name_idxs"][b][sel[b]].tolist()
                hit = {slots[c] for c in cols if c in slots}
                present[list(hit)] += 1
                n_dup_cells += len(cols) - len(set(cols))
                parents = nidx[b][is_parent_cell[b]].unique().numel()
                n_parent_rows.append(parents)
            n_seen += B

            # Equivalence: norm_out over the block-12 tap must reproduce the
            # released feature exactly, since that is what forward() returns.
            tap12 = net.norm_out(taps[11])
            gap = (tap12.float() - x_final.float()).abs().max().item()
            max_gap = max(max_gap, gap)

            sort_keys = batch["col_name_idxs"].masked_fill(
                pad, torch.iinfo(batch["col_name_idxs"].dtype).max
            )
            si = sort_keys.argsort(dim=-1, stable=True)
            sorted_is_targets = batch["is_targets"].gather(1, si)
            mine = tap12.gather(
                1, si.unsqueeze(-1).expand(-1, -1, tap12.shape[-1])
            )[sorted_is_targets].float().cpu().numpy()
            theirs = blob[start : start + B]
            d = np.abs(mine - theirs).max()
            print(
                f"  rows {start}-{start + B}: |norm_out(tap12)-blob| max {d:.3e}",
                flush=True,
            )

    for h in handles:
        h.remove()

    print(f"\n  seed-cell definitions disagree on {n_seed_mismatch}/{n_seen} rows")
    print(f"  norm_out(tap12) vs forward() max abs gap: {max_gap:.3e}")
    print(f"  duplicate col_name_idx cells in selection: {n_dup_cells}")
    pr = np.array(n_parent_rows)
    print(f"  parent rows per seed: min {pr.min()} max {pr.max()} mean {pr.mean():.2f}")
    print(f"\n  coverage over {n_seen} rows (fraction of rows where the column appears):")
    for s, n in enumerate(names):
        print(f"    {present[s] / n_seen:6.3f}  {n}")
    print(f"  mean coverage {present.mean() / n_seen:.3f}")
