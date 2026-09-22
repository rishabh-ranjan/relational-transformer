import json
import os
import time
from pathlib import Path

LOCAL_CTX = 256
BFS_WIDTH = 32


def mem_gib():
    out = {}
    for line in open("/proc/self/status"):
        k = line.split(":")[0]
        if k in ("VmRSS", "RssAnon", "RssFile"):
            out[k] = int(line.split()[1]) / 2**20
    return out


def make_dataset(tasks, pre_dir, config, mmap_populate):
    from rt.data import RustlerDataset

    return RustlerDataset(
        tasks=tasks,
        pre_dir=pre_dir,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_size_list=[LOCAL_CTX],
        bfs_width_list=[BFS_WIDTH],
        num_walks=0,
        walk_length=0,
        prefer_latest_list=[False],
        mask_prob_max=0.0,
        embedder=config["embedder"],
        d_text=config["d_text"],
        shuffle_seed=0,
        context_seed=0,
        items_per_task=10_000_000,
        quiet=True,
        ignore_data_errors=False,
        mmap_populate=mmap_populate,
        timeout_per_item=60.0,
        vector_db_path=None,
        db_cutoff=None,
        legacy_boolean=False,
    )


def sample_rows(ds, dataset_idx, nodes):
    from rt.data import process_batch

    tup = ds.sampler.batch_for_nodes_py([int(v) for v in nodes], dataset_idx, LOCAL_CTX)
    batch = process_batch(tup, ds.d_text)
    batch.pop("batch_mask", None)
    return batch


def sorted_frame(batch, keys):
    import torch

    k = batch["col_name_idxs"].masked_fill(
        batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
    )
    si = k.argsort(dim=-1, stable=True)
    return {name: batch[name].gather(1, si) for name in keys}


def full_mixture(report, dump, log, features_root, join_pre_dir, config, n_sample_tasks, rows_per_sample, populate_dbs, seed):
    import numpy as np

    from expts.repaper.adapter.train_adapter import load_index
    from rt.data import get_tasks

    entries = load_index(features_root, 512)
    t0 = time.perf_counter()
    tasks = []
    for e in entries:
        got = get_tasks(join_pre_dir, [(e["db"], e["task"])], ("train",))
        assert got, f"{e['db']}/{e['task']} does not resolve"
        tasks.append(got[0])
    sec = {"n_tasks": len(tasks), "n_dbs": len({e["db"] for e in entries})}
    sec["resolve_tasks_s"] = time.perf_counter() - t0
    log(f"resolved {len(tasks)} tasks over {sec['n_dbs']} dbs in {sec['resolve_tasks_s']:.0f}s")

    m0 = mem_gib()
    t0 = time.perf_counter()
    ds = make_dataset(tasks, join_pre_dir, config, mmap_populate=False)
    sec["setup_s_no_populate"] = time.perf_counter() - t0
    m1 = mem_gib()
    sec["mem_after_setup_gib"] = {k: m1[k] - m0[k] for k in m1}
    log(f"full-mixture sampler, no populate: {sec['setup_s_no_populate']:.0f}s, mem {sec['mem_after_setup_gib']}")
    report["full_mixture"] = sec
    dump()

    rng = np.random.default_rng(seed)
    w = np.array([e["weight"] for e in entries])
    pick = rng.choice(len(entries), size=n_sample_tasks, replace=False, p=w / w.sum())
    plan = []
    for i in pick:
        nodes = np.load(entries[i]["rows"])["node_idxs"]
        plan.append((int(i), np.sort(rng.choice(nodes, size=min(rows_per_sample, len(nodes)), replace=False))))
    for tag in ("cold", "warm"):
        t0 = time.perf_counter()
        n = sub = 0
        for i, nodes in plan:
            b = sample_rows(ds, i, nodes)
            sf = sorted_frame(b, ("is_targets", "node_idxs"))
            got = (sf["node_idxs"] * sf["is_targets"]).sum(1).numpy()
            sub += int((got != nodes).sum())
            n += len(nodes)
        dt = time.perf_counter() - t0
        sec[f"sample_{tag}"] = {"rows": n, "rows_per_s": n / dt, "substituted": sub}
        log(f"full-mixture sampling {tag}: {n / dt:.0f} rows/s over {n_sample_tasks} tasks, {sub} substituted")
    sec["mem_after_sampling_gib"] = {k: v - m0[k] for k, v in mem_gib().items()}
    log(f"mem after sampling {sec['mem_after_sampling_gib']}")
    dump()
    del ds

    dbs = sorted({e["db"] for e in entries})
    sub_dbs = [dbs[i] for i in rng.choice(len(dbs), size=min(populate_dbs, len(dbs)), replace=False)]
    sub_tasks = [t for t, e in zip(tasks, entries) if e["db"] in set(sub_dbs)]
    disk = 0
    for db in sub_dbs:
        for p in (Path(join_pre_dir).expanduser() / db).rglob("*.rkyv"):
            disk += p.stat().st_size
    m0 = mem_gib()
    t0 = time.perf_counter()
    ds = make_dataset(sub_tasks, join_pre_dir, config, mmap_populate=True)
    pop = {
        "n_dbs": len(sub_dbs),
        "n_tasks": len(sub_tasks),
        "setup_s": time.perf_counter() - t0,
        "mem_gib": {k: v - m0[k] for k, v in mem_gib().items()},
        "rkyv_apparent_gib": disk / 2**30,
    }
    pop["extrapolated_setup_s_all_dbs"] = pop["setup_s"] * len(dbs) / len(sub_dbs)
    pop["extrapolated_rss_gib_all_dbs"] = pop["mem_gib"]["VmRSS"] * len(dbs) / len(sub_dbs)
    sec["populate_subset"] = pop
    log(f"populate on {len(sub_dbs)} dbs: {pop}")
    dump()
    del ds


def leakage(report, dump, log, relbench_pre_dir, db_task_list, config, max_val_rows, seed):
    import numpy as np

    from expts.repaper.baselines.rel2tab.featurizer import get_table_splits, load_table_info
    from rt.data import get_tasks
    from rt.data.resolve import get_column_index

    tasks = get_tasks(relbench_pre_dir, db_task_list, ("test",))
    rng = np.random.default_rng(seed)
    out = []
    for task in tasks:
        db, table = task.db_name, task.table_name
        splits = get_table_splits(load_table_info(relbench_pre_dir, db), table)
        ranges = {s: (v["node_idx_offset"], v["node_idx_offset"] + v["num_nodes"]) for s, v in splits.items()}
        lo, hi = ranges["val"]
        nodes = np.arange(lo, hi)
        if len(nodes) > max_val_rows:
            nodes = np.sort(rng.choice(nodes, size=max_val_rows, replace=False))
        tgt_col = get_column_index(task.target_column, table, db, relbench_pre_dir)
        ds = make_dataset([task], relbench_pre_dir, config, mmap_populate=True)

        def split_of(v):
            res = np.full(v.shape, "other", dtype=object)
            for s, (a, b) in ranges.items():
                res[(v >= a) & (v < b)] = s
            return res

        counts = {"rows": 0, "substituted": 0, "target_col_mismatch": 0}
        cell_counts = {}
        rows_with = {}
        for s in range(0, len(nodes), 512):
            chunk = nodes[s : s + 512]
            b = sample_rows(ds, 0, chunk)
            sf = sorted_frame(
                b,
                ("is_targets", "node_idxs", "is_padding", "is_task_nodes", "col_name_idxs", "timestamps", "table_name_idxs"),
            )
            it = sf["is_targets"].numpy()
            nd = sf["node_idxs"].numpy()
            seed_node = (nd * it).sum(1)
            ok = seed_node == chunk
            counts["rows"] += len(chunk)
            counts["substituted"] += int((~ok).sum())
            tcol = (sf["col_name_idxs"].numpy() * it).sum(1)
            counts["target_col_mismatch"] += int((tcol != tgt_col).sum())
            seed_ts = (sf["timestamps"].numpy() * it).sum(1)
            live = ~sf["is_padding"].numpy() & sf["is_task_nodes"].numpy() & ~it
            other = live & (nd != seed_node[:, None])
            label_cell = other & (sf["col_name_idxs"].numpy() == tgt_col)
            sp = split_of(nd)
            ts = sf["timestamps"].numpy()
            later = ts >= seed_ts[:, None]
            for kind, mask in (("task_cells", other), ("label_cells", label_cell)):
                for split in ("train", "val", "test", "other"):
                    m = mask & (sp == split)
                    cell_counts[f"{kind}_{split}"] = cell_counts.get(f"{kind}_{split}", 0) + int(m.sum())
                    rows_with[f"{kind}_{split}"] = rows_with.get(f"{kind}_{split}", 0) + int(m.any(1).sum())
                m = mask & later
                cell_counts[f"{kind}_ts_ge_seed"] = cell_counts.get(f"{kind}_ts_ge_seed", 0) + int(m.sum())
                rows_with[f"{kind}_ts_ge_seed"] = rows_with.get(f"{kind}_ts_ge_seed", 0) + int(m.any(1).sum())
            same_entity_self = live & (nd == seed_node[:, None]) & (sf["col_name_idxs"].numpy() == tgt_col)
            counts["own_label_visible"] = counts.get("own_label_visible", 0) + int(same_entity_self.sum())
        rec = {"task": f"{db}/{table}", "val_rows_total": int(hi - lo), **counts, "cells": cell_counts, "rows_with": rows_with}
        out.append(rec)
        log(
            f"leakage {db}/{table}: {counts}, label cells by split "
            + str({k: v for k, v in cell_counts.items() if k.startswith("label")})
        )
        report["leakage"] = out
        dump()
        del ds


def main(
    *,
    features_root: str,
    join_pre_dir: str,
    relbench_pre_dir: str,
    db_task_list: str,
    ckpt: str,
    out_dir: str,
    n_sample_tasks: int,
    rows_per_sample: int,
    populate_dbs: int,
    max_val_rows: int,
    seed: int,
) -> None:
    from rt.model.checkpoints import resolve_checkpoint

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report = {"host": os.uname().nodename, "cpus": len(os.sched_getaffinity(0))}

    def dump():
        (out / "report.json").write_text(json.dumps(report, indent=2))

    def log(msg):
        print(msg, flush=True)

    config, _ = resolve_checkpoint(ckpt)
    log(f"embedder {config['embedder']} d_text {config['d_text']}")

    leakage(report, dump, log, relbench_pre_dir, db_task_list, config, max_val_rows, seed)
    full_mixture(report, dump, log, features_root, join_pre_dir, config, n_sample_tasks, rows_per_sample, populate_dbs, seed)
    log("done")
