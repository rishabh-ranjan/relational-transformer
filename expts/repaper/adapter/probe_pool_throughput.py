import json
import os
import time
from pathlib import Path

LOCAL_CTX = 256
BFS_WIDTH = 32


def rss_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def resolve_tasks(pre_dir, pairs):
    from rt.data import get_tasks

    out = []
    for db, name in pairs:
        got = get_tasks(pre_dir, [(db, name)], ("train",))
        assert got, f"{db}/{name} does not resolve"
        out.append(got[0])
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


def sample(ds, dataset_idx, nodes):
    from rt.data import process_batch

    t0 = time.perf_counter()
    tup = ds.sampler.batch_for_nodes_py([int(v) for v in nodes], dataset_idx, LOCAL_CTX)
    t1 = time.perf_counter()
    batch = process_batch(tup, ds.d_text)
    batch.pop("batch_mask", None)
    t2 = time.perf_counter()
    return batch, t1 - t0, t2 - t1


def worker(pre_dir, pairs, config, jobs, q, done):
    tasks = resolve_tasks(pre_dir, pairs)
    ds = make_dataset(tasks, pre_dir, config, mmap_populate=True)
    for dataset_idx, nodes in jobs:
        batch, _, _ = sample(ds, dataset_idx, nodes)
        for v in batch.values():
            v.share_memory_()
        q.put(batch)
    q.put(None)
    done.wait()


def consume(model, q, n_workers, cs, device, pinned):
    import torch

    copy_stream = torch.cuda.Stream()
    compute = torch.cuda.current_stream()

    def chunks():
        finished = 0
        while finished < n_workers:
            b = q.get()
            if b is None:
                finished += 1
                continue
            n = b["node_idxs"].shape[0]
            for s in range(0, n, cs):
                yield {k: v[s : s + cs] for k, v in b.items()}

    def stage(host):
        if pinned:
            host = {k: v.pin_memory() for k, v in host.items()}
        with torch.cuda.stream(copy_stream):
            dev = {k: v.to(device, non_blocking=True) for k, v in host.items()}
        ev = torch.cuda.Event()
        ev.record(copy_stream)
        return host, dev, ev

    rows = 0
    t_first = None
    it = chunks()
    prev = stage(next(it))
    t_first = time.perf_counter()
    with torch.inference_mode():
        for host in it:
            nxt = stage(host)
            _, dev, ev = prev
            compute.wait_event(ev)
            for v in dev.values():
                v.record_stream(compute)
            model(dev, return_embeddings=True)
            rows += dev["node_idxs"].shape[0]
            prev = nxt
        _, dev, ev = prev
        compute.wait_event(ev)
        model(dev, return_embeddings=True)
        rows += dev["node_idxs"].shape[0]
    torch.cuda.synchronize()
    return rows, t_first, time.perf_counter()


def run_overlap(report, dump, log, model, cs, plan, pairs, pre_dir, config, n_overlap_workers, device):
    import multiprocessing

    report["overlap"] = {"chunk": cs}
    ctx = multiprocessing.get_context("spawn")
    for nw in n_overlap_workers:
        for pinned in (False, True):
            jobs = [(di, nodes) for di, _e, _idx, nodes in plan]
            shards = [jobs[i::nw] for i in range(nw)]
            q = ctx.Queue(maxsize=4 * nw)
            done = ctx.Event()
            procs = [
                ctx.Process(target=worker, args=(pre_dir, pairs, config, sh, q, done), daemon=True)
                for sh in shards
            ]
            tp = time.perf_counter()
            for p in procs:
                p.start()
            rows, t_first, t_end = consume(model, q, nw, cs, device, pinned)
            done.set()
            for p in procs:
                p.join()
            key = f"workers_{nw}_{'pinned' if pinned else 'unpinned'}"
            report["overlap"][key] = {
                "rows_per_s_after_first_chunk": rows / (t_end - t_first),
                "startup_s": t_first - tp,
            }
            log(f"overlap {key}: {report['overlap'][key]}")
            dump()


def main(
    *,
    features_root: str,
    pre_dir: str,
    ckpt: str,
    out_dir: str,
    n_tasks: int,
    rows_per_task: int,
    chunk_sizes: list[int],
    n_overlap_workers: list[int],
    overlap_chunk: int,
    overlap_only: bool,
    seed: int,
) -> None:
    import ml_dtypes
    import numpy as np
    import torch

    from expts.repaper.adapter.pool_head import PoolHead
    from expts.repaper.adapter.train_adapter import load_index
    from rt.model import load_rt_model

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "host": os.uname().nodename,
        "cpus": len(os.sched_getaffinity(0)),
        "gpu": torch.cuda.get_device_name(0),
        "n_tasks": n_tasks,
        "rows_per_task": rows_per_task,
    }

    def dump():
        (out / "report.json").write_text(json.dumps(report, indent=2))

    def log(msg):
        print(msg, flush=True)

    entries = [e for e in load_index(features_root, rows_per_task)]
    w = np.array([e["weight"] for e in entries])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(entries), size=n_tasks, replace=False, p=w / w.sum())
    chosen = [entries[i] for i in pick]
    pairs = [(e["db"], e["task"]) for e in chosen]
    report["tasks"] = [f"{d}/{t}" for d, t in pairs]
    report["n_dbs"] = len({d for d, _ in pairs})
    log(f"{n_tasks} tasks over {report['n_dbs']} dbs")

    device = "cuda"
    net, config = load_rt_model(ckpt, device=device, compile=False)
    net = net.to(torch.bfloat16).eval()

    plan = []
    for dataset_idx, e in enumerate(chosen):
        rows = np.load(e["rows"])
        idx = np.sort(rng.choice(e["n"], size=rows_per_task, replace=False))
        plan.append((dataset_idx, e, idx, rows["node_idxs"][idx]))

    if overlap_only:
        net_c, _ = load_rt_model(ckpt, device=device, compile=True)
        net_c = net_c.to(torch.bfloat16).eval()
        run_overlap(report, dump, log, net_c, overlap_chunk, plan, pairs, pre_dir, config, n_overlap_workers, device)
        log("done")
        return

    rss0 = rss_gib()
    t0 = time.perf_counter()
    tasks = resolve_tasks(pre_dir, pairs)
    ds = make_dataset(tasks, pre_dir, config, mmap_populate=True)
    report["sampler_setup_s"] = time.perf_counter() - t0
    report["sampler_rss_gib"] = rss_gib() - rss0
    log(f"sampler setup {report['sampler_setup_s']:.1f}s, +{report['sampler_rss_gib']:.1f} GiB rss")

    batches = []
    samp_s = proc_s = 0.0
    sub_total = 0
    per_task = []
    t_host = 0.0
    for dataset_idx, e, idx, nodes in plan:
        batch, a, b = sample(ds, dataset_idx, nodes)
        samp_s += a
        proc_s += b
        batches.append(batch)
    n_rows = n_tasks * rows_per_task
    report["sample_rows_per_s"] = n_rows / samp_s
    report["process_batch_rows_per_s"] = n_rows / proc_s
    report["host_bytes_per_row"] = sum(
        v.numel() * v.element_size() for v in batches[0].values()
    ) / rows_per_task
    log(
        f"sampling {report['sample_rows_per_s']:.0f} rows/s, process_batch "
        f"{report['process_batch_rows_per_s']:.0f} rows/s, "
        f"{report['host_bytes_per_row'] / 2**20:.2f} MiB/row on host"
    )

    t0 = time.perf_counter()
    for dataset_idx, e, idx, nodes in plan[:4]:
        sample(ds, dataset_idx, nodes)
    report["sample_warm_rows_per_s"] = 4 * rows_per_task / (time.perf_counter() - t0)
    log(f"sampling, second pass over 4 tasks: {report['sample_warm_rows_per_s']:.0f} rows/s")
    dump()

    def sorted_frame(batch):
        keys = batch["col_name_idxs"].masked_fill(
            batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
        )
        si = keys.argsort(dim=-1, stable=True)
        return {
            k: batch[k].gather(1, si)
            for k in ("is_targets", "is_padding", "node_idxs")
        }

    with torch.inference_mode():
        tok_counts = []
        for (dataset_idx, e, idx, nodes), batch in zip(plan, batches):
            gb = {k: v.to(device) for k, v in batch.items()}
            x = net(gb, return_embeddings=True)
            sf = sorted_frame(gb)
            tgt = sf["is_targets"]
            assert (tgt.sum(1) == 1).all()
            got = (sf["node_idxs"] * tgt).sum(1).cpu().numpy()
            ok = got == nodes
            sub_total += int((~ok).sum())
            emb = x[tgt].float().cpu().numpy()
            ref = np.memmap(e["bin"], dtype=ml_dtypes.bfloat16, mode="r", shape=(e["n"], e["d"]))
            ref = np.asarray(ref[idx]).astype(np.float32)
            dev = np.linalg.norm(emb - ref, axis=1) / np.linalg.norm(ref, axis=1)
            dev = dev[ok]
            per_task.append(
                {
                    "task": f"{e['db']}/{e['task']}",
                    "substituted": int((~ok).sum()),
                    "dev_p50": float(np.median(dev)),
                    "dev_p99": float(np.quantile(dev, 0.99)),
                    "dev_max": float(dev.max()),
                }
            )
            tok_counts.append((~sf["is_padding"]).sum(1).cpu().numpy())
        tok = np.concatenate(tok_counts)
        report["tokens_per_row"] = {
            q: float(np.quantile(tok, p))
            for q, p in (("p1", 0.01), ("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("max", 1.0))
        }
        report["tokens_per_row"]["mean"] = float(tok.mean())
        report["target_equivalence"] = per_task
        report["substituted_total"] = sub_total
        p99 = [t["dev_p99"] for t in per_task]
        log(
            f"target-cell vs features_join_u12: worst task p99 {max(p99):.2e}, "
            f"{sub_total} substituted; tokens/row {report['tokens_per_row']}"
        )
        dump()

        cat = {
            k: torch.cat([b[k] for b in batches[:4]], 0) for k in batches[0]
        }
        sep = torch.cat(
            [net({k: v.to(device) for k, v in b.items()}, return_embeddings=True) for b in batches[:4]]
        )
        mix = net({k: v.to(device) for k, v in cat.items()}, return_embeddings=True)
        report["cross_task_batch_max_abs_diff"] = float((sep.float() - mix.float()).abs().max())
        log(f"4 tasks in one forward vs separately: max abs diff {report['cross_task_batch_max_abs_diff']:.3e}")

        allb = {k: torch.cat([b[k] for b in batches], 0) for k in batches[0]}
        report["forward"] = {}
        for mode in ("eager", "compile"):
            if mode == "compile":
                try:
                    net_c, _ = load_rt_model(ckpt, device=device, compile=True)
                    net_c = net_c.to(torch.bfloat16).eval()
                except Exception as exc:
                    report["forward"]["compile_error"] = repr(exc)
                    break
                model = net_c
            else:
                model = net
            for cs in chunk_sizes:
                key = f"{mode}_{cs}"
                try:
                    torch.cuda.reset_peak_memory_stats()
                    first = {k: v[:cs].to(device) for k, v in allb.items()}
                    tw = time.perf_counter()
                    model(first, return_embeddings=True)
                    torch.cuda.synchronize()
                    warm_s = time.perf_counter() - tw
                    n = (n_rows // cs) * cs
                    h2d_s = fwd_s = 0.0
                    for s in range(0, n, cs):
                        th = time.perf_counter()
                        gb = {k: v[s : s + cs].to(device, non_blocking=True) for k, v in allb.items()}
                        torch.cuda.synchronize()
                        tf = time.perf_counter()
                        model(gb, return_embeddings=True)
                        torch.cuda.synchronize()
                        h2d_s += tf - th
                        fwd_s += time.perf_counter() - tf
                    report["forward"][key] = {
                        "rows_per_s_fwd": n / fwd_s,
                        "rows_per_s_h2d": n / h2d_s,
                        "first_call_s": warm_s,
                        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                except torch.OutOfMemoryError:
                    report["forward"][key] = "OOM"
                    torch.cuda.empty_cache()
                log(f"forward {key}: {report['forward'][key]}")
                dump()

    head = PoolHead(512, 64).to(device)
    report["head"] = {}
    for rows in (850 * 8, 850 * 32):
        key = f"rows_{rows}"
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(rows, LOCAL_CTX, 512, device=device, dtype=torch.bfloat16)
            pad = torch.zeros(rows, LOCAL_CTX, dtype=torch.bool, device=device)
            base = torch.cuda.max_memory_allocated()
            ts = []
            for _ in range(4):
                torch.cuda.synchronize()
                t = time.perf_counter()
                head(x, pad).sum().backward()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            report["head"][key] = {
                "fwd_bwd_s": float(np.median(ts[1:])),
                "peak_over_input_gib": (torch.cuda.max_memory_allocated() - base) / 2**30,
            }
            del x, pad
        except torch.OutOfMemoryError:
            report["head"][key] = "OOM"
        torch.cuda.empty_cache()
        log(f"head {key}: {report['head'][key]}")
    dump()

    del batches, allb
    run_overlap(report, dump, log, net_c, overlap_chunk, plan, pairs, pre_dir, config, n_overlap_workers, device)
    log("done")
