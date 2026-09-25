import json
import math
import os
import signal
import time
import zlib
from pathlib import Path

CHUNK = 1024


def embed_rows(net, ds, didx, nodes, tokens, pad, labels, colkeys, need, device):
    import numpy as np
    import torch

    from expts.repaper.adapter.pool_head import cell_keys
    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX
    from rt.data import process_batch

    filled = n_sub = 0
    samp_s = fwd_s = 0.0
    for start in range(0, len(nodes), CHUNK):
        if filled == need:
            break
        chunk = np.asarray(nodes[start : start + CHUNK])
        n_real = len(chunk)
        if n_real < CHUNK:
            chunk = np.concatenate([chunk, np.repeat(chunk[:1], CHUNK - n_real)])
        t = time.perf_counter()
        tup = ds.sampler.batch_for_nodes_py([int(v) for v in chunk], didx, LOCAL_CTX)
        batch = process_batch(tup, ds.d_text)
        batch.pop("batch_mask", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        samp_s += time.perf_counter() - t
        t = time.perf_counter()
        x = net(batch, return_embeddings=True)
        keys = batch["col_name_idxs"].masked_fill(
            batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
        )
        si = keys.argsort(dim=-1, stable=True)
        is_t = batch["is_targets"].gather(1, si)
        assert (is_t.sum(1) == 1).all()
        got = (batch["node_idxs"].gather(1, si) * is_t).sum(1)
        ok = got == torch.as_tensor(chunk, device=device, dtype=got.dtype)
        ok[n_real:] = False
        n_sub += n_real - int(ok.sum())
        vals = batch["number_values"].gather(
            1, si.unsqueeze(-1).expand_as(batch["number_values"])
        ).squeeze(-1)
        lab = (vals * is_t.to(vals.dtype)).sum(1).float()
        take = min(int(ok.sum()), need - filled)
        keep = ok.nonzero().squeeze(1)[:take]
        tokens[filled : filled + take] = x[keep]
        pad[filled : filled + take] = batch["is_padding"].gather(1, si)[keep]
        labels[filled : filled + take] = lab[keep]
        colkeys[filled : filled + take] = cell_keys(
            batch["col_name_idxs"].gather(1, si),
            batch["table_name_idxs"].gather(1, si),
            is_t,
            batch["is_padding"].gather(1, si),
        )[keep]
        filled += take
        torch.cuda.synchronize(device)
        fwd_s += time.perf_counter() - t
    return filled, n_sub, samp_s, fwd_s


def colnorm_slice(cn, i, j):
    return None if cn is None else (cn[0][i:j], cn[1], cn[2])


def dump_spike(path, head, h, g, gh, y, pad, n_ctx, meta):
    import numpy as np
    import torch

    with torch.no_grad():
        u = head.norm(h, n_ctx) if head.swiglu_norm == "context" else head.norm(h)
        pre1, pre3 = head.w1(u), head.w3(u)
        hid = torch.nn.functional.silu(pre1) * pre3
        f = h + head.w2(hid)
        ctx_h = h[:n_ctx]
        arrs = {
            "h": h, "u": u, "pre1": pre1, "pre3": pre3, "hid": hid, "f": f, "dL_df": g, "dL_dh": gh, "y": y,
            "h_ctx_mean": ctx_h.mean(0), "h_ctx_std": ctx_h.std(0, unbiased=False), "row_live": (~pad).sum(dim=1),
        }
        arrs = {k: v.detach().float().cpu().numpy() for k, v in arrs.items()}
        grads = {f"grad/{n}": p.grad.detach().float().cpu().numpy() for n, p in head.named_parameters()}
    np.savez_compressed(path, **arrs, **grads, w2_grad=grads["grad/w2.weight"], n_ctx=np.array(n_ctx), meta=np.array(json.dumps(meta)))


def dedup_context(h, y, n_ctx):
    import torch

    key = torch.cat([h[:n_ctx].float().contiguous().view(torch.int32), y[:n_ctx, None].float().contiguous().view(torch.int32)], dim=1)
    _, inv = torch.unique(key, dim=0, return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), n_ctx, dtype=torch.long, device=h.device)
    first.scatter_reduce_(0, inv, torch.arange(n_ctx, device=h.device), reduce="amin")
    kept = first.sort().values
    return torch.cat([kept, torch.arange(n_ctx, len(h), device=h.device)]), len(kept)


def head_features(head, tokens, pad, n, n_ctx, chunk, cn):
    import torch

    with torch.no_grad():
        h = torch.cat(
            [
                head.pool(tokens[i : min(i + chunk, n)], pad[i : min(i + chunk, n)], colnorm_slice(cn, i, min(i + chunk, n)))
                for i in range(0, n, chunk)
            ]
        )
        return h, head.mix(h, n_ctx)


def build_relbench(pre_dir, db_task_list, config, n_ctx_max, n_query_max, seed):
    import numpy as np

    from expts.repaper.adapter.probe_pool_throughput import make_dataset
    from expts.repaper.baselines.rel2tab.featurizer import get_table_splits, load_table_info
    from rt.data import get_tasks

    by_db = {}
    for t in get_tasks(pre_dir, db_task_list, ("test",)):
        by_db.setdefault(t.db_name, []).append(t)
    evals = []
    for db, ts in sorted(by_db.items()):
        ds = make_dataset(ts, pre_dir, config, mmap_populate=False)
        info = load_table_info(pre_dir, db)
        for didx, t in enumerate(ts):
            sp = get_table_splits(info, t.table_name)
            r = np.random.default_rng([seed, zlib.crc32(f"{db}/{t.table_name}".encode())])

            def draw(s, k):
                lo, n = sp[s]["node_idx_offset"], sp[s]["num_nodes"]
                return lo + np.sort(r.choice(n, size=min(n, k), replace=False))

            evals.append(
                {
                    "name": f"{db}/{t.table_name}",
                    "kind": t.task_type,
                    "ds": ds,
                    "didx": didx,
                    "ctx": draw("train", n_ctx_max),
                    "query": draw("val", n_query_max),
                    "train_lo": sp["train"]["node_idx_offset"],
                    "train_n": sp["train"]["num_nodes"],
                }
            )
    return evals


def main(
    *,
    join_pre_dir: str,
    relbench_pre_dir: str,
    relbench_task_list: str,
    task_list: str,
    train_source: str,
    ckpt: str,
    tabpfn_dir: str,
    stats_path: str,
    out_dir: str,
    n_ctx: int,
    n_query: int,
    relbench_n_ctx: int,
    relbench_n_query: int,
    n_queries: int,
    swiglu_norm: str,
    input_norm: str,
    signsoftmax_temp: float,
    colnorm_tau: float | None,
    live_scale: bool,
    signsoftmax_scale_by_dim: bool,
    head_chunk: int,
    tabpfn_device: str,
    tabpfn_precision: str,
    offload_cells: bool,
    total_steps: int,
    lr: float,
    lr_min: float,
    wd: float,
    warmup_steps: int,
    grad_norm_max: float,
    swa_momentum: float,
    eval_every: int,
    save_every: int,
    resume_save_mins: float,
    spike_dump_gnorm: float | None,
    dedup_ctx: bool,
    grad_accum: int,
    task_batch: int,
    seed: int,
    run_id: str,
    run_name: str,
    project: str,
    entity: str,
    wandb_disabled: bool,
) -> None:
    params = dict(locals())
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import copy

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from expts.repaper.adapter.pool_head import PoolHead, column_stats
    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX, make_dataset
    from expts.repaper.adapter.train_adapter import batched_outputs, make_ests
    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len
    from rt.data import get_tasks
    from rt.model import load_rt_model
    from rt.train._train import seed_everything
    from rt.train.swa import SwaState

    dev0, dev1 = "cuda:0", tabpfn_device
    assert torch.cuda.device_count() > int(dev1.split(":")[1]), torch.cuda.device_count()
    need = n_ctx + n_query
    cap = max(task_batch * need, relbench_n_ctx + relbench_n_query)
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if spike_dump_gnorm is not None:
        (out / "spikes").mkdir(exist_ok=True)
    t_start = time.time()

    def log(msg):
        print(f"[{(time.time() - t_start) / 60:7.1f} min] {msg}", flush=True)

    use_wandb = not wandb_disabled
    if use_wandb:
        import wandb

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            id=run_id,
            group=run_id,
            resume="allow",
            config=params,
            dir=str(out),
            settings=wandb.Settings(console_multipart=True, console_chunk_max_seconds=60),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    with torch.cuda.device(dev0):
        t = time.perf_counter()
        net, config = load_rt_model(ckpt, device=dev0, compile=True)
        net = net.to(torch.bfloat16).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        d_model = net.d_model
    log(f"rt-j loaded, compile=True, {time.perf_counter() - t:.0f}s")

    assert train_source in ("join", "relbench"), train_source
    relbench = build_relbench(relbench_pre_dir, relbench_task_list, config, relbench_n_ctx, relbench_n_query, seed)
    log(f"relbench: {len(relbench)} eval tasks")
    if train_source == "join":
        rows = json.loads(Path(__file__).with_name(task_list).read_text())
        t = time.perf_counter()
        tasks = []
        for db, name, kind, _n, _sub in rows:
            (tk,) = get_tasks(join_pre_dir, [(db, name)], ("train",))
            assert tk.task_type == kind, (db, name, tk.task_type, kind)
            lo, n = table_offset_and_len(join_pre_dir, db, tk.table_name)
            assert n >= need, (db, name, n)
            tasks.append({"db": db, "task": name, "kind": kind, "t": tk, "lo": lo, "n": n})
        log(f"resolved {len(tasks)} tasks over {len({e['db'] for e in tasks})} dbs in {time.perf_counter() - t:.0f}s")
        t = time.perf_counter()
        ds = make_dataset([e["t"] for e in tasks], join_pre_dir, config, mmap_populate=False)
        for i, e in enumerate(tasks):
            e["ds"], e["didx"] = ds, i
        log(f"join sampler built in {time.perf_counter() - t:.0f}s")
    else:
        tasks = [
            {
                "db": ev["name"].split("/")[0],
                "task": ev["name"].split("/")[1],
                "kind": ev["kind"],
                "ds": ev["ds"],
                "didx": ev["didx"],
                "lo": ev["train_lo"],
                "n": ev["train_n"],
            }
            for ev in relbench
        ]
        log(f"training on relbench train splits: {len(tasks)} tasks, train rows {sorted(e['n'] for e in tasks)}")
    def task_stream():
        yield from np.random.default_rng([seed, 1]).permutation(len(tasks))
        ep = 1
        while True:
            yield from np.random.default_rng([seed, 1, ep]).permutation(len(tasks))
            ep += 1

    stream = task_stream()
    step_tasks = []
    queues = {"clf": [], "reg": []}
    while len(step_tasks) < total_steps:
        i = int(next(stream))
        q = queues[tasks[i]["kind"]]
        q.append(i)
        if len(q) == task_batch:
            step_tasks.append(list(q))
            q.clear()
    n_used = total_steps * task_batch
    log(f"{total_steps} steps x {task_batch} tasks over {len(tasks)} tasks: {n_used / len(tasks):.2f} epochs; "
        f"{sum(tasks[b[0]]['kind'] == 'clf' for b in step_tasks)} clf steps")

    st = np.load(Path(stats_path).expanduser())
    seed_everything(seed)
    with torch.cuda.device(dev0):
        head = PoolHead(d_model, n_queries, swiglu_norm, mean=st["mean"], scale=st["scale"], input_norm=input_norm, signsoftmax_temp=signsoftmax_temp, colnorm_tau=colnorm_tau, live_scale=live_scale, signsoftmax_scale_by_dim=signsoftmax_scale_by_dim).to(dev0)
        swa_head = copy.deepcopy(head)
        for p in swa_head.parameters():
            p.requires_grad_(False)
        buf = {"tokens": torch.empty(cap, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev0)}
        pad = torch.empty(cap, LOCAL_CTX, dtype=torch.bool, device=dev0)
        labels = torch.empty(cap, dtype=torch.float32, device=dev0)
        keys = torch.empty(cap, LOCAL_CTX, dtype=torch.int64, device=dev0)
    host = torch.empty(task_batch * need, LOCAL_CTX, d_model, dtype=torch.bfloat16, pin_memory=True) if offload_cells else None
    log(
        f"head: {sum(p.numel() for p in head.parameters())} params; cell buffer "
        f"{buf['tokens'].numel() * 2 / 2**30:.1f} GiB; tabpfn on {dev1}; offload_cells {offload_cells}"
    )

    def cells():
        if buf["tokens"] is None:
            buf["tokens"] = torch.empty(cap, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev0)
        return buf["tokens"]
    init_flat = torch.cat([p.detach().flatten() for p in head.parameters()]).clone()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    swa = SwaState(head.named_parameters(), momentum=swa_momentum)

    with torch.cuda.device(dev1):
        ests = make_ests(tabpfn_dir, dev1, seed, n_queries, tabpfn_precision)

    acc = [torch.zeros_like(p) for p in head.parameters()]
    acc_n = {"n": 0}
    resume_path = out / "resume.pt"
    start_step = 0
    counts = {"skipped": 0, "dropped": 0, "substituted": 0}
    elapsed = 0.0
    if resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu", weights_only=True)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        swa.load_state_dict(ck["swa"])
        init_flat = ck["init_flat"].to(dev0)
        start_step, counts, elapsed = ck["step"], ck["counts"], ck["elapsed"]
        assert grad_accum == 1 or "acc" in ck, "resume.pt has no gradient accumulator"
        if "acc" in ck:
            for a, v in zip(acc, ck["acc"]):
                a.copy_(v.to(dev0))
            acc_n["n"] = ck["acc_n"]
        log(f"resumed from {resume_path} at step {start_step}, swa n {swa.n}, accumulated {acc_n['n']}")

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        tt = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * tt))

    def save_resume(next_step):
        tmp = out / f"resume.pt.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            torch.save(
                {
                    "step": next_step,
                    "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "opt": opt.state_dict(),
                    "swa": swa.state_dict(),
                    "init_flat": init_flat.detach().cpu(),
                    "counts": counts,
                    "acc": [a.detach().cpu() for a in acc],
                    "acc_n": acc_n["n"],
                    "elapsed": elapsed + time.time() - t_start,
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, resume_path)

    def save_ckpt(step, name):
        swa.sync_to(swa_head.named_parameters())
        torch.save(
            {
                "step": step,
                "n_queries": n_queries,
                "swiglu_norm": swiglu_norm,
                "input_norm": input_norm,
                "signsoftmax_temp": signsoftmax_temp,
                "colnorm_tau": colnorm_tau,
                "live_scale": live_scale,
                "signsoftmax_scale_by_dim": signsoftmax_scale_by_dim,
                "stats_path": stats_path,
                "state_dict": head.state_dict(),
                "swa_state_dict": swa_head.state_dict(),
                "swa_n": swa.n,
                "swa_momentum": swa_momentum,
            },
            out / name,
        )

    def evaluate(step):
        swa.sync_to(swa_head.named_parameters())
        res = {"head": {}, "swa": {}}
        t0 = time.perf_counter()
        n_sub = 0
        for ev in relbench:
            if caught["flag"]:
                log(f"step {step:>5} eval abandoned for a stop signal")
                return
            t1 = time.perf_counter()
            tokens = cells()
            with torch.cuda.device(dev0), torch.inference_mode():
                nc, s1, _a, _b = embed_rows(
                    net, ev["ds"], ev["didx"], ev["ctx"], tokens, pad, labels, keys, len(ev["ctx"]), dev0
                )
                nq, s2, _a, _b = embed_rows(
                    net, ev["ds"], ev["didx"], ev["query"], tokens[nc:], pad[nc:], labels[nc:], keys[nc:], len(ev["query"]), dev0
                )
                cn = column_stats(tokens, keys, nc + nq, nc, 512)[:3] if input_norm != "fixed" else None
            n_sub += s1 + s2
            y = labels[: nc + nq].clone()
            for key, h in (("head", head), ("swa", swa_head)):
                with torch.cuda.device(dev0):
                    _h, f = head_features(h, tokens, pad, nc + nq, nc, head_chunk, cn)
                with torch.cuda.device(dev1), torch.no_grad():
                    _loss, pred, tgt = batched_outputs(
                        ests[ev["kind"]], ev["kind"], [f.to(dev1)], [y.to(dev1)], nc, dev1
                    )
                pred, tgt = pred[0].float().cpu().numpy(), tgt[0].float().cpu().numpy()
                if ev["kind"] == "clf":
                    res[key][ev["name"]] = float(roc_auc_score(tgt, pred))
                else:
                    res[key][ev["name"]] = float(np.abs(pred - tgt).mean())
            log(
                f"step {step:>5} eval {ev['name']}: ctx {nc} query {nq} "
                f"head {res['head'][ev['name']]:.4f} swa {res['swa'][ev['name']]:.4f} "
                f"{time.perf_counter() - t1:.0f}s"
            )
        kinds = {ev["name"]: ev["kind"] for ev in relbench}
        summary = {}
        for key in res:
            for kind, metric in (("clf", "auroc"), ("reg", "nmae")):
                v = [x for name, x in res[key].items() if kinds[name] == kind]
                summary[f"val/{key}/{metric}"] = float(np.mean(v))
        log(
            f"step {step:>5} relbench val ({time.perf_counter() - t0:.0f}s, {n_sub} substituted): "
            + " ".join(f"{k} {v:.4f}" for k, v in summary.items())
        )
        if use_wandb:
            wandb.log(
                {
                    "step": step,
                    **summary,
                    **{f"val/{key}/{name}": v for key in res for name, v in res[key].items()},
                    "val/seconds": time.perf_counter() - t0,
                    "train/swa_n": swa.n,
                },
                step=step,
            )
        (out / f"val_step{step}.json").write_text(json.dumps({"summary": summary, "per_task": res}, indent=1))

    caught = {"flag": False}
    cur = {"next": start_step, "busy": False, "saved": False}

    def on_signal(signum, frame):
        caught["flag"] = True
        print(f"caught signal {signum} during step {cur['next']}", flush=True)
        if not cur["busy"] and not cur["saved"]:
            save_resume(cur["next"])
            cur["saved"] = True
            print(f"resume saved at step {cur['next']} from the signal handler", flush=True)

    def settle(next_step):
        cur["next"], cur["busy"] = next_step, False
        if caught["flag"] and not cur["saved"]:
            save_resume(next_step)
            cur["saved"] = True
            print(f"resume saved at step {next_step} after the update", flush=True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)
    last_resume = time.perf_counter()

    def stop(next_step):
        save_resume(next_step)
        log(f"resume saved at step {next_step}, exiting to be requeued")
        if use_wandb:
            wandb.finish()

    for step in range(start_step, total_steps):
        cur["next"] = step
        if caught["flag"]:
            return stop(step)
        if eval_every and step % eval_every == 0:
            evaluate(step)
            if caught["flag"]:
                return stop(step)
        if save_every and step % save_every == 0:
            save_ckpt(step, f"pool_step{step}.pt")

        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        batch = step_tasks[step]
        e = tasks[batch[0]]
        kind = e["kind"]
        tm = {"sample_s": 0.0, "rtj_s": 0.0, "colstats_s": 0.0, "head_fwd_s": 0.0}
        tokens = cells()
        items = []
        n_sub = 0
        off = 0
        for j, ti in enumerate(batch):
            et = tasks[ti]
            need_t = min(need, et["n"])
            nctx_t = n_ctx if need_t == need else int(need_t * n_ctx / need)
            rng = np.random.default_rng([seed, step] if task_batch == 1 else [seed, step, j])
            cand = rng.permutation(et["n"]) + et["lo"]
            sl = slice(off, off + need_t)
            with torch.cuda.device(dev0), torch.inference_mode():
                filled, s_sub, s_t, r_t = embed_rows(
                    net, et["ds"], et["didx"], cand, tokens[sl], pad[sl], labels[sl], keys[sl], need_t, dev0
                )
                tm["sample_s"] += s_t
                tm["rtj_s"] += r_t
                cn = None
                if input_norm != "fixed" and filled == need_t:
                    t = time.perf_counter()
                    *cn, _q = column_stats(tokens[sl], keys[sl], need_t, nctx_t, 512)
                    tm["colstats_s"] += time.perf_counter() - t
            n_sub += s_sub
            counts["substituted"] += s_sub
            y = labels[sl].clone()
            reason = None
            if filled < need_t:
                reason = f"only {filled} buildable rows"
            elif kind == "reg" and float(y[:nctx_t].std()) < 1e-6:
                reason = "constant context label"
            elif kind == "clf" and (y[:nctx_t] > 0).float().mean().item() in (0.0, 1.0):
                reason = "one class in context"
            t = time.perf_counter()
            with torch.cuda.device(dev0):
                hp, f = head_features(head, tokens[sl], pad[sl], need_t, nctx_t, head_chunk, cn) if reason is None else (None, None)
            if reason is None and float(f[:nctx_t].std(dim=0).max()) == 0.0:
                reason = "constant features"
            tm["head_fwd_s"] += time.perf_counter() - t
            if reason is not None:
                counts["skipped"] += 1
                log(f"step {step:>5} {et['db']}/{et['task']} skipped: {reason}")
                continue
            sel, nctx_tp = dedup_context(hp, y, nctx_t) if dedup_ctx else (torch.arange(need_t, device=hp.device), nctx_t)
            items.append({"e": et, "j": j, "sl": sl, "need_t": need_t, "nctx_t": nctx_t, "cn": cn, "y": y, "hp": hp, "f": f, "sel": sel, "nctx_tp": nctx_tp})
            off += need_t
        if items:
            assert len({it["need_t"] - it["nctx_t"] for it in items}) == 1, [(it["need_t"], it["nctx_t"]) for it in items]
            m_ctx = min(it["nctx_tp"] for it in items)
            for it in items:
                it["idx"] = torch.cat([it["sel"][:m_ctx], it["sel"][it["nctx_tp"]:]])
            tm["dedup_dropped"] = float(sum(it["nctx_t"] - it["nctx_tp"] for it in items))
            if task_batch > 1:
                tm["ctx_trimmed"] = float(sum(it["nctx_tp"] - m_ctx for it in items))

        opt.zero_grad(set_to_none=True)
        loss = gnorm = None
        task_gn = []
        if items:
            if offload_cells:
                t = time.perf_counter()
                host[:off].copy_(tokens[:off])
                del tokens
                buf["tokens"] = None
                tm["offload_s"] = time.perf_counter() - t
            t = time.perf_counter()
            with torch.cuda.device(dev1):
                torch.cuda.reset_peak_memory_stats(dev1)
                leaves = [it["f"][it["idx"]].to(dev1).requires_grad_(True) for it in items]
                ys = [it["y"][it["idx"]].to(dev1) for it in items]
                loss, _p, _t = batched_outputs(ests[kind], kind, leaves, ys, m_ctx, dev1)
                ok = bool(torch.isfinite(loss))
                if ok:
                    loss.backward()
                    gs = [
                        torch.zeros_like(it["f"]).index_copy_(0, it["idx"], lf.grad.to(dev0) * len(items))
                        for it, lf in zip(items, leaves)
                    ]
                    ok = all(bool(torch.isfinite(g).all()) for g in gs)
                torch.cuda.synchronize(dev1)
            tm["tabpfn_s"] = time.perf_counter() - t
            del leaves
            if offload_cells:
                t = time.perf_counter()
                tokens = cells()
                tokens[:off].copy_(host[:off])
                tm["reload_s"] = time.perf_counter() - t
            if not ok:
                counts["dropped"] += len(items)
                log(f"step {step:>5} {e['db']}/{e['task']} (+{len(items) - 1}) non-finite loss or dL/df, dropped")
            else:
                t = time.perf_counter()
                spike_saved = False
                for it, g in zip(items, gs):
                    for p in head.parameters():
                        p.grad = None
                    sl, need_t, nctx_t, cn = it["sl"], it["need_t"], it["nctx_t"], it["cn"]
                    with torch.cuda.device(dev0):
                        hl = it["hp"].detach().requires_grad_(True)
                        head.mix(hl, nctx_t).backward(g)
                        gh = hl.grad
                        tk = tokens[sl]
                        pk = pad[sl]
                        for i in range(0, need_t, head_chunk):
                            jj = min(i + head_chunk, need_t)
                            head.pool(tk[i:jj], pk[i:jj], colnorm_slice(cn, i, jj)).backward(gh[i:jj])
                    if not all(bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
                        counts["dropped"] += 1
                        log(f"step {step:>5} {it['e']['db']}/{it['e']['task']} non-finite head grad, dropped")
                        continue
                    if step == 0:
                        assert sum(float(p.grad.abs().sum()) for p in head.parameters()) > 0
                    pre = float(torch.stack([p.grad.detach().double().norm() for p in head.parameters()]).norm())
                    task_gn.append(pre)
                    if spike_dump_gnorm is not None and pre > spike_dump_gnorm:
                        t2 = time.perf_counter()
                        if not spike_saved:
                            save_ckpt(step, f"spikes/pool_step{step}.pt")
                            spike_saved = True
                        name = f"step{step}_tensors.npz" if task_batch == 1 else f"step{step}_task{it['j']}_tensors.npz"
                        dump_spike(out / "spikes" / name, head, it["hp"], g, gh, it["y"], pk, nctx_t,
                                   {"step": step, "task_index_in_batch": it["j"], "task": f"{it['e']['db']}/{it['e']['task']}",
                                    "kind": kind, "loss_batch_mean": float(loss), "grad_norm": pre, "batch_tasks": len(items),
                                    "tabpfn_n_ctx": m_ctx})
                        tm["spike_dump_s"] = tm.get("spike_dump_s", 0.0) + time.perf_counter() - t2
                        log(f"step {step:>5} spike dump: {it['e']['db']}/{it['e']['task']} pre-clip gnorm {pre:.4g} > {spike_dump_gnorm}")
                    cur["busy"] = True
                    for a, p in zip(acc, head.parameters()):
                        a.add_(p.grad)
                    acc_n["n"] += 1
                torch.cuda.synchronize(dev0)
                tm["head_bwd_s"] = time.perf_counter() - t
                if task_gn:
                    gnorm = max(task_gn)

        agnorm = None
        if ((step + 1) % grad_accum == 0 or step == total_steps - 1) and acc_n["n"] > 0:
            cur["busy"] = True
            for a, p in zip(acc, head.parameters()):
                p.grad = a / acc_n["n"]
            agnorm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), grad_norm_max))
            opt.step()
            swa.update(head.named_parameters())
            for a in acc:
                a.zero_()
            acc_n["n"] = 0
        if cur["busy"]:
            settle(step + 1)

        flat = torch.cat([p.detach().flatten() for p in head.parameters()])
        drift = float((flat - init_flat).norm())
        rec = {
            "step": step,
            "train/lr": opt.param_groups[0]["lr"],
            "train/dist_from_init": drift,
            "train/rows_substituted": n_sub,
            "train/steps_skipped": counts["skipped"],
            "train/steps_dropped": counts["dropped"],
            "train/peak_gib_dev0": torch.cuda.max_memory_allocated(dev0) / 2**30,
            "train/peak_gib_dev1": torch.cuda.max_memory_allocated(dev1) / 2**30,
            "train/minutes": (elapsed + time.time() - t_start) / 60,
            **{f"time/{k}": v for k, v in tm.items()},
        }
        if loss is not None and gnorm is not None:
            rec.update(
                {
                    "train/loss": float(loss),
                    f"train/loss/{kind}": float(loss),
                    "train/grad_norm": gnorm,
                    "train/grad_norm_task_mean": float(np.mean(task_gn)),
                    "train/tasks_in_step": len(task_gn),
                }
            )
        if agnorm is not None:
            rec.update({"train/grad_norm_update": agnorm, "train/clipped": float(agnorm > grad_norm_max)})
        log(
            f"step {step:>5} {kind} {e['db']}/{e['task']} "
            + (f"(+{len(batch) - 1} tasks, {len(task_gn)} used) " if task_batch > 1 else "")
            + (f"loss {float(loss):.4f} gnorm {gnorm:.4g} " if gnorm is not None else "")
            + (f"task gnorm mean {np.mean(task_gn):.4g} " if task_batch > 1 and task_gn else "")
            + (f"update gnorm {agnorm:.4g} " if agnorm is not None and grad_accum > 1 else "")
            + f"drift {drift:.4f} lr {rec['train/lr']:.2e} sub {n_sub} "
            + f"peak {rec['train/peak_gib_dev1']:.1f}GiB "
            + " ".join(f"{k} {v:.1f}" for k, v in tm.items())
        )
        if use_wandb:
            wandb.log(rec, step=step)

        if caught["flag"]:
            return stop(step + 1)
        cur["next"] = step + 1
        if time.perf_counter() - last_resume >= resume_save_mins * 60:
            cur["busy"] = True
            save_resume(step + 1)
            settle(step + 1)
            last_resume = time.perf_counter()
            log(f"step {step:>5} resume saved")

    evaluate(total_steps)
    save_ckpt(total_steps, "pool_final.pt")
    save_resume(total_steps)
    log("done")
    if use_wandb:
        wandb.finish()
