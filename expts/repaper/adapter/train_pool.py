import json
import math
import os
import signal
import time
import zlib
from pathlib import Path

CHUNK = 1024


def embed_rows(net, ds, didx, nodes, tokens, pad, labels, need, device):
    import numpy as np
    import torch

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
        filled += take
        torch.cuda.synchronize(device)
        fwd_s += time.perf_counter() - t
    return filled, n_sub, samp_s, fwd_s


def head_features(head, tokens, pad, n, chunk):
    import torch

    with torch.no_grad():
        return torch.cat(
            [head(tokens[i : min(i + chunk, n)], pad[i : min(i + chunk, n)]) for i in range(0, n, chunk)]
        )


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
                }
            )
    return evals


def main(
    *,
    join_pre_dir: str,
    relbench_pre_dir: str,
    relbench_task_list: str,
    task_list: str,
    ckpt: str,
    tabpfn_dir: str,
    stats_path: str,
    out_dir: str,
    n_ctx: int,
    n_query: int,
    relbench_n_ctx: int,
    relbench_n_query: int,
    n_queries: int,
    head_chunk: int,
    tabpfn_device: str,
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

    from expts.repaper.adapter.pool_head import PoolHead
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
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
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
    log(f"join sampler built in {time.perf_counter() - t:.0f}s")
    order = np.random.default_rng([seed, 1]).permutation(len(tasks))
    assert total_steps <= len(order), (total_steps, len(order))

    relbench = build_relbench(relbench_pre_dir, relbench_task_list, config, relbench_n_ctx, relbench_n_query, seed)
    log(f"relbench: {len(relbench)} eval tasks")
    assert relbench_n_ctx + relbench_n_query <= need

    st = np.load(Path(stats_path).expanduser())
    seed_everything(seed)
    with torch.cuda.device(dev0):
        head = PoolHead(d_model, n_queries, mean=st["mean"], scale=st["scale"]).to(dev0)
        swa_head = copy.deepcopy(head)
        for p in swa_head.parameters():
            p.requires_grad_(False)
        buf = {"tokens": torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev0)}
        pad = torch.empty(need, LOCAL_CTX, dtype=torch.bool, device=dev0)
        labels = torch.empty(need, dtype=torch.float32, device=dev0)
    host = torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, pin_memory=True) if offload_cells else None
    log(
        f"head: {sum(p.numel() for p in head.parameters())} params; cell buffer "
        f"{buf['tokens'].numel() * 2 / 2**30:.1f} GiB; tabpfn on {dev1}; offload_cells {offload_cells}"
    )

    def cells():
        if buf["tokens"] is None:
            buf["tokens"] = torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev0)
        return buf["tokens"]
    init_flat = torch.cat([p.detach().flatten() for p in head.parameters()]).clone()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    swa = SwaState(head.named_parameters(), momentum=swa_momentum)

    with torch.cuda.device(dev1):
        ests = make_ests(tabpfn_dir, dev1, seed, n_queries, "bf16")

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
        log(f"resumed from {resume_path} at step {start_step}, swa n {swa.n}")

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
                nc, s1, _a, _b = embed_rows(net, ev["ds"], ev["didx"], ev["ctx"], tokens, pad, labels, len(ev["ctx"]), dev0)
                nq, s2, _a, _b = embed_rows(
                    net, ev["ds"], ev["didx"], ev["query"], tokens[nc:], pad[nc:], labels[nc:], len(ev["query"]), dev0
                )
            n_sub += s1 + s2
            y = labels[: nc + nq].clone()
            for key, h in (("head", head), ("swa", swa_head)):
                with torch.cuda.device(dev0):
                    f = head_features(h, tokens, pad, nc + nq, head_chunk)
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

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        e = tasks[order[step]]
        rng = np.random.default_rng([seed, step])
        cand = rng.permutation(e["n"]) + e["lo"]
        tm = {}
        tokens = cells()
        with torch.cuda.device(dev0), torch.inference_mode():
            filled, n_sub, tm["sample_s"], tm["rtj_s"] = embed_rows(
                net, ds, int(order[step]), cand, tokens, pad, labels, need, dev0
            )
        counts["substituted"] += n_sub
        y = labels.clone()
        reason = None
        if filled < need:
            reason = f"only {filled} buildable rows"
        elif e["kind"] == "reg" and float(y[:n_ctx].std()) < 1e-6:
            reason = "constant context label"
        elif e["kind"] == "clf" and (y[:n_ctx] > 0).float().mean().item() in (0.0, 1.0):
            reason = "one class in context"
        t = time.perf_counter()
        with torch.cuda.device(dev0):
            f = head_features(head, tokens, pad, need, head_chunk) if reason is None else None
        if reason is None and float(f[:n_ctx].std(dim=0).max()) == 0.0:
            reason = "constant features"
        tm["head_fwd_s"] = time.perf_counter() - t

        opt.zero_grad(set_to_none=True)
        loss = gnorm = None
        if reason is not None:
            counts["skipped"] += 1
            log(f"step {step:>5} {e['db']}/{e['task']} skipped: {reason}")
        else:
            if offload_cells:
                t = time.perf_counter()
                host.copy_(tokens)
                del tokens
                buf["tokens"] = None
                tm["offload_s"] = time.perf_counter() - t
            t = time.perf_counter()
            with torch.cuda.device(dev1):
                torch.cuda.reset_peak_memory_stats(dev1)
                leaf = f.to(dev1).requires_grad_(True)
                loss, _p, _t = batched_outputs(ests[e["kind"]], e["kind"], [leaf], [y.to(dev1)], n_ctx, dev1)
                ok = bool(torch.isfinite(loss))
                if ok:
                    loss.backward()
                    g = leaf.grad.to(dev0)
                    ok = bool(torch.isfinite(g).all())
                torch.cuda.synchronize(dev1)
            tm["tabpfn_s"] = time.perf_counter() - t
            del leaf
            if offload_cells:
                t = time.perf_counter()
                buf["tokens"] = host.to(dev0)
                tokens = buf["tokens"]
                tm["reload_s"] = time.perf_counter() - t
            if ok:
                t = time.perf_counter()
                with torch.cuda.device(dev0):
                    for i in range(0, need, head_chunk):
                        j = min(i + head_chunk, need)
                        head(tokens[i:j], pad[i:j]).backward(g[i:j])
                    torch.cuda.synchronize(dev0)
                tm["head_bwd_s"] = time.perf_counter() - t
                ok = all(torch.isfinite(p.grad).all() for p in head.parameters())
            if step == 0 and ok:
                assert sum(float(p.grad.abs().sum()) for p in head.parameters()) > 0
            if ok:
                cur["busy"] = True
                gnorm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), grad_norm_max))
                opt.step()
                swa.update(head.named_parameters())
                settle(step + 1)
            else:
                counts["dropped"] += 1
                log(f"step {step:>5} {e['db']}/{e['task']} non-finite, dropped")

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
                    f"train/loss/{e['kind']}": float(loss),
                    "train/grad_norm": gnorm,
                    "train/clipped": float(gnorm > grad_norm_max),
                }
            )
        log(
            f"step {step:>5} {e['kind']} {e['db']}/{e['task']} "
            + (f"loss {float(loss):.4f} gnorm {gnorm:.4g} " if gnorm is not None else "")
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
