import json
import os
import time
from pathlib import Path


def main(
    *,
    pre_dir: str,
    ckpt: str,
    tabpfn_dir: str,
    out_dir: str,
    db: str,
    task: str,
    n_rows: int,
    n_ctx: int,
    local_ctx_size: int,
    embed_chunk: int,
    head_chunk: int,
    stats_rows: int,
    n_queries: int,
    d_out: int,
    check_rows: int,
    check_ctx: int,
    check_chunk: int,
    tabpfn_modes: list[str],
    lr: float,
    seed: int,
) -> None:
    import numpy as np
    import torch

    from expts.repaper.adapter.pool_head import PoolHead
    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX, make_dataset, resolve_tasks
    from expts.repaper.adapter.train_adapter import loss_and_pred, make_ests
    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len
    from rt.data import process_batch
    from rt.model import load_rt_model

    assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
    assert local_ctx_size == LOCAL_CTX, (local_ctx_size, LOCAL_CTX)
    dev_rt, dev_pfn = "cuda:0", "cuda:1"
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "host": os.uname().nodename,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "db": db,
        "task": task,
        "n_rows": n_rows,
        "n_ctx": n_ctx,
        "n_query": n_rows - n_ctx,
    }

    def dump():
        (out / "report.json").write_text(json.dumps(report, indent=2))

    def log(msg):
        print(msg, flush=True)

    def gib(dev):
        return torch.cuda.max_memory_allocated(dev) / 2**30

    def sync():
        torch.cuda.synchronize(dev_rt)
        torch.cuda.synchronize(dev_pfn)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    net, config = load_rt_model(ckpt, device=dev_rt, compile=False)
    net = net.to(torch.bfloat16).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    d_model = net.d_model

    t0 = time.perf_counter()
    (tsk,) = resolve_tasks(pre_dir, [(db, task)])
    ds = make_dataset([tsk], pre_dir, config, mmap_populate=True)
    report["sampler_setup_s"] = time.perf_counter() - t0
    lo, n_table = table_offset_and_len(pre_dir, db, tsk.table_name)
    report["task_type"] = tsk.task_type
    report["table_rows"] = n_table
    assert tsk.task_type == "clf", tsk.task_type
    want = min(n_table, int(n_rows * 1.05) + embed_chunk)
    cand = np.sort(rng.choice(n_table, size=want, replace=False) + lo)
    log(f"{db}/{task}: {n_table} table rows, drawing up to {want} for {n_rows}")

    tokens = torch.empty(n_rows, local_ctx_size, d_model, dtype=torch.bfloat16, device=dev_rt)
    pad = torch.empty(n_rows, local_ctx_size, dtype=torch.bool, device=dev_rt)
    labels = torch.empty(n_rows, dtype=torch.float32)
    report["token_buffer_gib"] = tokens.numel() * 2 / 2**30
    filled = n_sub = 0
    samp_s = fwd_s = 0.0
    torch.cuda.reset_peak_memory_stats(dev_rt)
    with torch.inference_mode():
        for start in range(0, len(cand), embed_chunk):
            if filled == n_rows:
                break
            nodes = [int(v) for v in cand[start : start + embed_chunk]]
            ts = time.perf_counter()
            tup = ds.sampler.batch_for_nodes_py(nodes, 0, local_ctx_size)
            batch = process_batch(tup, ds.d_text)
            batch.pop("batch_mask", None)
            batch = {k: v.to(dev_rt, non_blocking=True) for k, v in batch.items()}
            samp_s += time.perf_counter() - ts
            tf = time.perf_counter()
            x = net(batch, return_embeddings=True)
            keys = batch["col_name_idxs"].masked_fill(
                batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
            )
            si = keys.argsort(dim=-1, stable=True)
            is_t = batch["is_targets"].gather(1, si)
            assert (is_t.sum(1) == 1).all()
            got = (batch["node_idxs"].gather(1, si) * is_t).sum(1)
            ok = got == torch.as_tensor(nodes, device=dev_rt, dtype=got.dtype)
            n_sub += int((~ok).sum())
            vals = batch["number_values"].gather(1, si.unsqueeze(-1).expand_as(batch["number_values"])).squeeze(-1)
            lab = (vals * is_t.to(vals.dtype)).sum(1).float()
            take = min(int(ok.sum()), n_rows - filled)
            keep = ok.nonzero().squeeze(1)[:take]
            tokens[filled : filled + take] = x[keep].bfloat16()
            pad[filled : filled + take] = batch["is_padding"].gather(1, si)[keep]
            labels[filled : filled + take] = lab[keep].cpu()
            filled += take
            torch.cuda.synchronize(dev_rt)
            fwd_s += time.perf_counter() - tf
    assert filled == n_rows, (filled, n_rows)
    live = (~pad).sum(1).float()
    report["embed"] = {
        "sample_s": samp_s,
        "rtj_forward_s": fwd_s,
        "rows_per_s": n_rows / (samp_s + fwd_s),
        "substituted": n_sub,
        "live_cells_per_row": {"mean": float(live.mean()), "p1": float(live.quantile(0.01)), "min": float(live.min())},
        "peak_gib_dev0": gib(dev_rt),
        "label_frac_positive": float((labels > 0).float().mean()),
    }
    log(f"embed: {report['embed']}")
    del net
    torch.cuda.empty_cache()
    torch.cuda.set_device(dev_pfn)
    dump()

    s = tokens[:stats_rows][~pad[:stats_rows]].float()
    mean, std = s.mean(0), s.std(0).clamp_min(1e-4)
    del s
    head = PoolHead(d_model, n_queries, d_out, mean=mean, scale=std).to(dev_rt)
    ests = make_ests(tabpfn_dir, dev_pfn, seed, d_out, "fp32")
    for est in ests.values():
        for m in est.models_:
            assert not any(p.requires_grad for p in m.parameters())

    def features(idx, chunk):
        with torch.no_grad():
            return torch.cat([head(tokens[idx[i : i + chunk]], pad[idx[i : i + chunk]]) for i in range(0, len(idx), chunk)])

    def backprop(idx, chunk, g):
        for i in range(0, len(idx), chunk):
            j = idx[i : i + chunk]
            f = head(tokens[j], pad[j])
            f.backward(g[i : i + chunk])

    def tabpfn_grad(f, y, nctx, mode):
        leaf = f.detach().to(dev_pfn).requires_grad_(True)
        ac = torch.bfloat16 if mode == "autocast_bf16" else None
        sync()
        t = time.perf_counter()
        loss, _p, _l = loss_and_pred(ests, lambda e: e, "clf", leaf, y, nctx, dev_pfn, ac)
        assert loss is not None
        sync()
        t_fwd = time.perf_counter() - t
        t = time.perf_counter()
        loss.backward()
        sync()
        t_bwd = time.perf_counter() - t
        assert all(p.grad is None for m in ests["clf"].models_ for p in m.parameters())
        return float(loss), leaf.grad.to(dev_rt), t_fwd, t_bwd

    def head_grads():
        return torch.cat([p.grad.flatten() for p in head.parameters()])

    ci = torch.arange(check_rows, device=dev_rt)
    cy = labels[:check_rows]
    head.zero_grad()
    f = head(tokens[ci], pad[ci])
    leaf = f.to(dev_pfn)
    loss_a, _p, _l = loss_and_pred(ests, lambda e: e, "clf", leaf, cy, check_ctx, dev_pfn)
    loss_a.backward()
    g_single = head_grads().clone()
    head.zero_grad()
    f2 = features(ci, check_chunk)
    loss_b, g_f, _a, _b = tabpfn_grad(f2, cy, check_ctx, "fp32")
    backprop(ci, check_chunk, g_f)
    g_two = head_grads().clone()
    head.zero_grad()
    report["two_pass_check"] = {
        "rows": check_rows,
        "n_ctx": check_ctx,
        "loss_single_graph": float(loss_a),
        "loss_two_pass": loss_b,
        "feature_max_abs_diff": float((f.detach() - f2).abs().max()),
        "grad_norm_single": float(g_single.norm()),
        "grad_rel_diff": float((g_single - g_two).norm() / g_single.norm()),
        "grad_cosine": float(torch.nn.functional.cosine_similarity(g_single, g_two, dim=0)),
    }
    log(f"two-pass check: {report['two_pass_check']}")
    del f, leaf, f2, g_f
    torch.cuda.empty_cache()
    dump()

    perm = torch.as_tensor(rng.permutation(n_rows), device=dev_rt)
    y = labels[perm.cpu()]
    torch.cuda.reset_peak_memory_stats(dev_rt)
    sync()
    t = time.perf_counter()
    f = features(perm, head_chunk)
    sync()
    report["head_pass1_s"] = time.perf_counter() - t
    report["head_pass1_peak_gib_dev0"] = gib(dev_rt)
    log(f"head pass 1: {report['head_pass1_s']:.2f}s, dev0 peak {report['head_pass1_peak_gib_dev0']:.1f} GiB")

    report["tabpfn"] = {}
    grad = None
    for mode in tabpfn_modes:
        torch.cuda.reset_peak_memory_stats(dev_pfn)
        try:
            loss, g, t_fwd, t_bwd = tabpfn_grad(f, y, n_ctx, mode)
            report["tabpfn"][mode] = {
                "loss": loss,
                "fwd_s": t_fwd,
                "bwd_s": t_bwd,
                "peak_gib_dev1": gib(dev_pfn),
                "feature_grad_norm": float(g.norm()),
                "feature_grad_finite": bool(torch.isfinite(g).all()),
                "ctx_query_grad_norm": [float(g[:n_ctx].norm()), float(g[n_ctx:].norm())],
            }
            if grad is None:
                grad, grad_mode, loss0 = g, mode, loss
        except torch.OutOfMemoryError as exc:
            report["tabpfn"][mode] = {"OOM": str(exc)[:300], "peak_gib_dev1": gib(dev_pfn)}
        torch.cuda.empty_cache()
        log(f"tabpfn {mode}: {report['tabpfn'][mode]}")
        dump()
    assert grad is not None, "every tabpfn mode ran out of memory"

    head.zero_grad()
    torch.cuda.reset_peak_memory_stats(dev_rt)
    sync()
    t = time.perf_counter()
    backprop(perm, head_chunk, grad)
    sync()
    report["head_pass2_s"] = time.perf_counter() - t
    report["head_pass2_peak_gib_dev0"] = gib(dev_rt)
    report["head_grad"] = {
        n: {"norm": float(p.grad.norm()), "finite": bool(torch.isfinite(p.grad).all())}
        for n, p in head.named_parameters()
    }
    log(f"head pass 2: {report['head_pass2_s']:.2f}s, grads {report['head_grad']}")

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    opt.step()
    f_after = features(perm, head_chunk)
    with torch.no_grad():
        leaf = f_after.to(dev_pfn)
        ac = torch.bfloat16 if grad_mode == "autocast_bf16" else None
        loss1, _p, _l = loss_and_pred(ests, lambda e: e, "clf", leaf, y, n_ctx, dev_pfn, ac)
    report["step"] = {"mode": grad_mode, "lr": lr, "loss_before": loss0, "loss_after": float(loss1)}
    report["peak_gib"] = {"dev0": gib(dev_rt), "dev1": gib(dev_pfn)}
    log(f"step: {report['step']}")
    dump()
    log("done")
