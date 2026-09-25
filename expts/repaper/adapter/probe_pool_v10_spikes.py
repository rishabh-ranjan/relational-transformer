import json
import os
import time
from pathlib import Path

PARAMS = ["q", "wk.weight", "wv.weight", "w1.weight", "w1.bias", "w3.weight", "w3.bias", "w2.weight", "norm.weight", "norm.bias"]


def fnorm(t):
    return float(t.detach().double().norm())


def row_stats(t, n_ctx):
    r = t.detach().double().norm(dim=-1)
    return {
        "fro": float(r.square().sum().sqrt()),
        "fro_ctx": float(r[:n_ctx].square().sum().sqrt()),
        "fro_query": float(r[n_ctx:].square().sum().sqrt()),
        "row_mean": float(r.mean()),
        "row_max_ctx": float(r[:n_ctx].max()),
        "row_max_query": float(r[n_ctx:].max()),
        "row_argmax": int(r.argmax()),
        "top1_sq_share": float(r.max().square() / r.square().sum().clamp_min(1e-300)),
    }


def z_halves(head, x, pad, cn):
    import torch

    idx, col_mean, col_std = cn
    xf = x.float()
    rms = xf.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
    second = xf.sign() * torch.softmax(xf.abs() / (head.signsoftmax_temp * rms), dim=-1)
    first = (xf - col_mean[idx]) / col_std[idx]
    live = ~pad
    return first.abs()[live], second.abs()[live]


def main(
    *,
    join_pre_dir: str,
    task_list: str,
    ckpt: str,
    tabpfn_dir: str,
    ckpt_dir: str,
    out_dir: str,
    spikes: list,
    n_ctx: int,
    n_query: int,
    n_queries: int,
    head_chunk: int,
    n_tasks_total: int,
    seed: int,
    dump: bool = False,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    from expts.repaper.adapter.pool_head import PoolHead, column_stats
    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX, make_dataset
    from expts.repaper.adapter.train_adapter import batched_outputs, make_ests
    from expts.repaper.adapter.train_pool import colnorm_slice, embed_rows, head_features
    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len
    from rt.data import get_tasks
    from rt.model import load_rt_model

    t_start = time.time()

    def log(msg):
        print(f"[{(time.time() - t_start) / 60:7.1f} min] {msg}", flush=True)

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in spikes if not (out / f"step{s['step']}.json").exists()]
    log(f"todo {[s['step'] for s in todo]}")
    if not todo:
        return
    dev = "cuda:0"
    need = n_ctx + n_query
    rows = json.loads(Path(__file__).with_name(task_list).read_text())
    assert len(rows) == n_tasks_total, (len(rows), n_tasks_total)
    order = np.random.default_rng([seed, 1]).permutation(len(rows))

    net, config = load_rt_model(ckpt, device=dev, compile=True)
    net = net.to(torch.bfloat16).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    d_model = net.d_model
    log("rt-j loaded")

    tasks = []
    for s in todo:
        db, name, kind, _n, _sub = rows[int(order[s["step"]])]
        assert f"{db}/{name}" == s["task"], (s, db, name)
        (tk,) = get_tasks(join_pre_dir, [(db, name)], ("train",))
        assert tk.task_type == kind, (db, name, tk.task_type, kind)
        lo, n = table_offset_and_len(join_pre_dir, db, tk.table_name)
        assert n >= need, (db, name, n)
        tasks.append({"db": db, "task": name, "kind": kind, "t": tk, "lo": lo, "n": n})
    ds = make_dataset([e["t"] for e in tasks], join_pre_dir, config, mmap_populate=False)
    log(f"sampler over {len(tasks)} tasks built")
    ests = make_ests(tabpfn_dir, dev, seed, n_queries, "fp32")
    log("tabpfn fp32 loaded")

    buf = {"tokens": torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev)}
    host = torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, pin_memory=True)
    pad = torch.empty(need, LOCAL_CTX, dtype=torch.bool, device=dev)
    labels = torch.empty(need, dtype=torch.float32, device=dev)
    colkeys = torch.empty(need, LOCAL_CTX, dtype=torch.int64, device=dev)
    ck_dir = Path(ckpt_dir).expanduser()

    for didx, (s, e) in enumerate(zip(todo, tasks)):
        step = s["step"]
        ck = torch.load(ck_dir / f"pool_step{s['ckpt_step']}.pt", map_location="cpu", weights_only=True)
        assert ck["n_queries"] == n_queries
        head = PoolHead(
            d_model, n_queries, ck["swiglu_norm"], input_norm=ck["input_norm"], signsoftmax_temp=ck["signsoftmax_temp"]
        ).to(dev)
        head.load_state_dict(ck["state_dict"])
        log(f"step {step}: head from pool_step{s['ckpt_step']}.pt ({ck['swiglu_norm']}, {ck['input_norm']}, T={ck['signsoftmax_temp']})")

        if buf["tokens"] is None:
            buf["tokens"] = torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev)
        tokens = buf["tokens"]
        cand = np.random.default_rng([seed, step]).permutation(e["n"]) + e["lo"]
        with torch.inference_mode():
            filled, n_sub, _a, _b = embed_rows(net, ds, didx, cand, tokens, pad, labels, colkeys, need, dev)
            assert filled == need, (filled, need)
            idx, cmean, cstd, qonly = column_stats(tokens, colkeys, need, n_ctx, 512)
        cn = (idx.clone(), cmean.clone(), cstd.clone())
        y = labels[:need].clone()
        log(f"step {step}: embedded {filled} rows, sub {n_sub}, query-only cols {qonly}")

        zmax = {"first_ctx": 0.0, "first_query": 0.0, "second_ctx": 0.0, "second_query": 0.0}
        zbig = {"first_gt10": 0, "first_gt100": 0, "first_live": 0}
        with torch.no_grad():
            for i in range(0, need, head_chunk):
                j = min(i + head_chunk, need)
                for part, a, b in (("ctx", i, min(j, n_ctx)), ("query", max(i, n_ctx), j)):
                    if a >= b:
                        continue
                    z1, z2 = z_halves(head, tokens[a:b], pad[a:b], colnorm_slice(cn, a, b))
                    zmax[f"first_{part}"] = max(zmax[f"first_{part}"], float(z1.max()))
                    zmax[f"second_{part}"] = max(zmax[f"second_{part}"], float(z2.max()))
                    zbig["first_gt10"] += int((z1 > 10).sum())
                    zbig["first_gt100"] += int((z1 > 100).sum())
                    zbig["first_live"] += int(z1.numel())
            hp, f = head_features(head, tokens, pad, need, n_ctx, head_chunk, cn)
            u = head.norm(hp, n_ctx)
            pre1, pre3 = head.w1(u), head.w3(u)
            hid = torch.nn.functional.silu(pre1) * pre3
            if dump:
                rowz = torch.empty(need, device=dev)
                rowlive = torch.empty(need, device=dev)
                rowattn = torch.empty(need, device=dev)
                rowzkey = torch.empty(need, dtype=torch.int64, device=dev)
                for i in range(0, need, head_chunk):
                    j = min(i + head_chunk, need)
                    idx_c, cm, cs = colnorm_slice(cn, i, j)
                    xf = tokens[i:j].float()
                    zc = ((xf - cm[idx_c]) / cs[idx_c]).abs().masked_fill(pad[i:j, :, None], 0)
                    cellmax = zc.amax(dim=-1)
                    rowz[i:j], arg = cellmax.max(dim=1)
                    rowzkey[i:j] = colkeys[i:j].gather(1, arg[:, None])[:, 0]
                    rowlive[i:j] = (~pad[i:j]).sum(dim=1)
                    rms = xf.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
                    second = xf.sign() * torch.softmax(xf.abs() / (head.signsoftmax_temp * rms), dim=-1)
                    z = torch.cat([(xf - cm[idx_c]) / cs[idx_c], second], dim=-1)
                    logits = (z @ (head.wk.weight.t() @ head.q.t())) / (head.d_model ** 0.5)
                    attn = logits.masked_fill(pad[i:j, :, None], float("-inf")).softmax(dim=1)
                    rowattn[i:j] = attn.amax(dim=1).mean(dim=-1)
        log(f"step {step}: features, zmax {zmax}")

        host.copy_(tokens)
        del tokens
        buf["tokens"] = None
        head.zero_grad(set_to_none=True)
        leaf = f.detach().requires_grad_(True)
        loss, _p, _t = batched_outputs(ests[e["kind"]], e["kind"], [leaf], [y], n_ctx, dev)
        loss.backward()
        g = leaf.grad.detach().clone()
        del leaf
        torch.cuda.empty_cache()
        log(f"step {step}: loss {float(loss):.4f} |dL/df| {fnorm(g):.4g}")
        tokens = buf["tokens"] = torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev)
        tokens.copy_(host)
        hl = hp.detach().requires_grad_(True)
        head.mix(hl, n_ctx).backward(g)
        gh = hl.grad.detach().clone()
        mix_grads = {n: p.grad.detach().clone() for n, p in head.named_parameters() if p.grad is not None}
        for i in range(0, need, head_chunk):
            j = min(i + head_chunk, need)
            head.pool(tokens[i:j], pad[i:j], colnorm_slice(cn, i, j)).backward(gh[i:j])
        torch.cuda.synchronize(dev)

        grads = {n: p.grad for n, p in head.named_parameters()}
        assert set(grads) == set(PARAMS), sorted(grads)
        sq = {n: fnorm(grads[n]) ** 2 for n in PARAMS}
        total = sum(sq.values())
        split = {}
        for n in ("wk.weight", "wv.weight"):
            gw = grads[n]
            half = gw.shape[1] // 2
            split[n] = {"colctx_first512": fnorm(gw[:, :half]), "signsoftmax_last512": fnorm(gw[:, half:])}
        wk_rows = grads["wk.weight"].double().norm(dim=1)
        rec = {
            "step": step,
            "task": s["task"],
            "kind": e["kind"],
            "ckpt": f"pool_step{s['ckpt_step']}.pt",
            "logged_grad_norm": s["logged_grad_norm"],
            "logged_loss": s.get("logged_loss"),
            "grad_norm": total ** 0.5,
            "loss": float(loss),
            "per_param": {n: {"norm": sq[n] ** 0.5, "sq_share": sq[n] / total} for n in PARAMS},
            "split": split,
            "mix_pass_norms": {n: fnorm(v) for n, v in mix_grads.items()},
            "dL_df": row_stats(g, n_ctx),
            "dL_dh": row_stats(gh, n_ctx),
            "h": row_stats(hp, n_ctx),
            "u": row_stats(u, n_ctx),
            "swiglu_hidden": row_stats(hid, n_ctx),
            "f": row_stats(f, n_ctx),
            "zmax": zmax,
            "z_first_counts": zbig,
            "wk_grad_row_top8_share": float(wk_rows.square().topk(8).values.sum() / wk_rows.square().sum()),
            "y": {
                "ctx_mean": float(y[:n_ctx].mean()), "ctx_std": float(y[:n_ctx].std()),
                "ctx_min": float(y[:n_ctx].min()), "ctx_max": float(y[:n_ctx].max()),
                "query_min": float(y[n_ctx:].min()), "query_max": float(y[n_ctx:].max()),
            },
            "param_norms": {n: fnorm(p) for n, p in head.named_parameters()},
            "rows_substituted": n_sub,
            "query_only_cols": qonly,
        }
        if dump:
            ctx_h = hp[:n_ctx]
            np.savez_compressed(
                out / f"step{step}_tensors.npz",
                h=hp.float().cpu().numpy(), u=u.float().cpu().numpy(),
                pre1=pre1.float().cpu().numpy(), pre3=pre3.float().cpu().numpy(),
                hid=hid.float().cpu().numpy(), f=f.float().cpu().numpy(),
                dL_df=g.float().cpu().numpy(), dL_dh=gh.float().cpu().numpy(), y=y.cpu().numpy(),
                h_ctx_mean=ctx_h.mean(0).float().cpu().numpy(), h_ctx_std=ctx_h.std(0, unbiased=False).float().cpu().numpy(),
                row_zmax=rowz.cpu().numpy(), row_zmax_key=rowzkey.cpu().numpy(),
                row_live=rowlive.cpu().numpy(), row_attn_max=rowattn.cpu().numpy(),
                w2_grad=grads["w2.weight"].float().cpu().numpy(),
                n_ctx=np.array(n_ctx),
            )
        (out / f"step{step}.json").write_text(json.dumps(rec, indent=1))
        log(
            f"step {step} {s['task']}: grad_norm {rec['grad_norm']:.4g} (logged {s['logged_grad_norm']}) loss {rec['loss']:.4f} "
            + " ".join(f"{n} {rec['per_param'][n]['sq_share']:.3f}" for n in PARAMS)
        )
    log("done")
