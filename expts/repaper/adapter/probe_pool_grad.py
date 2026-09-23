import json
import math
import os
import time
from pathlib import Path


def sqsum(t):
    t = t.detach().reshape(-1)
    acc = None
    for i in range(0, t.numel(), 2**26):
        v = t[i : i + 2**26].float().square().sum()
        acc = v if acc is None else acc + v
    return 0.0 if acc is None else float(acc)


def tensors(o):
    import torch

    if torch.is_tensor(o):
        yield o
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from tensors(v)
    elif isinstance(o, dict):
        for v in o.values():
            yield from tensors(v)


def qs(v, levels=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)):
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    return {f"p{round(q * 100)}": float(np.quantile(v, q)) for q in levels}


def spearman(a, b):
    from scipy.stats import spearmanr

    r = spearmanr(a, b).statistic
    return float(r) if r == r else None


class Tracer:
    def __init__(self, model, trace):
        self.model, self.trace = model, trace
        self.mods, self.order, self.handles = {}, [], []
        self.znorm, self.scaler, self.io = [], [], {}
        self.heads_out = None

    def __enter__(self):
        import tabpfn.preprocessing.steps.differentiable_z_norm_step as zmod
        import tabpfn.preprocessing.torch.torch_standard_scaler as smod

        import expts.repaper.adapter.train_adapter as ta

        self.zmod, self.smod, self.ta = zmod, smod, ta
        self.z_orig = zmod.DifferentiableZNormStep._transform
        self.s_orig = smod.TorchStandardScaler.transform
        self.p_orig = ta.preprocess_draw
        tr = self

        def keep(d, key):
            def h(g):
                d[key] = g.detach().float().clone()

            return h

        def keep_sq(d, key):
            def h(g):
                d[key] = d.get(key, 0.0) + sqsum(g)

            return h

        def z_patched(step, X, *a, **kw):
            out = tr.z_orig(step, X, *a, **kw)
            s = {
                "is_test": bool(kw.get("is_test", a[0] if a else False)),
                "shape": list(X.shape),
                "stds": step.stds.detach().flatten().float().cpu().tolist(),
                "means_absmax": float(step.means.detach().abs().max()),
            }
            tr.znorm.append(s)
            if out[0].requires_grad:
                out[0].register_hook(keep(s, "g_out"))
            if X.requires_grad:
                X.register_hook(keep(s, "g_in"))
            return out

        def s_patched(sc, x, fitted_cache):
            out = tr.s_orig(sc, x, fitted_cache)
            s = {
                "shape": list(x.shape),
                "std": qs(fitted_cache["std"].detach().float().flatten().cpu().numpy()),
                "frac_clipped": float((out.detach().abs() >= 100).float().mean()),
            }
            tr.scaler.append(s)
            if out.requires_grad:
                out.register_hook(keep_sq(s, "g_out_sq"))
            if x.requires_grad:
                x.register_hook(keep_sq(s, "g_in_sq"))
            return out

        def p_patched(est, x, y_ctx, n_ctx):
            a, b, c = tr.p_orig(est, x, y_ctx, n_ctx)
            for key, t in (("model_in_ctx", a), ("model_in_query", b)):
                if t.requires_grad:
                    t.register_hook(keep(tr.io, key))
            return a, b, c

        zmod.DifferentiableZNormStep._transform = z_patched
        smod.TorchStandardScaler.transform = s_patched
        ta.preprocess_draw = p_patched

        def make(name):
            r = self.mods.setdefault(name, {"calls": 0, "go_sq": 0.0, "gi_sq": 0.0})

            def go(g):
                r["go_sq"] += sqsum(g)
                if name not in self.order:
                    self.order.append(name)

            def gi(g):
                r["gi_sq"] += sqsum(g)

            def fwd(m, args, kwargs, out):
                r["type"] = type(m).__name__
                r["calls"] += 1
                if name == "heads":
                    self.heads_out = out
                if not self.trace:
                    return None
                for t in tensors(out):
                    if t.requires_grad and t.is_floating_point():
                        t.register_hook(go)
                for t in list(tensors(args)) + list(tensors(kwargs)):
                    if t.requires_grad and t.is_floating_point():
                        t.register_hook(gi)
                return None

            return fwd

        for name, mod in self.model.named_modules():
            if self.trace or name == "heads":
                self.handles.append(mod.register_forward_hook(make(name or "<model>"), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        self.zmod.DifferentiableZNormStep._transform = self.z_orig
        self.smod.TorchStandardScaler.transform = self.s_orig
        self.ta.preprocess_draw = self.p_orig
        for h in self.handles:
            h.remove()
        return False

    def report(self):
        return [
            {
                "name": n,
                "type": self.mods[n].get("type"),
                "calls": self.mods[n]["calls"],
                "go": self.mods[n]["go_sq"] ** 0.5,
                "gi": self.mods[n]["gi_sq"] ** 0.5,
            }
            for n in self.order
        ]


def label_stats(y, n_ctx, kind):
    import numpy as np
    from scipy.stats import kurtosis, skew

    y = np.asarray(y, dtype=np.float64)
    c, q = y[:n_ctx], y[n_ctx:]
    vals, cnt = np.unique(c, return_counts=True)
    sd = float(c.std())
    zc = (c - c.mean()) / max(sd, 1e-20)
    out = {
        "ctx_mean": float(c.mean()),
        "ctx_std_pop": sd,
        "ctx_skew": float(skew(c)),
        "ctx_excess_kurtosis": float(kurtosis(c)),
        "ctx_n_unique": int(len(vals)),
        "ctx_mode": float(vals[cnt.argmax()]),
        "ctx_frac_mode": float(cnt.max() / len(c)),
        "ctx_frac_zero": float((c == 0).mean()),
        "ctx_frac_abs_z_gt3": float((np.abs(zc) > 3).mean()),
        "ctx_frac_abs_z_gt10": float((np.abs(zc) > 10).mean()),
        "ctx_max_abs_z": float(np.abs(zc).max()),
        "ctx_quantiles": qs(c, (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)),
        "ctx_iqr_over_std": float((np.quantile(c, 0.75) - np.quantile(c, 0.25)) / max(sd, 1e-20)),
        "query_frac_mode": float((q == vals[cnt.argmax()]).mean()),
        "query_std_pop": float(q.std()),
        "query_max_abs_z": float(np.abs((q - c.mean()) / max(sd, 1e-20)).max()),
    }
    if kind == "clf":
        out["ctx_pos_rate"] = float((c > 0).mean())
        out["query_pos_rate"] = float((q > 0).mean())
    return out


class Fingerprint:
    def __init__(self, fp):
        self.fp = fp

    def __enter__(self):
        import numpy as np
        import torch
        import tabpfn.preprocessing.steps.add_fingerprint_features_step as fmod

        self.fmod, self.orig = fmod, fmod.AddFingerprintFeaturesStep._transform
        if self.fp is None:
            return self
        fp = self.fp

        def patched(step, X, *, is_test=False):
            n = X.shape[0]
            if fp == "const":
                h = np.full(n, 0.5, dtype=np.float32)
            else:
                h = np.random.default_rng([int(fp), int(is_test), n]).random(n).astype(np.float32)
            added = torch.from_numpy(h).reshape(-1, 1).to(X.device) if isinstance(X, torch.Tensor) else h.reshape(-1, 1)
            return X, added, fmod.FeatureModality.NUMERICAL

        fmod.AddFingerprintFeaturesStep._transform = patched
        return self

    def __exit__(self, *exc):
        self.fmod.AddFingerprintFeaturesStep._transform = self.orig
        return False


def tabpfn_pass(ests, kind, f, y, n_ctx, dev, trace, loss_kind="nll", fp=None):
    from expts.repaper.adapter.train_adapter import batched_outputs

    bardist = ests["reg"].znorm_space_bardist_
    leaf = f.detach().clone().requires_grad_(True)
    with Fingerprint(fp), Tracer(ests[kind].models_[0], trace) as tr:
        loss, pred, tgt = batched_outputs(ests[kind], kind, [leaf], [y], n_ctx, dev)
        lg = tr.heads_out
        used = loss
        if loss_kind == "mse":
            lgf = lg.float()
            lgf = lgf[0] if lgf.shape[0] == 1 else lgf[:, 0]
            yf = y[:n_ctx].float()
            zq = (y[n_ctx:].float() - yf.mean()) / yf.std(correction=0).clamp(min=1e-20)
            used = (bardist.mean(lgf) - zq).square().mean()
        used.backward()
    mods = tr.report() if trace else None
    return {
        "loss": float(loss),
        "loss_used": float(used),
        "g": leaf.grad.detach().clone(),
        "pred": pred.detach().float()[0],
        "tgt": tgt.detach().float()[0],
        "lg": lg.detach().float(),
        "znorm": tr.znorm,
        "scaler": tr.scaler,
        "io": tr.io,
        "modules": mods,
        "heads_go": next((m["go"] for m in mods or [] if m["name"] == "heads"), None),
    }


def head_parts(head, x, p):
    import torch.nn.functional as F

    z = ((x.float() - head.mean) / head.scale).detach().requires_grad_(True)
    qk = head.wk.weight.t() @ head.q.t()
    logits = (z @ qk) / math.sqrt(head.d_model)
    logits = logits.masked_fill(p[..., None], float("-inf"))
    attn = logits.softmax(dim=1)
    v = head.wv(z)
    h = (attn * v).sum(dim=1)
    a = F.silu(head.w1(h)) * head.w3(h)
    f = h + head.w2(a)
    return z, logits, attn, v, h, a, f


def head_decomp(head, tokens, pad, g, n_ctx, chunk):
    import torch

    need = len(g)
    names = [n for n, _ in head.named_parameters()]
    params = [p for _, p in head.named_parameters()]
    acc = {"ctx": [torch.zeros_like(p) for p in params], "query": [torch.zeros_like(p) for p in params]}
    sq = {k: {"ctx": 0.0, "query": 0.0} for k in ("df", "dh", "dv", "dlogits", "dz")}
    amax, check = [], None
    mom = {k: [0.0, 0.0, []] for k in ("h", "a")}
    for i in range(0, need, chunk):
        j = min(i + chunk, need)
        part = "ctx" if i < n_ctx else "query"
        z, lo, attn, v, h, a, f = head_parts(head, tokens[i:j], pad[i:j])
        for k, t in (("h", h), ("a", a)):
            t = t.detach().double()
            mom[k][0] = mom[k][0] + t.sum(0)
            mom[k][1] = mom[k][1] + t.square().sum(0)
            mom[k][2].append(t.norm(dim=1).float())
        if check is None:
            with torch.no_grad():
                check = float((f - head(tokens[i:j], pad[i:j])).abs().max())
        gs = torch.autograd.grad(f, [z, lo, v, h] + params, grad_outputs=g[i:j])
        dz, dlo, dv, dh = gs[:4]
        for a, b in zip(acc[part], gs[4:]):
            a += b
        sq["df"][part] += sqsum(g[i:j])
        sq["dh"][part] += sqsum(dh)
        sq["dv"][part] += sqsum(dv)
        sq["dlogits"][part] += sqsum(dlo.masked_fill(pad[i:j][..., None], 0.0))
        sq["dz"][part] += sqsum(dz)
        amax.append(attn.detach().max(dim=1).values.mean(dim=1))
    per = {}
    for k, name in enumerate(names):
        gc, gq = acc["ctx"][k], acc["query"][k]
        per[name] = {
            "total": float((gc + gq).norm()),
            "ctx": float(gc.norm()),
            "query": float(gq.norm()),
            "param_norm": float(params[k].detach().norm()),
        }
    tot = math.sqrt(sum(v["total"] ** 2 for v in per.values()))
    for v in per.values():
        v["share_sq"] = v["total"] ** 2 / max(tot**2, 1e-30)
    sum_g = g.double().sum(0)
    a_mean = mom["a"][0] / need
    acts = {
        "sum_g_norm": float(sum_g.norm()),
        "sum_g_over_rss": float(sum_g.norm() / g.double().norm().clamp(min=1e-30)),
        "w2_mean_term_norm": float(sum_g.norm() * a_mean.norm()),
    }
    for k, (s1, s2, rn) in mom.items():
        mean = s1 / need
        std = (s2 / need - mean.square()).clamp(min=0).sqrt()
        acts[k] = {
            "row_norm": qs(torch.cat(rn).cpu().numpy()),
            "mean_vec_norm": float(mean.norm()),
            "coord_std": qs(std.cpu().numpy()),
            "coord_absmean": qs(mean.abs().cpu().numpy()),
            "rms_std_over_rms_mean": float(std.square().mean().sqrt() / mean.square().mean().sqrt().clamp(min=1e-30)),
        }
    return {
        "acts": acts,
        "gnorm": tot,
        "gnorm_ctx_only": math.sqrt(sum(float(a.square().sum()) for a in acc["ctx"])),
        "gnorm_query_only": math.sqrt(sum(float(a.square().sum()) for a in acc["query"])),
        "per_param": per,
        "chain": {
            k: {"ctx": math.sqrt(v["ctx"]), "query": math.sqrt(v["query"]), "total": math.sqrt(v["ctx"] + v["query"])}
            for k, v in sq.items()
        },
        "attn_max_weight_mean": float(torch.cat(amax).mean()),
        "parts_vs_module_maxdiff": check,
    }


def pgrad(head, tokens, pad, g, chunk):
    need = len(g)
    head.zero_grad(set_to_none=True)
    for i in range(0, need, chunk):
        j = min(i + chunk, need)
        head(tokens[i:j], pad[i:j]).backward(g[i:j])
    per = {n: float(p.grad.norm()) for n, p in head.named_parameters()}
    head.zero_grad(set_to_none=True)
    return math.sqrt(sum(v * v for v in per.values())), per


def probe_rows(ests, weights, cells, pad, y, kind, n_ctx, chunk, jitter_frac, winsor_q, seed, log, n_fp_seeds=0):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from expts.repaper.adapter.train_pool import head_features

    dev = pad.device
    need = len(y)
    bardist = ests["reg"].znorm_space_bardist_
    rec = {"labels": label_stats(y.cpu().numpy(), n_ctx, kind)}
    rec["cells_per_row"] = qs((~pad).sum(1).float().cpu().numpy(), (0.0, 0.5, 1.0))

    feats = {}
    for wname, head in weights.items():
        f = head_features(head, cells["tokens"], pad, need, chunk)
        with torch.no_grad():
            hh = torch.cat(
                [
                    head_parts(head, cells["tokens"][i : min(i + chunk, need)], pad[i : min(i + chunk, need)])[4].detach()
                    for i in range(0, need, chunk)
                ]
            )
        feats[wname] = (f, hh)
    if cells.get("host") is not None:
        cells["host"].copy_(cells["tokens"])
        cells["tokens"] = None
        torch.cuda.empty_cache()

    runs = {}
    for wname in weights:
        runs[wname] = tabpfn_pass(ests, kind, feats[wname][0], y, n_ctx, dev, True)
        peak = torch.cuda.max_memory_allocated(dev) / 2**30 if torch.cuda.is_available() else 0.0
        log(f"tabpfn {wname}: loss {runs[wname]['loss']:.4f} |dL/df| {float(runs[wname]['g'].norm()):.4g} peak {peak:.1f}GiB")
    fck = feats["ckpt"][0]
    mu, sig = fck[:n_ctx].mean(0), fck[:n_ctx].std(0)
    controls = {
        "repeat": tabpfn_pass(ests, kind, fck, y, n_ctx, dev, False),
        "std_input": tabpfn_pass(ests, kind, (fck - mu) / sig, y, n_ctx, dev, False),
    }
    if kind == "reg":
        yc = y[:n_ctx]
        lo_q, hi_q = torch.quantile(yc, winsor_q), torch.quantile(yc, 1 - winsor_q)
        controls["winsor"] = tabpfn_pass(ests, kind, fck, y.clamp(lo_q, hi_q), n_ctx, dev, False)
        gen = torch.Generator(device=dev).manual_seed(seed)
        noise = (torch.rand(need, device=dev, generator=gen) - 0.5) * jitter_frac * yc.std(correction=0)
        controls["jitter"] = tabpfn_pass(ests, kind, fck, y + noise, n_ctx, dev, False)
        controls["mse_loss"] = tabpfn_pass(ests, kind, fck, y, n_ctx, dev, False, loss_kind="mse")
    for k in range(1, n_fp_seeds + 1):
        controls[f"fp_seed{k}"] = tabpfn_pass(ests, kind, fck, y, n_ctx, dev, False, fp=k)
    if n_fp_seeds:
        controls["fp_seed1_std_input"] = tabpfn_pass(ests, kind, (fck - mu) / sig, y, n_ctx, dev, False, fp=1)
        controls["fp_const"] = tabpfn_pass(ests, kind, fck, y, n_ctx, dev, False, fp="const")
        controls["fp_const_std_input"] = tabpfn_pass(ests, kind, (fck - mu) / sig, y, n_ctx, dev, False, fp="const")
        if kind == "reg":
            controls["fp_const_winsor"] = tabpfn_pass(ests, kind, fck, y.clamp(lo_q, hi_q), n_ctx, dev, False, fp="const")
    log("controls " + ", ".join(f"{k} loss {v['loss_used']:.4f}" for k, v in controls.items()))

    if cells.get("host") is not None:
        cells["tokens"] = cells["host"].to(dev)
    tokens = cells["tokens"]
    head = weights["ckpt"]
    main_run = runs["ckpt"]
    g = main_run["g"]
    rec["head"] = {w: head_decomp(weights[w], tokens, pad, runs[w]["g"], n_ctx, chunk) for w in weights}
    rec["gnorm"] = rec["head"]["ckpt"]["gnorm"]
    rec["loss"] = main_run["loss"]
    log(f"gnorm {' '.join(f'{w} {v['gnorm']:.4g}' for w, v in rec['head'].items())}")

    def invariant_part(gg, ff):
        fc = ff - ff.mean(0)
        g1 = gg - gg.mean(0)
        return g1 - (g1 * fc).sum(0) / (fc * fc).sum(0).clamp(min=1e-30) * fc

    def proj_stats(gg, ff):
        gp = invariant_part(gg, ff)
        return {
            "violating_frac": float((gg - gp).norm() / gg.norm().clamp(min=1e-30)),
            "mean_frac": float(gg.mean(0).norm() * math.sqrt(len(gg)) / gg.norm().clamp(min=1e-30)),
            "gnorm_invariant_part": pgrad(head, tokens, pad, gp, chunk)[0],
            "gnorm_violating_part": pgrad(head, tokens, pad, gg - gp, chunk)[0],
        }

    rec["projection"] = {w: proj_stats(runs[w]["g"], feats[w][0]) for w in weights}
    log("projection " + " ".join(f"{w} viol {v['violating_frac']:.3f} inv {v['gnorm_invariant_part']:.4g} viol-part {v['gnorm_violating_part']:.4g}" for w, v in rec["projection"].items()))
    rec["controls"] = {}
    for k, v in controls.items():
        gk = v["g"] / sig if k.endswith("std_input") else v["g"]
        gn, per = pgrad(head, tokens, pad, gk, chunk)
        pair = {"fp_seed1_std_input": "fp_seed1", "fp_const_std_input": "fp_const"}.get(k)
        rec["controls"][k] = {
            "projection": proj_stats(gk, fck),
            "cos_dLdf_vs_pair": (
                float(F.cosine_similarity(gk.flatten(), (controls[pair]["g"]).flatten(), dim=0)) if pair else None
            ),
            "loss": v["loss"],
            "loss_used": v["loss_used"],
            "gnorm": gn,
            "per_param": per,
            "dLdf_norm": float(v["g"].norm()),
            "cos_dLdf_vs_main": float(F.cosine_similarity(gk.flatten(), g.flatten(), dim=0)),
        }

    rec["tabpfn"] = {}
    for wname, r in runs.items():
        zs = []
        for zz in r["znorm"]:
            gi, go = zz.get("g_in"), zz.get("g_out")
            zs.append(
                {
                    "is_test": zz["is_test"],
                    "shape": zz["shape"],
                    "means_absmax": zz["means_absmax"],
                    "stds": qs(zz["stds"]),
                    "g_in": float(gi.norm()) if gi is not None else None,
                    "g_out": float(go.norm()) if go is not None else None,
                }
            )
        rec["tabpfn"][wname] = {
            "loss": r["loss"],
            "dLdf": float(r["g"].norm()),
            "dLdf_ctx": float(r["g"][:n_ctx].norm()),
            "dLdf_query": float(r["g"][n_ctx:].norm()),
            "heads_out_grad": r["heads_go"],
            "model_input_grad": {k: float(v.norm()) for k, v in r["io"].items()},
            "znorm": zs,
            "internal_scaler": [
                {
                    "shape": sc["shape"],
                    "std": sc["std"],
                    "frac_clipped": sc["frac_clipped"],
                    "g_in": sc.get("g_in_sq", 0.0) ** 0.5,
                    "g_out": sc.get("g_out_sq", 0.0) ** 0.5,
                }
                for sc in r["scaler"]
            ],
            "modules": r["modules"],
        }

    sig_np = sig.float().cpu().numpy()
    g_col = g.norm(dim=0).cpu().numpy()
    col = {
        "sigma_ctx": sig_np.tolist(),
        "mean_ctx": mu.float().cpu().numpy().tolist(),
        "sigma_h_ctx": feats["ckpt"][1][:n_ctx].std(0).cpu().numpy().tolist(),
        "g_ctx": g[:n_ctx].norm(dim=0).cpu().numpy().tolist(),
        "g_query": g[n_ctx:].norm(dim=0).cpu().numpy().tolist(),
        "g_total": g_col.tolist(),
    }
    zc = [zz for zz in main_run["znorm"] if not zz["is_test"] and zz.get("g_out") is not None]
    if zc:
        zst = np.asarray(zc[0]["stds"])
        col["znorm_stds_match_sigma_maxrel"] = float(np.abs(zst / sig_np - 1).max()) if len(zst) == len(sig_np) else None
        col["dL_dxhat_ctx"] = zc[0]["g_out"].norm(dim=0).cpu().numpy().tolist()
        if zc[0].get("g_in") is not None:
            col["dL_dx_znorm_in_ctx"] = zc[0]["g_in"].norm(dim=0).cpu().numpy().tolist()
    zq = [zz for zz in main_run["znorm"] if zz["is_test"] and zz.get("g_out") is not None]
    if zq:
        col["dL_dxhat_query"] = zq[0]["g_out"].norm(dim=0).cpu().numpy().tolist()
    pc = []
    for j in range(g.shape[1]):
        m = torch.zeros_like(g)
        m[:, j] = g[:, j]
        pc.append(pgrad(head, tokens, pad, m, chunk)[0])
    pc = np.asarray(pc)
    col["param_gnorm_from_col"] = pc.tolist()
    ordc = np.argsort(-pc)
    topk = {}
    for k in (1, 4, 8, 16):
        idx = torch.as_tensor(ordc[:k].copy(), device=dev)
        m = torch.zeros_like(g)
        m[:, idx] = g[:, idx]
        topk[str(k)] = pgrad(head, tokens, pad, m, chunk)[0]
    gs_ = g_col * sig_np
    col["summary"] = {
        "sigma": qs(sig_np),
        "sigma_ratio_max_min": float(sig_np.max() / sig_np.min()),
        "g_total": qs(g_col),
        "g_times_sigma": qs(gs_),
        "spread_p90_p10_g": float(np.quantile(g_col, 0.9) / np.quantile(g_col, 0.1)),
        "spread_p90_p10_g_times_sigma": float(np.quantile(gs_, 0.9) / np.quantile(gs_, 0.1)),
        "spearman_g_vs_inv_sigma": spearman(g_col, 1.0 / sig_np),
        "spearman_paramcol_vs_inv_sigma": spearman(pc, 1.0 / sig_np),
        "spearman_paramcol_vs_g": spearman(pc, g_col),
        "top_cols_by_param": ordc[:8].tolist(),
        "param_share_sq_top1": float(pc[ordc[0]] ** 2 / (pc**2).sum()),
        "param_share_sq_top4": float((pc[ordc[:4]] ** 2).sum() / (pc**2).sum()),
        "param_gnorm_topk_cols": topk,
        "param_gnorm_sum_over_cols": float(pc.sum()),
    }
    rec["columns"] = col

    rn = g.norm(dim=1)
    rsq = rn.square()
    tot = float(rsq.sum())
    srt = torch.sort(rsq, descending=True).values
    zlab = (y - y[:n_ctx].mean()) / y[:n_ctx].std(correction=0).clamp(min=1e-20)
    is_mode = y == rec["labels"]["ctx_mode"]
    tail = zlab.abs() > 3
    rowd = {
        "ctx_share_sq": float(rsq[:n_ctx].sum() / tot),
        "per_row_ctx_mean": float(rn[:n_ctx].mean()),
        "per_row_query_mean": float(rn[n_ctx:].mean()),
        "row_norm_ctx": qs(rn[:n_ctx].cpu().numpy()),
        "row_norm_query": qs(rn[n_ctx:].cpu().numpy()),
        "top1_share_sq": float(srt[0] / tot),
        "top10_share_sq": float(srt[:10].sum() / tot),
        "top1pct_share_sq": float(srt[: max(1, need // 100)].sum() / tot),
        "top10pct_share_sq": float(srt[: max(1, need // 10)].sum() / tot),
        "mode_rows_frac": float(is_mode.float().mean()),
        "mode_rows_share_sq": float(rsq[is_mode].sum() / tot),
        "tail_rows_frac": float(tail.float().mean()),
        "tail_rows_share_sq": float(rsq[tail].sum() / tot),
        "spearman_rownorm_vs_abs_zlabel_ctx": spearman(rn[:n_ctx].cpu().numpy(), zlab[:n_ctx].abs().cpu().numpy()),
        "spearman_rownorm_vs_abs_zlabel_query": spearman(rn[n_ctx:].cpu().numpy(), zlab[n_ctx:].abs().cpu().numpy()),
    }
    thr = srt[max(1, need // 100) - 1]
    ar = torch.arange(need, device=dev)
    masks = {
        "ctx_rows": ar < n_ctx,
        "query_rows": ar >= n_ctx,
        "top1pct_rows": rsq >= thr,
        "mode_rows": is_mode,
        "non_mode_rows": ~is_mode,
        "tail_rows": tail,
    }
    rowd["param_gnorm_by_rowset"] = {k: pgrad(head, tokens, pad, g * m[:, None].float(), chunk)[0] for k, m in masks.items()}
    rec["rows"] = rowd

    pred, tgt = main_run["pred"], main_run["tgt"]
    qd = {}
    if kind == "clf":
        pt = torch.where(tgt > 0, pred, 1 - pred).clamp(min=1e-12)
        nll = -pt.log()
        qd["p_true"] = qs(pt.cpu().numpy())
        qd["recomputed_minus_reported"] = float(nll.mean()) - main_run["loss"]
    else:
        lg = main_run["lg"]
        lg = lg[0] if lg.shape[0] == 1 else lg[:, 0]
        yf = y[:n_ctx]
        zq = (y[n_ctx:] - yf.mean()) / yf.std(correction=0).clamp(min=1e-20)
        with torch.no_grad():
            nll = bardist(lg.unsqueeze(1), zq.unsqueeze(1)).squeeze(1)
            bidx = bardist.map_to_bucket_idx(zq).clamp(0, bardist.num_bars - 1)
            width = bardist.bucket_widths[bidx]
            probs = lg.softmax(-1)
            pb = probs.gather(-1, bidx[:, None]).squeeze(1)
            ent = -(probs * probs.clamp(min=1e-30).log()).sum(-1)
        bv, bc = torch.unique(bidx, return_counts=True)
        top_b = int(bv[bc.argmax()])
        in_top = bidx == top_b
        qd.update(
            {
                "logits_shape": list(main_run["lg"].shape),
                "recomputed_loss": float(nll.mean()),
                "recomputed_minus_reported": float(nll.mean()) - main_run["loss"],
                "width_at_target": qs(width.cpu().numpy()),
                "p_target_bucket": qs(pb.cpu().numpy()),
                "density_at_target": qs(torch.exp(-nll).cpu().numpy()),
                "entropy_buckets": qs(ent.cpu().numpy()),
                "most_common_bucket": top_b,
                "most_common_bucket_width": float(bardist.bucket_widths[top_b]),
                "most_common_bucket_frac": float(in_top.float().mean()),
                "nll_in_top_bucket_mean": float(nll[in_top].mean()),
                "nll_other_mean": float(nll[~in_top].mean()) if bool((~in_top).any()) else None,
                "bucket_width_all": qs(bardist.bucket_widths.cpu().numpy()),
                "n_buckets": int(bardist.num_bars),
                "n_distinct_target_buckets": int(len(bv)),
                "frac_query_in_tail_buckets": float(((bidx == 0) | (bidx == bardist.num_bars - 1)).float().mean()),
                "rownorm_query_top_bucket_mean": float(rn[n_ctx:][in_top].mean()),
                "rownorm_query_other_mean": float(rn[n_ctx:][~in_top].mean()) if bool((~in_top).any()) else None,
            }
        )
    qd["nll"] = qs(nll.cpu().numpy())
    qd["spearman_rownorm_vs_nll_query"] = spearman(rn[n_ctx:].cpu().numpy(), nll.cpu().numpy())
    rec["query_loss"] = qd
    return rec, g




def setup(join_pre_dir, task_list, ckpt, tabpfn_dir, ckpt_dir, steps, n_tasks_total, n_queries, need, seed, log):
    import numpy as np
    import torch

    from expts.repaper.adapter.pool_head import PoolHead
    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX, make_dataset
    from expts.repaper.adapter.train_adapter import make_ests
    from expts.repaper.baselines.rel2tab.featurizer import table_offset_and_len
    from rt.data import get_tasks
    from rt.model import load_rt_model

    dev = "cuda:0"
    rows = json.loads(Path(__file__).with_name(task_list).read_text())
    assert len(rows) == n_tasks_total, (len(rows), n_tasks_total)
    order = np.random.default_rng([seed, 1]).permutation(len(rows))
    net, config = load_rt_model(ckpt, device=dev, compile=True)
    net = net.to(torch.bfloat16).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    d_model = net.d_model
    tasks, didx = [], {}
    for k, ti in enumerate(sorted({int(order[s]) for s in steps})):
        db, name, kind, _n, _sub = rows[ti]
        (tk,) = get_tasks(join_pre_dir, [(db, name)], ("train",))
        assert tk.task_type == kind, (db, name, tk.task_type, kind)
        lo, n = table_offset_and_len(join_pre_dir, db, tk.table_name)
        tasks.append({"db": db, "task": name, "kind": kind, "t": tk, "lo": lo, "n": n})
        didx[ti] = k
    ds = make_dataset([e["t"] for e in tasks], join_pre_dir, config, mmap_populate=False)
    log(f"sampler over {len(tasks)} tasks built")
    ests = make_ests(tabpfn_dir, dev, seed, n_queries, "bf16")
    log(f"tabpfn loaded; bar distribution {ests['reg'].znorm_space_bardist_.num_bars} buckets")
    ck_dir = Path(ckpt_dir).expanduser()
    avail = sorted(int(p.stem.replace("pool_step", "")) for p in ck_dir.glob("pool_step*.pt"))

    def load_head(step):
        ck = torch.load(ck_dir / f"pool_step{step}.pt", map_location="cpu", weights_only=True)
        h = PoolHead(d_model, n_queries).to(dev)
        h.load_state_dict(ck["state_dict"])
        return h

    cells = {
        "tokens": torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev),
        "host": torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, pin_memory=True),
    }
    pad = torch.empty(need, LOCAL_CTX, dtype=torch.bool, device=dev)
    labels = torch.empty(need, dtype=torch.float32, device=dev)

    def embed(step):
        from expts.repaper.adapter.train_pool import embed_rows

        ti = int(order[step])
        e = tasks[didx[ti]]
        cand = np.random.default_rng([seed, step]).permutation(e["n"]) + e["lo"]
        if cells["tokens"] is None:
            cells["tokens"] = torch.empty(need, LOCAL_CTX, d_model, dtype=torch.bfloat16, device=dev)
        with torch.inference_mode():
            filled, n_sub, _a, _b = embed_rows(net, ds, didx[ti], cand, cells["tokens"], pad, labels, need, dev)
        return e, filled, n_sub

    return {"ests": ests, "avail": avail, "load_head": load_head, "cells": cells, "pad": pad, "labels": labels, "embed": embed, "dev": dev}


def main(
    *,
    join_pre_dir: str,
    task_list: str,
    ckpt: str,
    tabpfn_dir: str,
    ckpt_dir: str,
    out_dir: str,
    steps: list[int],
    n_ctx: int,
    n_query: int,
    n_queries: int,
    head_chunk: int,
    n_tasks_total: int,
    jitter_frac: float,
    winsor_q: float,
    seed: int,
    n_fp_seeds: int,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    need = n_ctx + n_query
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(msg):
        print(f"[{(time.time() - t_start) / 60:7.1f} min] {msg}", flush=True)

    todo = [s for s in steps if not (out / f"step{s}.json").exists()]
    log(f"{len(steps)} steps requested, {len(todo)} to do: {todo}")
    if not todo:
        return
    env = setup(join_pre_dir, task_list, ckpt, tabpfn_dir, ckpt_dir, todo, n_tasks_total, n_queries, need, seed, log)
    heads = {}

    def get_head(step):
        if step not in heads:
            heads[step] = env["load_head"](step)
        return heads[step]

    for s in todo:
        t0 = time.perf_counter()
        ck_step = max(a for a in env["avail"] if a <= s)
        e, filled, n_sub = env["embed"](s)
        name = f"{e['db']}/{e['task']}"
        rec = {"step": s, "task": name, "kind": e["kind"], "ckpt_step": ck_step, "filled": filled, "substituted": n_sub}
        if filled < need:
            rec["skipped"] = f"only {filled} rows"
            (out / f"step{s}.json").write_text(json.dumps(rec, indent=1))
            log(f"step {s}: skipped, only {filled} rows")
            continue
        log(f"step {s} {e['kind']} {name}: embedded {filled} rows ({n_sub} substituted) in {time.perf_counter() - t0:.0f}s (ckpt pool_step{ck_step})")
        weights = {"ckpt": get_head(ck_step)}
        if ck_step != 0:
            weights["init"] = get_head(0)
        part, _g = probe_rows(
            env["ests"], weights, env["cells"], env["pad"], env["labels"].clone(), e["kind"], n_ctx, head_chunk,
            jitter_frac, winsor_q, seed, lambda m, s=s: log(f"step {s}: {m}"), n_fp_seeds,
        )
        rec.update(part)
        rec["seconds"] = time.perf_counter() - t0
        (out / f"step{s}.json").write_text(json.dumps(rec, indent=1))
        log(f"step {s} done {rec['seconds']:.0f}s: loss {rec['loss']:.4f} gnorm {rec['gnorm']:.4g}")
        torch.cuda.empty_cache()
    log("done")


def replay(
    *,
    join_pre_dir: str,
    task_list: str,
    ckpt: str,
    tabpfn_dir: str,
    ckpt_dir: str,
    out_dir: str,
    probe_steps: list[int],
    last_step: int,
    n_ctx: int,
    n_query: int,
    n_queries: int,
    head_chunk: int,
    n_tasks_total: int,
    lr: float,
    lr_min: float,
    warmup_steps: int,
    total_steps: int,
    grad_norm_max: float,
    jitter_frac: float,
    winsor_q: float,
    seed: int,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from expts.repaper.adapter.train_adapter import batched_outputs
    from expts.repaper.adapter.train_pool import head_features

    need = n_ctx + n_query
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(msg):
        print(f"[{(time.time() - t_start) / 60:7.1f} min] {msg}", flush=True)

    env = setup(join_pre_dir, task_list, ckpt, tabpfn_dir, ckpt_dir, list(range(last_step + 1)), n_tasks_total, n_queries, need, seed, log)
    dev, cells, pad, labels = env["dev"], env["cells"], env["pad"], env["labels"]
    head = env["load_head"](0)
    init_head = env["load_head"](0)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    state_path = out / "replay.pt"
    start = 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        start = st["step"]
        log(f"resumed replay at step {start}")

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        tt = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * tt))

    for step in range(start, last_step + 1):
        t0 = time.perf_counter()
        rec = {"step": step}
        if step in env["avail"]:
            ref = env["load_head"](step)
            rec["vs_saved_ckpt"] = {
                n: float((p.detach() - q.detach()).norm() / q.detach().norm().clamp(min=1e-12))
                for (n, p), (_m, q) in zip(head.named_parameters(), ref.named_parameters())
            }
            log(f"step {step}: replay vs pool_step{step}.pt rel diff " + " ".join(f"{n} {v:.2e}" for n, v in rec["vs_saved_ckpt"].items()))
        e, filled, n_sub = env["embed"](step)
        kind = e["kind"]
        rec.update({"task": f"{e['db']}/{e['task']}", "kind": kind, "filled": filled, "substituted": n_sub})
        y = labels.clone()
        reason = None
        if filled < need:
            reason = f"only {filled} rows"
        elif kind == "reg" and float(y[:n_ctx].std()) < 1e-6:
            reason = "constant context label"
        elif kind == "clf" and (y[:n_ctx] > 0).float().mean().item() in (0.0, 1.0):
            reason = "one class in context"
        g = None
        if reason is None and step in probe_steps and not (out / f"step{step}.json").exists():
            prec, g = probe_rows(
                env["ests"], {"ckpt": head, "init": init_head}, cells, pad, y, kind, n_ctx, head_chunk,
                jitter_frac, winsor_q, seed, lambda m, s=step: log(f"step {s}: {m}"),
            )
            prec.update({"step": step, "task": rec["task"], "kind": kind, "ckpt_step": "replay", "filled": filled, "substituted": n_sub})
            (out / f"step{step}.json").write_text(json.dumps(prec, indent=1))
            rec["probed"] = True
        elif reason is None:
            f = head_features(head, cells["tokens"], pad, need, head_chunk)
            cells["host"].copy_(cells["tokens"])
            cells["tokens"] = None
            leaf = f.to(dev).requires_grad_(True)
            loss, _p, _t = batched_outputs(env["ests"][kind], kind, [leaf], [y], n_ctx, dev)
            loss.backward()
            g = leaf.grad.detach()
            rec["loss"] = float(loss)
            del leaf, loss
            cells["tokens"] = cells["host"].to(dev)
        if g is not None and bool(torch.isfinite(g).all()):
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            for i in range(0, need, head_chunk):
                j = min(i + head_chunk, need)
                head(cells["tokens"][i:j], pad[i:j]).backward(g[i:j])
            rec["per_param"] = {n: float(p.grad.norm()) for n, p in head.named_parameters()}
            rec["gnorm"] = float(torch.nn.utils.clip_grad_norm_(head.parameters(), grad_norm_max))
            opt.step()
        else:
            rec["skipped"] = reason or "non-finite"
        rec["seconds"] = time.perf_counter() - t0
        with open(out / "replay.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        tmp = out / f"replay.pt.{os.getpid()}.tmp"
        torch.save({"step": step + 1, "head": head.state_dict(), "opt": opt.state_dict()}, tmp)
        os.replace(tmp, state_path)
        log(f"replay step {step} {kind} {rec['task']} loss {rec.get('loss', float('nan')):.4f} gnorm {rec.get('gnorm', float('nan')):.4g} {rec['seconds']:.0f}s" + (" probed" if rec.get("probed") else ""))
        torch.cuda.empty_cache()
    log("done")
