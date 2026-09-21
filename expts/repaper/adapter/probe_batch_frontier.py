import gc
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from expts.repaper.adapter.train_adapter import (
    Pool,
    build_adapter,
    draw_fixed,
    load_index,
)

GIB = 1024**3


def _fresh_est(task_type, tabpfn_dir, device, seed, n_features, precision, fingerprint):
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    model_path = Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors"
    cls = TabPFNClassifier if task_type == "clf" else TabPFNRegressor
    est = cls(
        n_estimators=1,
        model_path=model_path,
        device=device,
        fit_mode="fit_preprocessors",
        differentiable_input=True,
        random_state=seed,
        inference_precision={"fp32": torch.float32, "bf16": torch.bfloat16}[precision],
        inference_config={
            "OUTLIER_REMOVAL_STD": None,
            "FINGERPRINT_FEATURE": fingerprint,
        },
    )
    warm = torch.randn(32, n_features, device=device)
    y = torch.arange(32, device=device) % 2
    est.fit_with_differentiable_input(warm, y if task_type == "clf" else y.float())
    if task_type == "reg":
        est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
    for model in est.models_:
        for prm in model.parameters():
            prm.requires_grad_(False)
    return est


def _perf_options(est, recompute):
    import dataclasses

    base = est.models_[0].get_default_performance_options()
    return dataclasses.replace(base, force_recompute_layer=recompute)


def _batched_clf_loss(est, adapter, emb, lab, n_ctx, perf, device):
    from torch import nn

    x = adapter(emb.to(device).float())
    y = lab.to(device)
    b = x.shape[0]
    cat_ix = [[[]] for _ in range(b)]
    cfgs = [[c] * b for c in est.ensemble_configs_]
    est.fit_from_preprocessed(
        [x[:, :n_ctx]],
        [y[:, :n_ctx].float()],
        cat_ix,
        cfgs,
        performance_options=perf,
    )
    logits = est.forward([x[:, n_ctx:]], use_inference_mode=False, return_logits=True)
    return nn.functional.cross_entropy(logits.float(), y[:, n_ctx:]), logits


def _legacy_clf_loss(est, adapter, emb, lab, n_ctx, device):
    from torch import nn

    x = adapter(emb[0].to(device).float()).float()
    y = lab[0].to(device)
    est.fit_with_differentiable_input(x[:n_ctx], y[:n_ctx])
    logits = est.forward(x[n_ctx:], use_inference_mode=True, return_logits=True).float()
    return nn.functional.cross_entropy(logits, y[n_ctx:]), logits


def _legacy_reg_loss(est, adapter, emb, yy, n_ctx, device):
    x = adapter(emb[0].to(device).float()).float()
    y = yy[0].to(device)
    est.fit_with_differentiable_input(x[:n_ctx], y[:n_ctx])
    logits, _per, _b = est.forward(x[n_ctx:], use_inference_mode=True)
    logits = logits.float()
    z = (y[n_ctx:] - est.y_train_mean_) / est.y_train_std_
    loss = est.znorm_space_bardist_(
        logits.transpose(0, 1).unsqueeze(0), z.unsqueeze(0)
    ).mean()
    return loss, logits


def _batched_reg_loss(est, adapter, emb, yy, n_ctx, perf, device, via_engine):
    import torch

    x = adapter(emb.to(device).float())
    y = yy.to(device).float()
    b = x.shape[0]
    yc = y[:, :n_ctx]
    mean = yc.mean(dim=1, keepdim=True)
    std = torch.clamp(yc.std(dim=1, keepdim=True, correction=0), min=1e-20)
    zc = (yc - mean) / std
    zq = (y[:, n_ctx:] - mean) / std
    cat_ix = [[[]] for _ in range(b)]
    cfgs = [[c] * b for c in est.ensemble_configs_]
    est.fit_from_preprocessed(
        [x[:, :n_ctx]], [zc], cat_ix, cfgs, performance_options=perf
    )
    if not via_engine:
        logits, _per, _b = est.forward([x[:, n_ctx:]], use_inference_mode=False)
        logits = logits.float()
    else:
        est.executor_.use_torch_inference_mode(use_inference=False)
        outs = list(
            est.executor_.iter_outputs(
                [x[:, n_ctx:]], autocast=est.use_autocast_, task_type="regression"
            )
        )
        assert len(outs) == 1, len(outs)
        out, cfg = outs[0]
        assert all(c.target_transform is None for c in cfg), "target_transform set"
        logits = out.float().transpose(0, 1)
    loss = est.znorm_space_bardist_(logits, zq).mean()
    return loss, logits


def _time_config(fn, adapter, warmup, iters):
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for it in range(warmup + iters):
        adapter.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = fn()
        loss.backward()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0, float(loss.detach())))
        del loss
        if it == warmup - 1:
            torch.cuda.reset_peak_memory_stats()
    peak = torch.cuda.max_memory_allocated() / GIB
    tail = ts[warmup:]
    return statistics.median(t[0] for t in tail), tail[0][1], peak


def _oom(e):
    return "out of memory" in str(e).lower() or "CUDA out of memory" in str(e)


def main(
    *,
    features_root: str,
    tabpfn_dir: str,
    stats_path: str,
    d_feat: int,
    min_rows: int,
    n_query: int,
    sweep_n_ctx: list[int],
    sweep_batch: list[int],
    sweep_precision: list[str],
    sweep_recompute: list[bool],
    n_tasks: int,
    warmup: int,
    iters: int,
    budget_s: float,
    probe_d_out: int,
    seed: int,
) -> None:
    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch

    t_start = time.perf_counter()
    device = "cuda"
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}",
        flush=True,
    )
    print(
        f"tf32 matmul {torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn {torch.backends.cudnn.allow_tf32} "
        f"precision {torch.get_float32_matmul_precision()}",
        flush=True,
    )

    entries = load_index(features_root, min_rows)
    need = max(sweep_n_ctx) + n_query
    by_type = {}
    for tt in ("clf", "reg"):
        cand = sorted(
            (e for e in entries if e["task_type"] == tt and e["n"] >= need),
            key=lambda e: (-e["n"], e["db"], e["task"]),
        )
        by_type[tt] = cand[: n_tasks // 2]
        print(
            f"{tt}: {len(cand)} tasks with >= {need} rows, using "
            + ", ".join(f"{e['db']}/{e['task']}(n={e['n']})" for e in by_type[tt]),
            flush=True,
        )
    assert by_type["clf"] and by_type["reg"], "not enough tasks"

    pool = Pool(entries)
    rng = np.random.default_rng(seed)
    bmax = max(sweep_batch)
    draws = {}
    for tt in ("clf", "reg"):
        for n_ctx in sweep_n_ctx:
            got = []
            tries = 0
            while len(got) < bmax and tries < bmax * 8:
                tries += 1
                e = by_type[tt][len(got) % len(by_type[tt])]
                emb, y = draw_fixed(rng, pool, e, n_ctx, n_query)
                if tt == "clf":
                    lab = (y > 0).long()
                    if int(lab[:n_ctx].min()) == int(lab[:n_ctx].max()):
                        continue
                    got.append((emb, lab))
                else:
                    if float(y[:n_ctx].std()) < 1e-6:
                        continue
                    got.append((emb, y.float()))
            assert len(got) == bmax, (tt, n_ctx, len(got))
            draws[(tt, n_ctx)] = (
                torch.stack([g[0] for g in got]),
                torch.stack([g[1] for g in got]),
            )
    print(f"staged {len(draws)} draw stacks of {bmax}", flush=True)

    adapter = build_adapter("linear", d_feat, d_feat, 0, stats_path, device)
    rows = []

    def emit(rec):
        rows.append(rec)
        print("RESULT " + json.dumps(rec), flush=True)

    def left():
        return budget_s - (time.perf_counter() - t_start)

    # ---------------- phase 1: correctness at B=1 ----------------
    print("\n=== B=1 loss equivalence: legacy vs fit_from_preprocessed ===", flush=True)
    n_ctx0 = min(sweep_n_ctx)
    emb, lab = draws[("clf", n_ctx0)]
    clf_fp = _fresh_est("clf", tabpfn_dir, device, seed, d_feat, "fp32", True)
    clf_nofp = _fresh_est("clf", tabpfn_dir, device, seed, d_feat, "fp32", False)
    clf_bat = _fresh_est("clf", tabpfn_dir, device, seed, d_feat, "fp32", False)
    perf0 = _perf_options(clf_bat, False)

    def gnorm(loss):
        adapter.zero_grad(set_to_none=True)
        loss.backward()
        g = adapter[1].weight.grad
        return (
            float(g.norm()),
            bool(torch.isfinite(g).all()),
            float(g.abs().max()),
        )

    l_fp, _ = _legacy_clf_loss(clf_fp, adapter, emb[:1], lab[:1], n_ctx0, device)
    v_fp = float(l_fp)
    g_fp = gnorm(l_fp)
    l_no, _ = _legacy_clf_loss(clf_nofp, adapter, emb[:1], lab[:1], n_ctx0, device)
    v_no = float(l_no)
    g_no = gnorm(l_no)
    l_ba, _ = _batched_clf_loss(
        clf_bat, adapter, emb[:1], lab[:1], n_ctx0, perf0, device
    )
    v_ba = float(l_ba)
    g_ba = gnorm(l_ba)
    emit(
        {
            "check": "clf_b1_equivalence",
            "n_ctx": n_ctx0,
            "loss_legacy_fingerprint": v_fp,
            "loss_legacy_nofingerprint": v_no,
            "loss_batched_nofingerprint": v_ba,
            "rel_batched_vs_legacy_nofp": abs(v_ba - v_no) / abs(v_no),
            "rel_batched_vs_legacy_fp": abs(v_ba - v_fp) / abs(v_fp),
            "rel_fp_vs_nofp_legacy": abs(v_fp - v_no) / abs(v_fp),
            "gnorm_legacy_fingerprint": g_fp,
            "gnorm_legacy_nofingerprint": g_no,
            "gnorm_batched": g_ba,
        }
    )

    print("\n=== B=1 batched agreement across n_ctx ===", flush=True)
    for n_ctx in sweep_n_ctx:
        e2, l2 = draws[("clf", n_ctx)]
        a, _ = _legacy_clf_loss(clf_nofp, adapter, e2[:1], l2[:1], n_ctx, device)
        b, _ = _batched_clf_loss(clf_bat, adapter, e2[:1], l2[:1], n_ctx, perf0, device)
        emit(
            {
                "check": "clf_b1_by_nctx",
                "n_ctx": n_ctx,
                "loss_legacy_nofp": float(a),
                "loss_batched": float(b),
                "rel": abs(float(a) - float(b)) / abs(float(a)),
            }
        )

    print("\n=== batched path independence: B=2 vs B=1 per-draw losses ===", flush=True)
    try:
        e4, l4 = draws[("clf", n_ctx0)]
        from torch import nn

        nb = 2
        _lb, logits4 = _batched_clf_loss(
            clf_bat, adapter, e4[:nb], l4[:nb], n_ctx0, perf0, device
        )
        per_b4 = [
            float(
                nn.functional.cross_entropy(
                    logits4[i].float().unsqueeze(0),
                    l4[i, n_ctx0:].to(device).unsqueeze(0),
                )
            )
            for i in range(nb)
        ]
        per_b1 = []
        for i in range(nb):
            li, _ = _batched_clf_loss(
                clf_bat, adapter, e4[i : i + 1], l4[i : i + 1], n_ctx0, perf0, device
            )
            per_b1.append(float(li))
        emit(
            {
                "check": "clf_batch_independence",
                "n_ctx": n_ctx0,
                "B": nb,
                "per_draw_in_batch": per_b4,
                "per_draw_at_B1": per_b1,
                "max_rel": max(
                    abs(a - b) / abs(b) for a, b in zip(per_b4, per_b1, strict=True)
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        emit(
            {"check": "clf_batch_independence", "error": f"{type(exc).__name__}: {exc}"}
        )
        adapter.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== regressor batched path ===", flush=True)
    reg_leg = _fresh_est("reg", tabpfn_dir, device, seed, d_feat, "fp32", True)
    reg_bat = _fresh_est("reg", tabpfn_dir, device, seed, d_feat, "fp32", False)
    remb, ry = draws[("reg", n_ctx0)]
    rl, _ = _legacy_reg_loss(reg_leg, adapter, remb[:1], ry[:1], n_ctx0, device)
    for b, via in ((1, False), (2, False), (2, True)):
        try:
            v, _ = _batched_reg_loss(
                reg_bat, adapter, remb[:b], ry[:b], n_ctx0, perf0, device, via
            )
            rec = {
                "check": "reg_batched",
                "B": b,
                "via_engine": via,
                "loss": float(v),
                "loss_legacy_b1": float(rl),
            }
        except Exception as exc:  # noqa: BLE001
            rec = {
                "check": "reg_batched",
                "B": b,
                "via_engine": via,
                "error": f"{type(exc).__name__}: {exc}",
            }
        emit(rec)

    del clf_fp, clf_nofp, clf_bat, reg_leg, reg_bat
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- phase 2: legacy baselines ----------------
    print("\n=== legacy fit_with_differentiable_input baselines (B=1) ===", flush=True)
    for prec in sweep_precision:
        est = _fresh_est("clf", tabpfn_dir, device, seed, d_feat, prec, True)
        for n_ctx in sweep_n_ctx:
            e1, l1 = draws[("clf", n_ctx)]
            try:
                s, lv, peak = _time_config(
                    lambda e=e1, l=l1, n=n_ctx, es=est: _legacy_clf_loss(
                        es, adapter, e[:1], l[:1], n, device
                    )[0],
                    adapter,
                    warmup,
                    iters,
                )
                emit(
                    {
                        "path": "legacy",
                        "precision": prec,
                        "n_ctx": n_ctx,
                        "B": 1,
                        "recompute": False,
                        "s_per_draw": s,
                        "peak_gib": peak,
                        "loss": lv,
                    }
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                emit(
                    {
                        "path": "legacy",
                        "precision": prec,
                        "n_ctx": n_ctx,
                        "B": 1,
                        "recompute": False,
                        "oom": _oom(exc),
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()
        del est
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------- phase 3: batched sweep ----------------
    print("\n=== batched fit_from_preprocessed sweep ===", flush=True)
    for prec in sweep_precision:
        est = _fresh_est("clf", tabpfn_dir, device, seed, d_feat, prec, False)
        for recompute in sweep_recompute:
            perf = _perf_options(est, recompute)
            for n_ctx in sweep_n_ctx:
                e1, l1 = draws[("clf", n_ctx)]
                for b in sweep_batch:
                    if left() < 60:
                        emit(
                            {
                                "path": "batched",
                                "precision": prec,
                                "n_ctx": n_ctx,
                                "B": b,
                                "recompute": recompute,
                                "skipped": "budget",
                            }
                        )
                        continue
                    try:
                        s, lv, peak = _time_config(
                            lambda e=e1, l=l1, n=n_ctx, p=perf, bb=b, es=est: (
                                _batched_clf_loss(
                                    es, adapter, e[:bb], l[:bb], n, p, device
                                )[0]
                            ),
                            adapter,
                            warmup,
                            iters,
                        )
                        emit(
                            {
                                "path": "batched",
                                "precision": prec,
                                "n_ctx": n_ctx,
                                "B": b,
                                "recompute": recompute,
                                "s_per_draw": s / b,
                                "s_per_iter": s,
                                "peak_gib": peak,
                                "loss": lv,
                            }
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        oom = _oom(exc)
                        emit(
                            {
                                "path": "batched",
                                "precision": prec,
                                "n_ctx": n_ctx,
                                "B": b,
                                "recompute": recompute,
                                "oom": oom,
                                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                            }
                        )
                        adapter.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if oom:
                            break
        del est
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------- phase 4: narrow adapter output ----------------
    if probe_d_out and probe_d_out != d_feat and left() > 240:
        print(f"\n=== d_out={probe_d_out} spot check ===", flush=True)
        narrow = build_adapter("linear", d_feat, probe_d_out, 0, stats_path, device)
        for prec in sweep_precision:
            est = _fresh_est("clf", tabpfn_dir, device, seed, probe_d_out, prec, False)
            perf = _perf_options(est, False)
            for n_ctx in sweep_n_ctx:
                e1, l1 = draws[("clf", n_ctx)]
                for b in sweep_batch:
                    if left() < 60:
                        continue
                    try:
                        s, lv, peak = _time_config(
                            lambda e=e1, l=l1, n=n_ctx, bb=b, es=est, p=perf: (
                                _batched_clf_loss(
                                    es, narrow, e[:bb], l[:bb], n, p, device
                                )[0]
                            ),
                            narrow,
                            warmup,
                            iters,
                        )
                        emit(
                            {
                                "path": "batched",
                                "d_out": probe_d_out,
                                "precision": prec,
                                "n_ctx": n_ctx,
                                "B": b,
                                "recompute": False,
                                "s_per_draw": s / b,
                                "peak_gib": peak,
                                "loss": lv,
                            }
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        oom = _oom(exc)
                        emit(
                            {
                                "path": "batched",
                                "d_out": probe_d_out,
                                "precision": prec,
                                "n_ctx": n_ctx,
                                "B": b,
                                "recompute": False,
                                "oom": oom,
                                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                            }
                        )
                        narrow.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if oom:
                            break
            del est
            gc.collect()
            torch.cuda.empty_cache()

    # ---------------- table ----------------
    print("\n=== TABLE ===", flush=True)
    print(
        f"{'path':9s} {'dout':>5s} {'prec':5s} {'n_ctx':>6s} {'B':>3s} {'recomp':>6s} "
        f"{'peakGiB':>8s} {'s/draw':>8s} {'x base':>7s}",
        flush=True,
    )
    base = {
        r["n_ctx"]: r["s_per_draw"]
        for r in rows
        if r.get("path") == "legacy"
        and r.get("precision") == "fp32"
        and "s_per_draw" in r
    }
    for r in rows:
        if "path" not in r:
            continue
        n_ctx = r["n_ctx"]
        if "s_per_draw" in r:
            b0 = base.get(n_ctx)
            sp = f"{b0 / r['s_per_draw']:7.2f}" if b0 else "      -"
            print(
                f"{r['path']:9s} {r.get('d_out', d_feat):5d} {r['precision']:5s} "
                f"{n_ctx:6d} {r['B']:3d} {r['recompute']!s:>6s} "
                f"{r['peak_gib']:8.1f} {r['s_per_draw']:8.4f} {sp}",
                flush=True,
            )
        else:
            tag = "OOM" if r.get("oom") else r.get("skipped", "ERR")
            print(
                f"{r['path']:9s} {r.get('d_out', d_feat):5d} {r['precision']:5s} "
                f"{n_ctx:6d} {r['B']:3d} {r['recompute']!s:>6s} {tag:>8s}",
                flush=True,
            )
    print(f"\nelapsed {time.perf_counter() - t_start:.1f}s", flush=True)
