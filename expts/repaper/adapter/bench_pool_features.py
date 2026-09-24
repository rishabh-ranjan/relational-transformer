import json
import math
import os
import time
from pathlib import Path


def featurize(net, ds, didx, nodes, lut, n_slots, device):
    import numpy as np
    import torch

    from expts.repaper.adapter.probe_pool_throughput import LOCAL_CTX
    from expts.repaper.adapter.train_pool import CHUNK
    from rt.data import process_batch

    tgts, unis, labs = [], [], []
    n_sub = n_no_parent = 0
    for start in range(0, len(nodes), CHUNK):
        chunk = np.asarray(nodes[start : start + CHUNK])
        n_real = len(chunk)
        if n_real < CHUNK:
            chunk = np.concatenate([chunk, np.repeat(chunk[:1], CHUNK - n_real)])
        tup = ds.sampler.batch_for_nodes_py([int(v) for v in chunk], didx, LOCAL_CTX)
        batch = process_batch(tup, ds.d_text)
        batch.pop("batch_mask", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        x = net(batch, return_embeddings=True)
        keys = batch["col_name_idxs"].masked_fill(
            batch["is_padding"], torch.iinfo(batch["col_name_idxs"].dtype).max
        )
        si = keys.argsort(dim=-1, stable=True)
        is_t = batch["is_targets"].gather(1, si)
        assert (is_t.sum(1) == 1).all()
        nidx = batch["node_idxs"].gather(1, si)
        pad = batch["is_padding"].gather(1, si)
        cols = batch["col_name_idxs"].gather(1, si)
        f2p = batch["f2p_nbr_idxs"].gather(1, si.unsqueeze(-1).expand_as(batch["f2p_nbr_idxs"]))
        seeds = torch.as_tensor(chunk, device=device, dtype=nidx.dtype).view(-1, 1)
        ok = (nidx * is_t).sum(1) == seeds.squeeze(1)
        ok[n_real:] = False
        n_sub += n_real - int(ok.sum())
        vals = batch["number_values"].gather(1, si.unsqueeze(-1).expand_as(batch["number_values"])).squeeze(-1)
        lab = (vals * is_t.to(vals.dtype)).sum(1).float()

        is_seed = (nidx == seeds) & ~pad
        seed_f2p = f2p.masked_fill(~is_seed.unsqueeze(-1), -1)
        is_parent = (
            (nidx.unsqueeze(-1).unsqueeze(-1) == seed_f2p.unsqueeze(1)).any(-1).any(-1) & ~pad & ~is_seed
        )
        safe = cols.long().clamp(0, lut.numel() - 1)
        slot_ids = torch.where(cols.long() < lut.numel(), lut[safe], torch.full_like(safe, -1))
        take = (is_seed | is_parent) & (slot_ids >= 0)
        uni = torch.full((len(chunk), n_slots, x.shape[-1]), float("nan"), dtype=torch.float32, device=device)
        b_idx, s_idx = take.nonzero(as_tuple=True)
        uni[b_idx, slot_ids[b_idx, s_idx]] = x[b_idx, s_idx].float()

        keep = ok.nonzero().squeeze(1)
        tgts.append(x[is_t][keep].float())
        unis.append(uni[keep])
        labs.append(lab[keep])
        n_no_parent += int((~is_parent.any(dim=1))[keep].sum())
    return torch.cat(tgts), torch.cat(unis), torch.cat(labs), n_sub, n_no_parent


def tabpfn_predict(kind, X, y, n_ctx, tabpfn_dir, device, seed):
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.preprocessing.configs import PreprocessorConfig

    per_est = PreprocessorConfig("none", differentiable=True).max_features_per_estimator
    n_est = math.ceil(X.shape[1] / per_est)
    cls = TabPFNClassifier if kind == "clf" else TabPFNRegressor
    est = cls(
        n_estimators=n_est,
        model_path=Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors",
        device=device,
        fit_mode="fit_preprocessors",
        differentiable_input=True,
        random_state=seed,
        inference_precision=torch.float32,
        inference_config={"OUTLIER_REMOVAL_STD": None},
    )
    with torch.no_grad():
        if kind == "clf":
            lab = (y > 0).long()
            est.fit_with_differentiable_input(X[:n_ctx], lab[:n_ctx])
            probs = est.forward(X[n_ctx:], use_inference_mode=True)
            probs = probs.reshape(-1, probs.shape[-1])
            assert probs.shape[0] == len(X) - n_ctx, probs.shape
            return probs[:, 1].float(), lab[n_ctx:], n_est
        est.fit_with_differentiable_input(X[:n_ctx], y[:n_ctx])
        est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
        logits, _per, _b = est.forward(X[n_ctx:], use_inference_mode=True)
        mean = est.znorm_space_bardist_.mean(logits.transpose(0, 1))
        return (mean * est.y_train_std_ + est.y_train_mean_).float(), y[n_ctx:], n_est


def main(
    *,
    relbench_pre_dir: str,
    relbench_task_list: str,
    ckpt: str,
    tabpfn_dir: str,
    out_dir: str,
    relbench_n_ctx: int,
    relbench_n_query: int,
    modes: list[str],
    tasks: list[str] | None,
    seed: int,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from expts.repaper.adapter.train_pool import build_relbench
    from expts.repaper.baselines.featurize_rt_taps import union_slots
    from rt.model import load_rt_model

    assert set(modes) <= {"target", "union"}, modes
    device = "cuda"
    out = Path(out_dir).expanduser()
    for m in modes:
        (out / m).mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(msg):
        print(f"[{(time.time() - t_start) / 60:7.1f} min] {msg}", flush=True)

    net, config = load_rt_model(ckpt, device=device, compile=True)
    net = net.to(torch.bfloat16).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    relbench = build_relbench(relbench_pre_dir, relbench_task_list, config, relbench_n_ctx, relbench_n_query, seed)
    if tasks is not None:
        relbench = [ev for ev in relbench if ev["name"] in tasks]
        assert len(relbench) == len(tasks), ([ev["name"] for ev in relbench], tasks)
    log(f"{len(relbench)} relbench tasks, modes {modes}")

    for ev in relbench:
        name = ev["name"]
        path = {m: out / m / f"{name.replace('/', '__')}.json" for m in modes}
        if all(p.exists() for p in path.values()):
            log(f"{name}: done, skipping")
            continue
        db, table = name.split("/")
        slot_names, slot_of, target_name = union_slots(relbench_pre_dir, db, table)
        lut = torch.full((max(slot_of) + 1,), -1, dtype=torch.long, device=device)
        for idx, s in slot_of.items():
            lut[idx] = s
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            ct, cu, cy, s1, p1 = featurize(net, ev["ds"], ev["didx"], ev["ctx"], lut, len(slot_names), device)
            qt, qu, qy, s2, p2 = featurize(net, ev["ds"], ev["didx"], ev["query"], lut, len(slot_names), device)
        nc, nq = len(ct), len(qt)
        y = torch.cat([cy, qy]).clone()
        embed_s = time.perf_counter() - t0
        filled = (~torch.isnan(torch.cat([cu, qu])[:, :, 0])).sum(0).cpu().numpy()
        live = [i for i, c in enumerate(filled) if c > 0]
        feats = {
            "target": torch.cat([ct, qt]).clone(),
            "union": torch.cat([cu, qu])[:, live].flatten(1).clone(),
        }
        del cu, qu
        common = {
            "task": name,
            "kind": ev["kind"],
            "n_ctx": nc,
            "n_query": nq,
            "substituted": s1 + s2,
            "rows_without_parent": p1 + p2,
            "slots": slot_names,
            "live_slots": [slot_names[i] for i in live],
            "slot_filled_frac": {slot_names[i]: float(filled[i] / (nc + nq)) for i in range(len(slot_names))},
            "target_slot": target_name,
            "embed_s": embed_s,
        }
        log(f"{name}: embedded ctx {nc} query {nq} in {embed_s:.0f}s, {len(live)}/{len(slot_names)} live slots, "
            f"{s1 + s2} substituted, {p1 + p2} rows without parent")
        for m in modes:
            if path[m].exists():
                continue
            X = feats[m]
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            pred, tgt, n_est = tabpfn_predict(ev["kind"], X, y, nc, tabpfn_dir, device, seed)
            pred, tgt = pred.cpu().numpy(), tgt.float().cpu().numpy()
            assert np.isfinite(pred).all(), f"{name} {m}: non-finite predictions"
            metric = float(roc_auc_score(tgt, pred)) if ev["kind"] == "clf" else float(np.abs(pred - tgt).mean())
            rec = {
                **common,
                "mode": m,
                "metric_name": "roc_auc" if ev["kind"] == "clf" else "nmae",
                "metric": metric,
                "n_features": int(X.shape[1]),
                "nan_frac": float(torch.isnan(X).float().mean()),
                "n_estimators": n_est,
                "tabpfn_s": time.perf_counter() - t0,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            path[m].write_text(json.dumps(rec, indent=1))
            log(f"{name} {m}: {rec['metric_name']} {metric:.4f} features {rec['n_features']} "
                f"estimators {n_est} nan {rec['nan_frac']:.3f} tabpfn {rec['tabpfn_s']:.0f}s peak {rec['peak_gib']:.1f}GiB")
        del feats
        torch.cuda.empty_cache()

    summary = {}
    for m in modes:
        recs = [json.loads(p.read_text()) for p in sorted((out / m).glob("*.json"))]
        for kind, metric in (("clf", "roc_auc"), ("reg", "nmae")):
            v = [r["metric"] for r in recs if r["kind"] == kind]
            summary[f"{m}/{metric}"] = float(np.mean(v)) if v else None
            summary[f"{m}/n_{kind}"] = len(v)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"summary {summary}")
    log("done")
