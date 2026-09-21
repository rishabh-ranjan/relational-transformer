import json
import re
from pathlib import Path


def main(
    *,
    features_root: str,
    tabpfn_dir: str,
    ckpt_dir: str,
    out_json: str,
    n_tasks: int,
    n_ctx: int,
    n_query: int,
    d_feat: int,
    min_rows: int,
    seed: int,
) -> None:
    import time

    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch
    from torch import nn
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    from rt.eval.metrics import metric_for

    from expts.repaper.adapter.train_adapter import build_adapter

    # Did the adapter learn its own objective at all? The per-window training
    # loss cannot answer that: every window averages a different random sample
    # of tasks, and that between-task variance swamps any plausible gain
    # (measured: slope +-0.029 +- 0.046 on clf over 10k steps). This holds the
    # sample fixed instead -- same tasks, same context and query rows, same
    # TabPFN config -- and sweeps it across the saved checkpoints, so the only
    # thing that varies is the adapter.
    device = "cuda"
    root = Path(features_root).expanduser()
    model_path = Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors"
    assert model_path.exists(), f"TabPFN checkpoint {model_path} not found"

    entries = []
    for mp in sorted(root.glob("*/*_meta.json")):
        m = json.loads(mp.read_text())
        if m["shape"][0] < min_rows or (m["n_distinct_labels"] or 0) < 2:
            continue
        entries.append(
            {
                "db": mp.parent.name,
                "task": m["task"],
                "task_type": m["task_type"],
                "n": m["shape"][0],
                "bin": mp.with_name(mp.name.replace("_meta.json", "_target.bin")),
                "rows": mp.with_name(mp.name.replace("_meta.json", "_rows.npz")),
                "weight": float(min(m["total_nodes"], 100_000)),
            }
        )
    w = np.array([e["weight"] for e in entries])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(entries), size=n_tasks, replace=False, p=w / w.sum())

    probe = []
    for i in pick:
        e = entries[int(i)]
        n_c = min(n_ctx, e["n"] - n_query)
        if n_c < 64:
            continue
        idx = np.sort(rng.choice(e["n"], size=n_c + n_query, replace=False))
        x = np.memmap(e["bin"], dtype=ml_dtypes.bfloat16, mode="r", shape=(e["n"], d_feat))
        emb = torch.from_numpy(
            np.ascontiguousarray(x[idx]).view(np.uint16)
        ).view(torch.bfloat16)
        y = torch.from_numpy(np.load(e["rows"])["labels"][idx].astype("float32"))
        if e["task_type"] == "reg" and float(y[:n_c].std()) < 1e-6:
            continue
        probe.append({**e, "emb": emb, "y": y, "n_ctx": n_c})
    by = {t: sum(1 for p in probe if p["task_type"] == t) for t in ("clf", "reg")}
    print(f"probe set: {len(probe)} tasks {by}", flush=True)

    def tabpfn(task_type):
        cls = TabPFNClassifier if task_type == "clf" else TabPFNRegressor
        est = cls(
            n_estimators=1,
            model_path=model_path,
            device=device,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            random_state=seed,
            inference_precision=torch.float32,
            inference_config={"OUTLIER_REMOVAL_STD": None},
        )
        warm = torch.randn(32, d_feat, device=device)
        yy = torch.arange(32, device=device) % 2
        est.fit_with_differentiable_input(warm, yy if task_type == "clf" else yy.float())
        if task_type == "reg":
            est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
        for model in est.models_:
            for p in model.parameters():
                p.requires_grad_(False)
        return est

    ests = {"clf": tabpfn("clf"), "reg": tabpfn("reg")}

    def build(ck):
        if ck is None:
            return nn.Identity().to(device)
        a = build_adapter(
            ck["adapter_kind"],
            d_feat,
            ck.get("d_out", d_feat),
            ck["hidden_dim"],
            ck["stats_path"],
            device,
        )
        a.load_state_dict(ck["state_dict"])
        return a.eval()

    @torch.no_grad()
    def score(adapter):
        losses = {"clf": [], "reg": []}
        metrics = {"clf": [], "reg": []}
        for p in probe:
            x = adapter(p["emb"].to(device).float())
            yy = p["y"].to(device)
            n_c = p["n_ctx"]
            est = ests[p["task_type"]]
            if p["task_type"] == "clf":
                lab = (yy > 0).long()
                est.fit_with_differentiable_input(x[:n_c], lab[:n_c])
                logits = est.forward(
                    x[n_c:], use_inference_mode=True, return_logits=True
                )
                loss = nn.functional.cross_entropy(logits, lab[n_c:])
                pred, truth = torch.softmax(logits, -1)[:, 1], lab[n_c:]
            else:
                est.fit_with_differentiable_input(x[:n_c], yy[:n_c])
                logits, _pe, _b = est.forward(x[n_c:], use_inference_mode=True)
                z = (yy[n_c:] - est.y_train_mean_) / est.y_train_std_
                loss = est.znorm_space_bardist_(
                    logits.transpose(0, 1).unsqueeze(0), z.unsqueeze(0)
                ).mean()
                mean = est.znorm_space_bardist_.mean(logits.transpose(0, 1))
                pred = mean * est.y_train_std_ + est.y_train_mean_
                truth = yy[n_c:]
            losses[p["task_type"]].append(float(loss))
            try:
                _n, v = metric_for(
                    p["task_type"],
                    truth.float().cpu().numpy(),
                    pred.float().cpu().numpy(),
                )
                metrics[p["task_type"]].append(float(v))
            except ValueError:
                pass
        return {
            f"{kind}_{what}": (float(np.mean(vals)) if vals else None)
            for what, d in (("loss", losses), ("metric", metrics))
            for kind, vals in d.items()
        }

    ck = Path(ckpt_dir).expanduser()
    paths = sorted(
        ck.glob("adapter_step*.pt"),
        key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)),
    )
    final = ck / "adapter_final.pt"
    rows = []
    t0 = time.time()
    # "identity" is the reference the linear arm was initialised at, scored by
    # the same code on the same rows.
    for label, path in [("identity", None), *[(p.name, p) for p in paths]] + (
        [("final", final)] if final.exists() else []
    ):
        ck = (
            None
            if path is None
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        step = 0 if ck is None else ck["step"]
        res = score(build(ck))
        rows.append({"label": label, "step": step, **res})
        print(
            f"  {label:<24} step {step:>5}  "
            f"clf loss {res['clf_loss']:.4f} auroc {res['clf_metric']:.4f}  |  "
            f"reg loss {res['reg_loss']:.4f} mae {res['reg_metric']:.4f}  "
            f"({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )

    Path(out_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).expanduser().write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_json}", flush=True)
