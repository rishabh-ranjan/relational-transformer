import json
from pathlib import Path


def main(
    *,
    features_root: str,
    tabpfn_dir: str,
    tasks: list[tuple[str, str]],
    n_ctx: int,
    n_query: int,
    d_feat: int,
    seed: int,
) -> None:
    import ml_dtypes  # noqa: F401
    import numpy as np
    import torch
    from torch import nn
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    # Three hypotheses about the non-finite gradient have been wrong: labels,
    # degenerate feature dims, and fp16 (a GradScaler halved 65536 -> 0, and
    # forcing fp32 changed nothing). Rather than guess a fourth, this asks
    # autograd where it happens: detect_anomaly names the backward op, and a
    # hook on the adapter output says whether the inf is already there when the
    # gradient leaves TabPFN or is manufactured on our side of the boundary.
    device = "cuda"
    root = Path(features_root).expanduser()
    model_path = Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors"

    def make(task_type, precision):
        cls = TabPFNClassifier if task_type == "clf" else TabPFNRegressor
        est = cls(
            n_estimators=1,
            model_path=model_path,
            device=device,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            random_state=seed,
            inference_precision=precision,
        )
        warm = torch.randn(32, d_feat, device=device)
        y = torch.arange(32, device=device) % 2
        est.fit_with_differentiable_input(warm, y if task_type == "clf" else y.float())
        if task_type == "reg":
            est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
        for model in est.models_:
            for p in model.parameters():
                p.requires_grad_(False)
        return est

    def load(db, task):
        m = json.loads((root / db / f"{task}_meta.json").read_text())
        n, d = m["shape"]
        x = np.memmap(
            root / db / f"{task}_target.bin",
            dtype=ml_dtypes.bfloat16,
            mode="r",
            shape=(n, d),
        )
        y = np.load(root / db / f"{task}_rows.npz")["labels"]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(n_ctx + n_query, n), replace=False))
        emb = torch.from_numpy(
            np.ascontiguousarray(x[idx]).view(np.uint16)
        ).view(torch.bfloat16)
        return emb, torch.from_numpy(y[idx].astype("float32")), m["task_type"]

    for db, task in tasks:
        emb, y, task_type = load(db, task)
        for precision in (torch.float32, "auto"):
            adapter = nn.Linear(d_feat, d_feat, bias=True).to(device)
            with torch.no_grad():
                adapter.weight.copy_(torch.eye(d_feat))
                adapter.bias.zero_()
            est = make(task_type, precision)
            n_c = min(n_ctx, len(y) - n_query)
            # The forward has to run inside detect_anomaly too: it only
            # records "Traceback of forward call that caused the error" for
            # ops *created* while anomaly mode is on, so wrapping backward
            # alone yields the op name and nothing else.
            anomaly = torch.autograd.detect_anomaly(check_nan=True)
            anomaly.__enter__()
            x = adapter(emb.to(device).float())
            seen = {}
            x.register_hook(lambda g: seen.__setitem__("dx", g.detach()))
            yy = y.to(device)

            if task_type == "clf":
                lab = (yy > 0).long()
                est.fit_with_differentiable_input(x[:n_c], lab[:n_c])
                logits = est.forward(
                    x[n_c:], use_inference_mode=True, return_logits=True
                )
                loss = nn.functional.cross_entropy(logits, lab[n_c:])
            else:
                est.fit_with_differentiable_input(x[:n_c], yy[:n_c])
                logits, _p, _b = est.forward(x[n_c:], use_inference_mode=True)
                z = (yy[n_c:] - est.y_train_mean_) / est.y_train_std_
                loss = est.znorm_space_bardist_(
                    logits.transpose(0, 1).unsqueeze(0), z.unsqueeze(0)
                ).mean()

            fin_logits = bool(torch.isfinite(logits).all())
            print(
                f"\n=== {db}/{task} [{task_type}] precision={precision} "
                f"n_ctx={n_c} n_q={len(y) - n_c}",
                flush=True,
            )
            print(
                f"  logits finite {fin_logits}  |logits|max "
                f"{float(logits.abs().max()):.4g}  loss {float(loss):.6g}",
                flush=True,
            )
            err = None
            try:
                loss.backward()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                anomaly.__exit__(None, None, None)
            g = adapter.weight.grad
            dx = seen.get("dx")
            print(
                f"  d(loss)/dx  finite {None if dx is None else bool(torch.isfinite(dx).all())}"
                f"  max {None if dx is None else float(dx.abs().max()):}",
                flush=True,
            )
            print(
                f"  dW finite {None if g is None else bool(torch.isfinite(g).all())}"
                f"  max {None if g is None else float(g.abs().max()):}",
                flush=True,
            )
            if err:
                print(f"  ANOMALY: {err}", flush=True)
