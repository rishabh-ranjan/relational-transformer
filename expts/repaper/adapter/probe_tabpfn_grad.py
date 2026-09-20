from pathlib import Path


def main(
    *,
    tabpfn_dir: str,
    d_feat: int,
    shapes: list[tuple[int, int]],
    sanity_steps: int,
    sanity_shape: tuple[int, int],
    sanity_signal_dims: int,
    lr: float,
    seed: int,
) -> None:
    import time

    import torch
    from torch import nn
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    device = "cuda"
    model_path = Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors"
    assert model_path.exists(), (
        f"TabPFN checkpoint {model_path} not found; fetch it once with "
        f"`pixi run python -m expts.repaper.baselines.fetch_tabpfn`"
    )
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}", flush=True)

    def adapter():
        # Identity, not Xavier: step 0 is then exactly the un-adapted
        # featurizer, so anything the training moves is attributable.
        a = nn.Linear(d_feat, d_feat, bias=True).to(device)
        with torch.no_grad():
            a.weight.copy_(torch.eye(d_feat))
            a.bias.zero_()
        return a

    def tabpfn(task_type):
        # differentiable_input is TabPFN's prompt-tuning path: the ensemble
        # preprocessing collapses to one z-norm step and the executor is built
        # with inference_mode=False, so gradients reach both the context and
        # the query features. n_estimators=1 is exact here, not an
        # approximation: 512 features is under the 768 an estimator sees.
        cls = TabPFNClassifier if task_type == "clf" else TabPFNRegressor
        est = cls(
            n_estimators=1,
            model_path=model_path,
            device=device,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            random_state=seed,
        )
        # models_ does not exist until the first fit, and freezing has to
        # happen before the first backward or the weights collect gradients
        # this probe then reports as a leak. So: one throwaway fit, freeze,
        # and everything after that is measured.
        warm = torch.randn(32, d_feat, device=device)
        y = torch.arange(32, device=device) % 2
        est.fit_with_differentiable_input(warm, y if task_type == "clf" else y.float())
        if task_type == "reg":
            # The standard fit moves the bar distribution to the device; the
            # differentiable one never does, so its borders sit on the cpu and
            # the loss dies in searchsorted.
            est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
        n = 0
        for model in est.models_:
            for p in model.parameters():
                p.requires_grad_(False)
                n += 1
        print(f"[{task_type}] froze {n} TabPFN parameters", flush=True)
        return est

    def leaked(est):
        return [
            name
            for model in est.models_
            for name, p in model.named_parameters()
            if p.grad is not None
        ]

    def loss_for(est, task_type, x_ctx, y_ctx, x_q, y_q):
        est.fit_with_differentiable_input(x_ctx, y_ctx)
        if task_type == "clf":
            logits = est.forward(x_q, use_inference_mode=True, return_logits=True)
            assert logits.shape == (x_q.shape[0], 2), logits.shape
            return nn.functional.cross_entropy(logits, y_q)
        logits, _per_est, _borders = est.forward(x_q, use_inference_mode=True)
        # forward hands back the ensemble mean transposed to (n_bars, n_query);
        # the bar distribution wants (batch, n_query, n_bars) against targets in
        # the z-space fit_with_differentiable_input normalised y into.
        znormed = (y_q - est.y_train_mean_) / est.y_train_std_
        return est.znorm_space_bardist_(
            logits.transpose(0, 1).unsqueeze(0), znormed.unsqueeze(0)
        ).mean()

    print("\n--- can the adapter learn? ---", flush=True)
    n_ctx, n_q = sanity_shape
    for task_type in ("clf", "reg"):
        est = tabpfn(task_type)
        gen = torch.Generator(device=device).manual_seed(seed + 1)
        a = adapter()
        # Only sanity_signal_dims of the d_feat dimensions carry signal and the
        # rest is noise, so a 512-wide identity is a bad featurizer and the
        # adapter has somewhere to go. If the loss does not fall here, nothing
        # downstream will.
        e = torch.randn(n_ctx + n_q, d_feat, device=device, generator=gen)
        w = torch.zeros(d_feat, device=device)
        w[:sanity_signal_dims] = torch.randn(
            sanity_signal_dims, device=device, generator=gen
        )
        score = e @ w
        y = (score > 0).long() if task_type == "clf" else score
        opt = torch.optim.AdamW(a.parameters(), lr=lr)
        first = None
        for step in range(sanity_steps):
            perm = torch.randperm(e.shape[0], device=device, generator=gen)
            x, yy = a(e[perm]), y[perm]
            loss = loss_for(
                est, task_type, x[:n_ctx], yy[:n_ctx], x[n_ctx:], yy[n_ctx:]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            first = float(loss) if step == 0 else first
            if step % max(1, sanity_steps // 10) == 0 or step == sanity_steps - 1:
                print(
                    f"[{task_type}] step {step:>3}  loss {float(loss):.4f}", flush=True
                )
        print(
            f"[{task_type}] {sanity_steps} steps: {first:.4f} -> {float(loss):.4f} "
            f"({'down' if float(loss) < first else 'UP'})",
            flush=True,
        )
        assert not leaked(est), f"[{task_type}] TabPFN took a gradient"

    rows = []
    for task_type in ("clf", "reg"):
        est = tabpfn(task_type)
        for n_ctx, n_q in shapes:
            gen = torch.Generator(device=device).manual_seed(seed)
            a = adapter()
            e = torch.randn(n_ctx + n_q, d_feat, device=device, generator=gen)
            w = torch.randn(d_feat, device=device, generator=gen)
            score = (e @ w) / d_feat**0.5
            y = (score > 0).long() if task_type == "clf" else score
            x = a(e)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            loss = loss_for(est, task_type, x[:n_ctx], y[:n_ctx], x[n_ctx:], y[n_ctx:])
            torch.cuda.synchronize()
            t_fwd = time.time() - t0

            t0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            t_bwd = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30

            g = a.weight.grad
            assert g is not None, f"[{task_type}] no gradient reached the adapter"
            assert torch.isfinite(g).all(), f"[{task_type}] non-finite adapter grad"
            assert g.abs().sum() > 0, f"[{task_type}] adapter grad is all zero"
            bad = leaked(est)
            assert not bad, f"[{task_type}] TabPFN took a gradient: {bad[:5]}"

            rows.append((task_type, n_ctx, n_q, t_fwd, t_bwd, peak))
            print(
                f"[{task_type}] n_ctx={n_ctx:>6} n_q={n_q:>4} "
                f"loss {float(loss):.4f}  fwd {t_fwd:6.2f}s  bwd {t_bwd:6.2f}s  "
                f"peak {peak:6.2f} GiB  |grad| {float(g.norm()):.3e}",
                flush=True,
            )
            del loss, x, a
            torch.cuda.empty_cache()

    print("\n--- summary ---", flush=True)
    for task_type, n_ctx, n_q, t_fwd, t_bwd, peak in rows:
        print(
            f"{task_type} {n_ctx}x{n_q}: fwd {t_fwd:.2f}s bwd {t_bwd:.2f}s "
            f"peak {peak:.2f} GiB",
            flush=True,
        )
