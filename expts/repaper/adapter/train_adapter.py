import json
import math
import os
import time
from pathlib import Path


def load_index(features_root: str, min_rows: int):
    import numpy as np

    root = Path(features_root).expanduser()
    tasks = []
    skipped = {"rows": 0, "degenerate": 0}
    for mp in sorted(root.glob("*/*_meta.json")):
        m = json.loads(mp.read_text())
        n, d = m["shape"]
        if n < min_rows:
            skipped["rows"] += 1
            continue
        # A task whose label never varies teaches nothing, and TabPFN's
        # regressor refuses a constant target outright.
        if (m["n_distinct_labels"] or 0) < 2:
            skipped["degenerate"] += 1
            continue
        db = mp.parent.name
        entry = {
            "db": db,
            "task": m["task"],
            "task_type": m["task_type"],
            "n": n,
            "d": d,
            "bin": str(mp.with_name(mp.name.replace("_meta.json", "_target.bin"))),
            "rows": str(mp.with_name(mp.name.replace("_meta.json", "_rows.npz"))),
            # P(task) as pretraining draws it: a pool holding min(rows, 100_000)
            # of every task, sampled uniformly. What we dumped is capped far
            # lower, so the weight has to come off the true row count.
            "weight": float(min(m["total_nodes"], 100_000)),
        }
        tasks.append(entry)
    print(
        f"index: {len(tasks)} train tasks "
        f"({len({e['db'] for e in tasks})} dbs), skipped {skipped}",
        flush=True,
    )
    by = {t: sum(1 for e in tasks if e["task_type"] == t) for t in ("clf", "reg")}
    print(f"  train: {by}", flush=True)
    return tasks


class Pool:
    def __init__(self, entries):
        import numpy as np

        self.entries = entries
        w = np.array([e["weight"] for e in entries], dtype=np.float64)
        self.p = w / w.sum()
        self._cache = {}

    def read(self, e, idx):
        import ml_dtypes
        import numpy as np
        import torch

        key = e["bin"]
        if key not in self._cache:
            x = np.memmap(
                e["bin"], dtype=ml_dtypes.bfloat16, mode="r", shape=(e["n"], e["d"])
            )
            y = np.load(e["rows"])["labels"]
            self._cache[key] = (x, y)
        x, y = self._cache[key]
        emb = torch.from_numpy(np.ascontiguousarray(x[idx]).view(np.uint16)).view(
            torch.bfloat16
        )
        return emb, torch.from_numpy(y[idx].astype("float32"))


def draw(rng, pool, e, n_ctx_lo, n_ctx_hi, n_query):
    import numpy as np

    hi = min(n_ctx_hi, e["n"] - n_query)
    lo = min(n_ctx_lo, hi)
    n_ctx = int(math.exp(rng.uniform(math.log(lo), math.log(hi + 1))))
    n_ctx = max(lo, min(n_ctx, hi))
    idx = rng.choice(e["n"], size=n_ctx + n_query, replace=False)
    emb, y = pool.read(e, np.sort(idx))
    return emb, y, n_ctx


def build_adapter(adapter_kind, d_feat, d_out, hidden_dim, stats_path, device):
    import numpy as np
    import torch

    from torch import nn

    # rt-j's norm_out is an RMSNorm: it fixes the norm of each row (measured
    # sd/mean 0.0049) and says nothing about a column across rows. Four
    # massive-activation dims (119, 206, 303, 399) carry ~90% of the squared
    # row norm, dim 303 alone has mean -13.07 with sigma 0.02, and
    # per-dimension sigma spans 79x. dL/dW is an outer product with that
    # input, so those four dims own the gradient and the other 508 are
    # invisible to it.
    #
    # The fix is a frozen (x - mu) / sigma, measured once over the whole Join
    # dump under the training mixture (feature_stats.py). Frozen and global,
    # not a BatchNorm: the statistics we need are of the training marginal,
    # and a batch here is one task's draw, whose per-dimension sigma moves by
    # 60x from task to task on exactly the high-energy dims -- a running
    # estimate would disagree with itself between train and eval mode where it
    # matters most, and would couple the context rows to the query rows.
    #
    # For the linear arm this is free at the function level: TabPFN z-norms
    # every column against the context, and z-norm is invariant to a positive
    # affine map per column, so standardise-then-identity is the same
    # predictor as identity. Only the geometry of the parameter space changes.
    class Standardize(nn.Module):
        def __init__(self, mean, scale):
            super().__init__()
            self.register_buffer("mean", mean)
            self.register_buffer("scale", scale)

        def forward(self, x):
            return (x - self.mean) / self.scale

    st = np.load(Path(stats_path).expanduser())
    assert st["mean"].shape == (d_feat,), st["mean"].shape
    standardize = Standardize(
        torch.from_numpy(st["mean"].copy()), torch.from_numpy(st["scale"].copy())
    )
    print(
        f"standardiser from {stats_path}: "
        f"|mu| max {float(np.abs(st['mean']).max()):.3f}, "
        f"sigma [{float(st['std'].min()):.5f}, {float(st['std'].max()):.4f}], "
        f"{int(st['dead'].sum())} dims below the floor left unscaled",
        flush=True,
    )

    # No bias anywhere the output is read: TabPFN z-norms every column against
    # the context, so an output bias is subtracted straight back off and its
    # gradient is zero by construction.
    if adapter_kind == "linear":
        head = nn.Linear(d_feat, d_out, bias=False)
        with torch.no_grad():
            if d_out == d_feat:
                # Identity, so step 0 is exactly the un-adapted rt-j featurizer
                # (the standardiser in front is invisible to TabPFN, see above)
                # and every later step is attributable to training.
                head.weight.copy_(torch.eye(d_feat))
            else:
                # A projection cannot be identity-initialised, so step 0 is a
                # random 512->d_out projection and is expected to start *below*
                # the un-adapted featurizer, not at it.
                nn.init.xavier_uniform_(head.weight)
    elif adapter_kind == "mlp":
        # A bottleneck cannot be initialised at identity, so unlike the linear
        # arm this one does NOT start from the un-adapted featurizer: its step
        # 0 is a random projection through hidden_dim dims and is expected to
        # be worse.
        head = nn.Sequential(
            nn.Linear(d_feat, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_out, bias=False),
        )
        with torch.no_grad():
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.zero_()
    else:
        raise AssertionError(f"unknown adapter_kind {adapter_kind!r}")
    return nn.Sequential(standardize, head).to(device)


def draw_fixed(rng, pool, e, n_ctx, n_query):
    import numpy as np

    idx = rng.choice(e["n"], size=n_ctx + n_query, replace=False)
    emb, y = pool.read(e, np.sort(idx))
    return emb, y


def pick_task(rng, entries, p, n_rows, tries=64):
    # A ladder rung fixes n_ctx, so a task with too few rows cannot supply the
    # draw at all and has to be resampled rather than clamped -- clamping would
    # reintroduce the ragged shapes the ladder exists to remove. This does bias
    # the mixture toward larger tasks at the long rungs; that is the price of
    # fixed shapes and it is why the rung is drawn per micro-step rather than
    # per task.
    for _ in range(tries):
        e = entries[int(rng.choice(len(entries), p=p))]
        if e["n"] >= n_rows:
            return e
    return None


def summarize(per_task):
    import numpy as np

    out = {}
    for tt in ("clf", "reg"):
        vs = {k: v for k, (t, v) in per_task.items() if t == tt}
        out[tt] = (float(np.mean(list(vs.values()))) if vs else None, len(vs), vs)
    return out


def load_relbench(pre_dir, features_root, labels_root, n_ctx, n_query, seed):
    import numpy as np
    import torch

    # The val monitor the checkpoint is selected on. rt_features is the same
    # quantity the Join dump holds, and labels_relbench carries every val row
    # plus the context draw -- and no test row, so this cannot score test.
    pre, feat, lab = (Path(x).expanduser() for x in (pre_dir, features_root, labels_root))
    rng = np.random.default_rng(seed)
    out = []
    for db, name in sorted(
        map(tuple, json.loads((pre / "db-task-lists/forecast.json").read_text()))
    ):
        spec = {t["name"]: t for t in json.loads((pre / db / "meta.json").read_text())["tasks"]}[name]
        task_type = {"binary_classification": "clf", "regression": "reg"}[spec["task_type"]]
        z = np.load(lab / db / f"{name}.npz")
        idx, y = z["node_idxs"], z["labels"]
        is_val = (idx >= z["val_lo"]) & (idx < z["val_hi"])
        fm = json.loads((feat / db / "rt_features" / f"{name}_meta.json").read_text())
        vecs = np.memmap(
            feat / db / "rt_features" / f"{name}_vectors.bin",
            dtype=np.float32,
            mode="r",
            shape=(fm["total_nodes"], fm["n_features"]),
        )

        def take(mask, k):
            where = np.flatnonzero(mask)
            if len(where) > k:
                where = np.sort(rng.choice(where, size=k, replace=False))
            rows = idx[where] - fm["min_offset"]
            return (
                torch.from_numpy(np.ascontiguousarray(vecs[rows])),
                torch.from_numpy(y[where].astype("float32")),
            )

        ce, cy = take(~is_val, n_ctx)
        qe, qy = take(is_val, n_query)
        out.append(
            {
                "db": db,
                "task": name,
                "task_type": task_type,
                "ctx": (ce, cy),
                "query": (qe, qy),
            }
        )
        print(
            f"  relbench {db}/{name}: {len(cy)} ctx + {len(qy)} val, {task_type}",
            flush=True,
        )
    return out


def make_ests(tabpfn_dir, device, seed, n_features, precision):
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    model_path = Path(tabpfn_dir).expanduser() / "tabpfn-v3.5-20260909.safetensors"
    assert model_path.exists(), f"TabPFN checkpoint {model_path} not found"
    ests = {}
    for task_type in ("clf", "reg"):
        cls = TabPFNClassifier if task_type == "clf" else TabPFNRegressor
        est = cls(
            n_estimators=1,
            model_path=model_path,
            device=device,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            random_state=seed,
            # fp32 because it is what the A/B that found the fix below was
            # run under; autocast was never the cause (PowBackward0 nan'd
            # identically in both) and reverting to it, with n_ctx_hi back at
            # 2048, is an untested speedup rather than a correctness question.
            inference_precision={
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
            }[precision],
            # The one step in TabPFN's torch preprocessing pipeline that runs
            # unconditionally is TorchSoftClipOutliersStep -- everything else
            # sits behind enable_gpu_preprocessing, which defaults off. It
            # divides by a per-column robust scale and repairs degenerate
            # columns forward-only, with a masked_fill_ after the division, so
            # the forward is finite while autograd still differentiates
            # through a divide by zero and 0 * inf = nan lands on the ** 2.
            # Turning it off made both failing probe tasks finite and left the
            # already-working one identical to 6 significant figures.
            inference_config={"OUTLIER_REMOVAL_STD": None},
        )
        warm = torch.randn(32, n_features, device=device)
        y = torch.arange(32, device=device) % 2
        est.fit_with_differentiable_input(warm, y if task_type == "clf" else y.float())
        if task_type == "reg":
            # The standard fit moves the bar distribution to the device and the
            # differentiable one does not, so its borders sit on the cpu and the
            # loss dies in searchsorted.
            est.znorm_space_bardist_ = est.znorm_space_bardist_.to(device)
        for model in est.models_:
            for prm in model.parameters():
                prm.requires_grad_(False)
        ests[task_type] = est
    return ests


def loss_and_pred(ests, adapter, task_type, emb, y, n_ctx, device, autocast_dtype=None):
    import contextlib

    import torch
    from torch import nn

    # autocast_dtype=torch.bfloat16 puts the matmuls on the tensor cores and,
    # more importantly, makes q/k/v bf16 so torch's sdpa_kernel priority list
    # (FLASH, EFFICIENT, CUDNN, MATH -- attention/scaled_dot_product_attention
    # .py:181) can finally select FlashAttention: flash needs fp16/bf16 and
    # sm80+, and on an a100 dtype was the only thing disqualifying it.
    #
    # bf16 rather than fp16 on purpose. TabPFN's own "autocast" setting is fp16
    # on gpu (utils.py:269) and fp16's 5-bit exponent with a 65504 ceiling is
    # the shape of the original inf; bf16 keeps fp32's 8-bit exponent and only
    # loses mantissa, which cannot manufacture an inf that fp32 would not.
    ctx = (
        torch.autocast("cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with ctx:
        # .float() back immediately: autocast makes the adapter's own Linear
        # emit bf16, and TabPFN's AddFingerprintFeaturesStep hashes its input
        # via X.detach().cpu().numpy(), which raises "unsupported ScalarType
        # BFloat16" because numpy has no bf16. Handing TabPFN fp32 costs
        # nothing -- autocast casts per *op*, so the projections inside the
        # transformer are still bf16 and flash attention is still eligible;
        # only this one 512x512 matmul stays fp32.
        x = adapter(emb.to(device).float()).float()
        yy = y.to(device)
        est = ests[task_type]
        # A task can vary over its whole pool and still draw a context that
        # does not; fit_with_differentiable_input raises on std<=1e-12 rather
        # than returning, so the draw has to be rejected before it gets there.
        if task_type == "reg" and float(yy[:n_ctx].std()) < 1e-6:
            return None, None, None
        # RemoveConstantFeaturesStep raises rather than returning when every
        # column is constant, which a context of near-identical rows can be.
        if float(x[:n_ctx].std(dim=0).max()) == 0.0:
            return None, None, None
        if task_type == "clf":
            lab = (yy > 0).long()
            est.fit_with_differentiable_input(x[:n_ctx], lab[:n_ctx])
            logits = est.forward(
                x[n_ctx:], use_inference_mode=True, return_logits=True
            )
        else:
            est.fit_with_differentiable_input(x[:n_ctx], yy[:n_ctx])
            logits, _per, _b = est.forward(x[n_ctx:], use_inference_mode=True)
    # The loss stays fp32 whatever the forward ran in. The bar distribution
    # searchsorts the logits against fp32 bucket borders, and a reduction is
    # where reduced precision costs accuracy without buying throughput.
    logits = logits.float()
    if task_type == "clf":
        loss = nn.functional.cross_entropy(logits, lab[n_ctx:])
        return loss, torch.softmax(logits, -1)[:, 1], lab[n_ctx:]
    z = (yy[n_ctx:] - est.y_train_mean_) / est.y_train_std_
    loss = est.znorm_space_bardist_(
        logits.transpose(0, 1).unsqueeze(0), z.unsqueeze(0)
    ).mean()
    mean = est.znorm_space_bardist_.mean(logits.transpose(0, 1))
    return loss, mean * est.y_train_std_ + est.y_train_mean_, yy[n_ctx:]


def evaluate_relbench(relbench, ests, adapter, device):
    import torch

    from rt.eval.metrics import metric_for

    # auroc for clf, MAE for reg. The preprocessed target is already
    # z-scored -- the space rt.eval.relbench denormalises out of -- so MAE
    # on it is the nmae the result tables carry, comparable across tasks
    # and against the 0.7173 / 0.3584 rt-j baseline.
    per_task = {}
    with torch.no_grad():
        for t in relbench:
            ce, cy = t["ctx"]
            qe, qy = t["query"]
            _loss, pred, truth = loss_and_pred(
                ests,
                adapter,
                t["task_type"],
                torch.cat([ce, qe]),
                torch.cat([cy, qy]),
                len(cy),
                device,
            )
            try:
                _n, v = metric_for(
                    t["task_type"],
                    truth.float().cpu().numpy(),
                    pred.float().cpu().numpy(),
                )
            except ValueError:
                continue
            per_task[f"{t['db']}/{t['task']}"] = (t["task_type"], float(v))
    return summarize(per_task)


def main(
    *,
    features_root: str,
    relbench_pre_dir: str,
    relbench_features_root: str,
    relbench_labels_root: str,
    relbench_n_ctx: int,
    relbench_n_query: int,
    tabpfn_dir: str,
    stats_path: str,
    out_dir: str,
    d_feat: int,
    min_rows: int,
    n_ctx_lo: int,
    n_ctx_hi: int,
    n_query: int,
    tasks_per_step: int,
    total_steps: int,
    lr: float,
    lr_min: float,
    wd: float,
    adapter_kind: str,
    hidden_dim: int,
    warmup_steps: int,
    grad_norm_max: float,
    eval_every: int,
    save_every: int,
    seed: int,
    targets: dict[str, float],
    run_id: str,
    run_name: str | None,
    project: str,
    entity: str | None,
    wandb_disabled: bool,
) -> None:
    params = dict(locals())

    import numpy as np
    import torch

    from rt.train._train import seed_everything

    device = "cuda"
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    assert min_rows > n_query + 64, (
        f"min_rows {min_rows} leaves no context after {n_query} queries"
    )

    use_wandb = not wandb_disabled
    if use_wandb:
        import wandb

        job = os.environ.get("SLURM_JOB_ID")
        attempt = (
            f"{job}.{os.environ.get('SLURM_STEP_ID', '0')}"
            f".{os.environ.get('SLURM_RESTART_COUNT', '0')}"
            if job
            else f"{int(time.time())}"
        )
        wandb.init(
            project=project,
            entity=entity,
            name=f"{run_name}-{attempt}" if run_name else attempt,
            id=f"{run_id}-{attempt}",
            group=run_id,
            resume="never",
            config=params,
            dir=str(out),
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_seconds=60,
            ),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    train_entries = load_index(features_root, min_rows)
    train_pool = Pool(train_entries)

    seed_everything(seed)
    adapter = build_adapter(
        adapter_kind, d_feat, d_feat, hidden_dim, stats_path, device
    )
    init_flat = torch.cat([p.detach().flatten() for p in adapter.parameters()]).clone()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)

    def lr_at(step):
        # v1's schedule: warmup, then flat. v2 ran cosine lr -> lr_min and it
        # changed nothing, but nothing was learning in v2 either, so the decay
        # was never tested. With the conditioning fixed it is a confound to add
        # back later, not now. Uncomment the two lines for cosine.
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr
        # t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))

    ests = make_ests(tabpfn_dir, device, seed, d_feat, "fp32")
    print("tabpfn frozen; only the adapter trains", flush=True)

    relbench = load_relbench(
        relbench_pre_dir,
        relbench_features_root,
        relbench_labels_root,
        relbench_n_ctx,
        relbench_n_query,
        seed,
    )

    rng = np.random.default_rng(seed)
    t0 = time.time()
    logged_step, logged_t = 0, t0
    n_bad = n_degenerate = 0
    # Every step's pre-clip norm, not just the last one in the window: with
    # grad_norm_max left at 1.0 deliberately, the question v3 has to answer is
    # whether the clip still binds, and a single sampled step 20 apart cannot
    # say what fraction of steps are clipped.
    gnorms = []
    running = {"clf": [], "reg": []}
    for step in range(total_steps):
        cur_lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        # One task is one very correlated gradient, so a step is several of
        # them accumulated rather than a bigger draw from one task.
        #
        # Every task is checked for a non-finite gradient before the step, and
        # the whole step is dropped if any produced one. This is not belt and
        # braces: an inf gradient is *converted into NaN* by clipping, since
        # clip_coef = max_norm / (inf + eps) = 0 and inf * 0 = NaN, and AdamW
        # then writes NaN into the weights permanently. That killed the first
        # run at step 0 with a finite loss and a NaN |W-I|. The offending task
        # is named so a systematic source shows itself rather than being
        # silently skipped 10,000 times.
        bad = []
        for _ in range(tasks_per_step):
            e = train_entries[int(rng.choice(len(train_entries), p=train_pool.p))]
            emb, y, n_ctx = draw(rng, train_pool, e, n_ctx_lo, n_ctx_hi, n_query)
            loss, _pred, _truth = loss_and_pred(
                ests, adapter, e["task_type"], emb, y, n_ctx, device
            )
            if loss is None:
                n_degenerate += 1
                continue
            if not torch.isfinite(loss):
                bad.append((e["db"], e["task"], e["task_type"], n_ctx, "loss"))
                continue
            (loss / tasks_per_step).backward()
            running[e["task_type"]].append(float(loss))
        # The guard stays even though fp32 should remove the cause: clipping
        # an inf gradient silently turns it into NaN, so a single bad step
        # would poison the weights for the rest of the run.
        if bad or not all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in adapter.parameters()
        ):
            n_bad += 1
            if n_bad <= 20 or n_bad % 100 == 0:
                print(
                    f"step {step:>6} non-finite, step dropped: {bad}", flush=True
                )
            opt.zero_grad(set_to_none=True)
        else:
            # clip_grad_norm_ returns the total norm *before* clipping, which
            # is the quantity worth logging: after clipping it is just
            # min(norm, grad_norm_max).
            gnorms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(
                        adapter.parameters(), grad_norm_max
                    )
                )
            )
            opt.step()

        if step % 20 == 0:
            msg = " ".join(
                f"{k} {np.mean(v):.4f}" for k, v in running.items() if v
            )
            flat = torch.cat([p.detach().flatten() for p in adapter.parameters()])
            # dist_from_init is the comparable drift measure across both arms;
            # w_minus_i only means anything for the identity-initialised
            # linear one, where the two coincide.
            drift = float((flat - init_flat).norm())
            g = np.array(gnorms) if gnorms else np.array([np.nan])
            print(
                f"step {step:>6} {msg} drift {drift:.4f} "
                f"gnorm {np.mean(g):.4g} max {np.max(g):.4g} "
                f"clipped {float(np.mean(g > grad_norm_max)):.2f} "
                f"dropped {n_bad} degen {n_degenerate} "
                f"{(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
            if use_wandb:
                now = time.time()
                wandb.log(
                    {
                        "step": step,
                        **{
                            f"train/loss/{k}": float(np.mean(v))
                            for k, v in running.items()
                            if v
                        },
                        "train/loss/mean": float(
                            np.mean([x for v in running.values() for x in v])
                        ),
                        "train/dist_from_init": drift,
                        "train/grad_norm": float(np.mean(g)),
                        "train/grad_norm_max": float(np.max(g)),
                        "train/frac_clipped": float(np.mean(g > grad_norm_max)),
                        "train/steps_dropped": n_bad,
                        "train/ctx_degenerate": n_degenerate,
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/minutes": (now - t0) / 60,
                        **(
                            {
                                "train/steps_per_sec": (step - logged_step)
                                / (now - logged_t)
                            }
                            if step > logged_step
                            else {}
                        ),
                    },
                    step=step,
                )
                logged_step, logged_t = step, now
            running = {"clf": [], "reg": []}
            gnorms = []

        if eval_every and step % eval_every == 0:
            rb = evaluate_relbench(relbench, ests, adapter, device)
            print(
                f"step {step:>6} relbench val: "
                f"auroc {rb['clf'][0]} (n={rb['clf'][1]}) "
                f"nmae {rb['reg'][0]} (n={rb['reg'][1]})"
                f"   [rt-j baseline 0.7173 / 0.3584 at 2**18 context]",
                flush=True,
            )
            if use_wandb:
                metric = {"clf": "auroc", "reg": "nmae"}
                wandb.log(
                    {
                        "step": step,
                        **{
                            f"val/{metric[tt]}": mean
                            for tt, (mean, _n, _per) in rb.items()
                            if mean is not None
                        },
                        **{
                            f"val/n_{tt}": n for tt, (_mean, n, _per) in rb.items()
                        },
                        **{
                            f"val/{metric[tt]}/{task}": v
                            for tt, (_mean, _n, per) in rb.items()
                            for task, v in per.items()
                        },
                        **{f"target/{k}": v for k, v in targets.items()},
                    },
                    step=step,
                )

        if save_every and step % save_every == 0:
            torch.save(
                {
                    "step": step,
                    "adapter_kind": adapter_kind,
                    "hidden_dim": hidden_dim,
                    "stats_path": stats_path,
                    "state_dict": adapter.state_dict(),
                },
                out / f"adapter_step{step}.pt",
            )

    torch.save(
        {
            "step": total_steps,
            "adapter_kind": adapter_kind,
            "hidden_dim": hidden_dim,
            "stats_path": stats_path,
            "state_dict": adapter.state_dict(),
        },
        out / "adapter_final.pt",
    )
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if use_wandb:
        wandb.finish()


def preprocess_draw(est, x, y_ctx, n_ctx):
    from tabpfn.preprocessing.configs import (
        FeatureSubsamplingMethod,
        SampleSubsamplingMethod,
    )
    from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
    from tabpfn.utils import infer_random_state

    # Verbatim what fit_with_differentiable_input does internally
    # (classifier.py:1306-1324), so the stacked path sees the *same* input
    # tensor as the one-at-a-time path -- z-norm against the context, the
    # fingerprint hash column, and the column shuffle all still applied. The
    # shuffle is a real permutation even at n_estimators=1 and ColumnAggregator
    # puts RoPE on the column axis, so dropping it would change the model's
    # input, not just its bookkeeping.
    static_seed, _ = infer_random_state(est.random_state)
    pre = TabPFNEnsemblePreprocessor(
        configs=est.ensemble_configs_,
        n_samples=n_ctx,
        feature_schema=est.inferred_feature_schema_,
        random_state=static_seed,
        n_preprocessing_jobs=1,
        feature_subsampling_method=FeatureSubsamplingMethod(
            est.inference_config_.FEATURE_SUBSAMPLING_METHOD
        ),
        constant_feature_count=(
            est.inference_config_.FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT
        ),
        subsample_samples=est.inference_config_.SUBSAMPLE_SAMPLES,
        sample_subsampling_method=SampleSubsamplingMethod(
            est.inference_config_.SAMPLE_SUBSAMPLING_METHOD
        ),
        y_train=y_ctx,
        task_type=est.estimator_type,
    )
    mem = pre.fit_transform_ensemble_members(X_train=x[:n_ctx], y_train=y_ctx)[0]
    return mem.X_train, mem.transform_X_test(x[n_ctx:]), mem.config


def batched_loss(est, task_type, xs, ys, n_ctx, device):
    import torch
    from torch import nn
    from tabpfn.preprocessing.datamodel import FeatureModality

    # One fused forward for B draws that share (n_ctx, n_query, n_features).
    # Verified equivalent to running them one at a time: the only reduction in
    # the architecture that crosses the dataset axis is num_present_classes
    # (tabpfn_v3_5.py:2261), and it only widens a one-hot whose absent columns
    # contribute exactly zero. Re-running a draw at B=1 is bit-identical; the
    # residual under batching is fp32 reassociation (~1e-6) and does not depend
    # on which draws share the batch.
    Xtr, Xte, cfgs, ytr, extra = [], [], [], [], []
    for x, y in zip(xs, ys):
        if task_type == "clf":
            yb = (y > 0).long()[:n_ctx].float()
            extra.append((y > 0).long()[n_ctx:])
        else:
            yf = y[:n_ctx].float()
            mu = yf.mean()
            # Population std, as regressor.py:1183-1201. torch's default
            # correction=1 would not match.
            sd = torch.clamp(yf.std(correction=0), min=1e-20)
            yb = (yf - mu) / sd
            extra.append((mu, sd, y[n_ctx:].float()))
        a, b, c = preprocess_draw(est, x, yb, n_ctx)
        Xtr.append(a)
        Xte.append(b)
        cfgs.append(c)
        ytr.append(yb)
    # RemoveConstantFeaturesStep drops columns per draw off the *context*, so
    # two draws in one batch can disagree on width. Fail with the reason named
    # rather than inside torch.stack.
    widths = {t.shape[1] for t in Xtr}
    assert len(widths) == 1, f"draws disagree on feature width: {sorted(widths)}"

    est.fit_from_preprocessed(
        [torch.stack(Xtr)],
        [torch.stack(ytr)],
        [[[]] for _ in range(len(xs))],
        [cfgs],
        performance_options=est.models_[0].get_default_performance_options(),
    )
    Xte_s = [torch.stack(Xte)]
    if task_type == "clf":
        logits = est.forward(Xte_s, use_inference_mode=True, return_logits=True)
        yq = torch.stack(extra).to(device)
        return nn.functional.cross_entropy(logits.permute(1, 2, 0), yq)
    # TabPFNRegressor.forward raises for B>1 (regressor.py:2123), so go to the
    # executor. Nothing is lost: on the differentiable path target_transforms
    # is [None], temperature is 1.0 and there is one estimator, so the
    # wrapper's post-processing is the identity.
    est.executor_.use_torch_inference_mode(use_inference=False)
    ((out, _cfg),) = est.executor_.iter_outputs(
        Xte_s, autocast=est.use_autocast_, task_type="regression"
    )
    lg = out.float().permute(1, 0, 2)
    z = torch.stack([(q - mu) / sd for mu, sd, q in extra])
    return est.znorm_space_bardist_(lg, z).mean()
