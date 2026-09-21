import os
import time
from pathlib import Path


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
    total_tasks_per_step: int,
    tasks_per_micro: int,
    total_steps: int,
    lr: float,
    wd: float,
    adapter_kind: str,
    hidden_dim: int,
    warmup_steps: int,
    grad_norm_max: float,
    autocast: str | None,
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

    import fnmatch
    import socket
    from datetime import timedelta

    import numpy as np
    import torch
    import torch.distributed as dist

    from expts.repaper.adapter.train_adapter import (
        Pool,
        build_adapter,
        draw,
        evaluate_relbench,
        load_index,
        load_relbench,
        loss_and_pred,
        make_ests,
    )

    # Why data parallel at all: rt-j pretraining averaged `total_bs=1024`
    # *independently drawn* items per optimiser step across 8 a100s
    # (_train.py:513, DDP), while the single-gpu adapter run averaged 4 task
    # draws. Rows inside one task share a context and are strongly correlated,
    # so 4 x 256 query rows is 4 samples against between-task variance, not
    # 1024 -- and between-task variance is what made v1's loss curve
    # unreadable. This buys the batch with hardware, as pretraining did.
    #
    # No DDP wrapper: the adapter is 262,144 parameters, so one all_reduce of
    # the accumulated gradient per optimiser step is cheaper and far more
    # legible than DDP's per-backward bucketed hooks, which would fire on every
    # one of the `accum` micro-steps unless each were wrapped in no_sync().
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if fnmatch.fnmatch(socket.getfqdn(), "ampere*.stanford.edu"):
        os.environ["NCCL_NET_GDR_LEVEL"] = "0"
    dist.init_process_group(
        "nccl", timeout=timedelta(hours=1), device_id=torch.device(device)
    )
    is_main = rank == 0
    # Eval deliberately stays fp32 whatever training runs in, so val numbers
    # are comparable across precision arms and only the training path varies.
    autocast_dtype = {None: None, "bf16": torch.bfloat16}[autocast]

    assert total_tasks_per_step % (world_size * tasks_per_micro) == 0, (
        f"total_tasks_per_step {total_tasks_per_step} not divisible by "
        f"world_size {world_size} * tasks_per_micro {tasks_per_micro}"
    )
    accum = total_tasks_per_step // (world_size * tasks_per_micro)
    assert min_rows > n_query + 64, (
        f"min_rows {min_rows} leaves no context after {n_query} queries"
    )

    out = Path(out_dir).expanduser()
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    use_wandb = not wandb_disabled and is_main
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
            config={**params, "world_size": world_size, "accum": accum},
            dir=str(out),
            settings=wandb.Settings(
                console_multipart=True, console_chunk_max_seconds=60
            ),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    train_entries = load_index(features_root, min_rows)
    train_pool = Pool(train_entries)

    adapter = build_adapter(adapter_kind, d_feat, hidden_dim, stats_path, device)
    # Identity init is deterministic so the ranks already agree, but a mlp arm
    # would not be: broadcast regardless, or a silent divergence at step 0
    # turns into ranks optimising different functions.
    for prm in adapter.parameters():
        dist.broadcast(prm.data, src=0)
    init_flat = torch.cat([p.detach().flatten() for p in adapter.parameters()]).clone()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)

    def lr_at(step):
        # Warmup then flat, as rt-j pretraining (lr_decay_steps=0).
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr

    ests = make_ests(tabpfn_dir, device, seed, d_feat)

    # Every rank evaluates the same 21 val tasks so the metric does not depend
    # on which rank reports it; only rank 0 logs. Loading is per-rank memmap
    # reads off the same files, so this costs nothing but page cache.
    relbench = load_relbench(
        relbench_pre_dir,
        relbench_features_root,
        relbench_labels_root,
        relbench_n_ctx,
        relbench_n_query,
        seed,
    )

    # THE point of this script: each rank must draw *different* tasks, or four
    # ranks compute one gradient four times and the batch is still 4 tasks.
    rng = np.random.default_rng(seed + 10_000 * rank)
    if is_main:
        print(
            f"ddp: world_size {world_size}, {tasks_per_micro} tasks x {accum} "
            f"accum x {world_size} ranks = {total_tasks_per_step} tasks/step",
            flush=True,
        )

    t0 = time.time()
    logged_step, logged_t = 0, t0
    n_bad = n_degenerate = 0
    gnorms = []
    scale = 1.0 / (world_size * accum * tasks_per_micro)
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        local_bad = 0
        loss_sum = torch.zeros(2, device=device)
        loss_cnt = torch.zeros(2, device=device)
        for _ in range(accum):
            for _ in range(tasks_per_micro):
                e = train_entries[int(rng.choice(len(train_entries), p=train_pool.p))]
                emb, y, n_ctx = draw(rng, train_pool, e, n_ctx_lo, n_ctx_hi, n_query)
                loss, _pred, _truth = loss_and_pred(
                    ests,
                    adapter,
                    e["task_type"],
                    emb,
                    y,
                    n_ctx,
                    device,
                    autocast_dtype,
                )
                if loss is None:
                    n_degenerate += 1
                    continue
                if not torch.isfinite(loss):
                    local_bad += 1
                    continue
                # Fixed denominator, not the number that survived: a rank that
                # skips a draw contributes zero for it, which is what the
                # single-gpu run does, and needs no extra collective.
                (loss * scale).backward()
                i = 0 if e["task_type"] == "clf" else 1
                loss_sum[i] += float(loss)
                loss_cnt[i] += 1

        # A step must be dropped on *every* rank or on none: if one rank skips
        # opt.step() the ranks' weights diverge permanently and the all_reduce
        # below silently averages gradients of different models. Note an inf
        # gradient is turned into NaN by clipping (clip_coef = 1/inf = 0,
        # inf*0 = NaN), so this cannot be left to the optimiser.
        nonfinite_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in adapter.parameters()
        )
        bad = torch.tensor(
            float(local_bad + nonfinite_grad), device=device, dtype=torch.float64
        )
        dist.all_reduce(bad, op=dist.ReduceOp.SUM)
        if float(bad) > 0:
            n_bad += 1
            if is_main and (n_bad <= 20 or n_bad % 100 == 0):
                print(f"step {step:>6} non-finite on some rank, dropped", flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            # SUM, matching the 1/(world_size*accum*tasks_per_micro) already
            # folded into each backward, so the result is the mean over all
            # total_tasks_per_step draws.
            for p in adapter.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            # Clipped after the all_reduce: the norm that matters is the global
            # gradient's, not any one rank's slice of it.
            gnorms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(
                        adapter.parameters(), grad_norm_max
                    )
                )
            )
            opt.step()

        if step % 20 == 0:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_cnt, op=dist.ReduceOp.SUM)
            means = {
                k: float(loss_sum[i] / loss_cnt[i])
                for i, k in enumerate(("clf", "reg"))
                if float(loss_cnt[i]) > 0
            }
            flat = torch.cat([p.detach().flatten() for p in adapter.parameters()])
            drift = float((flat - init_flat).norm())
            g = np.array(gnorms) if gnorms else np.array([np.nan])
            if is_main:
                print(
                    f"step {step:>6} "
                    + " ".join(f"{k} {v:.4f}" for k, v in means.items())
                    + f" drift {drift:.4f} gnorm {np.mean(g):.4g} "
                    f"max {np.max(g):.4g} "
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
                        **{f"train/loss/{k}": v for k, v in means.items()},
                        "train/loss/mean": float(
                            loss_sum.sum() / loss_cnt.sum().clamp(min=1)
                        ),
                        "train/dist_from_init": drift,
                        "train/grad_norm": float(np.mean(g)),
                        "train/grad_norm_max": float(np.max(g)),
                        "train/frac_clipped": float(np.mean(g > grad_norm_max)),
                        "train/steps_dropped": n_bad,
                        "train/ctx_degenerate": n_degenerate,
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/minutes": (now - t0) / 60,
                        "train/tasks_seen": (step + 1) * total_tasks_per_step,
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
            gnorms = []

        if eval_every and step % eval_every == 0:
            rb = evaluate_relbench(relbench, ests, adapter, device)
            if is_main:
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
                        **{f"val/n_{tt}": n for tt, (_mean, n, _per) in rb.items()},
                        **{
                            f"val/{metric[tt]}/{task}": v
                            for tt, (_mean, _n, per) in rb.items()
                            for task, v in per.items()
                        },
                        **{f"target/{k}": v for k, v in targets.items()},
                    },
                    step=step,
                )

        if save_every and step % save_every == 0 and is_main:
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

    if is_main:
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
    dist.destroy_process_group()
