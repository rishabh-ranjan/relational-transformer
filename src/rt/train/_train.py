#!/usr/bin/env python
"""Pretrain a Relational Transformer on preprocessed data (the Join).

Self-supervised pretraining over every task in the preprocessed datasets at
``--pre-dir`` (a local directory; download it up front, see docs/train.md). Features: Muon+AdamW optimization, stochastic
weight averaging (SWA), periodic validation, checkpointing, and automatic
selection of the best classifier / regressor checkpoint by mean validation
metric across all live and SWA evaluations.

Robust to preemption (the default config matches the released RT-J runs):

* checkpoints + a full ``resume.pt`` (model, optimizers, schedulers, SWA, step,
  best-so-far) are written every eval; a SIGTERM/SIGUSR1 handler saves and exits
  cleanly so the job can be requeued.
* resume is **GPU-count flexible**: data parallelism keeps the full model +
  optimizer on every rank (no sharding), so a run preempted on 16 GPUs across 2
  nodes can resume on, say, 4 GPUs. The training data stream is re-seeded by the
  resumed step so no items are replayed, and ops are seeded for determinism.

Single-node multi-GPU and multi-node (preemptible queue) both run under
``torchrun`` -- see the README for the exact launch commands.

    torchrun --standalone --nproc-per-node=auto -m rt.cli.train \\
        --train.pre-dir data/the-join-preprocessed \\
        --eval.pre-dir data/relbench-preprocessed \\
        --logger.out-root ~/ckpts
"""

import json
import os
import random
import shutil
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader

from rt.data import TrainDataset, get_tasks
from rt.model import (
    RelationalTransformer,
    load_model,
    resolve_checkpoint,
    save_model,
)
from rt.train.muon import Muon
from rt.train.swa import SwaState
from rt.eval import metric_for
from rt.eval import Evaluator
from rt.progress import log
import wandb

# Released model dims (RT-J). Override via CLI for a different size.
# Re-seed offset applied per resumed step so a resumed stream does not replay.
SEED_STRIDE = 1_000_003


def setup_dist():
    """Return (device, rank, local_rank, world_size, ddp). Honors torchrun env."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Long timeout: the first eval/compile keeps non-participating ranks idle
        # at a collective for many minutes; the default 10-min NCCL watchdog would
        # otherwise abort the job. (Slow first-step compile + full validation pass.)
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        return f"cuda:{local_rank}", rank, local_rank, world_size, True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, 0, 0, 1, False


def run_subdir(entity: str | None, project: str, run_id: str) -> Path:
    """``<entity>/<project>/<run_id>``, so every output directory is uniquely
    associated with its run. Derived from the arguments alone, so every rank
    computes it without communicating."""
    return Path(entity or "no-entity", project, run_id)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN where it does not conflict with the compiled kernels.
    torch.backends.cudnn.benchmark = False


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.inference_mode()
def eval_avg_metrics(evaluator, nets_with_prefix, ctx_size_list):
    """Per-task and mean metric, per net prefix, eval split and metric name::

        {prefix: {split: {"auc": {"mean": v, "rel-f1/driver-dnf": v, ...},
                          "mae": {"mean": v, ...}}}}

    Metrics are named after what they are -- ``auc`` for clf tasks, ``mae``
    for reg -- and ``"mean"`` is the average over that split's tasks of that
    type. Splits are kept apart: an evaluator built with ``eval_splits=["val",
    "test"]`` yields both, and averaging them together would both hide the
    test curve and contaminate val-driven checkpoint selection.

    Each metric is itself averaged over the requested eval ctx sizes -- one
    evaluate_raw yield is one (task, ctx_size) slice, so a task appears once
    per ctx size and its per-task value spans all of them.
    """

    metric_names = {"clf": "auc", "reg": "mae"}
    # split -> metric_name -> task_key -> [values over ctx sizes]
    acc = {
        p: {s: {m: {} for m in metric_names.values()} for s in evaluator.eval_splits}
        for _, p in nets_with_prefix
    }
    for task, _ctx, labels, preds_by_prefix, _nl in evaluator.evaluate_raw(
        nets_with_prefix, ctx_size_list
    ):
        for _, prefix in nets_with_prefix:
            try:
                _, v = metric_for(task.task_type, labels, preds_by_prefix[prefix])
            except ValueError:
                # e.g. a single-class slice -> ROC AUC undefined; skip this task.
                continue
            # setdefault: a task with an empty split is absent from
            # ``eval_splits`` but still yielded, and still worth a curve.
            by_metric = acc[prefix].setdefault(
                task.split, {m: {} for m in metric_names.values()}
            )
            per_task = by_metric[metric_names[task.task_type]]
            per_task.setdefault(f"{task.db_name}/{task.table_name}", []).append(v)

    def _reduce(per_task):
        # Per task: mean over ctx sizes. Then "mean": mean over tasks.
        out = {k: float(np.mean(vs)) for k, vs in per_task.items()}
        out["mean"] = float(np.mean(list(out.values()))) if out else None
        return out

    return {
        p: {
            s: {m: _reduce(per_task) for m, per_task in by_metric.items()}
            for s, by_metric in by_split.items()
        }
        for p, by_split in acc.items()
    }


def main(
    *,
    # model
    embedder: str,
    d_text: int,
    num_blocks: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    compile: bool,
    materialize_attn_masks: bool,
    load_ckpt_path: str | None,
    # data + optimization
    db_task_list: list[tuple[str, str]] | str,
    pre_dir: str,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    ctx_size_list: list[int],
    local_ctx_size_list: list[int],
    bfs_width_list: list[int],
    prefer_latest_list: list[bool],
    num_walks: int,
    walk_length: int,
    mask_prob_max: float,
    items_per_task: int,
    lr: float,
    wd: float,
    warmup_steps: int,
    grad_norm_max: float,
    total_bs: int,
    total_steps: int,
    swa_momentum: float,
    seed: int,
    mmap_populate: bool,
    timeout_per_item: float,
    eval_freq: int | None,
    vector_db_path: str | None,
    resume_save_mins: float,
    # in-loop validation
    eval_splits: list[str],
    eval_db_task_list: list[tuple[str, str]] | str,
    eval_pre_dir: str,
    eval_tokens_per_gpu: int,
    eval_num_workers: int,
    eval_prefetch_factor: int,
    eval_num_walks: int,
    eval_walk_length: int,
    eval_items_per_task: int,
    eval_ctx_size_list: list[int],
    eval_mmap_populate: bool,
    eval_shuffle_seed: int,
    eval_context_seed: int,
    eval_vector_db_path: str | None,
    eval_lcs_bw_pl_grid: list[tuple[int, int, bool]],
    # logging
    run_id: str,
    project: str,
    entity: str | None,
    run_name: str | None,
    wandb_disabled: bool,
    out_root: str,
) -> None:
    """Pretrain a Relational Transformer under DDP.

    Every argument is required: a default here would be a hidden experimental
    choice, and the arguments *are* the record of the run (they are written to
    ``params.json`` beside the checkpoints and logged to wandb).
    """
    params = dict(locals())

    device, rank, local_rank, world_size, ddp = setup_dist()
    is_main = rank == 0

    use_wandb = (not wandb_disabled) and is_main
    if use_wandb:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            id=run_id,
            resume="allow",
            config=params,
        )
        # Log against our own step axis rather than wandb's internal counter.
        # A resumed run rewinds to the last resume.pt, so it re-logs steps the
        # previous attempt already sent; wandb's counter only moves forward and
        # would drop every one of them ("Tried to log to step N that is less
        # than the current step M ... this data will be ignored"), silently
        # losing the window between the last checkpoint and the preemption.
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    seed_everything(seed + rank)
    # Rank 0 is the only writer; the other ranks read exactly one thing from
    # here, resume.pt. That needs no agreement between ranks: a resume only
    # happens when --logger.id names an existing run, and then every rank
    # derives the same path from its own config. An unset id defaults to a
    # per-rank timestamp, which names a fresh directory holding no resume.pt --
    # on every rank alike, so they still agree there is nothing to resume.
    out_dir = Path(out_root).expanduser() / run_subdir(entity, project, run_id)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        # The arguments are the run's record; keep them next to what they made.
        (out_dir / "params.json").write_text(
            json.dumps(params, indent=1, sort_keys=True) + "\n"
        )
        log(out_dir=out_dir)
    compile = compile

    def build_net():
        return (
            RelationalTransformer(
                num_blocks=num_blocks,
                d_model=d_model,
                d_text=d_text,
                num_heads=num_heads,
                d_ff=d_ff,
                compile=compile,
                materialize_attn_masks=materialize_attn_masks,
            )
            .to(device)
            .to(torch.bfloat16)
        )

    # ---- model / optim / swa ----
    net = build_net()
    raw_net = net
    if is_main:
        log(params=f"{sum(p.numel() for p in net.parameters()):_}")
    muon_params = [p for p in net.parameters() if p.ndim == 2]
    other_params = [p for p in net.parameters() if p.ndim != 2]
    opts = [
        Muon(
            muon_params,
            lr=lr,
            momentum=0.95,
            weight_decay=wd,
            adjust_lr_fn="match_rms_adamw",
            ns_steps=5,
            compile=compile,
        ),
        optim.AdamW(
            other_params,
            lr=lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=device.startswith("cuda"),
        ),
    ]

    def lr_lambda(step):
        return (step + 1) / warmup_steps if step < warmup_steps else 1.0

    scheds = [optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in opts]
    swa = SwaState(raw_net.named_parameters(), momentum=swa_momentum)
    swa_net = build_net()

    # best (kind, step, value) trackers, persisted across resumes
    best = {"clf": None, "reg": None}
    start_step = 0

    # ---- warm start (model weights only; optimizer/SWA/step start fresh) ----
    # resume.pt takes precedence: a preempted warm-started run must continue,
    # not restart from the warm-start weights.
    resume_path = out_dir / "resume.pt"
    if load_ckpt_path is not None and not resume_path.exists():
        _, ckpt_path = resolve_checkpoint(load_ckpt_path)
        raw_net.load_state_dict(load_model(ckpt_path))
        if is_main:
            log(warm_started_from=load_ckpt_path)

    # ---- resume from preemption (GPU-count flexible: full model+opt per rank) ----
    if resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu")
        raw_net.load_state_dict(ck["model"])
        for o, sd in zip(opts, ck["optimizers"], strict=True):
            o.load_state_dict(sd)
        for s, sd in zip(scheds, ck["schedulers"], strict=True):
            s.load_state_dict(sd)
        swa.load_state_dict(ck["swa"])
        start_step = ck["step"]
        best = ck.get("best", best)
        if is_main:
            log(
                resumed_from=resume_path,
                step=start_step,
                world_size=world_size,
            )

    if ddp:
        # Multi-node comm tuning: gradient_as_bucket_view avoids a grad copy,
        # broadcast_buffers=False skips per-step buffer sync (no buffers needing
        # it here), static_graph enables comm/compute overlap for the fixed
        # compiled graph. find_unused_parameters stays False (all params used).
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=True,
        )

    # ---- data: re-seed by resumed step so the stream does not replay ----
    data_seed = seed + SEED_STRIDE * start_step
    train_tasks = get_tasks(pre_dir, db_task_list, ("train",))
    train_ds = TrainDataset(
        tasks=train_tasks,
        pre_dir=pre_dir,
        train_ctx_size_list=ctx_size_list,
        train_tokens_per_gpu=tokens_per_gpu,
        total_bs=total_bs,
        global_rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_ctx_size_list=local_ctx_size_list,
        bfs_width_list=bfs_width_list,
        num_walks=num_walks,
        walk_length=walk_length,
        prefer_latest_list=prefer_latest_list,
        mask_prob_max=mask_prob_max,
        embedder=embedder,
        d_text=d_text,
        seed=data_seed,
        items_per_task=items_per_task,
        mask_prob_max_shared=None,
        mmap_populate=mmap_populate,
        timeout_per_item=timeout_per_item,
        vector_db_path=vector_db_path,
        train_only_fallback=False,
    )
    if is_main:
        # total_bs items enter the model per optimizer step, so the whole run
        # consumes total_steps * total_bs items. Printed against the stream's
        # size so it is obvious how many times the data is repeated.
        #
        # That size is the sampler's own count, which is why this comes after
        # the dataset is built: items_per_task is a *cap*, and multiplying it by
        # the task count says what the run would see if every task were at least
        # that large. On a single small task it is not close -- rel-f1's
        # driver-top3 has 1_353 training items against a cap of 100_000 -- and
        # the epoch count printed from the cap was wrong by that factor.
        total_items = total_steps * total_bs
        stream_items = train_ds.num_items
        log(
            train_tasks=len(train_tasks),
            pre_dir=pre_dir,
            items=f"{total_items:_}",
            steps=f"{total_steps:_}",
            bs=total_bs,
            distinct_items=f"{stream_items:_}",
            epochs=f"{total_items / stream_items:.2f}",
        )
    loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        pin_memory=True,
    )
    # Per ctx size, train_bs = tokens_per_gpu // ctx and grad_accum makes the
    # global batch exactly total_bs. With multiple ctx sizes the dataloader
    # yields a *list* of grad_accum microbatches per optimizer step (one shared
    # ctx size per step); with a single ctx size it yields one microbatch at a
    # time. Validate total_bs splits exactly for every ctx size, mirroring
    # TrainDataset.__iter__: when world_size*train_bs would exceed total_bs the
    # per-gpu batch shrinks to total_bs/world_size with grad_accum=1.
    multi_ctx = len(ctx_size_list) > 1
    for c in ctx_size_list:
        tb = max(1, tokens_per_gpu // c)
        if total_bs < world_size * tb:
            assert total_bs % world_size == 0, (
                f"total_bs={total_bs} not divisible by world_size={world_size}"
                f" for ctx_size={c}"
            )
        else:
            assert total_bs % (world_size * tb) == 0, (
                f"total_bs={total_bs} must be divisible by world_size*train_bs="
                f"{world_size * tb} for ctx_size={c} (world_size={world_size}); "
                f"pick a GPU count dividing total_bs/train_bs={total_bs // tb}"
            )
    # grad_accum for the single-ctx loop; multi-ctx derives it from the yielded
    # list length each step.
    train_bs = max(1, tokens_per_gpu // max(ctx_size_list))
    if total_bs < world_size * train_bs:
        train_bs = max(1, total_bs // world_size)
        grad_accum = 1
    else:
        grad_accum = total_bs // (world_size * train_bs)

    # ---- evaluators (built once; one per context config in the eval grid) ----
    # The first grid entry is the primary config: its metrics keep the untagged
    # wandb keys and drive best-checkpoint tracking. Extra entries are evaluated
    # alongside it under a "lcs<l>-bw<b>-pl<p>_" tag. All evaluators share the
    # underlying mmap'd data (page cache), so extra entries cost eval compute
    # only, nothing between eval points.
    val_tasks = get_tasks(eval_pre_dir, eval_db_task_list, tuple(eval_splits))

    evaluators = (
        [
            (
                f"lcs{lcs}-bw{bw}-pl{int(pl)}_" if i else "",
                Evaluator(
                    tasks=val_tasks,
                    pre_dir=eval_pre_dir,
                    eval_bs=max(1, eval_tokens_per_gpu // max(eval_ctx_size_list)),
                    ctx_size_list=eval_ctx_size_list,
                    items_per_task=eval_items_per_task,
                    num_workers=eval_num_workers,
                    prefetch_factor=eval_prefetch_factor,
                    persistent_workers=False,
                    local_ctx_size=lcs,
                    bfs_width=bw,
                    num_walks=eval_num_walks,
                    walk_length=eval_walk_length,
                    prefer_latest=pl,
                    mmap_populate=eval_mmap_populate,
                    embedder=embedder,
                    d_text=d_text,
                    shuffle_seed=eval_shuffle_seed,
                    context_seed=eval_context_seed,
                    vector_db_path=eval_vector_db_path,
                    train_only_fallback=False,
                    global_rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    ddp=ddp,
                    device=device,
                ),
            )
            for i, (lcs, bw, pl) in enumerate(eval_lcs_bw_pl_grid)
        ]
        if val_tasks
        else []
    )

    # ---- preemption: SIGTERM/SIGUSR1 -> save + exit (cooperatively across ranks) ----
    preempt = {"flag": False}

    def _on_signal(signum, frame):
        preempt["flag"] = True
        # Log it: when a preempted run comes back at the last *periodic* save
        # instead of the step it died at, the question is always whether the
        # ranks ever saw the signal. Without this line that is unanswerable
        # after the fact.
        log(rank=rank, caught_signal=signum, action="save_next_step")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    if is_main:
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "embedder": embedder,
                    "d_text": d_text,
                    "checkpoint_file": "model.safetensors",
                    "model": {
                        "num_blocks": num_blocks,
                        "d_model": d_model,
                        "d_text": d_text,
                        "num_heads": num_heads,
                        "d_ff": d_ff,
                        "materialize_attn_masks": materialize_attn_masks,
                    },
                },
                indent=2,
            )
            + "\n"
        )

    def save_resume(step):
        # resume.pt stays a torch.save pickle: it holds non-tensor optimizer /
        # scheduler / SWA state that safetensors cannot store. It is internal
        # only (never distributed), used solely to resume a preempted run.
        if not is_main:
            return
        # Write beside the target and rename: rename(2) is atomic, so a reader
        # (or the next attempt) sees either the previous checkpoint or the new
        # one, never a half-written file -- being SIGKILLed mid-write only
        # leaves a stale .tmp behind. The pid in the name keeps two writers from
        # interleaving into the same temporary if a run is ever double-started.
        # fsync before the rename so the bytes are on the server, not just in
        # the client's cache, when the directory entry flips.
        tmp = out_dir / f"resume.pt.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as f:
                torch.save(
                    {
                        "model": raw_net.state_dict(),
                        "optimizers": [o.state_dict() for o in opts],
                        "schedulers": [s.state_dict() for s in scheds],
                        "swa": swa.state_dict(),
                        "step": step,
                        "best": best,
                    },
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, resume_path)  # atomic
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def checkpoint(step):
        if not is_main:
            return
        save_model(
            raw_net.state_dict(),
            out_dir / f"steps={step}.safetensors",
            metadata={"step": step},
        )
        if swa.n > 0:
            swa.sync_to(swa_net.named_parameters())
            save_model(
                swa_net.state_dict(),
                out_dir / f"swa_steps={step}.safetensors",
                metadata={"step": step, "swa_n": swa.n},
            )

    def consider(metrics, step):
        # Selection is val-only: a test split may be evaluated alongside for
        # its curves, but must never pick the checkpoint. With no val split
        # configured, nothing is selected.
        for prefix, kind in [("", "live"), ("swa_", "swa")]:
            if prefix not in metrics or "val" not in metrics[prefix]:
                continue
            for tt, metric, better in [("clf", "auc", max), ("reg", "mae", min)]:
                v = metrics[prefix]["val"][metric].get("mean")
                if v is None:
                    continue
                cur = best[tt]
                if cur is None or better(v, cur["value"]) == v:
                    best[tt] = {
                        "kind": kind,
                        "step": step,
                        "value": v,
                        "metric": metric,
                    }

    def run_eval(step):
        if not evaluators:
            return
        nets = [(raw_net, "")]
        if swa.n > 0:
            swa.sync_to(swa_net.named_parameters())
            nets.append((swa_net, "swa_"))
        metrics = {}
        for tag, evaluator in evaluators:
            tagged_nets = [(n, tag + p) for n, p in nets]
            metrics.update(eval_avg_metrics(evaluator, tagged_nets, eval_ctx_size_list))
        # Best-checkpoint tracking follows the primary (untagged) grid entry.
        consider(metrics, step)
        if is_main:
            with open(out_dir / "val_metrics.jsonl", "a") as f:
                f.write(
                    json.dumps({"step": step, "swa_n": swa.n, "metrics": metrics})
                    + "\n"
                )
            for prefix, by_split in metrics.items():
                label = prefix.rstrip("_") or "live"
                for split, by_metric in by_split.items():
                    log(
                        indent=1,
                        eval_model=label,
                        split=split,
                        step=step,
                        auc=by_metric["auc"].get("mean"),
                        mae=by_metric["mae"].get("mean"),
                    )
            if use_wandb:
                # {prefix}{metric}/{split}/mean and .../{db}/{table}
                wandb.log(
                    {
                        "step": step,
                        **{
                            f"{p}{metric}/{split}/{task_key}": v
                            for p, by_split in metrics.items()
                            for split, by_metric in by_split.items()
                            for metric, per_task in by_metric.items()
                            for task_key, v in per_task.items()
                            if v is not None
                        },
                    }
                )
        for n, _ in nets:
            n.train()

    def should_stop():
        """True if any rank caught a preemption signal."""
        flag = torch.tensor([1.0 if preempt["flag"] else 0.0], device=device)
        if ddp:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return flag.item() > 0

    # ---- training loop ----
    it = iter(loader)
    step = start_step
    step_t0 = time.perf_counter()
    # Time-based resume dump: in addition to the eval_freq save (~hours apart),
    # write resume.pt every --resume-save-mins of wall-clock so a preemption
    # loses at most that much progress. The save is atomic (tmp+rename) and rank
    # 0 only; we don't count it in sec/step (step_t0 is reset after).
    last_resume_t = time.perf_counter()
    while step < total_steps:
        if eval_freq and step % eval_freq == 0:
            run_eval(step)
            checkpoint(step)
            save_resume(step)
            step_t0 = time.perf_counter()  # don't count eval/ckpt in step time

        total_loss = 0.0
        # load_time = wall-clock spent waiting on the dataloader (next(it)).
        # With prefetch hiding data loading it is ~0; if it dominates, the
        # GPUs are data-starved (the failure mode this run is verifying).
        load_time = 0.0
        # Multi-ctx: one next(it) yields a list of grad_accum microbatches that
        # share a ctx size (so grad_accum can vary per step). Single-ctx: each
        # next(it) yields one microbatch, called grad_accum times.
        if multi_ctx:
            t_load = time.perf_counter()
            micro_batches = next(it)
            load_time += time.perf_counter() - t_load
        else:
            micro_batches = None
        step_grad_accum = len(micro_batches) if multi_ctx else grad_accum
        for micro in range(step_grad_accum):
            if multi_ctx:
                raw_batch = micro_batches[micro]
            else:
                t_load = time.perf_counter()
                raw_batch = next(it)
                load_time += time.perf_counter() - t_load
            batch = move(raw_batch, device)
            out = net(batch, return_embeddings=False)
            loss = out[0] / step_grad_accum
            if ddp and micro < step_grad_accum - 1:
                with net.no_sync():
                    loss.backward()
            else:
                loss.backward()
            total_loss += loss.item()

        norm = torch.nn.utils.get_total_norm(
            [p.grad for p in raw_net.parameters() if p.grad is not None]
        )
        torch.nn.utils.clip_grads_with_norm_(raw_net.parameters(), grad_norm_max, norm)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        for s in scheds:
            s.step()
        swa.update(raw_net.named_parameters())
        step += 1

        step_time = time.perf_counter() - step_t0
        step_t0 = time.perf_counter()

        if is_main:
            # Every step to wandb: a fine-tuning run is short, and a loss curve
            # sampled every 50th step hides exactly the early movement these
            # runs are about. stdout keeps the coarser cadence -- it is read by
            # a human, and 10k lines is not.
            if use_wandb:
                wandb.log(
                    {
                        "step": step,
                        "train/loss": total_loss,
                        "train/lr": scheds[0].get_last_lr()[0],
                        "train/grad_norm": float(norm),
                        "train/sec_per_step": step_time,
                        "train/load_time": load_time,
                    }
                )
            if step % 50 == 0:
                log(
                    step=step,
                    loss=f"{total_loss:.4f}",
                    grad_norm=f"{float(norm):.3f}",
                    sec_per_step=f"{step_time:.3f}",
                    load_time=f"{load_time:.3f}",
                )

        # Time-based resume checkpoint (preemption resilience), independent of
        # the eval_freq save. All ranks evaluate the same wall-clock condition;
        # save_resume itself only writes on rank 0.
        if time.perf_counter() - last_resume_t >= resume_save_mins * 60:
            save_resume(step)
            last_resume_t = time.perf_counter()
            step_t0 = time.perf_counter()  # don't count the save in sec/step
            if is_main:
                log(resume_saved_at_step=step, every_mins=resume_save_mins)

        if should_stop():
            if is_main:
                log(preempted_at_step=step, action="save_resume_and_exit")
            save_resume(step)
            if ddp:
                dist.barrier()
                dist.destroy_process_group()
            return

    # ---- final eval + best selection ----
    run_eval(step)
    checkpoint(step)
    save_resume(step)
    if is_main:
        for tt, label in [("clf", "best_clf"), ("reg", "best_reg")]:
            b = best[tt]
            if b is None:
                log(skipped=label, task_type=tt, reason="no_val_tasks")
                continue
            src = out_dir / (
                f"swa_steps={b['step']}.safetensors"
                if b["kind"] == "swa"
                else f"steps={b['step']}.safetensors"
            )
            if src.exists():
                shutil.copyfile(src, out_dir / f"{label}.safetensors")
            log(
                saved=label,
                kind=b["kind"],
                step=b["step"],
                metric=b["metric"],
                value=f"{b['value']:.4f}",
                path=f"{label}.safetensors",
            )
        log(load_with=f"rt.model.load_rt_model('{out_dir}/best_clf.safetensors')")
    if ddp:
        dist.destroy_process_group()
