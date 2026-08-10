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

import contextlib
import fnmatch
import json
import os
import random
import shutil
import signal
import socket
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader

from rt._env import _setup_env
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
from rt.progress import fmt_bytes, fmt_duration, log
import wandb

# Released model dims (RT-J). Override via CLI for a different size.
# Re-seed offset applied per resumed step so a resumed stream does not replay.
SEED_STRIDE = 1_000_003


def setup_dist(num_workers: int = 0):
    """Return (device, rank, local_rank, world_size, ddp). Honors torchrun env.

    ``num_workers`` is the loader width this rank will run, which is what the
    per-worker thread budget is divided by."""
    _setup_env(num_workers)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # GPUDirect RDMA hangs on the ampere nodes: a multi-node job wedges in
        # the init-time model broadcast, every rank enqueueing it and none ever
        # starting it. Must be set before init_process_group.
        if fnmatch.fnmatch(socket.getfqdn(), "ampere*.stanford.edu"):
            os.environ["NCCL_NET_GDR_LEVEL"] = "0"
        # Long timeout: the first eval/compile keeps non-participating ranks idle
        # at a collective for many minutes; the default 10-min NCCL watchdog would
        # otherwise abort the job. (Slow first-step compile + full validation pass.)
        # `device_id` binds the rank's device up front so NCCL initializes the
        # communicator eagerly and can abort cleanly instead of hanging.
        dist.init_process_group(
            "nccl",
            timeout=timedelta(hours=1),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return f"cuda:{local_rank}", rank, local_rank, world_size, True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, 0, 0, 1, False


def run_subdir(entity: str | None, project: str, run_id: str) -> Path:
    """``<entity>/<project>/<run_id>``, so every output directory is uniquely
    associated with its run. Derived from the arguments alone, so every rank
    computes it without communicating."""
    return Path(entity or "no-entity", project, run_id)


def seed_everything(seed):
    # Fixes str/bytes hash randomization, which otherwise varies per process and
    # leaks into anything that iterates a set or dict keyed by them.
    os.environ["PYTHONHASHSEED"] = str(seed)
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

        {prefix: {split: {"auroc": {"mean": v, "rel-f1/driver-dnf": v, ...},
                          "nmae": {"mean": v, ...}}}}

    Metrics are named after what they are -- ``auroc`` for clf tasks, ``nmae``
    for reg (the labels and predictions the evaluator hands over are on the
    normalized scale, so a plain MAE over them is already MAE / train-target
    std) -- and both are in percent, the units of the published baselines in
    ``expts/fine_tune/results.md`` and so of the targets plotted beside them.
    ``"mean"`` is the average over that split's tasks of that
    type. Splits are kept apart: an evaluator built with ``eval_splits=["val",
    "test"]`` yields both, and averaging them together would both hide the
    test curve and contaminate val-driven checkpoint selection.

    Each metric is itself averaged over the requested eval ctx sizes -- one
    evaluate_raw yield is one (task, ctx_size) slice, so a task appears once
    per ctx size and its per-task value spans all of them.
    """

    metric_names = {"clf": "auroc", "reg": "nmae"}
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
                # Percent, like results.md: a curve and the published target
                # it is plotted against have to be in the same units.
                v *= 100.0
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
    loss_fn: str,
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
    keep_all_ckpts: bool,
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
    targets: dict[str, float],
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

    # Reference point for time-to-first-step: everything before the first
    # optimizer step (dist setup, model build, data, the step-0 eval) is
    # startup cost, and it is what a run that "hangs" is usually stuck in.
    start_tic = time.time()

    device, rank, local_rank, world_size, ddp = setup_dist(num_workers)
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
                loss_fn=loss_fn,
            )
            .to(device)
            .to(torch.bfloat16)
        )

    # ---- model / optim / swa ----
    net = build_net()
    raw_net = net
    if is_main:
        log(params=f"{sum(p.numel() for p in net.parameters()):_}")

    # Muon orthogonalizes a 2-D update, which only means anything for a
    # hidden weight *matrix*. The per-sem-type encoders/decoders are the
    # model's embedding and output layers -- and the number/datetime/boolean
    # ones are (d_model, 1) besides, where Newton-Schulz degenerates to a
    # rescale. Those, and everything 0/1-D, go to AdamW, per Muon's own
    # guidance (no embeddings, no heads, no gains/biases).
    def _is_muon(name, p):
        if p.ndim != 2 or min(p.shape) == 1:
            return False
        return not name.startswith(("enc_dict.", "dec_dict."))

    # Baseline for the memory breakdown logged on the first step. Taken before
    # the optimizers exist, so it is weights and nothing else -- `swa_net` is
    # built later and lands in the framework term with the rest of the fixed
    # overhead.
    mem_params = torch.cuda.memory_allocated() if device.startswith("cuda") else 0

    named = list(net.named_parameters())
    muon_params = [p for n, p in named if _is_muon(n, p)]
    other_params = [p for n, p in named if not _is_muon(n, p)]
    assert len(muon_params) + len(other_params) == len(named)
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
        # warmup_steps=0 means no warmup: full lr from the first step.
        if step >= warmup_steps:
            return 1.0
        return (step + 1) / warmup_steps

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
        ck = torch.load(resume_path, map_location="cpu", weights_only=True)
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

    # ---- data: re-seed by resumed step so the stream does not replay ----
    data_seed = seed + SEED_STRIDE * start_step
    train_init_tic = time.time()
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
            train_tasks_loaded=len(train_tasks),
            elapsed=fmt_duration(time.time() - train_init_tic),
            items=f"{total_items:_}",
            epochs=f"{total_items / stream_items:.2f}",
        )
    loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        pin_memory=True,
        # The stream is infinite and iterated once, but an eval pass that
        # interrupts it would otherwise tear down and re-fork all `num_workers`
        # processes, each of which re-mmaps the mixture.
        persistent_workers=num_workers > 0,
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

    # ---- DDP (wrapped here: static_graph depends on grad_accum) ----
    # gradient_as_bucket_view avoids a grad copy and broadcast_buffers=False
    # skips a per-step buffer sync (no buffers need it here);
    # find_unused_parameters stays False (all params are used).
    #
    # static_graph is the one knob with a tradeoff. It buys comm/compute overlap
    # on a fixed graph, but it cannot coexist with working gradient
    # accumulation: DDP only skips a microbatch's all-reduce when the *forward*
    # runs inside no_sync(), and that combination dies under static_graph
    # (``expect_autograd_hooks_ INTERNAL ASSERT`` in the reducer). Accumulating
    # runs therefore drop it and save grad_accum-1 all-reduces per step --
    # strictly more comm saved than the overlap was worth. Runs that never
    # accumulate skip nothing, so they keep static_graph.
    accumulates = multi_ctx or grad_accum > 1
    if ddp:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            # The successor to broadcast_buffers=False. It differs in still
            # syncing buffers at init, which costs nothing here: the model
            # registers no buffers at all.
            forward_sync_buffers=False,
            static_graph=not accumulates,
        )

    # ---- evaluators (built once; one per context config in the eval grid) ----
    # The first grid entry is the primary config: its metrics keep the untagged
    # wandb keys and drive best-checkpoint tracking. Extra entries are evaluated
    # alongside it under a "lcs<l>-bw<b>-pl<p>_" tag. All evaluators share the
    # underlying mmap'd data (page cache), so extra entries cost eval compute
    # only, nothing between eval points.
    # Duplicates would map to the same metrics prefix, and the later entry
    # would silently overwrite the earlier one in ``metrics``.
    assert len(set(map(tuple, eval_lcs_bw_pl_grid))) == len(eval_lcs_bw_pl_grid), (
        f"duplicate entries in eval_lcs_bw_pl_grid: {eval_lcs_bw_pl_grid}"
    )
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
                    # The workers stay alive between eval passes, so the
                    # iterator each pass re-creates is already prefetching
                    # while training runs: an eval starts on data that is
                    # ready, instead of re-forking every worker and re-mmapping
                    # the split first.
                    persistent_workers=eval_num_workers > 0,
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
                        "loss_fn": loss_fn,
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

    def prune_ckpts():
        """Delete every periodic checkpoint no longer worth keeping.

        With ``keep_all_ckpts=False`` the only checkpoints that must survive
        are the ones ``best`` still points at -- a best from an earlier eval
        is copied to best_clf/best_reg only at the very end, so its file has
        to live until then. The latest step needs no file of its own:
        resume.pt is rewritten at every eval and once more at the end, and it
        carries the same weights.
        """
        if keep_all_ckpts or not is_main:
            return
        keep = {
            f"{'swa_' if b['kind'] == 'swa' else ''}steps={b['step']}.safetensors"
            for b in best.values()
            if b is not None
        }
        for f in out_dir.glob("*steps=*.safetensors"):
            if f.name not in keep:
                f.unlink(missing_ok=True)

    def consider(metrics, step):
        # Selection is val-only: a test split may be evaluated alongside for
        # its curves, but must never pick the checkpoint. With no val split
        # configured, nothing is selected.
        for prefix, kind in [("", "live"), ("swa/", "swa")]:
            if prefix not in metrics or "val" not in metrics[prefix]:
                continue
            for tt, metric, better in [
                ("clf", "auroc", max),
                ("reg", "nmae", min),
            ]:
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
        # Unconditionally, including at n == 0, where the average is still the
        # weights it was initialized from and the swa metrics just duplicate the
        # live ones. An eval that runs one net when a later eval runs two is an
        # eval that proves nothing about what the later one costs, and every
        # eval here is also the memory check for the evals after it.
        swa.sync_to(swa_net.named_parameters())
        nets.append((swa_net, "swa/"))
        metrics = {}
        for tag, evaluator in evaluators:
            tagged_nets = [(n, tag + p) for n, p in nets]
            metrics.update(eval_avg_metrics(evaluator, tagged_nets, eval_ctx_size_list))
        # Best-checkpoint tracking follows the primary (untagged) grid entry.
        # Rank 0 only: ``evaluate_raw`` yields there and nowhere else, so the
        # other ranks hold empty metrics and must not touch ``best`` (they
        # would write None-free garbage the moment that changes).
        if is_main:
            consider(metrics, step)
        if is_main:
            if use_wandb:
                # {metric}/{split}/mean and .../{db}/{table}, the swa twin
                # of each under a "swa/" prefix.
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

    def reduce_step_stats(local_loss):
        """(mean loss over ranks, any rank preempted?).

        One collective per step carries both: the loss so the logged curve is
        the batch's, not rank 0's slice of it, and the preemption flag so all
        ranks leave the loop together (a rank exiting alone hangs the rest at
        their next collective). Both reduce with SUM, so they ride in one
        tensor.
        """
        if not ddp:
            return local_loss, preempt["flag"]
        stats = torch.tensor(
            [local_loss, 1.0 if preempt["flag"] else 0.0], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats[0].item() / world_size, stats[1].item() > 0

    # ---- training loop ----
    it = iter(loader)
    # Everything fixed that is resident before a single step runs: the SWA
    # average and its net, and -- on a resumed run -- the optimizer state loaded
    # from resume.pt. A fresh run allocates that state inside the first
    # `step()` instead, which is why the two are reported apart below.
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        mem_resident = torch.cuda.memory_allocated() - mem_params
        torch.cuda.reset_peak_memory_stats()

    step = start_step
    step_t0 = time.perf_counter()
    # Time-based resume dump: in addition to the eval_freq save (~hours apart),
    # write resume.pt every --resume-save-mins of wall-clock so a preemption
    # loses at most that much progress. The save is atomic (tmp+rename) and rank
    # 0 only; we don't count it in sec/step (step_t0 is reset after).
    last_resume_t = time.perf_counter()
    # A resume starts at a step whose eval already ran (resume.pt is written at
    # every eval), so don't repeat it. A fresh run's step 0 is not that case.
    evaled_at = start_step if start_step > 0 else None
    # Memory guard: one eval-shaped batch, run once, as soon as a step at the
    # largest ctx size has completed its forward, backward and optimizer step.
    # An eval is the largest allocation the job makes, and it lands on top of
    # everything training leaves resident -- inductor workspaces per compiled
    # ctx shape, DDP buckets, cuBLAS workspaces -- so it is only worth checking
    # once the heaviest training shape has put its share of that in place.
    # Waiting for the real eval at the next eval_freq boundary means finding
    # out an hour later, which is how the step-39000 OOM happened.
    mem_guard_ctx = max(ctx_size_list)
    mem_guard_done = False
    is_cuda = device.startswith("cuda")
    # Measured on this process's first step, not on global step 0: a resumed job
    # never sees step 0, and its memory is what matters when it is the job that
    # runs out. `ctx_size_list` varies the activation term step to step, so this
    # is one sample of it, not the maximum.
    measure_mem = is_cuda
    while step < total_steps:
        if eval_freq and step % eval_freq == 0 and step != evaled_at:
            run_eval(step)
            checkpoint(step)
            prune_ckpts()
            save_resume(step)
            evaled_at = step
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
            # Every microbatch of a step shares its ctx size.
            step_ctx = next(iter(batch.values())).shape[1]
            # The forward belongs inside no_sync() as much as the backward:
            # DDP decides whether to reduce at forward time, so a forward
            # outside the context all-reduces this microbatch regardless.
            sync = not (ddp and micro < step_grad_accum - 1)
            with contextlib.nullcontext() if sync else net.no_sync():
                out = net(batch, return_embeddings=False)
                loss = out[0] / step_grad_accum
                loss.backward()
            total_loss += loss.item()

        # Snapshot after backward: framework overhead (cuBLAS/cuDNN workspaces,
        # DDP reduction buckets, compile buffers) is fully allocated, grads
        # exist, activations are freed, optimizer state is not yet allocated.
        if measure_mem:
            torch.cuda.synchronize()
            mem_post_bwd = torch.cuda.memory_allocated()

        norm = torch.nn.utils.get_total_norm(
            [p.grad for p in raw_net.parameters() if p.grad is not None]
        )
        torch.nn.utils.clip_grads_with_norm_(raw_net.parameters(), grad_norm_max, norm)
        for o in opts:
            o.step()
        if measure_mem:
            torch.cuda.synchronize()
            peak_step = torch.cuda.max_memory_allocated()
            mem_pre_zero = torch.cuda.memory_allocated()
        for o in opts:
            o.zero_grad(set_to_none=True)
        if measure_mem:
            # Grads and optimizer state are measured exactly, by what freeing
            # or allocating them moved. Activations are the remainder of the
            # peak and so an estimate: the peak falls during backward, where
            # grads are still accumulating, so the term absorbs some of them.
            torch.cuda.synchronize()
            mem_grads = mem_pre_zero - torch.cuda.memory_allocated()
            mem_opt = mem_pre_zero - mem_post_bwd
            mem_fixed = mem_params + mem_resident
            mem_framework = mem_post_bwd - mem_fixed - mem_grads
            if is_main:
                log(
                    gpu_mem_peak=fmt_bytes(peak_step),
                    reserved=fmt_bytes(torch.cuda.max_memory_reserved()),
                    params=fmt_bytes(mem_params),
                    # SWA + whatever optimizer state a resume brought back.
                    resident=fmt_bytes(mem_resident),
                    framework=fmt_bytes(mem_framework),
                    grads=fmt_bytes(mem_grads),
                    # Zero on a resume: the state was already in `resident`.
                    optimizer=fmt_bytes(mem_opt),
                    activations=fmt_bytes(peak_step - mem_fixed - mem_framework),
                )
            # The breakdown is a one-off; the peak from here on is the run's.
            torch.cuda.reset_peak_memory_stats()
            measure_mem = False
        for s in scheds:
            s.step()
        swa.update(raw_net.named_parameters())
        step += 1

        if not mem_guard_done and evaluators and step_ctx == mem_guard_ctx:
            mem_guard_done = True
            # The first grid entry only: the others differ in context-building
            # knobs, not in the shapes the net sees, so they cost time here
            # without reaching any peak the first one misses.
            evaluators[0][1].mem_guard([raw_net, swa_net])
            if is_main:
                if is_cuda:
                    torch.cuda.synchronize()
                log(
                    mem_guard_passed_at_ctx=step_ctx,
                    peak=fmt_bytes(torch.cuda.max_memory_allocated())
                    if is_cuda
                    else "-",
                )

        total_loss, stop = reduce_step_stats(total_loss)

        step_time = time.perf_counter() - step_t0
        step_t0 = time.perf_counter()

        if is_main and step == start_step + 1:
            # step_time for the first step starts where the loop reset it after
            # the step-0 eval, so it is the compile + first-batch cost alone;
            # time_to_first_step adds everything before that (setup, eval).
            log(
                time_to_first_step=fmt_duration(time.time() - start_tic),
                compile_time=fmt_duration(step_time),
            )

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
                        # A constant per metric key, logged every step, so it
                        # draws as a horizontal line across the whole x-range
                        # of the panel its metric lives in -- wandb has no
                        # reference-line primitive, a flat series is the line.
                        **{f"target/{k}": v for k, v in targets.items()},
                    }
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

        if stop:
            if is_main:
                log(preempted_at_step=step, action="save_resume_and_exit")
            save_resume(step)
            if use_wandb:
                wandb.finish()
            if ddp:
                dist.barrier()
                dist.destroy_process_group()
            return

    # ---- final eval + best selection ----
    run_eval(step)
    checkpoint(step)
    prune_ckpts()
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
            else:
                log(warning="best_ckpt_missing", label=label, expected=src)
            log(
                saved=label,
                kind=b["kind"],
                step=b["step"],
                metric=b["metric"],
                value=f"{b['value']:.4f}",
                path=f"{label}.safetensors",
            )
        log(load_with=f"rt.model.load_rt_model('{out_dir}/best_clf.safetensors')")
    if use_wandb:
        wandb.finish()
    if ddp:
        dist.destroy_process_group()
