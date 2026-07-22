#!/usr/bin/env python
"""Pretrain a Relational Transformer on preprocessed data (the Join).

Self-supervised pretraining over every task in the preprocessed datasets at
``--pre-dir`` (local or Hub). Features: Muon+AdamW optimization, stochastic
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

    pixi run pretrain --pre-dir stanford-star/the-join-preprocessed \\
        --val-pre-dir stanford-star/relbench-preprocessed --out-dir ~/ckpts/run1
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import shutil
import signal
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader

from rt.model import load_model, resolve_checkpoint, save_model
from rt.config import Config
from rt.data import TrainDataset
from rt.model import RelationalTransformer
from rt.data import eval_tasks, pretrain_tasks

# Released model dims (RT-J). Override via CLI for a different size.
# Re-seed offset applied per resumed step so a resumed stream does not replay.
SEED_STRIDE = 1_000_003


from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

_DEFAULT_A = 3.4445
_DEFAULT_B = -4.7750
_DEFAULT_C = 2.0315
_DEFAULT_NS_STEPS = 5
_EPS = 1e-7


def _batched_newton_schulz(
    X: Tensor,
    steps: int,
    eps: float,
    a: float,
    b: float,
    c: float,
) -> Tensor:
    """Newton-Schulz orthogonalization on a batch of matrices.

    Args:
        X: (batch, m, n) with m <= n.
        steps: number of NS iterations.
        eps: numerical stability epsilon.
        a, b, c: quintic polynomial coefficients.

    Returns:
        Orthogonalized tensor, same shape, bfloat16.
    """
    X = X.bfloat16()
    X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps))
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))  # (B, m, m)
        G = torch.baddbmm(A, A, A, beta=b, alpha=c)  # b*A + c*A@A
        X = torch.baddbmm(X, G, X, beta=a)  # a*X + G@X
    return X


def _lr_factor(adjust_lr_fn: Optional[str], shape: torch.Size) -> float:
    """Shape-dependent LR scaling factor (independent of base LR)."""
    m, n = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return math.sqrt(max(1, m / n))
    elif adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(m, n))
    return 1.0


class Muon(Optimizer):
    """Muon optimizer with batched Newton-Schulz orthogonalization.

    Drop-in replacement for ``torch.optim.Muon``.  Parameters that share
    the same shape are stacked and processed together via batched matrix
    operations, reducing thousands of individual kernel launches to a
    handful of batched ones.

    Pass ``compile=True`` to torch.compile the step method.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        ns_steps: int = _DEFAULT_NS_STEPS,
        ns_coefficients: tuple[float, float, float] = (
            _DEFAULT_A,
            _DEFAULT_B,
            _DEFAULT_C,
        ),
        eps: float = _EPS,
        adjust_lr_fn: Optional[str] = None,
        compile: bool = False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon requires 2D parameters, got shape {p.shape}"
                    )

        # Pre-compute shape groups (stable across steps).
        self._shape_indices: list[dict[torch.Size, list[int]]] = []
        for group in self.param_groups:
            by_shape: dict[torch.Size, list[int]] = defaultdict(list)
            for idx, p in enumerate(group["params"]):
                by_shape[p.shape].append(idx)
            self._shape_indices.append(dict(by_shape))

        # Precompute per-param LR adjustment factors (shape-dependent, constant).
        self._param_lr_factors: list[list[float]] = []
        for group in self.param_groups:
            self._param_lr_factors.append(
                [_lr_factor(group["adjust_lr_fn"], p.shape) for p in group["params"]]
            )

        if compile:
            self._step_impl = torch.compile(self._step_impl)

    @torch.no_grad()
    def _step_impl(self, lrs):
        for gidx, group in enumerate(self.param_groups):
            lr = lrs[gidx]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            wd = group["weight_decay"]
            a, b, c = group["ns_coefficients"]
            params = group["params"]

            # Collect active params (those with gradients).
            active_params: list[Tensor] = []
            active_grads: list[Tensor] = []
            active_bufs: list[Tensor] = []
            active_indices: list[int] = []

            for i, p in enumerate(params):
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                active_params.append(p)
                active_grads.append(p.grad)
                active_bufs.append(state["momentum_buffer"])
                active_indices.append(i)

            if not active_params:
                continue

            # Phase 1: momentum via foreach (single multi-tensor kernel).
            torch._foreach_lerp_(active_bufs, active_grads, 1 - momentum)

            # Nesterov updates.
            if nesterov:
                updates = torch._foreach_lerp(active_grads, active_bufs, momentum)
            else:
                updates = active_bufs

            # Phase 2: batched Newton-Schulz per shape group.
            ortho: list[Optional[Tensor]] = [None] * len(active_params)

            # Map from param-group index to active-list position.
            active_set = {idx: j for j, idx in enumerate(active_indices)}

            for shape, param_indices in self._shape_indices[gidx].items():
                local = [active_set[i] for i in param_indices if i in active_set]
                if not local:
                    continue

                m, n = shape
                need_T = m > n

                stacked = torch.stack([updates[j] for j in local])
                if need_T:
                    stacked = stacked.transpose(-2, -1)

                X = _batched_newton_schulz(stacked, ns_steps, eps, a, b, c)

                if need_T:
                    X = X.transpose(-2, -1)

                for k, j in enumerate(local):
                    ortho[j] = X[k]

            # Phase 3: weight decay + update via foreach.
            # Group by LR adjustment factor (shape-dependent, constant).
            factor_groups: dict[float, tuple[list[Tensor], list[Tensor]]] = defaultdict(
                lambda: ([], [])
            )
            for j, p in enumerate(active_params):
                if ortho[j] is None:
                    continue
                factor = self._param_lr_factors[gidx][active_indices[j]]
                p_list, o_list = factor_groups[factor]
                p_list.append(p)
                o_list.append(ortho[j])

            # Weight decay: all active params share the same wd factor.
            torch._foreach_mul_(active_params, 1 - lr * wd)

            # Apply orthogonalized updates per factor group.
            for factor, (p_list, o_list) in factor_groups.items():
                scaled = torch._foreach_mul(o_list, -(lr * factor))
                torch._foreach_add_(p_list, scaled)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Extract LRs as scalar tensors outside the compiled region to prevent
        # dynamo from guarding on the (changing) learning rate value.
        lrs = [
            torch.tensor(
                group["lr"], dtype=torch.float64, device=group["params"][0].device
            )
            for group in self.param_groups
        ]
        self._step_impl(lrs)
        return loss


class SwaState:
    """Equal-weight or EMA running average of a fixed set of named tensors.

    Backed by an fp32 dict of clones; updates are in-place via
    ``lerp_``. The averaging weight schedule depends on ``momentum``:

    - ``1.0``: equal-weight averaging (``alpha = 1/n``). After ``k``
      updates, ``params[name]`` equals the arithmetic mean of the
      ``k`` inputs.
    - ``< 1.0``: bias-corrected EMA (``alpha = (1-m) / (1-m^n)``).
      First update has ``alpha=1.0``, asymptotes to ``1-m`` as ``n``
      grows.

    Used from training (``rt.pretrain``: in-loop SWA over ``raw_net``
    parameters).
    """

    def __init__(self, named_tensors, momentum):
        """``named_tensors``: iterable of ``(name, tensor)`` pairs. The
        fp32 storage is allocated as clones on the source tensors'
        devices. Initial values are arbitrary — the first ``update``
        sets them exactly (``alpha=1.0``)."""
        self.momentum = momentum
        self.params = {name: t.detach().float().clone() for name, t in named_tensors}
        self.n = 0

    @torch.no_grad()
    def update(self, named_tensors):
        """Add one snapshot to the running average. Source key set must
        equal the stored key set."""
        self.n += 1
        if self.momentum == 1.0:
            alpha = 1.0 / self.n
        else:
            m = self.momentum
            alpha = (1.0 - m) / (1.0 - m**self.n)
        src = dict(named_tensors)
        assert src.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(src) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(src))}"
        )
        for name, target in self.params.items():
            target.lerp_(src[name].float(), alpha)

    def state_dict(self):
        """CPU-serializable snapshot for training resume."""
        return {
            "momentum": self.momentum,
            "n": self.n,
            "params": {k: v.detach().cpu().clone() for k, v in self.params.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        assert state["momentum"] == self.momentum, (
            f"momentum mismatch: ckpt={state['momentum']} cfg={self.momentum}"
        )
        assert state["params"].keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(state['params']) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(state['params']))}"
        )
        self.n = state["n"]
        for k, v in self.params.items():
            v.copy_(state["params"][k].to(v.device))

    @torch.no_grad()
    def sync_to(self, named_tensors):
        """Copy the running average into the target tensors in-place.
        Target key set must equal the stored key set."""
        dst = dict(named_tensors)
        assert dst.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(dst) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(dst))}"
        )
        for name, target in dst.items():
            target.copy_(self.params[name])


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
def eval_avg_metrics(evaluator, nets_with_prefix, ctx_sizes, reg_metric):
    """Mean val metric per net prefix: {prefix: {"clf": auc, "reg": mae}}.

    Averaged over both eval tasks and the requested eval ctx sizes -- each
    evaluate_raw yield is one (task, ctx_size) slice, so passing the full
    ctx-size list means the mean spans all of them.
    """
    from rt.eval import metric_for

    acc = {p: {"clf": [], "reg": []} for _, p in nets_with_prefix}
    for task, _ctx, labels, preds_by_prefix, _nl in evaluator.evaluate_raw(
        nets_with_prefix, ctx_sizes
    ):
        for _, prefix in nets_with_prefix:
            try:
                _, v = metric_for(task.task_type, labels, preds_by_prefix[prefix],
                                  reg_metric)
            except ValueError:
                # e.g. a single-class slice -> ROC AUC undefined; skip this task.
                continue
            acc[prefix][task.task_type].append(v)
    return {
        p: {k: (float(np.mean(vs)) if vs else None) for k, vs in d.items()}
        for p, d in acc.items()
    }


def main(cfg: Config) -> None:
    assert cfg.eval.ensemble_size == 1, (
        "in-loop eval does not ensemble; use rt.cli.eval on a saved checkpoint "
        "for eval.ensemble_size > 1"
    )
    assert not cfg.eval.write_csv and not cfg.eval.out_dir, (
        "in-loop eval computes metrics only; submission CSVs come from rt.cli.eval"
    )
    device, rank, local_rank, world_size, ddp = setup_dist()
    is_main = rank == 0

    use_wandb = (not cfg.logger.wandb_disabled) and is_main
    if use_wandb:
        import wandb
        wandb.init(project=cfg.logger.project, name=cfg.logger.wandb_run_name,
                   id=cfg.logger.wandb_run_name, resume="allow", config=dataclasses.asdict(cfg))
    seed_everything(cfg.train.seed + rank)
    out_dir = Path(cfg.train.out_dir).expanduser()
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    compile = cfg.model.compile

    def build_net():
        return RelationalTransformer(
            num_blocks=cfg.model.num_blocks, d_model=cfg.model.d_model, d_text=cfg.model.d_text,
            num_heads=cfg.model.num_heads, d_ff=cfg.model.d_ff, compile=compile,
            materialize_attn_masks=cfg.model.materialize_attn_masks,
        ).to(device).to(torch.bfloat16)

    # ---- model / optim / swa ----
    net = build_net()
    raw_net = net
    if is_main:
        print(f"params: {sum(p.numel() for p in net.parameters()):_}", flush=True)
    muon_params = [p for p in net.parameters() if p.ndim == 2]
    other_params = [p for p in net.parameters() if p.ndim != 2]
    opts = [
        Muon(muon_params, lr=cfg.train.lr, momentum=0.95, weight_decay=cfg.train.wd,
             adjust_lr_fn="match_rms_adamw", ns_steps=5, compile=compile),
        optim.AdamW(other_params, lr=cfg.train.lr, weight_decay=0.0, betas=(0.9, 0.999),
                    eps=1e-8, fused=device.startswith("cuda")),
    ]

    def lr_lambda(step):
        return (step + 1) / cfg.train.warmup_steps if step < cfg.train.warmup_steps else 1.0

    scheds = [optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in opts]
    swa = SwaState(raw_net.named_parameters(), momentum=cfg.train.swa_momentum)
    swa_net = build_net()

    # best (kind, step, value) trackers, persisted across resumes
    best = {"clf": None, "reg": None}
    start_step = 0

    # ---- warm start (model weights only; optimizer/SWA/step start fresh) ----
    # resume.pt takes precedence: a preempted warm-started run must continue,
    # not restart from the warm-start weights.
    resume_path = out_dir / "resume.pt"
    if cfg.model.load_ckpt_path is not None and not resume_path.exists():
        _, ckpt_path = resolve_checkpoint(cfg.model.load_ckpt_path)
        raw_net.load_state_dict(load_model(ckpt_path))
        if is_main:
            print(f"warm-started model weights from {cfg.model.load_ckpt_path}",
                  flush=True)

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
            print(f"resumed from {resume_path} at step {start_step} "
                  f"(world_size now {world_size})", flush=True)

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
    data_seed = cfg.train.seed + SEED_STRIDE * start_step
    train_tasks = pretrain_tasks(cfg.train.pre_dir)
    if cfg.train.include_dbs_file:
        with open(cfg.train.include_dbs_file) as f:
            include_dbs = {
                ln.strip() for ln in f
                if ln.strip() and not ln.lstrip().startswith("#")
            }
        before = len(train_tasks)
        train_tasks = [t for t in train_tasks if t.db_name in include_dbs]
        kept_dbs = {t.db_name for t in train_tasks}
        missing = sorted(include_dbs - kept_dbs)
        if is_main:
            print(
                f"include-dbs filter ({cfg.train.include_dbs_file}): kept "
                f"{len(train_tasks)}/{before} tasks across {len(kept_dbs)} dbs "
                f"(requested {len(include_dbs)})",
                flush=True,
            )
            if missing:
                print(
                    f"  warning: {len(missing)} requested dbs not present under "
                    f"--pre-dir, e.g. {missing[:5]}",
                    flush=True,
                )
    if is_main:
        print(f"pretraining on {len(train_tasks)} tasks from {cfg.train.pre_dir}", flush=True)
    train_ds = TrainDataset(
        tasks=train_tasks, pre_dir=cfg.train.pre_dir, train_ctx_sizes=cfg.train.ctx_sizes,
        train_tokens_per_gpu=cfg.train.tokens_per_gpu, total_bs=cfg.train.total_bs,
        global_rank=rank, local_rank=local_rank, world_size=world_size,
        local_ctx_sizes=cfg.train.local_ctx_sizes, bfs_widths=cfg.train.bfs_widths,
        num_walks=cfg.train.num_walks, walk_length=cfg.train.walk_length, prefer_latest=cfg.train.prefer_latest,
        mask_prob_max=cfg.train.mask_prob_max, embedding_model=cfg.model.embedding_model, d_text=cfg.model.d_text,
        seed=data_seed, items_per_task=cfg.train.items_per_task, mask_prob_max_shared=None,
        bool_as_num=cfg.train.bool_as_num, skip_text_cols=cfg.train.skip_text_cols, mmap_populate=cfg.train.mmap_populate,
        balance_labels=cfg.train.balance_labels, timeout_per_item=cfg.train.timeout_per_item, ablate_schema_semantics=False,
        vector_db_path=cfg.train.vector_db_path, train_only_fallback=False,
    )
    loader = DataLoader(train_ds, batch_size=None, num_workers=cfg.train.num_workers,
                        prefetch_factor=cfg.train.prefetch_factor if cfg.train.num_workers else None, pin_memory=True)
    # Per ctx size, train_bs = tokens_per_gpu // ctx and grad_accum makes the
    # global batch exactly total_bs. With multiple ctx sizes the dataloader
    # yields a *list* of grad_accum microbatches per optimizer step (one shared
    # ctx size per step); with a single ctx size it yields one microbatch at a
    # time. Validate total_bs splits exactly for every ctx size, mirroring
    # TrainDataset.__iter__: when world_size*train_bs would exceed total_bs the
    # per-gpu batch shrinks to total_bs/world_size with grad_accum=1.
    multi_ctx = len(cfg.train.ctx_sizes) > 1
    for c in cfg.train.ctx_sizes:
        tb = max(1, cfg.train.tokens_per_gpu // c)
        if cfg.train.total_bs < world_size * tb:
            assert cfg.train.total_bs % world_size == 0, (
                f"total_bs={cfg.train.total_bs} not divisible by world_size={world_size}"
                f" for ctx_size={c}"
            )
        else:
            assert cfg.train.total_bs % (world_size * tb) == 0, (
                f"total_bs={cfg.train.total_bs} must be divisible by world_size*train_bs="
                f"{world_size * tb} for ctx_size={c} (world_size={world_size}); "
                f"pick a GPU count dividing total_bs/train_bs={cfg.train.total_bs // tb}"
            )
    # grad_accum for the single-ctx loop; multi-ctx derives it from the yielded
    # list length each step.
    train_bs = max(1, cfg.train.tokens_per_gpu // max(cfg.train.ctx_sizes))
    if cfg.train.total_bs < world_size * train_bs:
        train_bs = max(1, cfg.train.total_bs // world_size)
        grad_accum = 1
    else:
        grad_accum = cfg.train.total_bs // (world_size * train_bs)

    # ---- evaluators (built once; one per context config in the eval grid) ----
    # The first grid entry is the primary config: its metrics keep the untagged
    # wandb keys and drive best-checkpoint tracking. Extra entries are evaluated
    # alongside it under a "lcs<l>-bw<b>-pl<p>_" tag. All evaluators share the
    # underlying mmap'd data (page cache), so extra entries cost eval compute
    # only, nothing between eval points.
    val_tasks = eval_tasks(cfg.eval.pre_dir, splits=tuple(cfg.eval.splits))
    from rt.evaluator import Evaluator

    evaluators = [
        (f"lcs{lcs}-bw{bw}-pl{int(pl)}_" if i else "", Evaluator(
            tasks=val_tasks, pre_dir=cfg.eval.pre_dir,
            eval_bs=max(1, cfg.eval.tokens_per_gpu // max(cfg.eval.ctx_sizes)),
            ctx_sizes=cfg.eval.ctx_sizes, items_per_task=cfg.eval.items_per_task,
            num_workers=cfg.eval.num_workers, prefetch_factor=cfg.eval.prefetch_factor,
            persistent_workers=False, local_ctx_size=lcs,
            bfs_width=bw, num_walks=cfg.eval.num_walks,
            walk_length=cfg.eval.walk_length, prefer_latest=pl,
            bool_as_num=cfg.eval.bool_as_num, skip_text_cols=cfg.eval.skip_text_cols,
            mmap_populate=cfg.eval.mmap_populate, balance_labels=cfg.eval.balance_labels,
            ablate_schema_semantics=cfg.eval.ablate_schema_semantics,
            embedding_model=cfg.model.embedding_model, d_text=cfg.model.d_text,
            shuffle_seed=cfg.eval.shuffle_seed, context_seed=cfg.eval.context_seed,
            vector_db_path=cfg.eval.vector_db_path, train_only_fallback=False,
            global_rank=rank, local_rank=local_rank,
            world_size=world_size, ddp=ddp, device=device,
        ))
        for i, (lcs, bw, pl) in enumerate(cfg.eval.lcs_bw_pl_grid)
    ] if val_tasks else []

    # ---- preemption: SIGTERM/SIGUSR1 -> save + exit (cooperatively across ranks) ----
    preempt = {"flag": False}

    def _on_signal(signum, frame):
        preempt["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    if is_main:
        (out_dir / "config.json").write_text(json.dumps({
            "embedding_model": cfg.model.embedding_model, "d_text": cfg.model.d_text,
            "checkpoint_file": "model.safetensors",
            "model": {"num_blocks": cfg.model.num_blocks, "d_model": cfg.model.d_model,
                      "d_text": cfg.model.d_text, "num_heads": cfg.model.num_heads,
                      "d_ff": cfg.model.d_ff,
                      "materialize_attn_masks": cfg.model.materialize_attn_masks},
        }, indent=2) + "\n")

    def save_resume(step):
        # resume.pt stays a torch.save pickle: it holds non-tensor optimizer /
        # scheduler / SWA state that safetensors cannot store. It is internal
        # only (never distributed), used solely to resume a preempted run.
        if not is_main:
            return
        tmp = out_dir / "resume.pt.tmp"
        torch.save({"model": raw_net.state_dict(),
                    "optimizers": [o.state_dict() for o in opts],
                    "schedulers": [s.state_dict() for s in scheds],
                    "swa": swa.state_dict(), "step": step, "best": best}, tmp)
        os.replace(tmp, resume_path)  # atomic

    def checkpoint(step):
        if not is_main:
            return
        save_model(raw_net.state_dict(), out_dir / f"steps={step}.safetensors",
                   metadata={"step": step})
        if swa.n > 0:
            swa.sync_to(swa_net.named_parameters())
            save_model(swa_net.state_dict(), out_dir / f"swa_steps={step}.safetensors",
                       metadata={"step": step, "swa_n": swa.n})

    def consider(metrics, step):
        for prefix, kind in [("", "live"), ("swa_", "swa")]:
            if prefix not in metrics:
                continue
            for tt, better in [("clf", max), ("reg", min)]:
                v = metrics[prefix].get(tt)
                if v is None:
                    continue
                cur = best[tt]
                if cur is None or better(v, cur["value"]) == v:
                    best[tt] = {"kind": kind, "step": step, "value": v,
                                "metric": "auc" if tt == "clf" else cfg.eval.reg_metric}

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
            metrics.update(eval_avg_metrics(evaluator, tagged_nets,
                                            cfg.eval.ctx_sizes, cfg.eval.reg_metric))
        # Best-checkpoint tracking follows the primary (untagged) grid entry.
        consider(metrics, step)
        if is_main:
            with open(out_dir / "val_metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "swa_n": swa.n, "metrics": metrics}) + "\n")
            for prefix, m in metrics.items():
                label = prefix.rstrip("_") or "live"
                print(f"  [eval step={step} {label}] clf_auc={m['clf']} "
                      f"{cfg.eval.reg_metric}={m['reg']}", flush=True)
            if use_wandb:
                wandb.log({
                    f"val/{p}{tt}": metrics[p][tt]
                    for p in metrics for tt in metrics[p]
                    if metrics[p][tt] is not None
                }, step=step)
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
    while step < cfg.train.total_steps:
        if cfg.eval.freq and step % cfg.eval.freq == 0:
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

        norm = torch.nn.utils.get_total_norm([p.grad for p in raw_net.parameters() if p.grad is not None])
        torch.nn.utils.clip_grads_with_norm_(raw_net.parameters(), cfg.train.grad_norm_max, norm)
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

        if is_main and step % 50 == 0:
            print(f"step {step}  loss {total_loss:.4f}  grad_norm {float(norm):.3f}  "
                  f"sec/step {step_time:.3f}  load_time {load_time:.3f}", flush=True)
            if use_wandb:
                wandb.log({"train/loss": total_loss,
                           "train/lr": scheds[0].get_last_lr()[0],
                           "train/grad_norm": float(norm),
                           "train/sec_per_step": step_time,
                           "train/load_time": load_time}, step=step)

        # Time-based resume checkpoint (preemption resilience), independent of
        # the eval_freq save. All ranks evaluate the same wall-clock condition;
        # save_resume itself only writes on rank 0.
        if time.perf_counter() - last_resume_t >= cfg.train.resume_save_mins * 60:
            save_resume(step)
            last_resume_t = time.perf_counter()
            step_t0 = time.perf_counter()  # don't count the save in sec/step
            if is_main:
                print(f"resume.pt saved at step {step} "
                      f"(every {cfg.train.resume_save_mins} min)", flush=True)

        if should_stop():
            if is_main:
                print(f"preemption signal at step {step}; saving resume and exiting", flush=True)
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
                print(f"{label}: no {tt} val tasks; skipped", flush=True)
                continue
            src = out_dir / (f"swa_steps={b['step']}.safetensors" if b["kind"] == "swa"
                             else f"steps={b['step']}.safetensors")
            if src.exists():
                shutil.copyfile(src, out_dir / f"{label}.safetensors")
            print(f"\n{label}: {b['kind']} model at step {b['step']}, "
                  f"val {b['metric']}={b['value']:.4f}  ->  {label}.safetensors", flush=True)
        print(f"(load with rt.checkpoints.load_rt_model('{out_dir}/best_clf.safetensors'))", flush=True)
    if ddp:
        dist.destroy_process_group()
