#!/usr/bin/env python
"""Context-scaling eval on the 4DBInfer tasks: one method, per-(task, ctx) metric.

Started as a copy of ``src/rt/cli/eval.py`` and diverges in three ways, all of
which that script cannot do:

1. **A context sweep, not a single point.** ``rt.eval.main`` asserts
   ``len(ctx_size_list) == 1``. Here the whole 256..8192 curve comes out of *one*
   pass per task: :meth:`Evaluator.evaluate_raw` builds each target's context once
   at ``max(ctx_size_list)`` and reads every shorter point off as a length-``ctx``
   prefix of that one context, which also shrinks the visible in-context label
   set. So the sweep costs one pass, and the points are exactly the points a
   256-only run would produce.
2. **Metrics computed here, not through RelBench's leaderboard evaluator.**
   4DBInfer is not a RelBench leaderboard benchmark, so there is nothing to submit
   to. AUROC / NMAE are computed on rustler's normalized target scale, which is
   what the leaderboard metric reduces to anyway: rustler normalizes a regression
   target by the same train std that NMAE divides out, and AUROC is invariant to
   the classification sigmoid.
3. **Per-(method, task) JSON output**, written atomically and skipped when
   present, so slurm array tasks can shard the task list and resume freely.

Two methods share one ``.predict(batch, ctx_sizes, device, task, bool_as_num)``
interface:

* ``rt`` -- RT-J, routed through the clf checkpoint for clf-type tasks and the reg
  checkpoint for reg-type ones.
* ``rdblearn_tabicl`` -- ``precomputed_rdblearn`` features (written offline by
  ``featurize.py``) fed to ``tabicl_batched``.

Every context knob is identical across methods, which is what makes the reported
``mean_labels`` comparable: it is a pure function of the sampler, so if the two
methods disagree on it at any ctx, the comparison is not on the same x-axis and
the numbers should not be believed. ``reduce.py`` checks that.

    python expts/dbinfer/eval.py --method rt --out-dir /dfs/user/$USER/dbinfer-scaling
    python expts/dbinfer/eval.py --method rdblearn_tabicl --tasks dbinfer-amazon/churn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import OrderedDict
from pathlib import Path

# rustler's Sampler is unpicklable, so force 'fork' before any DataLoader exists
# (3.14 defaults to forkserver/spawn, which would try to pickle it).
import multiprocessing as mp

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

import numpy as np
import torch

# Share DataLoader tensors through node-local /tmp files rather than /dev/shm,
# which dense multi-worker nodes exhaust.
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # local rel2tab/

from rt_tasks import get_tasks  # noqa: E402  (vendored; see _check_task_resolution)


def _check_task_resolution(pre_dir, db_task_list, splits) -> None:
    """Fail if the vendored task resolution has drifted from the repo's.

    Stage 2 cannot import ``rt`` (its env has no torch or rustler), so
    ``rt_tasks`` is a copy of ``rt.data.tasks``. This process *can* import both, so
    it is the one place the copy can be checked -- and it must be, or the featurizer
    could silently iterate a different task set than the eval scores.
    """
    from rt.data import get_tasks as upstream

    mine = get_tasks(pre_dir, db_task_list, splits)
    theirs = upstream(pre_dir, db_task_list, splits)
    key = lambda ts: sorted(  # noqa: E731
        (
            t.db_name,
            t.table_name,
            t.target_column,
            t.task_type,
            t.split,
            t.leakage_columns,
        )
        for t in ts
    )
    if key(mine) != key(theirs):
        raise RuntimeError(
            "expts/dbinfer/rt_tasks.py has drifted from rt.data.tasks:\n"
            f"  vendored: {key(mine)}\n"
            f"  upstream: {key(theirs)}\n"
            "re-copy src/rt/data/tasks.py into expts/dbinfer/rt_tasks.py"
        )


# The rel2tab baseline is featurized against these text embeddings, and RT-J's own
# config specifies the same model -- so a single preprocessed copy serves both and
# the two methods see byte-identical contexts.
EMBEDDER = "all-MiniLM-L12-v2"
D_TEXT = 384

CTX_SIZES = [256, 512, 1024, 2048, 4096, 8192]
METHODS = ("rt", "rdblearn_tabicl")

# Subdirectory under ``<pre_dir>/<db>/`` holding the precomputed feature matrices.
FEATURES_SUBDIR = "rdblearn_features"


def compute_metric(task_type: str, labels: np.ndarray, preds: np.ndarray):
    """``(metric_name, value)`` for one (task, ctx) slice, on the normalized scale.

    reg -> ``nmae``: MAE against rustler's normalized target, which equals RelBench
    NMAE (same train std). clf -> ``roc_auc`` of ``labels > 0`` against ``preds``.
    A slice with a single class present yields nan rather than raising -- possible
    on a 1024-row subsample of a rare-positive task.
    """
    import sklearn.metrics as M

    if task_type == "reg":
        return "nmae", float(M.mean_absolute_error(labels, preds))
    y = (np.asarray(labels) > 0).astype(int)
    try:
        return "roc_auc", float(M.roc_auc_score(y, preds))
    except ValueError as e:
        print(f"    roc_auc undefined ({e}); recording nan", flush=True)
        return "roc_auc", float("nan")


def make_evaluator(tasks, args, device):
    from rt.eval.evaluator import Evaluator

    return Evaluator(
        tasks=tasks,
        pre_dir=args.pre_dir,
        eval_bs=max(1, args.tokens_per_gpu // max(args.ctx_sizes)),
        ctx_size_list=sorted(args.ctx_sizes),
        items_per_task=args.items_per_task,
        num_workers=args.num_workers,
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0,
        local_ctx_size=args.local_ctx_size,
        bfs_width=args.bfs_width,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        prefer_latest=args.prefer_latest,
        mmap_populate=args.mmap_populate,
        embedder=EMBEDDER,
        d_text=D_TEXT,
        shuffle_seed=args.shuffle_seed,
        context_seed=args.context_seed,
        vector_db_path=None,
        train_only_fallback=False,
        global_rank=args.global_rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        ddp=args.world_size > 1,
        device=device,
    )


def build_rdblearn_tabicl(args, device):
    """``precomputed_rdblearn`` features + ``tabicl_batched`` predictor."""
    from rel2tab.config import Rel2TabModelConfig
    from rel2tab.featurizers import PrecomputedFeaturizerConfig
    from rel2tab.predictors import TabICLBatchedPredictorConfig

    cfg = Rel2TabModelConfig(
        featurizer=PrecomputedFeaturizerConfig(
            pre_dir=args.pre_dir,
            db_task_list=args.db_task_list,
            eval_splits=["test"],
            features_subdir=FEATURES_SUBDIR,
        ),
        predictor=TabICLBatchedPredictorConfig(
            max_batch_size=args.tabicl_max_batch_size,
            min_bin_size=args.tabicl_min_bin_size,
            softmax_temperature=args.tabicl_softmax_temperature,
            use_amp=False,
        ),
        featurize_batch_size=4096,
        embedding_model=EMBEDDER,
        d_text=D_TEXT,
    )
    return cfg.build(device)


def build_rt(args, device, task_type: str):
    """RT-J, clf or reg head depending on the task."""
    from rt.model.checkpoints import load_rt_model

    spec = args.rt_clf_ckpt if task_type == "clf" else args.rt_reg_ckpt
    kwargs = {}
    if os.environ.get("RT_MATERIALIZE_ATTN_MASKS", "") == "0":
        kwargs["model_kwargs"] = {"materialize_attn_masks": False}
    net, _config = load_rt_model(spec, device=device, compile=args.compile, **kwargs)
    return net.to(torch.bfloat16)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.json"
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def task_json_path(out_dir: Path, method: str, task) -> Path:
    return out_dir / method / f"{task.db_name}__{task.table_name}.json"


def flush_task(out_dir, method, task, per_ctx, config, rank0: bool) -> None:
    if not rank0:
        return
    path = task_json_path(Path(out_dir), method, task)
    if path.exists():
        return
    _atomic_write_json(
        path,
        {
            "method": method,
            "task": f"{task.db_name}/{task.table_name}",
            "db": task.db_name,
            "table": task.table_name,
            "task_type": task.task_type,
            "per_ctx": {str(c): per_ctx[c] for c in sorted(per_ctx)},
            "config": config,
        },
    )
    summary = "  ".join(
        f"ctx={c}:{per_ctx[c]['metric_name']}="
        + (
            "nan"
            if per_ctx[c]["metric_value"] is None
            else f"{per_ctx[c]['metric_value']:.4f}"
        )
        + f"(lbl={per_ctx[c]['mean_labels']:.1f})"
        for c in sorted(per_ctx)
    )
    print(
        f"[{method}] {task.db_name}/{task.table_name} ({task.task_type})  {summary}",
        flush=True,
    )


def run(model, evaluator, ctx_sizes, method, out_dir, config, rank0: bool) -> None:
    """Drive ``evaluate_raw`` and write one JSON per task.

    ``evaluate_raw`` yields ``(task, ctx, labels, preds_by_prefix, num_labels)`` with
    the ctx sweep as the inner loop, so a task's rows arrive consecutively; flush at
    each task boundary so a crash keeps every completed task.
    """
    per_ctx: dict[int, dict] = OrderedDict()
    cur = None
    for task, ctx, labels, preds_by_prefix, num_labels in evaluator.evaluate_raw(
        [(model, "")], ctx_sizes
    ):
        if cur is not None and task is not cur:
            flush_task(out_dir, method, cur, per_ctx, config, rank0)
            per_ctx = OrderedDict()
        cur = task
        name, value = compute_metric(task.task_type, labels, preds_by_prefix[""])
        per_ctx[int(ctx)] = {
            "metric_name": name,
            "metric_value": value if np.isfinite(value) else None,
            "n": int(labels.shape[0]),
            "mean_labels": float(np.mean(num_labels)),
        }
    if cur is not None:
        flush_task(out_dir, method, cur, per_ctx, config, rank0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pre-dir", default="/dfs/user/$USER/pre/dbinfer-preprocessed")
    ap.add_argument(
        "--db-task-list", default=str(Path(__file__).resolve().parent / "tasks.json")
    )
    ap.add_argument("--tasks", nargs="+", default=None, help="db or db/table filter")
    ap.add_argument("--ctx-sizes", nargs="+", type=int, default=CTX_SIZES)
    # 1024-row test subsample, identical for both methods: the sampler picks it
    # from (task, items_per_task, shuffle_seed) alone, nothing model-dependent.
    ap.add_argument("--items-per-task", type=int, default=1024)
    # Uniform context config -- rt/cli/eval.py's own default. The RelBench campaign
    # selected these per task from a validation grid; there is no such grid for
    # 4DBInfer, so one setting is used everywhere and the numbers are untuned.
    ap.add_argument("--local-ctx-size", type=int, default=256)
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument(
        "--prefer-latest", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--num-walks", type=int, default=10_000)
    ap.add_argument("--walk-length", type=int, default=20)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--context-seed", type=int, default=0)
    ap.add_argument("--tokens-per-gpu", type=int, default=2**18)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--mmap-populate", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--rt-clf-ckpt", default="stanford-star/rt-j/classification")
    ap.add_argument("--rt-reg-ckpt", default="stanford-star/rt-j/regression")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tabicl-max-batch-size", type=int, default=32)
    ap.add_argument("--tabicl-min-bin-size", type=int, default=64)
    ap.add_argument("--tabicl-softmax-temperature", type=float, default=0.9)
    args = ap.parse_args()

    args.pre_dir = os.path.expandvars(args.pre_dir)
    # torchrun sets these; single-process eval leaves them unset.
    args.global_rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank0 = args.global_rank == 0
    ctx_sizes = sorted(args.ctx_sizes)
    out_dir = Path(args.out_dir).expanduser()

    if args.world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"

    _check_task_resolution(args.pre_dir, args.db_task_list, ["test"])
    all_tasks = get_tasks(args.pre_dir, args.db_task_list, ["test"])
    sel = set(args.tasks) if args.tasks else None
    tasks = [
        t
        for t in all_tasks
        if sel is None or t.db_name in sel or f"{t.db_name}/{t.table_name}" in sel
    ]
    todo = [t for t in tasks if not task_json_path(out_dir, args.method, t).exists()]
    if rank0:
        print(
            f"method={args.method} device={device} world_size={args.world_size}\n"
            f"ctx={ctx_sizes} items_per_task={args.items_per_task}\n"
            f"tasks selected={len(tasks)} done={len(tasks) - len(todo)} todo={len(todo)}",
            flush=True,
        )
    if not todo:
        if rank0:
            print("nothing to do.")
        return

    config = {
        "method": args.method,
        "ctx_sizes": ctx_sizes,
        "items_per_task": args.items_per_task,
        "local_ctx_size": args.local_ctx_size,
        "bfs_width": args.bfs_width,
        "prefer_latest": args.prefer_latest,
        "num_walks": args.num_walks,
        "walk_length": args.walk_length,
        "shuffle_seed": args.shuffle_seed,
        "context_seed": args.context_seed,
        "tokens_per_gpu": args.tokens_per_gpu,
        "eval_bs": max(1, args.tokens_per_gpu // max(ctx_sizes)),
        "pre_dir": args.pre_dir,
        "embedder": EMBEDDER,
        "d_text": D_TEXT,
        "world_size": args.world_size,
    }

    if args.method == "rdblearn_tabicl":
        config["features_subdir"] = FEATURES_SUBDIR
        config["tabicl"] = {
            "max_batch_size": args.tabicl_max_batch_size,
            "min_bin_size": args.tabicl_min_bin_size,
            "softmax_temperature": args.tabicl_softmax_temperature,
        }
        model = build_rdblearn_tabicl(args, device)
        evaluator = make_evaluator(todo, args, device)
        run(model, evaluator, ctx_sizes, args.method, out_dir, config, rank0)
        return

    # RT-J has separate clf and reg checkpoints, so split the task list by type and
    # make one evaluator per group -- a group's tasks all route through one net.
    config["rt_clf_ckpt"] = args.rt_clf_ckpt
    config["rt_reg_ckpt"] = args.rt_reg_ckpt
    for task_type in ("clf", "reg"):
        group = [t for t in todo if t.task_type == task_type]
        if not group:
            continue
        if rank0:
            print(f"\n=== rt / {task_type}: {len(group)} task(s)", flush=True)
        model = build_rt(args, device, task_type)
        evaluator = make_evaluator(group, args, device)
        run(model, evaluator, ctx_sizes, args.method, out_dir, config, rank0)


if __name__ == "__main__":
    main()
