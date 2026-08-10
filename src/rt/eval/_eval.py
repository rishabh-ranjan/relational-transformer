"""Standalone evaluation drivers: simple runs, context-tuned + ensembled runs,
and the eval CLI entry (RT checkpoints)."""

import fnmatch
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rt._env import _setup_env
from rt.data import get_tasks
from rt.eval.evaluator import Evaluator
from rt.eval.metrics import metric_for
from rt.eval.relbench import _emit_and_score
from rt.model import load_rt_model
from rt.progress import log
from collections import defaultdict
import numpy as np


def setup_dist():
    """Return (device, global_rank, local_rank, world_size, ddp). Honors torchrun
    env, exactly like ``rt.train._train.setup_dist``; without torchrun this is a
    plain single-process run."""
    _setup_env()
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
        # Same long timeout and `device_id` rationale as training: the first
        # task's context build keeps other ranks parked at a collective for many
        # minutes.
        dist.init_process_group(
            "nccl",
            timeout=timedelta(minutes=20),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return f"cuda:{local_rank}", rank, local_rank, world_size, True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, 0, 0, 1, False


def main(
    *,
    # model: the checkpoint carries its own dims, so only what selects it
    load_ckpt_path: str,
    embedder: str,
    d_text: int,
    num_blocks: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    # what to evaluate
    splits: list[str],
    db_task_list: list[tuple[str, str]] | str,
    pre_dir: str,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    num_walks: int,
    walk_length: int,
    items_per_task: int,
    ctx_size_list: list[int],
    mmap_populate: bool,
    shuffle_seed: int,
    context_seed: int,
    vector_db_path: str | None,
    lcs_bw_pl_grid: list[tuple[int, int, bool]],
    ensemble_size: int,
    # where it lands
    run_id: str,
    project: str,
    entity: str | None,
    out_root: str,
    wandb_disabled: bool,
) -> None:
    """Evaluate a checkpoint and write a RelBench submission.

    Every argument is required; the arguments are the record of the evaluation.
    """
    assert wandb_disabled, "standalone eval does not log to wandb"
    assert len(ctx_size_list) == 1, (
        "standalone eval writes one submission per run and needs exactly one "
        "ctx size; multi-size ctx_size_list is an in-loop training-eval feature"
    )
    ctx_size = ctx_size_list[0]
    # Submission CSVs land with the run's other outputs:
    # <out_root>/<entity>/<project>/<id>/eval_out (same layout as training).
    csv_out_dir = (
        Path(out_root).expanduser()
        / (entity or "no-entity")
        / project
        / run_id
        / "eval_out"
    )
    device, global_rank, local_rank, world_size, ddp = setup_dist()

    checkpoint = load_ckpt_path
    assert checkpoint is not None, "model.load_ckpt_path is required"
    net, config = load_rt_model(checkpoint, device=device, compile=False)
    net = net.to(torch.bfloat16)
    embedder = config["embedder"]
    d_text = config["d_text"]
    if global_rank == 0:
        log(
            model_loaded=config.get("name", checkpoint),
            embedder=embedder,
            device=device,
            world_size=world_size,
        )
    if global_rank == 0 and config.get("task_type") in ("clf", "reg"):
        log(
            warning="ckpt_task_type_mismatch",
            ckpt_task_type=config["task_type"],
            evaluated_on="clf_and_reg",
        )
    # The checkpoint carries the dims that built it; the arguments here
    # are ignored here. Warn when they disagree so a stale CLI default is
    # visible rather than silently shadowed.
    ckpt_model = config.get("model", {})
    mismatches = [
        f"{k}: config={v} checkpoint={ckpt_model[k]}"
        for k, v in (
            ("num_blocks", num_blocks),
            ("d_model", d_model),
            ("d_text", d_text),
            ("num_heads", num_heads),
            ("d_ff", d_ff),
        )
        if k in ckpt_model and ckpt_model[k] != v
    ]
    if embedder != embedder:
        mismatches.append(f"embedder: config={embedder} checkpoint={embedder}")
    if mismatches and global_rank == 0:
        log(
            warning="model_config_ignored",
            mismatches=";".join(mismatches).replace(" ", "_"),
        )

    eval_kwargs = dict(
        embedder=embedder,
        d_text=d_text,
        device=device,
        num_walks=num_walks,
        walk_length=walk_length,
        tokens_per_gpu=tokens_per_gpu,
        items_per_task=items_per_task,
        num_workers=num_workers,
        shuffle_seed=shuffle_seed,
        prefetch_factor=prefetch_factor,
        mmap_populate=mmap_populate,
        vector_db_path=vector_db_path,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        ddp=ddp,
    )
    grid = lcs_bw_pl_grid

    if len(grid) > 1 or ensemble_size > 1:
        assert context_seed == 0, (
            "ensembling sweeps context seeds 0..ensemble_size-1; a fixed "
            "eval.context_seed only applies to single-config runs"
        )

        val_tasks = get_tasks(pre_dir, db_task_list, ("val",))
        test_tasks = get_tasks(pre_dir, db_task_list, ("test",))
        if not test_tasks:
            raise SystemExit(f"no tasks found in {pre_dir}")
        run_ensemble(
            net,
            pre_dir,
            val_tasks,
            test_tasks,
            grid=grid,
            ensemble_size=ensemble_size,
            ctx_size=ctx_size,
            csv_out_dir=csv_out_dir,
            **eval_kwargs,
        )
        _teardown_dist(ddp)
        return

    tasks = get_tasks(pre_dir, db_task_list, tuple(splits))
    if not tasks:
        raise SystemExit(f"no tasks found in {pre_dir}")
    lcs, bw, pl = grid[0]
    ev = build_evaluator(
        tasks,
        pre_dir,
        ctx_size=ctx_size,
        local_ctx_size=lcs,
        bfs_width=bw,
        prefer_latest=pl,
        context_seed=context_seed,
        **eval_kwargs,
    )
    run_and_report(
        net,
        tasks,
        pre_dir,
        ctx_size=ctx_size,
        csv_out_dir=csv_out_dir,
        evaluator=ev,
        embedder=embedder,
    )
    _teardown_dist(ddp)


def _teardown_dist(ddp):
    """Rank 0 keeps working (scoring, CSV writes) after the last collective;
    the barrier keeps the other ranks alive until it is done, so the process
    group is never torn down under an in-flight peer."""
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def build_evaluator(
    tasks,
    pre_dir,
    *,
    embedder,
    d_text,
    device,
    ctx_size,
    local_ctx_size,
    bfs_width,
    num_walks,
    walk_length,
    tokens_per_gpu,
    items_per_task,
    num_workers,
    context_seed,
    prefer_latest,
    shuffle_seed,
    mmap_populate,
    prefetch_factor,
    vector_db_path,
    global_rank=0,
    local_rank=0,
    world_size=1,
    ddp=False,
):
    """Evaluator over ``tasks`` at one context size (single process by default,
    or one shard per rank under DDP).

    Every knob is required: a default here would silently paper over a
    misconfigured caller. ``mmap_populate=True`` pre-faults the eval data into
    RAM so the context build is fed instead of cold-faulting it from shared
    storage per item (the same starvation that hits training without it).
    ``prefer_latest`` picks the same-table neighbor sort (recency vs
    frequency); ``shuffle_seed`` fixes the val/test subset selection + item
    shuffle, so an ``items_per_task`` subsample stays the *same* rows across
    configs (context tuning, ensembling).
    """
    return Evaluator(
        tasks=tasks,
        pre_dir=pre_dir,
        eval_bs=max(1, tokens_per_gpu // ctx_size),
        ctx_size_list=[ctx_size],
        items_per_task=items_per_task,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        local_ctx_size=local_ctx_size,
        bfs_width=bfs_width,
        num_walks=num_walks,
        walk_length=walk_length,
        prefer_latest=prefer_latest,
        mmap_populate=mmap_populate,
        embedder=embedder,
        d_text=d_text,
        shuffle_seed=shuffle_seed,
        context_seed=context_seed,
        vector_db_path=vector_db_path,
        train_only_fallback=False,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        ddp=ddp,
        device=device,
    )


def run_and_report(
    model, tasks, pre_dir, *, ctx_size, csv_out_dir, evaluator, embedder
):
    """Run inference, write relbench submission CSVs (when ``csv_out_dir`` is
    set), score via relbench's evaluator, print per-task + mean metrics.
    Returns a results dict."""
    # Under DDP only rank 0 receives yields from ``evaluate_raw`` (the other
    # ranks just drive the collectives), so it alone scores and writes CSVs.
    is_main = evaluator.global_rank == 0
    csv_out_dir = None if csv_out_dir is None else Path(csv_out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    if is_main:
        log(eval_mode="plain", ctx_size=ctx_size)
    for task, _ctx, labels, preds_by_prefix, _nl, node_idxs in evaluator.evaluate_raw(
        [(model, "")], [ctx_size], with_node_idxs=True
    ):
        preds = preds_by_prefix[""]
        mname, mval, n, align, _ = _emit_and_score(
            csv_out_dir,
            task,
            pre_dir,
            embedder,
            labels,
            preds,
            node_idxs,
        )
        nm, nv = metric_for(task.task_type, labels, preds)  # normalized-scale debug
        by_metric.setdefault(mname, []).append(mval)
        results[f"{task.db_name}/{task.table_name}"] = {
            "metric": mname,
            "value": mval,
            "n": n,
        }
        log(
            indent=1,
            task=f"{task.db_name}/{task.table_name}",
            metric=mname,
            value=f"{mval:.4f}",
            n=n,
            align=align,
            norm_metric=nm,
            norm_value=f"{nv:.4f}",
        )
    if not is_main:
        return results
    for name, vals in by_metric.items():
        log(
            mean_metric=name,
            value=f"{sum(vals) / len(vals):.4f}",
            over_tasks=len(vals),
        )
    if csv_out_dir is not None:
        log(csv_dir=csv_out_dir)
    return results


def _is_better(task_type, a, b):
    return a > b if task_type == "clf" else a < b  # higher auc / lower mae


def run_ensemble(
    model,
    pre_dir,
    val_tasks,
    test_tasks,
    *,
    grid,
    ensemble_size,
    ctx_size,
    csv_out_dir,
    **eval_kwargs,
):
    """Context-tuned + ensembled evaluation.

    Tune: for each task, pick the (local_ctx_size, bfs_width, prefer_latest) in
    ``grid`` with the best *validation* metric. Ensemble: on test, run that config with
    ``ensemble_size`` context seeds and average the per-item predictions, then
    score the averaged submission through relbench's evaluator.
    """

    embedder = eval_kwargs["embedder"]
    ddp = eval_kwargs.get("ddp", False)
    is_main = eval_kwargs.get("global_rank", 0) == 0

    # ---- tune on val: best context config per task ----
    best = {}  # (db, table) -> {"cfg", "value", "task_type"}
    for cfg in grid:
        lcs, bw, pl = cfg
        # Tuning always reads context seed 0; the seed sweep is a test-side
        # ensembling concern (below), not part of picking the best config.
        ev = build_evaluator(
            val_tasks,
            pre_dir,
            ctx_size=ctx_size,
            local_ctx_size=lcs,
            bfs_width=bw,
            prefer_latest=pl,
            context_seed=0,
            **eval_kwargs,
        )
        for task, _c, labels, preds_by_prefix, _nl in ev.evaluate_raw(
            [(model, "")], [ctx_size]
        ):
            _, v = metric_for(task.task_type, labels, preds_by_prefix[""])
            key = (task.db_name, task.table_name)
            if key not in best or _is_better(task.task_type, v, best[key]["value"]):
                best[key] = {"cfg": cfg, "value": v, "task_type": task.task_type}
            log(
                indent=1,
                tune_task=f"{task.db_name}/{task.table_name}",
                cfg=str(cfg).replace(" ", ""),
                value=f"{v:.4f}",
            )

    # Only rank 0 saw the tuning metrics, so only it knows the winning configs.
    # Every rank must group the test tasks identically -- otherwise the ranks
    # run different task/seed sequences and hang on mismatched collectives.
    if ddp:
        payload = [best if is_main else None]
        dist.broadcast_object_list(payload, src=0)
        best = payload[0]

    # ---- ensemble on test: best config per task, averaged over context seeds ----
    groups = defaultdict(list)
    for t in test_tasks:
        b = best.get((t.db_name, t.table_name))
        if b is not None:
            groups[b["cfg"]].append(t)

    csv_out_dir = None if csv_out_dir is None else Path(csv_out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    if is_main:
        log(eval_mode="ensembled", ctx_size=ctx_size)
    for cfg, tasks in groups.items():
        lcs, bw, pl = cfg
        acc = {}  # key -> [labels, sum_preds, task, node_idxs]
        for seed in range(ensemble_size):
            ev = build_evaluator(
                tasks,
                pre_dir,
                ctx_size=ctx_size,
                local_ctx_size=lcs,
                bfs_width=bw,
                prefer_latest=pl,
                context_seed=seed,
                **eval_kwargs,
            )
            for task, _c, labels, preds_by_prefix, _nl, node_idxs in ev.evaluate_raw(
                [(model, "")], [ctx_size], with_node_idxs=True
            ):
                key = (task.db_name, task.table_name)
                p = preds_by_prefix[""].astype(np.float64)
                if key not in acc:
                    acc[key] = [labels, np.zeros_like(p), task, node_idxs]
                acc[key][1] += p
        for key, (labels, sp, task, node_idxs) in acc.items():
            preds = sp / ensemble_size
            mname, mval, n, align, _ = _emit_and_score(
                csv_out_dir,
                task,
                pre_dir,
                embedder,
                labels,
                preds,
                node_idxs,
            )
            by_metric.setdefault(mname, []).append(mval)
            results[f"{task.db_name}/{task.table_name}"] = {
                "metric": mname,
                "value": mval,
                "cfg": cfg,
                "n": n,
            }
            log(
                indent=1,
                task=f"{task.db_name}/{task.table_name}",
                cfg=str(cfg).replace(" ", ""),
                metric=mname,
                value=f"{mval:.4f}",
                n=n,
                align=align,
            )
    if not is_main:
        return results
    for name, vals in by_metric.items():
        log(
            mean_metric=name,
            value=f"{sum(vals) / len(vals):.4f}",
            over_tasks=len(vals),
        )
    if csv_out_dir is not None:
        log(csv_dir=csv_out_dir)
    return results
