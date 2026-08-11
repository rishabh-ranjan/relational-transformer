"""Standalone evaluation drivers: simple runs, context-tuned + ensembled runs,
and the eval entry point (RT checkpoints)."""

import fnmatch
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import wandb

from rt._env import _setup_env
from rt.data import get_tasks
from rt.eval.evaluator import Evaluator
from rt.eval.metrics import metric_for
from rt.eval.relbench import _emit_and_score
from rt.model import load_rt_model
from rt.progress import log
from collections import defaultdict
import numpy as np


# What each task type's metric is called on the wandb axis it shares with the
# published targets, exactly as ``rt.train.eval_avg_metrics`` names them: the
# curve and the target it is drawn against have to be one key family.
METRIC_NAMES = {"clf": "auroc", "reg": "nmae"}


def setup_dist(num_workers: int = 0):
    """Return (device, global_rank, local_rank, world_size, ddp). Honors torchrun
    env, exactly like ``rt.train._train.setup_dist``; without torchrun this is a
    plain single-process run."""
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
    val_ensemble_size: int,
    test_ensemble_size: int,
    # where it lands
    run_id: str,
    run_name: str | None,
    targets: dict[str, float],
    project: str,
    entity: str | None,
    out_root: str,
    wandb_disabled: bool,
) -> None:
    """Evaluate a checkpoint and write a RelBench submission.

    Every argument is required; the arguments are the record of the evaluation.

    ``targets`` are published baselines keyed by the same
    ``{metric}/{split}/{db}/{task}`` names the curve is logged under; each is
    logged as a constant at every ensemble size, so it draws as a horizontal
    line in the panel of the curve it bounds. Only the ensembled test pass
    logs to wandb -- it is the only thing here with a curve.

    The tuning grid is ``ctx_size_list`` x ``lcs_bw_pl_grid``, minus the
    combinations with ``local_ctx_size > ctx_size``, which are not distinct
    from ``local_ctx_size == ctx_size``. More than one surviving combination is
    a tuning run, which picks one per task on validation; exactly one is a
    fixed configuration, and nothing reads validation at all.

    The ctx sizes cost almost nothing to add: ``Evaluator`` builds each item's
    context once at ``max(ctx_size_list)`` and scores every requested size off
    a prefix of it, so a whole ``ctx_size_list`` is one pass over the data, not
    one per size. Widening ``lcs_bw_pl_grid`` is what costs passes.
    """
    params = dict(locals())
    assert wandb_disabled or (test_ensemble_size > 1 and "test" in splits), (
        "the only wandb curve here is test metric vs ensemble size"
    )
    assert wandb_disabled or targets, "nothing to draw the curve against"
    # Submission CSVs land with the run's other outputs:
    # <out_root>/<entity>/<project>/<id>/eval_out (same layout as training).
    csv_out_dir = (
        Path(out_root).expanduser()
        / (entity or "no-entity")
        / project
        / run_id
        / "eval_out"
    )
    device, global_rank, local_rank, world_size, ddp = setup_dist(num_workers)

    use_wandb = (not wandb_disabled) and global_rank == 0
    if use_wandb:
        # One wandb run per *attempt*, grouped under the run_id, as in
        # ``rt.train``. An eval run does not checkpoint, so an attempt that is
        # preempted replays the whole curve from ensemble size 1; a fresh
        # wandb id per attempt is what keeps those from colliding on a step
        # axis that only ever increases.
        job = os.environ.get("SLURM_JOB_ID")
        attempt = (
            f"{job}.{os.environ.get('SLURM_RESTART_COUNT', '0')}"
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
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_seconds=60,
            ),
        )
        # Two phases, two x-axes: the tuning sweeps configurations and the
        # ensembling sweeps seeds. Neither is wandb's own step counter, which
        # just counts log calls and is left to do that.
        wandb.define_metric("ens_size")
        wandb.define_metric("tune/idx")
        wandb.define_metric("*", step_metric="ens_size")
        wandb.define_metric("tune/*", step_metric="tune/idx")

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
    # The checkpoint carries the dims that built it; the arguments here are
    # ignored. Warn when they disagree, so a stale argument is visible rather
    # than silently shadowed.
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
    # A config is a dict key downstream (it groups the tasks that chose it),
    # and a caller that came through JSON hands these over as lists.
    grid = [tuple(cfg) for cfg in lcs_bw_pl_grid]
    ctx_sizes = sorted(ctx_size_list)
    assert ctx_sizes, "nothing to evaluate at"
    # Every (ctx, lcs, bw, pl) the tuning is choosing between.
    n_cfgs = sum(1 for lcs, _, _ in grid for c in ctx_sizes if lcs <= c)
    assert n_cfgs, "every lcs exceeds every ctx size; no configuration survives"

    assert val_ensemble_size >= 1 and test_ensemble_size >= 1, "sizes are seed counts"
    assert n_cfgs > 1 or val_ensemble_size == 1, (
        "a one-configuration grid tunes nothing, so val_ensemble_size buys nothing"
    )

    if n_cfgs > 1 or test_ensemble_size > 1:
        assert context_seed == 0, (
            "ensembling sweeps context seeds 0..N-1; a fixed eval.context_seed "
            "only applies to single-config runs"
        )

        # `splits` without "test" stops after tuning: the val scores land in
        # tuning.json and a later run evaluates test with the winner.
        tune_only = "test" not in splits
        assert not (tune_only and n_cfgs == 1), (
            "one configuration has nothing to tune; ask for the test split"
        )
        # One configuration is a fixed context config: nothing to tune, so no
        # val split is read at all.
        val_tasks = get_tasks(pre_dir, db_task_list, ("val",)) if n_cfgs > 1 else None
        test_tasks = [] if tune_only else get_tasks(pre_dir, db_task_list, ("test",))
        if not test_tasks and not tune_only:
            raise SystemExit(f"no tasks found in {pre_dir}")
        run_ensemble(
            net,
            pre_dir,
            val_tasks,
            test_tasks,
            grid=grid,
            ctx_sizes=ctx_sizes,
            val_ensemble_size=val_ensemble_size,
            test_ensemble_size=test_ensemble_size,
            tune_only=tune_only,
            tuning_out_path=csv_out_dir.parent / "tuning.json",
            csv_out_dir=csv_out_dir,
            targets=targets,
            use_wandb=use_wandb,
            **eval_kwargs,
        )
        if use_wandb:
            wandb.finish()
        _teardown_dist(ddp)
        return

    tasks = get_tasks(pre_dir, db_task_list, tuple(splits))
    if not tasks:
        raise SystemExit(f"no tasks found in {pre_dir}")
    (ctx_size,) = ctx_sizes
    lcs, bw, pl = grid[0]
    ev = build_evaluator(
        tasks,
        pre_dir,
        ctx_size_list=[ctx_size],
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
    ctx_size_list,
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
    """Evaluator over ``tasks`` at every size in ``ctx_size_list`` (single
    process by default, or one shard per rank under DDP).

    The sizes share one pass: contexts are built at the largest and each size
    is scored off a prefix, so ``evaluate_raw`` yields one result per (task,
    ctx size) for the price of the largest alone.

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
        eval_bs=max(1, tokens_per_gpu // max(ctx_size_list)),
        ctx_size_list=list(ctx_size_list),
        items_per_task=items_per_task,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
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
    ctx_sizes,
    val_ensemble_size,
    test_ensemble_size,
    tune_only,
    tuning_out_path,
    csv_out_dir,
    targets,
    use_wandb,
    **eval_kwargs,
):
    """Context-tuned + ensembled evaluation.

    Tune: for each task, pick the (ctx_size, local_ctx_size, bfs_width,
    prefer_latest) in ``ctx_sizes`` x ``grid`` -- minus ``lcs > ctx``, which is
    not distinct from ``lcs == ctx`` -- with the best *validation* metric, each
    configuration scored on its prediction averaged over ``val_ensemble_size``
    context seeds. A single surviving configuration is a fixed context config
    instead: no val split is touched and ``val_tasks`` must be ``None``.

    One pass covers a whole ``ctx_sizes``: an evaluator is built per ``grid``
    entry, and every ctx size it can serve is scored off a prefix of the
    contexts it already built. So the tuning costs ``len(grid)`` passes
    however many ctx sizes are being chosen between. Ensemble: on test, run the chosen config
    with ``test_ensemble_size`` context seeds and average the per-item
    predictions, scoring the average through relbench's evaluator after *every*
    seed -- so one run yields the whole metric-vs-ensemble-size curve, and the
    submission CSVs are those of the full ensemble.

    The two sizes are independent: tuning over one seed is much cheaper and
    usually ranks the configs the same, while matching them scores a config on
    the quantity it will be used at.

    Tuning writes every cfg's val score, and the winner per task, to
    ``tuning_out_path``. With ``tune_only`` the run stops there and never reads
    test: a later run evaluates the winner it recorded.

    With ``use_wandb`` the test curve is logged as it is produced -- one
    ``wandb.log`` per ensemble size, so the panel fills in while the run is
    still going -- under ``rt.train``'s key names and units (percent), with
    each of ``targets`` repeated at every size as the flat line it bounds.
    """

    embedder = eval_kwargs["embedder"]
    ddp = eval_kwargs.get("ddp", False)
    is_main = eval_kwargs.get("global_rank", 0) == 0

    cfgs = [(c, lcs, bw, pl) for lcs, bw, pl in grid for c in ctx_sizes if lcs <= c]
    if len(cfgs) == 1:
        assert val_tasks is None, "one configuration tunes nothing; pass val_tasks=None"
        best = {(t.db_name, t.table_name): {"cfg": cfgs[0]} for t in test_tasks}
    else:
        # ---- tune on val: best context config per task ----
        best = {}  # (db, table) -> {"cfg", "value", "task_type"}
        scores = defaultdict(dict)  # (db, table) -> str(cfg) -> value
        tuned = []  # one row per scored configuration, for the wandb table
        for lcs, bw, pl in grid:
            # Every ctx size this entry can serve, in one pass.
            ctxs = [c for c in ctx_sizes if lcs <= c]
            if not ctxs:
                continue
            # Each configuration is scored on the average over its own seeds,
            # so it is picked on the quantity it will be used at -- as long as
            # val_ensemble_size matches the test one, which is the caller's
            # business, not this loop's.
            acc = {}  # (db, table, ctx) -> [labels, sum_preds, task_type]
            for seed in range(val_ensemble_size):
                ev = build_evaluator(
                    val_tasks,
                    pre_dir,
                    ctx_size_list=ctxs,
                    local_ctx_size=lcs,
                    bfs_width=bw,
                    prefer_latest=pl,
                    context_seed=seed,
                    **eval_kwargs,
                )
                for task, ctx, labels, preds_by_prefix, _nl in ev.evaluate_raw(
                    [(model, "")], ctxs
                ):
                    key = (task.db_name, task.table_name, ctx)
                    p = preds_by_prefix[""].astype(np.float64)
                    if key not in acc:
                        acc[key] = [labels, np.zeros_like(p), task.task_type]
                    acc[key][1] += p
            for (db, table, ctx), (labels, sp, task_type) in acc.items():
                key = (db, table)
                cfg = (ctx, lcs, bw, pl)
                _, v = metric_for(task_type, labels, sp / val_ensemble_size)
                if key not in best or _is_better(task_type, v, best[key]["value"]):
                    best[key] = {"cfg": cfg, "value": v, "task_type": task_type}
                scores[key][str(cfg)] = v
                log(
                    indent=1,
                    tune_task="/".join(key),
                    cfg=str(cfg).replace(" ", ""),
                    ens_size=val_ensemble_size,
                    value=f"{v:.4f}",
                )
                if use_wandb:
                    # The search as it happens: this configuration's score, the
                    # best so far beside it, and the knobs that produced it --
                    # all against `tune/idx`, so the panel is the trajectory
                    # and a hover says which configuration a point was.
                    metric = METRIC_NAMES[task_type]
                    tuned.append([f"{db}/{table}", ctx, lcs, bw, pl, metric, v * 100])
                    wandb.log(
                        {
                            "tune/idx": len(tuned),
                            f"tune/{metric}/val/{db}/{table}": v * 100,
                            f"tune/best/{metric}/val/{db}/{table}": (
                                best[key]["value"] * 100
                            ),
                            "tune/ctx_size": ctx,
                            "tune/local_ctx_size": lcs,
                            "tune/bfs_width": bw,
                            "tune/prefer_latest": int(pl),
                        }
                    )

        # Only rank 0 saw the tuning metrics, so only it writes them and only
        # it knows the winning configs. Every rank must group the test tasks
        # identically -- otherwise the ranks run different task/seed sequences
        # and hang on mismatched collectives.
        if is_main:
            tuning_out_path = Path(tuning_out_path).expanduser()
            tuning_out_path.parent.mkdir(parents=True, exist_ok=True)
            tuning_out_path.write_text(
                json.dumps(
                    {
                        f"{db}/{table}": {
                            "best_cfg": list(best[db, table]["cfg"]),
                            "best_value": best[db, table]["value"],
                            "task_type": best[db, table]["task_type"],
                            "val_ensemble_size": val_ensemble_size,
                            "val_scores": by_cfg,
                        }
                        for (db, table), by_cfg in scores.items()
                    },
                    indent=2,
                )
            )
            log(tuning_written=str(tuning_out_path), tasks=len(scores))
        if use_wandb:
            # The same rows as a table, which is sortable in the app and is
            # what a scatter of score against any one knob is built from.
            wandb.log(
                {
                    "tune/scores": wandb.Table(
                        columns=["task", "ctx", "lcs", "bw", "pl", "metric", "value"],
                        data=tuned,
                    )
                }
            )
        if tune_only:
            return
        if ddp:
            payload = [best if is_main else None]
            dist.broadcast_object_list(payload, src=0)
            best = payload[0]

    # ---- ensemble on test: best config per task, averaged over context seeds ----
    # Grouped by the sampler settings, which are what an evaluator is built
    # for; the ctx size a task won is a prefix of the contexts that evaluator
    # already builds, so tasks that agree on (lcs, bw, pl) share one pass even
    # when they won different ctx sizes.
    groups = defaultdict(list)
    won_ctx = {}  # (db, table) -> the ctx size that task is scored at
    for t in test_tasks:
        b = best.get((t.db_name, t.table_name))
        if b is not None:
            ctx, lcs, bw, pl = b["cfg"]
            groups[lcs, bw, pl].append(t)
            won_ctx[t.db_name, t.table_name] = ctx

    csv_out_dir = None if csv_out_dir is None else Path(csv_out_dir).expanduser()
    # size -> metric -> per-task values, so the mean curve is logged per size
    # too; `results` is the full ensemble, the run's headline number.
    curve: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results = {}
    if is_main:
        log(eval_mode="ensembled", configs=len(groups))
    # Each group replays the same 1..N sizes, and wandb's step axis only ever
    # moves forward: two groups would log the second one's curve nowhere.
    assert not use_wandb or len(groups) == 1, (
        "wandb logging needs one context config across the run's tasks"
    )
    for (lcs, bw, pl), tasks in groups.items():
        ctxs = sorted({won_ctx[t.db_name, t.table_name] for t in tasks})
        acc = {}  # key -> [labels, sum_preds, task, node_idxs]
        for seed in range(test_ensemble_size):
            ev = build_evaluator(
                tasks,
                pre_dir,
                ctx_size_list=ctxs,
                local_ctx_size=lcs,
                bfs_width=bw,
                prefer_latest=pl,
                context_seed=seed,
                **eval_kwargs,
            )
            for task, ctx, labels, preds_by_prefix, _nl, node_idxs in ev.evaluate_raw(
                [(model, "")], ctxs, with_node_idxs=True
            ):
                key = (task.db_name, task.table_name)
                # The other sizes in `ctxs` are some other task's winner.
                if ctx != won_ctx[key]:
                    continue
                p = preds_by_prefix[""].astype(np.float64)
                if key not in acc:
                    acc[key] = [labels, np.zeros_like(p), task, node_idxs]
                acc[key][1] += p
            size = seed + 1
            full = size == test_ensemble_size
            # metric name -> per-task wandb key -> value, this size's curve.
            point: dict[str, dict[str, float]] = defaultdict(dict)
            for key, (labels, sp, task, node_idxs) in acc.items():
                preds = sp / size
                # Only the full ensemble writes a submission: every size scores
                # into the same per-task path, so a partial one would be
                # overwritten anyway.
                mname, mval, n, align, _ = _emit_and_score(
                    csv_out_dir if full else None,
                    task,
                    pre_dir,
                    embedder,
                    labels,
                    preds,
                    node_idxs,
                )
                curve[size][mname].append(mval)
                if use_wandb:
                    # The wandb curve is on the normalized scale and in percent
                    # -- `rt.train`'s units, and so the published targets' --
                    # not the relbench submission metric logged beside it.
                    _, nv = metric_for(task.task_type, labels, preds)
                    point[METRIC_NAMES[task.task_type]][
                        f"{task.split}/{task.db_name}/{task.table_name}"
                    ] = nv * 100.0
                if full:
                    results[f"{task.db_name}/{task.table_name}"] = {
                        "metric": mname,
                        "value": mval,
                        "cfg": (won_ctx[key], lcs, bw, pl),
                        "n": n,
                    }
                log(
                    indent=1,
                    task=f"{task.db_name}/{task.table_name}",
                    cfg=str((won_ctx[key], lcs, bw, pl)).replace(" ", ""),
                    ens_size=size,
                    metric=mname,
                    value=f"{mval:.4f}",
                    n=n,
                    align=align,
                )
            if use_wandb:
                logged = {"ens_size": size}
                for metric, per_task in point.items():
                    logged.update({f"{metric}/{k}": v for k, v in per_task.items()})
                    # One mean per split, over that split's tasks: the same
                    # `{metric}/{split}/mean` key `rt.train` logs.
                    by_split = defaultdict(list)
                    for k, v in per_task.items():
                        by_split[k.split("/")[0]].append(v)
                    for split, vs in by_split.items():
                        logged[f"{metric}/{split}/mean"] = sum(vs) / len(vs)
                # A constant at every size, so it draws as the horizontal line
                # the curve is measured against.
                logged.update({f"target/{k}": v for k, v in targets.items()})
                wandb.log(logged)
    if not is_main:
        return results
    for size in sorted(curve):
        for name, vals in curve[size].items():
            log(
                ens_size=size,
                mean_metric=name,
                value=f"{sum(vals) / len(vals):.4f}",
                over_tasks=len(vals),
            )
    if csv_out_dir is not None:
        log(csv_dir=csv_out_dir)
    return results
