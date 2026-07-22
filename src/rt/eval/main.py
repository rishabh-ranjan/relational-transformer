"""Standalone evaluation drivers: simple runs, context-tuned + ensembled runs,
and the eval CLI entry (RT checkpoints)."""

from __future__ import annotations

from pathlib import Path

import torch

from rt.config import Config
from rt.data import get_tasks
from rt.eval.evaluator import Evaluator
from rt.eval.metrics import metric_for
from rt.eval.relbench import _emit_and_score
from rt.model import load_rt_model

def main(cfg: Config) -> None:
    ev_cfg = cfg.eval
    assert cfg.logger.wandb_disabled, "standalone eval does not log to wandb"
    assert ev_cfg.freq is None, "eval.freq is an in-loop training knob"
    assert len(ev_cfg.ctx_sizes) == 1, (
        "standalone eval writes one submission per run and needs exactly one "
        "ctx size; multi-size ctx_sizes is an in-loop training-eval feature"
    )
    ctx_size = ev_cfg.ctx_sizes[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(cfg.model, Rel2TabModelConfig):
        # rel2tab tabular baseline: (featurizer, predictor) through the same
        # eval path as RT.
        net = cfg.model.build(device)
        embedding_model = cfg.model.embedding_model
        d_text = cfg.model.d_text
        print(f"baseline: {type(cfg.model.featurizer).__name__} + "
              f"{type(cfg.model.predictor).__name__} on {device}")
    else:
        checkpoint = cfg.model.load_ckpt_path
        assert checkpoint is not None, "model.load_ckpt_path is required"
        net, config = load_rt_model(checkpoint, device=device, compile=False)
        net = net.to(torch.bfloat16)
        embedding_model = config["embedding_model"]
        d_text = config["d_text"]
        print(f"loaded {config.get('name', checkpoint)} "
              f"(embed={embedding_model}) on {device}")
        if config.get("task_type") in ("clf", "reg"):
            print(f"warning: this checkpoint was selected/trained for "
                  f"task_type={config['task_type']}; it will be evaluated on "
                  f"both clf and reg tasks", flush=True)
        # The checkpoint's own config drives model construction; cfg.model dims
        # are ignored here. Warn when they disagree so a stale CLI default is
        # visible rather than silently shadowed.
        ckpt_model = config.get("model", {})
        mismatches = [
            f"{k}: config={v} checkpoint={ckpt_model[k]}"
            for k, v in (("num_blocks", cfg.model.num_blocks),
                         ("d_model", cfg.model.d_model),
                         ("d_text", cfg.model.d_text),
                         ("num_heads", cfg.model.num_heads),
                         ("d_ff", cfg.model.d_ff))
            if k in ckpt_model and ckpt_model[k] != v
        ]
        if cfg.model.embedding_model != embedding_model:
            mismatches.append(f"embedding_model: config={cfg.model.embedding_model} "
                              f"checkpoint={embedding_model}")
        if mismatches:
            print("warning: model config ignored for checkpoint eval; differs from "
                  "the checkpoint's own config: " + "; ".join(mismatches), flush=True)


    eval_kwargs = dict(
        embedding_model=embedding_model, d_text=d_text, device=device,
        num_walks=ev_cfg.num_walks, walk_length=ev_cfg.walk_length,
        tokens_per_gpu=ev_cfg.tokens_per_gpu, items_per_task=ev_cfg.items_per_task,
        num_workers=ev_cfg.num_workers, shuffle_seed=ev_cfg.shuffle_seed,
        prefetch_factor=ev_cfg.prefetch_factor, bool_as_num=ev_cfg.bool_as_num,
        skip_text_cols=ev_cfg.skip_text_cols, mmap_populate=ev_cfg.mmap_populate,
        balance_labels=ev_cfg.balance_labels,
        ablate_schema_semantics=ev_cfg.ablate_schema_semantics,
        vector_db_path=ev_cfg.vector_db_path,
    )
    grid = ev_cfg.lcs_bw_pl_grid

    if len(grid) > 1 or ev_cfg.ensemble_size > 1:
        assert ev_cfg.context_seed == 0, (
            "ensembling sweeps context seeds 0..ensemble_size-1; a fixed "
            "eval.context_seed only applies to single-config runs"
        )

        val_tasks = get_tasks(ev_cfg.pre_dir, ev_cfg.db_task_list, ("val",))
        test_tasks = get_tasks(ev_cfg.pre_dir, ev_cfg.db_task_list, ("test",))
        if not test_tasks:
            raise SystemExit(f"no tasks found in {ev_cfg.pre_dir}")
        run_ensemble(net, ev_cfg.pre_dir, val_tasks, test_tasks, grid=grid,
                     ensemble_size=ev_cfg.ensemble_size, ctx_size=ctx_size,
                     reg_metric=ev_cfg.reg_metric, out_dir=ev_cfg.out_dir, no_csv=not ev_cfg.write_csv,
                     **eval_kwargs)
        return

        tasks = get_tasks(ev_cfg.pre_dir, ev_cfg.db_task_list, tuple(ev_cfg.splits))
    if not tasks:
        raise SystemExit(f"no tasks found in {ev_cfg.pre_dir}")
    lcs, bw, pl = grid[0]
    ev = build_evaluator(tasks, ev_cfg.pre_dir, ctx_size=ctx_size,
                         local_ctx_size=lcs, bfs_width=bw, prefer_latest=pl,
                         context_seed=ev_cfg.context_seed, **eval_kwargs)
    run_and_report(net, tasks, ev_cfg.pre_dir, ctx_size=ctx_size,
                   reg_metric=ev_cfg.reg_metric, out_dir=ev_cfg.out_dir, no_csv=not ev_cfg.write_csv,
                   evaluator=ev, embedding_model=embedding_model)


def build_evaluator(tasks, pre_dir, *, embedding_model, d_text, device, ctx_size=8192,
                    local_ctx_size=256, bfs_width=32, num_walks=10_000, walk_length=20,
                    tokens_per_gpu=2**18, items_per_task=None, num_workers=2, context_seed=0,
                    prefer_latest=True, shuffle_seed=0, mmap_populate=True,
                    prefetch_factor=2, bool_as_num=True, skip_text_cols=False,
                    balance_labels=False, ablate_schema_semantics=False,
                    vector_db_path=None):
    # mmap_populate=True by default: pre-fault the eval data into RAM so the
    # context build is fed instead of cold-faulting it from shared storage per
    # item (the same starvation that hits training without it).
    #
    # prefer_latest / shuffle_seed are real knobs (not constants): prefer_latest
    # picks the same-table neighbor sort (recency vs frequency); shuffle_seed
    # fixes the val/test subset selection + item shuffle, so an --items-per-task
    # subsample stays the *same* rows across configs (context tuning, ensembling).
    return Evaluator(
        tasks=tasks, pre_dir=pre_dir, eval_bs=max(1, tokens_per_gpu // ctx_size),
        ctx_sizes=[ctx_size], items_per_task=items_per_task, num_workers=num_workers,
        prefetch_factor=prefetch_factor, persistent_workers=False, local_ctx_size=local_ctx_size,
        bfs_width=bfs_width, num_walks=num_walks, walk_length=walk_length,
        prefer_latest=prefer_latest, bool_as_num=bool_as_num, skip_text_cols=skip_text_cols,
        mmap_populate=mmap_populate, balance_labels=balance_labels,
        ablate_schema_semantics=ablate_schema_semantics, embedding_model=embedding_model, d_text=d_text,
        shuffle_seed=shuffle_seed, context_seed=context_seed, vector_db_path=vector_db_path,
        train_only_fallback=False,
        global_rank=0, local_rank=0, world_size=1, ddp=False, device=device,
    )


def run_and_report(model, tasks, pre_dir, *, ctx_size, reg_metric, out_dir, no_csv,
                   evaluator, embedding_model):
    """Run inference, write relbench submission CSVs, score via relbench's
    evaluator, print per-task + mean metrics. Returns a results dict."""
    out_dir = Path(out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}  debug")
    for task, _ctx, labels, preds_by_prefix, _nl, node_idxs in evaluator.evaluate_raw(
        [(model, "")], [ctx_size], with_node_idxs=True
    ):
        preds = preds_by_prefix[""]
        mname, mval, n, align, _ = _emit_and_score(
            out_dir, task, pre_dir, embedding_model, labels, preds, node_idxs,
            keep_csv=not no_csv,
        )
        nm, nv = metric_for(task.task_type, labels, preds, reg_metric)  # normalized-scale debug
        by_metric.setdefault(mname, []).append(mval)
        results[f"{task.db_name}/{task.table_name}"] = {"metric": mname, "value": mval, "n": n}
        print(f"{task.db_name + '/' + task.table_name:40} {mname:8} {mval:>9.4f} {n:>7}  "
              f"{align:>11}  norm[{nm}]={nv:.4f}")
    print(f"\n{'mean':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    if not no_csv:
        print(f"\nsubmission CSVs written to {out_dir}/  "
              f"(validate: python -m relbench.leaderboard {out_dir})")
    return results


def _is_better(task_type, a, b):
    return a > b if task_type == "clf" else a < b  # higher auc / lower mae


def run_ensemble(model, pre_dir, val_tasks, test_tasks, *, grid, ensemble_size, ctx_size,
                 reg_metric, out_dir, no_csv, **eval_kwargs):
    """Context-tuned + ensembled evaluation.

    Tune: for each task, pick the (local_ctx_size, bfs_width, prefer_latest) in
    ``grid`` with the best *validation* metric. Ensemble: on test, run that config with
    ``ensemble_size`` context seeds and average the per-item predictions, then
    score the averaged submission through relbench's evaluator.
    """
    from collections import defaultdict

    import numpy as np

    embedding_model = eval_kwargs["embedding_model"]

    # ---- tune on val: best context config per task ----
    best = {}  # (db, table) -> {"cfg", "value", "task_type"}
    for cfg in grid:
        lcs, bw, pl = cfg
        ev = build_evaluator(val_tasks, pre_dir, ctx_size=ctx_size, local_ctx_size=lcs,
                             bfs_width=bw, prefer_latest=pl, **eval_kwargs)
        for task, _c, labels, preds_by_prefix, _nl in ev.evaluate_raw([(model, "")], [ctx_size]):
            _, v = metric_for(task.task_type, labels, preds_by_prefix[""], reg_metric)
            key = (task.db_name, task.table_name)
            if key not in best or _is_better(task.task_type, v, best[key]["value"]):
                best[key] = {"cfg": cfg, "value": v, "task_type": task.task_type}
            print(f"  tune {task.db_name}/{task.table_name} cfg={cfg}: {v:.4f}")

    # ---- ensemble on test: best config per task, averaged over context seeds ----
    groups = defaultdict(list)
    for t in test_tasks:
        b = best.get((t.db_name, t.table_name))
        if b is not None:
            groups[b["cfg"]].append(t)

    out_dir = Path(out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'cfg':14} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}")
    for cfg, tasks in groups.items():
        lcs, bw, pl = cfg
        acc = {}  # key -> [labels, sum_preds, task, node_idxs]
        for seed in range(ensemble_size):
            ev = build_evaluator(tasks, pre_dir, ctx_size=ctx_size, local_ctx_size=lcs,
                                 bfs_width=bw, prefer_latest=pl, context_seed=seed,
                                 **eval_kwargs)
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
                out_dir, task, pre_dir, embedding_model, labels, preds, node_idxs,
                keep_csv=not no_csv,
            )
            by_metric.setdefault(mname, []).append(mval)
            results[f"{task.db_name}/{task.table_name}"] = {"metric": mname, "value": mval,
                                                            "cfg": cfg, "n": n}
            print(f"{task.db_name + '/' + task.table_name:40} {str(cfg):14} {mname:8} "
                  f"{mval:>9.4f} {n:>7}  {align:>11}")
    print(f"\n{'mean (ensembled)':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    if not no_csv:
        print(f"\nsubmission CSVs written to {out_dir}/  "
              f"(validate: python -m relbench.leaderboard {out_dir})")
    return results
