#!/usr/bin/env python
"""Evaluate an RT checkpoint on the RelBench benchmark tasks.

Loads a checkpoint -- a local dir/file or a Hub model repo such as
``stanford-star/rt-j/classification`` -- evaluates every task of its kind (clf/reg) in
the preprocessed RelBench data (``--pre-dir``, local or Hub), and reports metrics
through RelBench's own leaderboard evaluator: regression predictions are
denormalized to the original target scale, classification logits are sigmoided
to probabilities, and each prediction is keyed back to its relbench
``(entity_col, time_col)``. ``--out-dir`` becomes a valid RelBench *submission
directory* -- one ``<dataset>__<task>.csv`` prediction table per task -- that is
scored with ``relbench.leaderboard.evaluate_task`` (AUROC for clf, NMAE for reg)
and can be re-validated with ``python -m relbench.leaderboard <out-dir>``.

Single-process (one GPU). Example:

    pixi run eval --checkpoint stanford-star/rt-j/classification \\
        --pre-dir stanford-star/relbench-preprocessed --out-dir eval_out
"""

from __future__ import annotations

import torch

from rt.checkpoints import load_rt_model
from rt.rel2tab.config import Rel2TabModelConfig
from rt.config import Config
from rt.recipes import get_tasks


def main(cfg: Config) -> None:
    ev_cfg = cfg.eval
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
        task_type = None if ev_cfg.task_type == "both" else ev_cfg.task_type
        print(f"baseline: {type(cfg.model.featurizer).__name__} + "
              f"{type(cfg.model.predictor).__name__} on {device}")
    else:
        checkpoint = cfg.model.load_ckpt_path
        assert checkpoint is not None, "model.load_ckpt_path is required"
        net, config = load_rt_model(checkpoint, device=device, compile=False)
        net = net.to(torch.bfloat16)
        embedding_model = config["embedding_model"]
        d_text = config["d_text"]
        task_type = config.get("task_type")
        print(f"loaded {config.get('name', checkpoint)} "
              f"(task_type={task_type}, embed={embedding_model}) on {device}")

    def of_kind(tasks):
        return [t for t in tasks if t.task_type == task_type] if task_type in ("clf", "reg") else tasks

    sel = set(ev_cfg.tasks) if ev_cfg.tasks else None

    def selected(tasks):
        if sel is None:
            return tasks
        return [t for t in tasks if t.db_name in sel or f"{t.db_name}/{t.table_name}" in sel]

    eval_kwargs = dict(
        embedding_model=embedding_model, d_text=d_text, device=device,
        num_walks=ev_cfg.num_walks, walk_length=ev_cfg.walk_length,
        tokens_per_gpu=ev_cfg.tokens_per_gpu, items_per_task=ev_cfg.items_per_task,
        num_workers=ev_cfg.num_workers, shuffle_seed=ev_cfg.shuffle_seed,
    )
    grid = ev_cfg.lcs_bw_pl_grid

    if len(grid) > 1 or ev_cfg.ensemble_size > 1:
        from rt.eval_utils import run_ensemble

        val_tasks = selected(of_kind(get_tasks("relbench_eval_val", ev_cfg.pre_dir)))
        test_tasks = selected(of_kind(get_tasks("relbench_eval_test", ev_cfg.pre_dir)))
        if not test_tasks:
            raise SystemExit(f"no {task_type} tasks found in {ev_cfg.pre_dir}")
        run_ensemble(net, ev_cfg.pre_dir, val_tasks, test_tasks, grid=grid,
                     ensemble_size=ev_cfg.ensemble_size, ctx_size=ctx_size,
                     reg_metric=ev_cfg.reg_metric, out_dir=ev_cfg.out_dir, no_csv=not ev_cfg.write_csv,
                     **eval_kwargs)
        return

    from rt.eval_utils import build_evaluator, run_and_report

    tasks = selected(of_kind(get_tasks(ev_cfg.recipe, ev_cfg.pre_dir)))
    if not tasks:
        raise SystemExit(f"no {task_type} tasks found in {ev_cfg.pre_dir}")
    lcs, bw, pl = grid[0]
    ev = build_evaluator(tasks, ev_cfg.pre_dir, ctx_size=ctx_size,
                         local_ctx_size=lcs, bfs_width=bw, prefer_latest=pl,
                         **eval_kwargs)
    run_and_report(net, tasks, ev_cfg.pre_dir, ctx_size=ctx_size,
                   reg_metric=ev_cfg.reg_metric, out_dir=ev_cfg.out_dir, no_csv=not ev_cfg.write_csv,
                   evaluator=ev, embedding_model=embedding_model)

