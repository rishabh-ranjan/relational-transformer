#!/usr/bin/env python
"""Run rel2tab tabular baselines on the RelBench benchmark tasks.

A baseline is a (featurizer, predictor) pair evaluated through the same data path
as ``rt.eval``: each task's in-context training labels (optionally
featurized) are fed to a tabular predictor. Prints per-task + mean metrics and
writes per-item prediction CSVs.

    pixi run baseline --featurizer entity --predictor ridge \\
        --pre-dir stanford-star/relbench-preprocessed --recipe relbench_eval_test

Featurizers: ``global`` (all in-context rows), ``entity`` (same-entity rows),
``rt`` (RelationalTransformer embeddings; needs --rt-ckpt). Predictors:
``mean``, ``linear``, ``ridge``, ``xgboost`` (val-tuned gradient-boosted trees;
pick the HP set with --xgb-features).
"""

from __future__ import annotations

import torch

from rt.config import Config
from rt.recipes import get_tasks
from rt.rel2tab.config import Rel2TabModelConfig

EMBEDDING_MODEL = "all-MiniLM-L12-v2"
D_TEXT = 384
RT_DIMS = dict(num_blocks=12, d_model=512, num_heads=8, d_ff=2048)


def make_featurizer(name: str, rt_ckpt: str | None, pre_dir: str, recipe: str):
    from rt.rel2tab.featurizers import (
        EntityFeaturizerConfig,
        GlobalFeaturizerConfig,
        RTFeaturizerConfig,
    )

    if name == "global":
        return GlobalFeaturizerConfig()
    if name == "entity":
        return EntityFeaturizerConfig()
    if name == "rt":
        return RTFeaturizerConfig(
            embedding_model=EMBEDDING_MODEL, d_text=D_TEXT, compile=False,
            materialize_attn_masks=True, load_ckpt_path=rt_ckpt, ctx_size=256, bfs_width=32,
            eval_recipe=recipe, pre_dir=pre_dir, shuffle_seed=0, context_seed=0,
            vector_db_path=None, **RT_DIMS,
        )
    raise ValueError(f"unknown featurizer {name!r}")


def make_predictor(name: str, alpha_clf: float, alpha_reg: float, xgb_features: str):
    from rt.rel2tab.predictors import (
        LinearPredictorConfig,
        MeanPredictorConfig,
        RidgePredictorConfig,
    )

    if name == "mean":
        return MeanPredictorConfig()
    if name == "linear":
        return LinearPredictorConfig()
    if name == "ridge":
        return RidgePredictorConfig(alpha_clf=alpha_clf, alpha_reg=alpha_reg)
    if name == "xgboost":
        # Global val-tuned HP set (shared across tasks within each task type).
        # XGB_TUNED_JSON overrides the baked-in winners; see xgboost_tuned.py.
        from rt.rel2tab.predictors.xgboost_tuned import tuned_xgboost_config

        return tuned_xgboost_config(xgb_features)
    raise ValueError(f"unknown predictor {name!r}")


def main(cfg: Config) -> None:
    ev_cfg = cfg.eval
    model_cfg = cfg.model
    assert isinstance(model_cfg, Rel2TabModelConfig), "model must be a Rel2TabModelConfig"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_cfg.build(device)
    print(f"baseline: {type(model_cfg.featurizer).__name__} + "
          f"{type(model_cfg.predictor).__name__} on {device}")

    tasks = get_tasks(ev_cfg.recipe, ev_cfg.pre_dir)
    if ev_cfg.task_type != "both":
        tasks = [t for t in tasks if t.task_type == ev_cfg.task_type]
    if not tasks:
        raise SystemExit(f"no tasks found in {ev_cfg.pre_dir}")

    from rt.eval_utils import build_evaluator, run_and_report

    ev = build_evaluator(
        tasks, ev_cfg.pre_dir, embedding_model=model_cfg.embedding_model, d_text=model_cfg.d_text, device=device,
        ctx_size=ev_cfg.ctx_sizes[0], local_ctx_size=ev_cfg.local_ctx_size, bfs_width=ev_cfg.bfs_width,
        num_walks=ev_cfg.num_walks, items_per_task=ev_cfg.items_per_task, num_workers=ev_cfg.num_workers,
    )
    run_and_report(model, tasks, ev_cfg.pre_dir, ctx_size=ev_cfg.ctx_sizes[0],
                   reg_metric=ev_cfg.reg_metric, out_dir=ev_cfg.out_dir, no_csv=not ev_cfg.write_csv,
                   evaluator=ev, embedding_model=model_cfg.embedding_model)

