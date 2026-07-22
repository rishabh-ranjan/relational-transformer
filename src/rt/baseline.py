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

import argparse

import torch

from rt.recipes import get_tasks

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featurizer", default="entity", choices=["global", "entity", "rt"])
    ap.add_argument("--predictor", default="ridge",
                    choices=["mean", "linear", "ridge", "xgboost"])
    ap.add_argument("--pre-dir", required=True, help="preprocessed RelBench (local or Hub)")
    ap.add_argument("--recipe", default="relbench_eval_test")
    ap.add_argument("--task-type", default="both", choices=["clf", "reg", "both"])
    ap.add_argument("--rt-ckpt", default=None, help="checkpoint for --featurizer rt")
    ap.add_argument("--out-dir", default="baseline_out")
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--local-ctx-size", type=int, default=256)
    ap.add_argument("--bfs-width", type=int, default=32)
    ap.add_argument("--num-walks", type=int, default=10_000)
    ap.add_argument("--items-per-task", type=int, default=10_000_000,
                    help="cap on items per task (default covers the full split)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--alpha-clf", type=float, default=10.0)
    ap.add_argument("--alpha-reg", type=float, default=100.0)
    ap.add_argument("--xgb-features", default="sql_features",
                    choices=["sql_features", "rdblearn_features"],
                    help="which val-tuned XGBoost HP set to use (--predictor xgboost)")
    ap.add_argument("--reg-metric", default="mae", choices=["mae", "r2"])
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from rt.rel2tab.config import Rel2TabModelConfig

    cfg = Rel2TabModelConfig(
        featurizer=make_featurizer(args.featurizer, args.rt_ckpt, args.pre_dir, args.recipe),
        predictor=make_predictor(args.predictor, args.alpha_clf, args.alpha_reg,
                                 args.xgb_features),
        featurize_batch_size=4096, embedding_model=EMBEDDING_MODEL, d_text=D_TEXT,
    )
    model = cfg.build(device)
    print(f"baseline: {args.featurizer} + {args.predictor} on {device}")

    tasks = get_tasks(args.recipe, args.pre_dir)
    if args.task_type != "both":
        tasks = [t for t in tasks if t.task_type == args.task_type]
    if not tasks:
        raise SystemExit(f"no tasks found in {args.pre_dir}")

    from rt.eval_utils import build_evaluator, run_and_report

    ev = build_evaluator(
        tasks, args.pre_dir, embedding_model=EMBEDDING_MODEL, d_text=D_TEXT, device=device,
        ctx_size=args.ctx_size, local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
        num_walks=args.num_walks, items_per_task=args.items_per_task, num_workers=args.num_workers,
    )
    run_and_report(model, tasks, args.pre_dir, ctx_size=args.ctx_size,
                   reg_metric=args.reg_metric, out_dir=args.out_dir, no_csv=args.no_csv,
                   evaluator=ev, embedding_model=EMBEDDING_MODEL)


if __name__ == "__main__":
    main()
