"""Unified rel2tab baseline runner.

    python -m rt.cli.baseline --featurizer entity --predictor ridge \\
        --pre-dir stanford-star/relbench-preprocessed

The (featurizer, predictor) pair is selected by name here and expanded into the
full nested :class:`rt.config.Config` (all defaults live in this file); the
shared eval path is :func:`rt.baseline.main`.
"""

from dataclasses import dataclass
from typing import Literal

import tyro

from rt.baseline import main
from rt.config import Config, EvalConfig, EvalOnlyConfig, LoggerConfig
from rt.rel2tab.config import Rel2TabModelConfig


@dataclass
class BaselineArgs:
    pre_dir: str
    """preprocessed RelBench (local or Hub)"""
    featurizer: Literal["global", "entity", "rt"]
    """global = all in-context rows, entity = same-entity rows, rt = RT embeddings"""
    predictor: Literal["mean", "linear", "ridge", "xgboost"]
    """xgboost = val-tuned gradient-boosted trees; pick the HP set with xgb_features"""
    rt_ckpt: str | None
    """checkpoint for --featurizer rt"""
    recipe: str
    task_type: Literal["clf", "reg", "both"]
    out_dir: str
    ctx_size: int
    local_ctx_size: int
    bfs_width: int
    num_walks: int
    items_per_task: int
    """cap on eval items per task"""
    num_workers: int
    alpha_clf: float
    alpha_reg: float
    xgb_features: Literal["sql_features", "rdblearn_features"]
    """HP set for --predictor xgboost"""
    reg_metric: Literal["mae", "r2"]
    write_csv: bool


def default_args() -> BaselineArgs:
    return BaselineArgs(
        pre_dir="stanford-star/relbench-preprocessed",
        featurizer="entity",
        predictor="ridge",
        rt_ckpt=None,
        recipe="relbench_eval_test",
        task_type="both",
        out_dir="baseline_out",
        ctx_size=8192,
        local_ctx_size=256,
        bfs_width=32,
        num_walks=10_000,
        items_per_task=10_000_000,
        num_workers=2,
        alpha_clf=10.0,
        alpha_reg=100.0,
        xgb_features="sql_features",
        reg_metric="mae",
        write_csv=True,
    )


def make_featurizer(a: BaselineArgs):
    from rt.rel2tab.featurizers import (
        EntityFeaturizerConfig,
        GlobalFeaturizerConfig,
        RTFeaturizerConfig,
    )

    if a.featurizer == "global":
        return GlobalFeaturizerConfig()
    if a.featurizer == "entity":
        return EntityFeaturizerConfig()
    if a.featurizer == "rt":
        return RTFeaturizerConfig(
            embedding_model="all-MiniLM-L12-v2", d_text=384,
            num_blocks=12, d_model=512, num_heads=8, d_ff=2048,
            compile=False, materialize_attn_masks=True,
            load_ckpt_path=a.rt_ckpt, ctx_size=256, bfs_width=32,
            eval_recipe=a.recipe, pre_dir=a.pre_dir,
            shuffle_seed=0, context_seed=0, vector_db_path=None,
        )
    raise ValueError(f"unknown featurizer {a.featurizer!r}")


def make_predictor(a: BaselineArgs):
    from rt.rel2tab.predictors import (
        LinearPredictorConfig,
        MeanPredictorConfig,
        RidgePredictorConfig,
    )

    if a.predictor == "mean":
        return MeanPredictorConfig()
    if a.predictor == "linear":
        return LinearPredictorConfig()
    if a.predictor == "ridge":
        return RidgePredictorConfig(alpha_clf=a.alpha_clf, alpha_reg=a.alpha_reg)
    if a.predictor == "xgboost":
        # Global val-tuned HP set (shared across tasks within each task type).
        # XGB_TUNED_JSON overrides the baked-in winners; see xgboost_tuned.py.
        from rt.rel2tab.predictors.xgboost_tuned import tuned_xgboost_config

        return tuned_xgboost_config(a.xgb_features)
    raise ValueError(f"unknown predictor {a.predictor!r}")


def build_config(a: BaselineArgs) -> Config:
    return Config(
        logger=LoggerConfig(
            project="rt",
            wandb_run_name=f"{a.featurizer}_{a.predictor}",
            wandb_disabled=True,
        ),
        model=Rel2TabModelConfig(
            featurizer=make_featurizer(a),
            predictor=make_predictor(a),
            featurize_batch_size=4096,
            embedding_model="all-MiniLM-L12-v2",
            d_text=384,
        ),
        train=EvalOnlyConfig(),
        eval=EvalConfig(
            recipe=a.recipe,
            pre_dir=a.pre_dir,
            tokens_per_gpu=2**18,
            num_workers=a.num_workers,
            prefetch_factor=2,
            local_ctx_size=a.local_ctx_size,
            bfs_width=a.bfs_width,
            num_walks=a.num_walks,
            walk_length=20,
            prefer_latest=True,
            freq=None,
            items_per_task=a.items_per_task,
            ctx_sizes=[a.ctx_size],
            bool_as_num=False,
            skip_text_cols=False,
            mmap_populate=False,
            balance_labels=False,
            ablate_schema_semantics=False,
            reg_metric=a.reg_metric,
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            mode="simple",
            tasks=None,
            grid=["256,32", "512,64"],
            ensemble_size=4,
            out_dir=a.out_dir,
            write_csv=a.write_csv,
            task_type=a.task_type,
        ),
    )


if __name__ == "__main__":
    main(build_config(tyro.cli(BaselineArgs, default=default_args())))
