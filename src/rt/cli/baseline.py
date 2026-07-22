"""Unified rel2tab baseline runner.

    python -m rt.cli.baseline --featurizer entity --predictor ridge \\
        --eval.pre-dir stanford-star/relbench-preprocessed

The (featurizer, predictor) pair is selected by name here and expanded into the
full nested :class:`rt.config.Config` (all defaults live in this file); the
shared eval path is :func:`rt.eval.main`.
"""

from dataclasses import dataclass
from typing import Literal

import tyro

from rt.config import Config, EvalConfig, LoggerConfig
from rt.eval import main
from rt.rel2tab.config import Rel2TabModelConfig


@dataclass
class BaselineConfig:
    featurizer: Literal["global", "entity", "rt"]
    """global = all in-context rows, entity = same-entity rows, rt = RT embeddings"""
    predictor: Literal["mean", "linear", "ridge", "xgboost"]
    """xgboost = val-tuned gradient-boosted trees; pick the HP set with xgb_features"""
    rt_ckpt: str | None
    """checkpoint for --featurizer rt"""
    alpha_clf: float
    alpha_reg: float
    xgb_features: Literal["sql_features", "rdblearn_features"]
    """HP set for --predictor xgboost"""
    eval: EvalConfig


def default_config() -> BaselineConfig:
    return BaselineConfig(
        featurizer="entity",
        predictor="ridge",
        rt_ckpt=None,
        alpha_clf=10.0,
        alpha_reg=100.0,
        xgb_features="sql_features",
        eval=EvalConfig(
            recipe="relbench_eval_test",
            pre_dir="stanford-star/relbench-preprocessed",
            tokens_per_gpu=2**18,
            num_workers=2,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            freq=None,
            items_per_task=10_000_000,
            ctx_sizes=[8192],
            bool_as_num=True,
            skip_text_cols=False,
            mmap_populate=True,
            balance_labels=False,
            ablate_schema_semantics=False,
            reg_metric="mae",
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            tasks=None,
            lcs_bw_pl_grid=[(256, 32, True)],
            ensemble_size=1,
            out_dir="baseline_out",
            write_csv=True,
            task_type="both",
        ),
    )


def make_featurizer(b: BaselineConfig):
    from rt.rel2tab.featurizers import (
        EntityFeaturizerConfig,
        GlobalFeaturizerConfig,
        RTFeaturizerConfig,
    )

    if b.featurizer == "global":
        return GlobalFeaturizerConfig()
    if b.featurizer == "entity":
        return EntityFeaturizerConfig()
    if b.featurizer == "rt":
        return RTFeaturizerConfig(
            embedding_model="all-MiniLM-L12-v2", d_text=384,
            num_blocks=12, d_model=512, num_heads=8, d_ff=2048,
            compile=False, materialize_attn_masks=True,
            load_ckpt_path=b.rt_ckpt, ctx_size=256, bfs_width=32,
            eval_recipe=b.eval.recipe, pre_dir=b.eval.pre_dir,
            shuffle_seed=0, context_seed=0, vector_db_path=None,
        )
    raise ValueError(f"unknown featurizer {b.featurizer!r}")


def make_predictor(b: BaselineConfig):
    from rt.rel2tab.predictors import (
        LinearPredictorConfig,
        MeanPredictorConfig,
        RidgePredictorConfig,
    )

    if b.predictor == "mean":
        return MeanPredictorConfig()
    if b.predictor == "linear":
        return LinearPredictorConfig()
    if b.predictor == "ridge":
        return RidgePredictorConfig(alpha_clf=b.alpha_clf, alpha_reg=b.alpha_reg)
    if b.predictor == "xgboost":
        # Global val-tuned HP set (shared across tasks within each task type).
        # XGB_TUNED_JSON overrides the baked-in winners; see xgboost_tuned.py.
        from rt.rel2tab.predictors.xgboost_tuned import tuned_xgboost_config

        return tuned_xgboost_config(b.xgb_features)
    raise ValueError(f"unknown predictor {b.predictor!r}")


def build_config(b: BaselineConfig) -> Config:
    return Config(
        logger=LoggerConfig(
            project="rt",
            wandb_run_name=f"{b.featurizer}_{b.predictor}",
            wandb_disabled=True,
        ),
        model=Rel2TabModelConfig(
            featurizer=make_featurizer(b),
            predictor=make_predictor(b),
            featurize_batch_size=4096,
            embedding_model="all-MiniLM-L12-v2",
            d_text=384,
        ),
        train=None,
        eval=b.eval,
    )


if __name__ == "__main__":
    main(build_config(tyro.cli(BaselineConfig, default=default_config(), description=__doc__)))
