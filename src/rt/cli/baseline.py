import tyro

import rt.config
from rt.baseline import EMBEDDING_MODEL, D_TEXT, main
from rt.config import Config, EvalConfig, EvalOnlyConfig, LoggerConfig
from rt.rel2tab.config import Rel2TabModelConfig
from rt.rel2tab.featurizers import EntityFeaturizerConfig
from rt.rel2tab.predictors import RidgePredictorConfig

# Config.model's annotation references Rel2TabModelConfig only under
# TYPE_CHECKING; make it resolvable at runtime for tyro.
rt.config.Rel2TabModelConfig = Rel2TabModelConfig


def default_config() -> Config:
    return Config(
        logger=LoggerConfig(project="rt", wandb_run_name=None, wandb_disabled=True),
        # was --featurizer entity --predictor ridge --alpha-clf 10 --alpha-reg 100;
        # pick other featurizers/predictors via tyro subcommands on model.*
        model=Rel2TabModelConfig(
            featurizer=EntityFeaturizerConfig(),
            predictor=RidgePredictorConfig(alpha_clf=10.0, alpha_reg=100.0),
            featurize_batch_size=4096,
            embedding_model=EMBEDDING_MODEL,
            d_text=D_TEXT,
        ),
        train=EvalOnlyConfig(),
        eval=EvalConfig(
            recipe="relbench_eval_test",
            pre_dir=tyro.MISSING,  # was --pre-dir (required)
            tokens_per_gpu=2**18,
            num_workers=2,
            prefetch_factor=2,
            local_ctx_size=256,
            bfs_width=32,
            num_walks=10_000,
            walk_length=20,
            prefer_latest=True,
            freq=None,
            items_per_task=10_000_000,
            ctx_sizes=[8192],  # was --ctx-size; only the first entry is used
            bool_as_num=False,
            skip_text_cols=False,
            mmap_populate=False,
            balance_labels=False,
            ablate_schema_semantics=False,
            reg_metric="mae",
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            mode="simple",
            tasks=None,
            grid=["256,32", "512,64"],
            ensemble_size=4,
            out_dir="baseline_out",
            write_csv=True,
            task_type="both",
        ),
    )


if __name__ == "__main__":
    main(tyro.cli(Config, default=default_config()))
