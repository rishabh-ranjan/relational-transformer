import tyro

import rt.config
from rt.config import Config, EvalConfig, EvalOnlyConfig, LoggerConfig, ModelConfig
from rt.eval import main
from rt.rel2tab.config import Rel2TabModelConfig

# Config.model's annotation references Rel2TabModelConfig only under
# TYPE_CHECKING; make it resolvable at runtime for tyro.
rt.config.Rel2TabModelConfig = Rel2TabModelConfig


def default_config() -> Config:
    return Config(
        profile=False,
        logger=LoggerConfig(project="rt", wandb_run_name=None, wandb_disabled=True),
        model=ModelConfig(
            embedding_model="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=False,
            materialize_attn_masks=False,
            load_ckpt_path=tyro.MISSING,  # was --checkpoint (required)
        ),
        train=EvalOnlyConfig(),
        eval=EvalConfig(
            recipe="relbench_eval_test",
            pre_dir=tyro.MISSING,  # was --pre-dir (required)
            tokens_per_gpu=2**18,  # 2**19 overflows RT-J eval kernel @ctx=8192
            num_workers=2,
            prefetch_factor=2,
            local_ctx_size=256,
            bfs_width=32,
            num_walks=10_000,
            walk_length=20,
            prefer_latest=True,
            freq=None,
            pow2=False,
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
            out_dir="eval_out",
            write_csv=True,
            task_type="both",
        ),
    )


if __name__ == "__main__":
    main(tyro.cli(Config, default=default_config()))
