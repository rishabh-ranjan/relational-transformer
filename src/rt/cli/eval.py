"""Evaluate an RT checkpoint on the RelBench tasks.

Loads --model.load-ckpt-path (local dir/file or Hub repo such as
stanford-star/rt-j/classification), evaluates every RelBench task of the
checkpoint's kind (clf/reg) via RelBench's own leaderboard evaluator, and
writes --eval.csv-out-dir as a valid RelBench submission directory. Single-process,
one GPU.
"""

import tyro

from rt.config import (
    Config,
    EvalConfig,
    LoggerConfig,
    ModelConfig,
    default_id,
)
from rt.eval import main



def default_config() -> Config:
    return Config(
        logger=LoggerConfig(
            project="rt",
            entity=None,
            id=default_id(),
            run_name=None,
            wandb_disabled=True,
            out_root="~/ckpts",
        ),
        model=ModelConfig(
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=False,
            materialize_attn_masks=False,
            load_ckpt_path="stanford-star/rt-j/classification",
        ),
        train=None,
        eval=EvalConfig(
            splits=["test"],
            db_task_list="stanford-star/relbench/db-task-lists/forecast.json",
            pre_dir="stanford-star/relbench-preprocessed",
            tokens_per_gpu=2**18,  # 2**19 overflows RT-J eval kernel @ctx=8192
            num_workers=2,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            items_per_task=10_000_000,
            ctx_size_list=[8192],
            mmap_populate=True,
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            lcs_bw_pl_grid=[(256, 32, True)],
            ensemble_size=1,
            csv_out_dir="eval_out",
        ),
    )


if __name__ == "__main__":
    main(tyro.cli(Config, default=default_config(), description=__doc__))
