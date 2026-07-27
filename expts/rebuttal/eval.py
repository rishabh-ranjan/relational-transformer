"""Evaluate an RT checkpoint on the RelBench tasks.

Loads --model.load-ckpt-path (local dir/file or Hub repo such as
stanford-star/rt-j/classification), evaluates every RelBench task of the
checkpoint's kind (clf/reg) via RelBench's own leaderboard evaluator, and
writes a valid RelBench submission directory to
<logger.out-root>/<entity>/<project>/<id>/eval_out.

Runs single-process on one GPU by default. Launch under torchrun for
multi-GPU eval (``torchrun --nproc-per-node=8 -m rt.cli.eval ...``): items are
sharded across ranks and gathered back on rank 0, which does the scoring and
writes the submission CSVs.
"""

import tyro

from rt.config import (
    Config,
    EvalConfig,
    LoggerConfig,
    ModelConfig,
    timestamp,
)
from rt.eval import main


def default_config() -> Config:
    return Config(
        logger=LoggerConfig(
            project="2026-07-26_eval",
            entity="rtv2",
            id=timestamp(),
            run_name=None,
            wandb_disabled=True,
            out_root="/dfs/user/ranjanr/eval_out",
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
            db_task_list="/dfs/user/ranjanr/pre/relbench-preprocessed/db-task-lists/forecast.json",
            pre_dir="/dfs/user/ranjanr/pre/relbench-preprocessed",
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
        ),
    )


if __name__ == "__main__":
    main(tyro.cli(Config, default=default_config(), description=__doc__))
