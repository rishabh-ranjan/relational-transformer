"""Evaluate a data-scaling RT checkpoint on the RelBench tasks.

Copy of expts/rebuttal/eval.py with the base config matched to the
data-scaling training runs' in-loop eval (expts/data-scaling/train.py):
test split, ctx 4096, items_per_task 1024 (tokens_per_gpu 2**18 for eval
throughput),
lcs_bw_pl_grid [(256, 32, True)], seeds 0. The submitter only overrides
--model.load-ckpt-path, --logger.id, and --eval.db-task-list.

Writes a RelBench submission directory to
<logger.out-root>/<entity>/<project>/<id>/eval_out and prints per-task +
mean metrics.
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
            project="2026-07-26_data-scaling-eval",
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
            load_ckpt_path=None,
        ),
        train=None,
        eval=EvalConfig(
            splits=["test"],
            db_task_list="/dfs/user/ranjanr/pre/relbench-preprocessed/db-task-lists/forecast.json",
            pre_dir="/dfs/user/ranjanr/pre/relbench-preprocessed",
            tokens_per_gpu=2**18,
            num_workers=2,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            items_per_task=1024,
            ctx_size_list=[4096],
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
