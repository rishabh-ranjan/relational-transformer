"""Pretrain a Relational Transformer with Muon+AdamW under DDP.

Streams training items from the preprocessed mixture at --train.pre-dir,
periodically evaluates on --eval.pre-dir, and writes checkpoints plus a
preemption-safe resume.pt to --logger.out-root (resume is automatic and
GPU-count flexible). Launch with torchrun; see docs/train.md.
"""

import tyro

from rt.config import (
    Config,
    EvalConfig,
    LoggerConfig,
    ModelConfig,
    TrainConfig,
    timestamp,
)
from rt.train import main



def default_config() -> Config:
    return Config(
        logger=LoggerConfig(
            project="rt-verify",
            entity=None,
            id=timestamp(),
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
            compile=True,
            materialize_attn_masks=True,
            load_ckpt_path=None,
        ),
        train=TrainConfig(
            db_task_list="data/the-join-preprocessed/db-task-lists/forecast.json",
            pre_dir="data/the-join-preprocessed",
            tokens_per_gpu=2**17,
            num_workers=16,
            prefetch_factor=2,
            ctx_size_list=[1024, 2048, 4096, 8192],
            local_ctx_size_list=[512, 1024, 2048],
            bfs_width_list=[16, 32, 64, 128],
            prefer_latest_list=[True],
            num_walks=10_000,
            walk_length=20,
            mask_prob_max=0.0,
            items_per_task=100_000,
            lr=5e-4,
            wd=0.1,
            warmup_steps=2000,
            grad_norm_max=1.0,
            total_bs=1024,
            total_steps=100_001,
            swa_momentum=0.9995,
            seed=0,
            mmap_populate=True,
            timeout_per_item=10.0,
            eval_freq=2000,
            vector_db_path=None,
            resume_save_mins=20.0,
        ),
        eval=EvalConfig(
            splits=["val"],
            db_task_list="data/relbench-preprocessed/db-task-lists/forecast.json",
            pre_dir="data/relbench-preprocessed",
            tokens_per_gpu=2**17,
            num_workers=1,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            items_per_task=1024,
            ctx_size_list=[4096, 8192],
            mmap_populate=True,
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            lcs_bw_pl_grid=[(256, 32, True)],
            ensemble_size=1,
            csv_out_dir=None,
        ),
    )


if __name__ == "__main__":
    main(tyro.cli(tyro.conf.AvoidSubcommands[Config], default=default_config(), description=__doc__))
