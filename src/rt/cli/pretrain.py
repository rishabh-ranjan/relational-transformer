"""Pretrain a Relational Transformer with Muon+AdamW under DDP.

Streams training items from the preprocessed mixture at --train.pre-dir,
periodically evaluates on --eval.pre-dir, and writes checkpoints plus a
preemption-safe resume.pt to --train.out-dir (resume is automatic and
GPU-count flexible). Launch with torchrun; see docs/pretrain.md.
"""

import tyro

from rt.config import Config, EvalConfig, LoggerConfig, ModelConfig, TrainConfig
from rt.pretrain import main



def default_config() -> Config:
    return Config(
        logger=LoggerConfig(
            project="rt-verify",
            wandb_run_name=None,
            wandb_disabled=True,
        ),
        model=ModelConfig(
            embedding_model="all-MiniLM-L12-v2",
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
            pre_dir="stanford-star/the-join-preprocessed",
            tokens_per_gpu=2**17,
            num_workers=16,
            prefetch_factor=2,
            ctx_sizes=[1024, 2048, 4096, 8192],
            local_ctx_sizes=[512, 1024, 2048],
            bfs_widths=[16, 32, 64, 128],
            num_walks=10_000,
            walk_length=20,
            prefer_latest=[True],
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
            bool_as_num=True,
            skip_text_cols=False,
            mmap_populate=True,
            balance_labels=[False],
            timeout_per_item=10.0,
            vector_db_path=None,
            out_dir="~/ckpts/run1",
            resume_save_mins=20.0,
            include_dbs_file=None,
        ),
        eval=EvalConfig(
            splits=["val"],
            pre_dir="stanford-star/relbench-preprocessed",
            tokens_per_gpu=2**17,
            num_workers=1,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            freq=2000,
            items_per_task=1024,
            ctx_sizes=[4096, 8192],
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
            out_dir="",
            write_csv=False,
            task_type="both",
        ),
    )


if __name__ == "__main__":
    main(tyro.cli(tyro.conf.AvoidSubcommands[Config], default=default_config(), description=__doc__))
