"""Smoke-test the pretraining launch path on rel-f1 for 20 steps.

    pixi run python -m expts.pretrain.smoke

Pass: `time_to_first_step` in the log within a few minutes of the ranks
starting, then a clean exit. Silence past that is a hang: do not start a real
run on that cluster until it is understood.

Nothing is logged to wandb.
"""

import dataclasses
from pathlib import Path

from roach.slurm.clusters import marlowe

from roach.slurm import submit

# 2026-08-22: the first job through roach on Marlowe: one H100 on the free
# preempt partition, exercising the remote submit, the clone on ~/roach_clones,
# the secrets and the per-job TMPDIR. relbench-preprocessed and rt-plurel are
# on Marlowe scratch; the-join-preprocessed is still copying.
resources = dataclasses.replace(marlowe.H100_PREEMPT, gpus="1", exclusive=False)

submit(
    "rt.train:main",
    args=dict(
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        compile=False,
        materialize_attn_masks=True,
        loss_fn="huber",
        load_ckpt_path="~/scratch/hf/stanford-star/rt-plurel/classification",
        db_task_list=[("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3")],
        train_splits=["train"],
        pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
        stage_dir=None,
        tokens_per_gpu=2**16,
        num_workers=4,
        prefetch_factor=2,
        ctx_size_list=[512, 1024, 2048, 4096, 8192],
        local_ctx_size_list=[256, 512, 1024, 2048, 4096, 8192],
        bfs_width_list=[8, 16, 32, 64, 128, 256],
        prefer_latest_list=[False, True],
        num_walks=10_000,
        walk_length=20,
        mask_prob_max=0.5,
        items_per_task=100_000,
        delta_finetune=True,
        optimizer="muon",
        lr=5e-4,
        wd=0.1,
        lr_warmup_steps=10,
        lr_decay_steps=0,
        grad_norm_max=1.0,
        total_bs=64,
        total_steps=20,
        early_stop_after_steps=None,
        can_select_init_model=False,
        swa_momentum=0.9995,
        seed=0,
        mmap_populate=True,
        timeout_per_item=10.0,
        eval_freq=None,
        keep_all_ckpts=False,
        vector_db_path=None,
        db_cutoff=None,
        resume_save_mins=20.0,
        eval_splits=["val"],
        eval_db_task_list=[("rel-f1", "driver-dnf")],
        eval_pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
        eval_tokens_per_gpu=2**16,
        eval_num_workers=2,
        eval_prefetch_factor=2,
        eval_num_walks=10_000,
        eval_walk_length=20,
        eval_items_per_task=1024,
        eval_ctx_size_list=[8192],
        eval_mmap_populate=True,
        eval_shuffle_seed=0,
        eval_context_seed=0,
        eval_ensemble_size=1,
        eval_vector_db_path=None,
        eval_lcs_bw_pl_grid=[(256, 32, True)],
        targets={},
        project="2026-08-07-pretrain",
        entity="rtv2",
        run_name="smoke",
        wandb_disabled=True,
        out_root="~/scratch/relational-transformer/pretrain/smoke",
    ),
    resources=resources,
    name="pretrain-smoke",
    run_id=None,
    repo_root=str(Path(__file__).resolve().parents[2]),
    cluster=marlowe.MARLOWE,
    job_env="expts/job_env.sh",
    log_root="~/scratch/relational-transformer/pretrain/smoke/slurm-logs",
    clone_root="~/roach_clones",
    secrets_dir="~/scratch/.secrets",
)
