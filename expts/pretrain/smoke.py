"""Smoke-test the pretraining launch path on rel-f1 for 20 steps.

    pixi run python -m expts.pretrain.smoke

Pass: `time_to_first_step` in the log within about a minute of the ranks
starting, then `saved: best_clf` and a clean exit. Silence past that is a hang:
do not start a multi-node run on those nodes.

Checkpoints go to `~/tmp`; nothing is logged to wandb.
"""

import dataclasses
from pathlib import Path

from roach.slurm.clusters import ilc

from roach.slurm import submit

tasks = [("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3")]
pre_dir = "~/scratch/share/stanford-star/relbench-preprocessed"

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
        load_ckpt_path="~/scratch/share/stanford-star/rt-plurel/classification",
        db_task_list=tasks,
        train_splits=["train"],
        pre_dir=pre_dir,
        tokens_per_gpu=2**15,
        num_workers=4,
        prefetch_factor=2,
        ctx_size_list=[512],
        local_ctx_size_list=[256],
        bfs_width_list=[8],
        prefer_latest_list=[False],
        num_walks=1_000,
        walk_length=20,
        mask_prob_max=0.5,
        items_per_task=1_000,
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
        mmap_populate=False,
        timeout_per_item=10.0,
        eval_freq=20,
        keep_all_ckpts=False,
        vector_db_path=None,
        db_cutoff="test",
        resume_save_mins=0.0,
        eval_splits=["val"],
        eval_db_task_list=tasks,
        eval_pre_dir=pre_dir,
        eval_tokens_per_gpu=2**15,
        eval_num_workers=1,
        eval_prefetch_factor=2,
        eval_num_walks=1_000,
        eval_walk_length=20,
        eval_items_per_task=64,
        eval_ctx_size_list=[512],
        eval_mmap_populate=False,
        eval_shuffle_seed=0,
        eval_context_seed=0,
        eval_ensemble_size=1,
        eval_vector_db_path=None,
        eval_lcs_bw_pl_grid=[(256, 32, True)],
        targets={},
        project="2026-08-07-pretrain-smoke",
        entity="rtv2",
        run_name="smoke",
        wandb_disabled=True,
        out_root="~/tmp/pretrain-smoke/ckpts",
    ),
    resources=dataclasses.replace(
        ilc.AMPERE_LO,
        nodes=2,
        qos="il-lo",
        time="0-01:00:00",
        exclusive=True,
        cpus_per_task=16,
        nodelist="ampere1,ampere2,ampere3,ampere4,ampere5,ampere6,ampere7,ampere8,ampere9",
    ),
    # resources=dataclasses.replace(
    #     ilc.BLACKWELL,
    #     gpus="b200:1",
    #     qos="il-interactive",
    #     time="0-01:00:00",
    #     cpus_per_task=4,
    #     mem="40000M",
    # ),
    name="pretrain-smoke",
    run_id=None,
    repo_root=str(Path(__file__).resolve().parents[2]),
    cluster=ilc.ILC,
    job_env="expts/job_env.sh",
    log_root="~/scratch/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain/smoke",
    clone_root="~/roach_clones",
    secrets_dir="~/scratch/.secrets",
)
