"""Submit the pretraining run. See [README.md](README.md)."""

import dataclasses
import json
from pathlib import Path

from roach.slurm.clusters import ilc

from roach.slurm import submit

resources = dataclasses.replace(
    ilc.AMPERE_LO,
    nodes=1,
    qos="il",
    time="7-00:00:00",
    exclusive=True,
    cpus_per_task=16,
    nodelist="ampere1,ampere2,ampere3,ampere4,ampere5,ampere6,ampere7,ampere8,ampere9",
)
# resources = dataclasses.replace(
#     ilc.BLACKWELL, gpus="b200:2", qos="il", time="7-00:00:00", mem="750000M"
# )

tokens_per_gpu = {"a100": 2**17, "b200": 2**18}[resources.gpus.rpartition(":")[0]]

submit(
    "rt.train:main",
    args=dict(
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        compile=True,
        materialize_attn_masks=True,
        loss_fn="huber",
        load_ckpt_path="~/scratch/hf/stanford-star/rt-plurel/classification",
        db_task_list="~/scratch/hf/stanford-star/the-join-preprocessed/db-task-lists/all.json",
        train_splits=["train"],
        pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        tokens_per_gpu=tokens_per_gpu,
        num_workers=resources.cpus_per_task,
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
        lr_warmup_steps=2_000,
        lr_decay_steps=0,
        grad_norm_max=1.0,
        total_bs=1024,
        total_steps=100_001,
        early_stop_after_steps=None,
        can_select_init_model=False,
        swa_momentum=0.9995,
        seed=0,
        mmap_populate=True,
        timeout_per_item=10.0,
        eval_freq=1_000,
        keep_all_ckpts=True,
        vector_db_path=None,
        db_cutoff=None,
        resume_save_mins=20.0,
        eval_splits=["val"],
        eval_db_task_list=json.loads(
            Path(__file__).with_name("eval-tasks.json").read_text()
        ),
        eval_pre_dir="~/scratch/share/stanford-star/relbench-preprocessed",
        eval_tokens_per_gpu=tokens_per_gpu // 2,
        eval_num_workers=1,
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
        run_name="rt-j-from-rt-plurel",
        wandb_disabled=False,
        out_root="~/scratch/ckpts",
    ),
    resources=resources,
    name="pretrain",
    run_id=None,
    repo_root=str(Path(__file__).resolve().parents[2]),
    cluster=ilc.ILC,
    job_env="expts/job_env.sh",
    log_root="~/scratch/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain",
    clone_root="~/roach_clones",
    secrets_dir="~/scratch/.secrets",
    timeout_grace_secs=1800,
)
