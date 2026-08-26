import dataclasses
from pathlib import Path

from roach.slurm.clusters import marlowe

from expts.repaper.config import project
from roach.slurm import submit

cluster = marlowe.MARLOWE
resources = dataclasses.replace(marlowe.H100, nodes=4)


def args():
    gpu = resources.gpus.rpartition(":")[0] or {"marlowe": "h100"}[cluster.name]
    return dict(
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        compile=True,
        materialize_attn_masks=True,
        loss_fn="huber",
        load_ckpt_path=None,
        db_task_list="expts/pretrain/all_5gb_cutoff.json",
        train_splits=["train"],
        pre_dir="~/scratch/hf/stanford-star/the-join-lite-preprocessed",
        stage_dir={"ilc": None, "marlowe": "$TMPDIR/hf"}[cluster.name],
        tokens_per_gpu={"a100": 2**17, "h100": 2**16, "b200": 2**18}[gpu],
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
        delta_finetune=False,
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
        eval_db_task_list="expts/pretrain/eval-tasks.json",
        eval_pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
        eval_tokens_per_gpu=2**17,
        eval_num_workers=3,
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
        targets={
            "swa/nmae/val/mean": 32.4110,
            "swa/auroc/val/mean": 73.3929,
            "nmae/val/mean": 32.9141,
            "auroc/val/mean": 72.8895,
        },
        project=project("pretrain"),
        entity="rtv2",
        run_name="base",
        wandb_disabled=False,
        out_root="~/scratch/relational-transformer/pretrain",
    )


if __name__ == "__main__":
    submit(
        "rt.train:main",
        args=args(),
        resources=resources,
        name="pretrain",
        run_id=None,
        inside=447124,
        repo_root=str(Path(__file__).resolve().parents[2]),
        cluster=cluster,
        job_env="expts/job_env.sh",
        log_root="~/scratch/relational-transformer/pretrain/slurm-logs",
        clone_root="~/roach_clones",
        secrets_dir="~/scratch/.secrets",
    )
