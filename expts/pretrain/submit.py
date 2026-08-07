"""Pretrain a Relational Transformer on the Join, and submit that one run.

    pixi run python expts/pretrain/submit.py

One run, not a sweep: the Join's mixture for training, the benchmark's forecast
tasks for in-loop validation. See [README.md](README.md).
"""

from roach.slurm import BLACKWELL, submit

PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/the-join-preprocessed"
EVAL_PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"


def main() -> None:
    submit(
        "rt.train:main",
        args=dict(
            # model: RT-J's dims
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=True,
            materialize_attn_masks=True,
            load_ckpt_path=None,
            # data: the Join's mixture
            db_task_list=f"{PRE_DIR}/db-task-lists/rt-j.json",
            pre_dir=PRE_DIR,
            tokens_per_gpu=2**17,
            # loader workers are processes, and the job only owns
            # `cpus_per_task` cores per task
            num_workers=16,
            prefetch_factor=2,
            ctx_size_list=[512, 1024, 2048, 4096],
            local_ctx_size_list=[256, 512, 1024, 2048, 4096],
            bfs_width_list=[16, 64, 256],
            prefer_latest_list=[False, True],
            num_walks=10_000,
            walk_length=20,
            mask_prob_max=0.5,
            items_per_task=100_000,
            # optimization
            lr=5e-4,
            wd=0.1,
            warmup_steps=2000,
            grad_norm_max=1.0,
            total_bs=512,
            total_steps=100_001,
            swa_momentum=0.9995,
            seed=0,
            mmap_populate=True,
            timeout_per_item=10.0,
            eval_freq=2000,
            keep_all_ckpts=False,
            vector_db_path=None,
            resume_save_mins=20.0,
            # in-loop validation: the benchmark's forecast tasks, val split
            eval_splits=["val"],
            eval_db_task_list=f"{EVAL_PRE_DIR}/db-task-lists/forecast.json",
            eval_pre_dir=EVAL_PRE_DIR,
            eval_tokens_per_gpu=2**17,
            eval_num_workers=1,
            eval_prefetch_factor=2,
            eval_num_walks=10_000,
            eval_walk_length=20,
            eval_items_per_task=1024,
            eval_ctx_size_list=[4096],
            eval_mmap_populate=True,
            eval_shuffle_seed=0,
            eval_context_seed=0,
            eval_vector_db_path=None,
            eval_lcs_bw_pl_grid=[(256, 32, True)],
            # logging
            targets={},
            project="2026-08-07-pretrain",
            entity="rtv2",
            run_name="rt-j",
            wandb_disabled=False,
            out_root="/dfs/user/ranjanr/ckpts",
        ),
        # 4xB200 on il-lo: preemptible, and the run checkpoints, so the low
        # priority queue costs wall clock, not work
        resources=BLACKWELL,
        name="pretrain",
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain",
        # the node's own big disk, not /tmp (the 280G root filesystem)
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
        # this run is long, and clones are swept after a week unused
        clone_ttl_days=7,
        # No setup: `pixi install` already builds the rustler extension into
        # src/rt/, and `pixi run build-sampler` on top of it rebuilds the whole
        # pyo3 stack for no different result.
    )


if __name__ == "__main__":
    main()
