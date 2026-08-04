"""The data-scaling arm: RT-J's recipe, with the task mixture as the variable.

One function, called with a different ``db_task_list`` per arm. Everything else
is held fixed -- including ``total_bs`` and ``total_steps``, so every arm sees
exactly the same number of items and only their diversity differs.
"""

from __future__ import annotations

from rt.train import main


def train(
    db_task_list: str,
    pre_dir: str,
    eval_pre_dir: str,
    out_root: str,
    project: str,
    run_id: str,
) -> None:
    main(
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        compile=True,
        materialize_attn_masks=True,
        load_ckpt_path=None,
        db_task_list=db_task_list,
        pre_dir=pre_dir,
        tokens_per_gpu=2**17,
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
        vector_db_path=None,
        resume_save_mins=20.0,
        eval_splits=["val"],
        eval_db_task_list=f"{eval_pre_dir}/db-task-lists/forecast.json",
        eval_pre_dir=eval_pre_dir,
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
        run_id=run_id,
        project=project,
        entity="rtv2",
        run_name=None,
        wandb_disabled=False,
        out_root=out_root,
    )
