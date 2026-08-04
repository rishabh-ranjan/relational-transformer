"""Pretrain a Relational Transformer on the Join -- the released RT-J recipe.

There is no CLI: copy this file, change what you want, run it. The arguments
here are the ones the released runs used; ``rt.train.main`` requires all of
them, so nothing is hidden in a default you did not choose.

    pixi run python examples/train.py                      # one process
    srun --ntasks-per-node=8 pixi run python examples/train.py   # DDP, one rank per GPU

Data is a local directory -- download it first (see docs/downloads.md). For
running this on a cluster without writing any slurm boilerplate, see
rt.slurm.submit and expts/data_scaling/submit.py.
"""

from __future__ import annotations

from rt.slurm import timestamp
from rt.train import main


def train(pre_dir: str, eval_pre_dir: str, out_root: str, run_id: str) -> None:
    main(
        # model (RT-J dims)
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        compile=True,
        materialize_attn_masks=True,
        load_ckpt_path=None,
        # data + optimization
        db_task_list=f"{pre_dir}/db-task-lists/rt-j.json",
        pre_dir=pre_dir,
        tokens_per_gpu=2**17,
        num_workers=16,
        prefetch_factor=2,
        ctx_size_list=[1024, 2048, 4096, 8192],
        local_ctx_size_list=[256, 512, 1024, 2048, 4096, 8192],
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
        total_bs=1024,
        total_steps=100_001,
        swa_momentum=0.9995,
        seed=0,
        mmap_populate=True,
        timeout_per_item=10.0,
        eval_freq=2000,
        vector_db_path=None,
        resume_save_mins=20.0,
        # in-loop validation on RelBench
        eval_splits=["val"],
        eval_db_task_list=f"{eval_pre_dir}/db-task-lists/forecast.json",
        eval_pre_dir=eval_pre_dir,
        eval_tokens_per_gpu=2**17,
        eval_num_workers=1,
        eval_prefetch_factor=2,
        eval_num_walks=10_000,
        eval_walk_length=20,
        eval_items_per_task=1024,
        eval_ctx_size_list=[4096, 8192],
        eval_mmap_populate=True,
        eval_shuffle_seed=0,
        eval_context_seed=0,
        eval_vector_db_path=None,
        eval_lcs_bw_pl_grid=[(256, 32, True)],
        # logging: the run id names the output directory and the wandb run, and
        # reusing it is how a preempted run resumes
        run_id=run_id,
        project="rt-train",
        entity=None,
        run_name=None,
        wandb_disabled=True,
        out_root=out_root,
    )


if __name__ == "__main__":
    train(
        pre_dir="data/the-join-preprocessed",
        eval_pre_dir="data/relbench-preprocessed",
        out_root="~/ckpts",
        run_id=timestamp(),
    )
