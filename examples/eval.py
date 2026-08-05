"""Evaluate a checkpoint on RelBench and write a submission.

No CLI: copy, edit, run. The defaults below are the published RT-J evaluation
setup; ``rt.eval._eval`` requires every argument, so nothing is implicit.

    pixi run python examples/eval.py

The checkpoint may be a local path or a Hub spec (checkpoints are still fetched
on demand); the data is a local directory (see docs/downloads.md).
"""

from __future__ import annotations

from rt.eval import main
from roach.slurm import timestamp


def evaluate(pre_dir: str, out_root: str, checkpoint: str, run_id: str) -> None:
    main(
        # which checkpoint, and the dims it was trained with
        load_ckpt_path=checkpoint,
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        # what to evaluate
        splits=["test"],
        db_task_list=f"{pre_dir}/db-task-lists/forecast.json",
        pre_dir=pre_dir,
        tokens_per_gpu=2**18,  # 2**19 overflows the RT-J eval kernel at ctx=8192
        num_workers=2,
        prefetch_factor=2,
        num_walks=10_000,
        walk_length=20,
        items_per_task=10_000_000,
        ctx_size_list=[8192],
        mmap_populate=True,
        shuffle_seed=0,
        context_seed=0,
        vector_db_path=None,
        lcs_bw_pl_grid=[(256, 32, True)],
        ensemble_size=1,
        # where the submission CSVs land
        run_id=run_id,
        project="rt-eval",
        entity=None,
        out_root=out_root,
        wandb_disabled=True,
    )


if __name__ == "__main__":
    evaluate(
        pre_dir="data/relbench-preprocessed",
        out_root="~/ckpts",
        checkpoint="stanford-star/rt-j/classification",
        run_id=timestamp(),
    )
