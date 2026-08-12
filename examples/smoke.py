"""A pretraining run small enough to finish in a minute on one GPU.

Not a benchmark: it exists so that a change to the training path fails here
first, in seconds, instead of an hour into a real run. Every argument the real
runs take is passed here too -- that is the point, since a missing one is a
submit-time error either way.

A GPU is required. The model attends with ``flex_attention``, which has no CPU
backward ("FlexAttention does not support backward on CPU"), so there is no
CPU-only training path to fall back on; ``tests/test_smoke.py`` skips itself
when there is no CUDA device, and the same function runs on a GPU through
roach.slurm.

    python examples/smoke.py                      # from a checkout, on a GPU
    from examples.smoke import smoke; smoke(...)  # or submitted with roach.slurm
"""

from rt.train import main as train
from roach.slurm import timestamp


def smoke(
    pre_dir: str, out_root: str, run_id: str, total_steps: int, compile: bool
) -> None:
    """Tiny model, tiny context, two databases' worth of tasks, a few steps."""
    train(
        # model: small enough to step on a CPU, real enough to exercise the code
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=1,
        d_model=64,
        num_heads=2,
        d_ff=128,
        compile=compile,
        materialize_attn_masks=True,
        loss_fn="huber",
        load_ckpt_path=None,
        # data + optimization
        db_task_list=[("rel-f1", "driver-dnf")],
        train_splits=["train"],
        pre_dir=pre_dir,
        tokens_per_gpu=256,
        num_workers=0,
        prefetch_factor=None,
        ctx_size_list=[128],
        local_ctx_size_list=[64],
        bfs_width_list=[8],
        prefer_latest_list=[True],
        num_walks=100,
        walk_length=4,
        mask_prob_max=0.5,
        items_per_task=8,
        delta_finetune=False,
        optimizer="muon",
        lr=1e-3,
        wd=0.0,
        lr_warmup_steps=1,
        lr_decay_steps=0,
        grad_norm_max=1.0,
        total_bs=2,
        total_steps=total_steps,
        early_stop_after_steps=None,
        swa_momentum=0.9,
        seed=0,
        mmap_populate=False,
        timeout_per_item=60.0,
        eval_freq=None,
        keep_all_ckpts=False,
        vector_db_path=None,
        db_cutoff=None,
        resume_save_mins=60.0,
        # in-loop validation: the final eval always runs, so keep it minimal
        eval_splits=["val"],
        eval_db_task_list=[("rel-f1", "driver-dnf")],
        eval_pre_dir=pre_dir,
        eval_tokens_per_gpu=256,
        eval_num_workers=0,
        eval_prefetch_factor=None,
        eval_num_walks=100,
        eval_walk_length=4,
        eval_items_per_task=4,
        eval_ctx_size_list=[128],
        eval_mmap_populate=False,
        eval_shuffle_seed=0,
        eval_context_seed=0,
        eval_ensemble_size=1,
        eval_vector_db_path=None,
        eval_lcs_bw_pl_grid=[(64, 8, True)],
        # logging
        run_id=run_id,
        targets={},
        project="smoke",
        entity=None,
        run_name=None,
        wandb_disabled=True,
        out_root=out_root,
    )


if __name__ == "__main__":
    smoke(
        pre_dir="/dfs/user/ranjanr/pre/relbench-preprocessed",
        out_root="/tmp/rt-smoke",
        run_id=timestamp(),
        total_steps=3,
        compile=False,
    )
