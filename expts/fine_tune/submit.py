"""Fine-tune a Relational Transformer on one task, and submit it.

    pixi run python expts/fine_tune/submit.py                # batch, 4xB200
    pixi run python expts/fine_tune/submit.py --interactive  # in a held allocation

One file, and every argument written where it is passed: `rt.train:main` is the
target, so there is no per-experiment wrapper to keep in step with it, and the
call below is both the recipe and the record of what ran. Change a number here
and the diff is the experiment.

The values are pretraining's (the released RT-J recipe, `examples/train.py`).
What fine-tuning changes is the data: one `(db, task)` pair instead of a mixture,
read from the *benchmark* directory rather than the Join, so a run trains where
it is evaluated and train/eval differ only in split. `load_ckpt_path=None` is the
random-init control -- what the architecture learns from the task alone, which is
the number the pretrained arm has to beat.

Run it from a clean, pushed checkout: the job clones the commit you submit from.
"""

import sys

from roach.slurm import BLACKWELL, BLACKWELL_INTERACTIVE_1GPU, interactive, submit


def main(use_interactive: bool = False) -> None:
    held = None
    if use_interactive:
        held = interactive.find()
        if held is None:
            raise SystemExit(
                "no interactive allocation is held; take one with\n"
                "  from roach.slurm import BLACKWELL_INTERACTIVE, interactive\n"
                "  interactive.hold(BLACKWELL_INTERACTIVE,"
                " log_root='/dfs/user/ranjanr/slurm-logs/fine-tune')"
            )
    submit(
        "rt.train:main",
        args=dict(
            # model: RT-J's dims, so a fine-tuned run and a pretrained
            # checkpoint are the same architecture
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=True,
            materialize_attn_masks=True,
            # the arm: None is random init, a checkpoint path is fine-tuning
            load_ckpt_path=None,
            # data: one task, from the benchmark data rather than the Join
            db_task_list=[("rel-f1", "driver-top3")],
            pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            tokens_per_gpu=2**17,
            num_workers=16,
            prefetch_factor=2,
            ctx_size_list=[8192],
            local_ctx_size_list=[8192],
            bfs_width_list=[256],
            prefer_latest_list=[True],
            num_walks=10_000,
            walk_length=20,
            mask_prob_max=0.0,
            items_per_task=1000_000_000,
            # optimization: pretraining's, unchanged
            lr=5e-4,
            wd=0.1,
            warmup_steps=1_000,
            grad_norm_max=1.0,
            total_bs=1024,
            # pretraining's 100k steps is a mixture's worth of data, not one
            # task's: the one number this experiment sets on its own
            total_steps=10_001,
            swa_momentum=0.9995,
            seed=0,
            mmap_populate=True,
            timeout_per_item=10.0,
            eval_freq=100,
            vector_db_path=None,
            resume_save_mins=20.0,
            # in-loop validation: the task it is trained on, on the val split
            eval_splits=["val", "test"],
            eval_db_task_list=[("rel-f1", "driver-top3")],
            eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            eval_tokens_per_gpu=2**18,
            eval_num_workers=1,
            eval_prefetch_factor=2,
            eval_num_walks=10_000,
            eval_walk_length=20,
            eval_items_per_task=1_000_000_000,
            eval_ctx_size_list=[8192],
            eval_mmap_populate=True,
            eval_shuffle_seed=0,
            eval_context_seed=0,
            eval_vector_db_path=None,
            eval_lcs_bw_pl_grid=[(8192, 256, True)],
            # logging
            project="2026-08-05-fine-tune",
            entity="rtv2",
            run_name=None,
            wandb_disabled=False,
            out_root="/dfs/user/ranjanr/ckpts",
        ),
        # A held allocation is 2 b200s and each run takes one, so a second arm
        # starts beside this one rather than after it -- and a run's world size
        # does not change with how much of the allocation happens to be free.
        resources=BLACKWELL_INTERACTIVE_1GPU if use_interactive else BLACKWELL,
        overlap=held,
        name="ft-rel-f1-driver-top3-scratch",
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        log_root="/dfs/user/ranjanr/slurm-logs/fine-tune",
        # the node's own big disk, not /tmp (the 280G root filesystem): clones
        # are shared per commit and hold the pixi env, which pixi hardlinks from
        # the package cache only when the two are on the same filesystem
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
        clone_ttl_days=7,
        # the rustler sampler is a compiled extension; build it in the clone
        setup=("pixi run build-sampler",),
    )


if __name__ == "__main__":
    main(use_interactive="--interactive" in sys.argv[1:])
