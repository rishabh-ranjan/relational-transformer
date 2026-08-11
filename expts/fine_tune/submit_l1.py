"""Submit one L1 fine-tuning job per regression task. See [README.md](README.md).

Four regression tasks under `loss_fn="l1"` instead of the sweep's huber, train
only: the arm `submit.py` already ran for each of them under huber, so the two
are read side by side and nothing but the loss differs.
"""

from pathlib import Path

from roach.slurm import Resources, submit

from submit import ckpt_for, ntest, targets_for

HERE = Path(__file__).parent

TASKS = (
    # already running from this file's previous submission
    # ("rel-stack", "post-votes"),
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "user-ltv"),
    # ("rel-event", "user-attendance"),
)


def b200(qos: str, time: str) -> Resources:
    """One B200. 36 cpus is blackwell1's 288 cores split eight ways, and the
    memory is that share of the node -- under the site's MaxMemPerCPU of 10700M
    times 36, which is what an explicit --mem is capped at."""
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="b200:1",
        cpus_per_task=36,
        ntasks=None,
        exclusive=False,
        mem="375000M",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
    )


def a100(qos: str, time: str) -> Resources:
    """One A100. 14 cpus is what the site allows per gpu on a job that is not
    --exclusive; no --mem, so the partition's DefMemPerGPU (240000M) applies,
    which is more than an explicit request would be given."""
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem=None,
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
    )


# Which slot each job goes in, laid out by hand -- one line per job, keyed by
# `(db, task)`. Commenting a line out is how a job is left out of a submission.
#
# NOT A DEFAULT TO INHERIT: work the assignment out again every time, following
# [Allocating a sweep](../README.md#allocating-a-sweep).
#
# Read at submission time: the huber sweep's two `il-interactive` jobs are
# finishing, handing that whole tier (2 gpus, the highest priority there is)
# back, and blackwell1 is 5 of 8 allocated with no reservation holding the rest
# -- so the two ltv jobs move off their amperes and onto b200. The other two
# were submitted an hour earlier and stay where they are: `il` a100.
#
# 12 hours is the tier's wall and these runs checkpoint every 20 minutes, so a
# job that outlives it resumes rather than starting over.
#
# A job with no line here stops the submission rather than taking a slot
# nobody chose for it.
RESOURCES: dict[tuple[str, str], Resources] = {
    ("rel-stack", "post-votes"): a100("il", "1-00:00:00"),
    ("rel-amazon", "item-ltv"): b200("il-interactive", "12:00:00"),
    ("rel-amazon", "user-ltv"): b200("il-interactive", "12:00:00"),
    ("rel-event", "user-attendance"): a100("il", "1-00:00:00"),
}


def main() -> None:
    tasks = sorted(TASKS, key=lambda p: ntest()[f"{p[0]}/{p[1]}"])
    for db, task in tasks:
        resources = RESOURCES[db, task]
        name = f"{db}/{task}-train-l1"
        print(f"  {name:38s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit(
            "rt.train:main",
            # Do not put comments inside this dict: it is a config block,
            # and reading it means scanning the values.
            args=dict(
                embedder="all-MiniLM-L12-v2",
                d_text=384,
                num_blocks=12,
                d_model=512,
                num_heads=8,
                d_ff=2048,
                compile=True,
                materialize_attn_masks=True,
                loss_fn="l1",
                load_ckpt_path=ckpt_for(db, task),
                db_task_list=[(db, task)],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**17 if resources.gpus.startswith("b200") else 2**16,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[1024],
                local_ctx_size_list=[1024],
                bfs_width_list=[128],
                prefer_latest_list=[True],
                num_walks=0,
                walk_length=20,
                mask_prob_max=0.0,
                items_per_task=1_000_000_000,
                lr=5e-4,
                wd=0.1,
                warmup_steps=500,
                grad_norm_max=1.0,
                total_bs=256,
                total_steps=10_001,
                early_stop_after_steps=None,
                swa_momentum=0.9999,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=500,
                keep_all_ckpts=False,
                vector_db_path=None,
                db_upto_test_timestamp=True,
                resume_save_mins=20.0,
                eval_splits=["test"],
                eval_db_task_list=[(db, task)],
                eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                eval_tokens_per_gpu=2**18,
                eval_num_workers=resources.cpus_per_task,
                eval_prefetch_factor=2,
                eval_num_walks=0,
                eval_walk_length=20,
                eval_items_per_task=2**16,
                eval_ctx_size_list=[1024],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(1024, 128, False)],
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune",
                entity="rtv2",
                run_name=name,
                wandb_disabled=False,
                out_root="/dfs/user/ranjanr/ckpts",
            ),
            resources=resources,
            name=f"{db}-{task}-train-l1",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
