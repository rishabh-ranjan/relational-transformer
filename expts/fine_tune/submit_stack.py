"""Submit one BCE fine-tuning job per rel-stack task. See [README.md](README.md).

The two rel-stack binary tasks under `loss_fn="bce"` instead of the sweep's
huber, train+val only: the arm the huber sweep reports from, so the two are
read side by side.
"""

from pathlib import Path

from roach.slurm import Resources, submit

from submit import ckpt_for, ntest, targets_for

HERE = Path(__file__).parent

TASKS = (
    ("rel-stack", "user-badge"),
    # already running from this file's previous submission
    # ("rel-stack", "user-engagement"),
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
# Read at submission time: the huber sweep of this directory holds my
# `il-interactive` cap (2 b200) outright, but two of its `il` a100 have since
# finished, so `il` has 2 of its 10 free and both jobs take them. blackwell1 is
# 7 of 8 allocated and the one free b200 looked untaken, but asking for it came
# back `ReqNodeNotAvail`: a reservation holds that card, and a b200 job pinned
# to blackwell1 has nowhere else to go. So both jobs are on amperes.
#
# A job with no line here stops the submission rather than taking a slot
# nobody chose for it.
RESOURCES: dict[tuple[str, str], Resources] = {
    ("rel-stack", "user-badge"): a100("il", "1-00:00:00"),
    ("rel-stack", "user-engagement"): a100("il", "1-00:00:00"),
}


def main() -> None:
    tasks = sorted(TASKS, key=lambda p: ntest()[f"{p[0]}/{p[1]}"])
    for db, task in tasks:
        resources = RESOURCES[db, task]
        name = f"{db}/{task}-trainval-bce"
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
                loss_fn="bce",
                load_ckpt_path=ckpt_for(db, task),
                db_task_list=[(db, task)],
                train_splits=["train", "val"],
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
            name=f"{db}-{task}-trainval-bce",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
