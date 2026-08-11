"""Tune the eval context per task on validation and score the winner on test,
both in one job per task, on the fine-tuned checkpoints. See
[README.md](README.md).

The undivided version of `submit_hpo_only.py` + `submit_ens.py`: nothing to
wait for and nothing to read back.

Tuning reads fewer rows than the test pass it feeds: ranking 36 configurations
against each other needs less of the split than the number being reported
does, and the tuning is 18 passes to the test phase's 4.

`splits` does not gate the tuning: with a grid of more than one entry
`rt.eval.main` reads the val split whatever `splits` says, and `splits` decides
only whether the test phase runs at all. `["test"]` is therefore both phases.

The task list, the checkpoint each job loads and how much of a split a pass
covers are `submit_ens_only`'s -- one home each, and the two experiments score
the same weights on the same rows.
"""

from roach.slurm import Resources, submit

# a100 / b200 are unused while RESOURCES is blank, and imported so that
# filling it in is one line and not an import hunt.
from submit import a100, b200, targets_for  # noqa: F401
from submit_ens_only import ckpt_for, in_flight, items_for, nval, ntest, ready


# Which slot each job goes in, laid out by hand -- one line per job, keyed by
# `(db, task)`. Commenting a line out is how a job is left out of a submission.
#
# NOT A DEFAULT TO INHERIT, and blank on purpose: whatever the last submission
# put here is a record of a different cluster and a different instruction. Work
# the assignment out again every time, following
# [Allocating a sweep](../README.md#allocating-a-sweep) -- read the cluster,
# subtract what your own jobs already hold, spend the tiers top down -- and
# write today's answer here, one line per job this submission sends:
#
#     ("rel-f1", "driver-dnf"): a100("il", "12:00:00"),
#
# A job with no line here stops the submission rather than taking a slot
# nobody chose for it.
RESOURCES: dict[tuple[str, str], Resources] = {}


def job_name(db: str, task: str) -> str:
    """The slurm job name this experiment gives that task."""
    return f"hpo-ens-{db}-{task}"


def cost(db: str, task: str) -> float:
    """Roughly what this task's job costs: the rows one pass scores.

    The slot order below is by this, not by train-set size: an eval job's wall
    clock is the split it reads, and `items_for` caps the biggest ones, so the
    two orders disagree.
    """
    cap = items_for(db, task)
    return min(nval()[f"{db}/{task}"], cap) + min(ntest()[f"{db}/{task}"], cap)


def main() -> None:
    tasks = [t for t in ready() if job_name(*t) not in in_flight()]
    # Slowest first, so the best slots go where they buy the most.
    tasks.sort(key=lambda t: -cost(*t))
    ckpts = {t: ckpt_for(*t) for t in tasks}
    for db, task in tasks:
        resources = RESOURCES[db, task]
        name = f"{db}/{task}"
        print(f"  {name:28s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit(
            "rt.eval:main",
            # Do not put comments inside this dict: it is a config block,
            # and reading it means scanning the values.
            args=dict(
                load_ckpt_path=ckpts[db, task],
                embedder="all-MiniLM-L12-v2",
                d_text=384,
                num_blocks=12,
                d_model=512,
                num_heads=8,
                d_ff=2048,
                splits=["test"],
                db_task_list=[(db, task)],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**18,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                num_walks=10_000,
                walk_length=20,
                items_per_task={"val": 2**14, "test": items_for(db, task)},
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                db_upto_test_timestamp=False,
                ctx_size_list=[512, 1024, 2048],
                lcs_bw_pl_grid=[
                    (lcs, bw, pl)
                    for lcs in (512, 1024, 2048)
                    for bw in (64, 128, 256)
                    for pl in (True, False)
                ],
                val_ensemble_size=1,
                test_ensemble_size=4,
                run_name=name,
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune_hpo_ens",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=job_name(db, task),
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-hpo-ens",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
