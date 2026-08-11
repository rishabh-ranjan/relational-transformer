"""What trimming the database to the test timestamp costs, on rel-f1.

Two jobs per task, identical but for `db_upto_test_timestamp`: with it, the
contexts are built from the database RelBench's
`get_db(upto_test_timestamp=True)` hands a model -- nothing past 2010-01-01 --
and without it, from the whole database, which for rel-f1 runs to 2023.

rel-f1 is the only RelBench dataset where the two differ at all: everywhere
else the raw tables stop at the test timestamp, so the trim removes nothing.
The task list is this experiment's own for that reason, not `submit.TASKS`.

Weights, context and item counts are `submit_ens_only.py`'s, so the arm without
the trim reproduces that experiment's rel-f1 numbers rather than being a
separate measurement of them.

    pixi run python expts/fine_tune/submit_cutoff.py

The table is `cutoff_table.py`. See [README.md](README.md).
"""

from roach.slurm import Resources, submit

from submit import a100, targets_for
from submit_ens_only import ckpt_for, items_for

TASKS = (
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
)

# One project per arm: the two runs of a task are then the same `run_name` in
# two places, which is what `cutoff_table.py` puts side by side.
PROJECTS = {
    False: "2026-08-11-fine_tune_cutoff_off",
    True: "2026-08-11-fine_tune_cutoff_on",
}


def plan(n: int) -> list[Resources]:
    """One slot per job, in the order `main` hands them out.

    Amperes, not blackwell. blackwell1's 8 cards are all held by other people's
    non-preemptible jobs and the soonest to free is ~11h out (12h limit, 1h
    elapsed); the rest run to 17h, 23h and 5d. One of these evals is minutes to
    an hour, so a card now beats a b200 tomorrow.

    Three `il` slots and no more: `il` is 10 gpus across all my sweeps and the
    fine-tuning sweep holds 7, `il-interactive`'s 2 are held by it too. The
    three that fit go there, the other three take `il-lo`.

    Which three matters here, unlike in a training sweep: an eval run does not
    checkpoint, so a preemption on `il-lo` restarts it from ensemble size 1.
    `il` is non-preemptible and 7 days, so 1 day is more than these need.

    Recount and rewrite this before every submission.
    """
    return [a100("il", "1-00:00:00")] * min(n, 3) + [a100("il-lo", "1-00:00:00")] * max(
        0, n - 3
    )


def main() -> None:
    jobs = [(db, task, upto) for db, task in TASKS for upto in (False, True)]
    for (db, task, upto), resources in zip(jobs, plan(len(jobs)), strict=True):
        name = f"{db}/{task}"
        print(f"  {name:28s} db_upto_test_timestamp={upto} {resources.qos}")
        submit(
            "rt.eval:main",
            # Do not put comments inside this dict: it is a config block,
            # and reading it means scanning the values.
            args=dict(
                load_ckpt_path=ckpt_for(db, task),
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
                num_workers=14,
                prefetch_factor=2,
                num_walks=10_000,
                walk_length=20,
                items_per_task={"test": items_for(db, task)},
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                db_upto_test_timestamp=upto,
                ctx_size_list=[2048],
                lcs_bw_pl_grid=[(2048, 128, True)],
                val_ensemble_size=1,
                test_ensemble_size=16,
                run_name=name,
                targets=targets_for(db, task),
                project=PROJECTS[upto],
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=f"cutoff-{int(upto)}-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-cutoff",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
