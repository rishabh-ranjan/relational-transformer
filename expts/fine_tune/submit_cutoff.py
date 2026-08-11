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

from roach.slurm import submit

# a100 is unused while every line that asked for one is commented out, and
# imported so that putting a job back on one is a line and not an import hunt.
from submit import a100, b200, targets_for  # noqa: F401
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


# Which slot each job goes in, laid out by hand. Commenting a line out is how a
# job is left out of this submission.
#
# NOT A DEFAULT TO INHERIT. It records one cluster at one moment, and it is
# stale by the next submission. Work the assignment out again every time,
# following [Allocating a sweep](../README.md#allocating-a-sweep) -- read the
# cluster, subtract what your own jobs already hold, spend the tiers top down --
# and rewrite every line below with today's answer.
#
# 2026-08-10: `il-interactive` is full and `il` holds 10 -- 7 ensembling evals
# from another session, 3 of these. blackwell1 has 2 of its 8 b200 free, so the two jobs still
# waiting take them on `il` -- its whole b200 sub-cap, and those cost 2 of the
# ten, which the two `il` jobs demoted to `il-lo` here pay for. Demoting these
# and not the ensembling evals is the cheap direction: an eval does not
# checkpoint, and these are two minutes old against their forty. The seven are
# another session's jobs in any case: they count against the same per-user cap,
# but they are not this script's to cancel.
RESOURCES = {
    # ("rel-f1", "driver-dnf", False): a100("il", "1-00:00:00"),  # done
    # ("rel-f1", "driver-dnf", True): a100("il", "1-00:00:00"),  # done
    # ("rel-f1", "driver-position", False): a100("il", "1-00:00:00"),  # done
    ("rel-f1", "driver-position", True): b200("il", "1-00:00:00"),
    ("rel-f1", "driver-top3", False): b200("il", "1-00:00:00"),
    # ("rel-f1", "driver-top3", True): a100("il-lo", "1-00:00:00"),  # already queued
}


def main() -> None:
    for (db, task, upto), resources in RESOURCES.items():
        name = f"{db}/{task}"
        print(
            f"  {name:28s} upto={upto!s:5s} {resources.gpus} "
            f"{resources.qos:15s} {resources.time}"
        )
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
                num_workers=resources.cpus_per_task,
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
