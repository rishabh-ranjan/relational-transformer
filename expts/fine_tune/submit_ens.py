"""Score each task on test with the context its tuning run picked on
validation, ensembled over context seeds, one job per task, over the *whole*
test split. See [README.md](README.md).

`submit_hpo_ens.py` caps the biggest test sets (`submit_ens_only.items_for`),
which makes those numbers incomparable to a published one. This is the pass
that removes the cap, so it only has the tasks that were capped: everything
else already has a whole-split number and rerunning it would buy nothing.
"""

import json
from pathlib import Path

from roach.slurm import Resources, submit

# a100 / b200 are unused while RESOURCES is blank, and imported so that
# filling it in is one line and not an import hunt.
from submit import a100, b200, targets_for  # noqa: F401
from submit_ens_only import TASKS, ckpt_for, items_for, ntest

TUNING_ROOT = Path("/dfs/user/ranjanr/ckpts/rtv2/2026-08-10-fine_tune_hpo_ens")


def capped() -> list[tuple[str, str]]:
    """The tasks `submit_hpo_ens.py` subsampled, costliest first.

    A task whose whole test split already fits under the cap was never
    subsampled and is not rerun here.
    """
    out = [t for t in TASKS if items_for(*t) < ntest()[f"{t[0]}/{t[1]}"]]
    return sorted(out, key=lambda t: -ntest()[f"{t[0]}/{t[1]}"])


def cfg_for(db: str, task: str) -> tuple[int, int, int, bool]:
    """The `(ctx_size, local_ctx_size, bfs_width, prefer_latest)` this task's
    tuning run chose on validation.

    Read from the `tuning.json` every tuning run writes beside its `eval_out`,
    over every run under `TUNING_ROOT`: which run covered which task is not in
    the path. A task can have several -- a cancelled attempt that got as far as
    writing one leaves it behind -- so the newest wins, run directories being
    named by timestamp.
    """
    hits = [
        entry
        for p in sorted(TUNING_ROOT.glob("*/tuning.json"), key=lambda q: q.parent.name)
        for key, entry in json.loads(p.read_text()).items()
        if key == f"{db}/{task}"
    ]
    assert hits, f"no tuning entry for {db}/{task} in {TUNING_ROOT}"
    ctx, lcs, bw, pl = hits[-1]["best_cfg"]
    return ctx, lcs, bw, pl


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


def main() -> None:
    tasks = capped()
    # Every checkpoint and winning config before any job: both assert, and a
    # task whose fine-tuning or tuning run has not got that far must abort the
    # submission rather than leave the tasks ahead of it queued and the rest
    # not.
    ckpts = {t: ckpt_for(*t) for t in tasks}
    cfgs = {t: cfg_for(*t) for t in tasks}
    for db, task in tasks:
        resources = RESOURCES[db, task]
        name = f"{db}/{task}"
        ctx, lcs, bw, pl = cfgs[db, task]
        print(f"  {name:28s} {(ctx, lcs, bw, pl)} {resources.qos:15s} {resources.time}")
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
                items_per_task={"test": 10_000_000},
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                db_upto_test_timestamp=False,
                ctx_size_list=[ctx],
                lcs_bw_pl_grid=[(lcs, bw, pl)],
                val_ensemble_size=1,
                test_ensemble_size=4,
                run_name=name,
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune_ens_full",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=f"ens-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-ens",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
