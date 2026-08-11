"""Score the fine-tuned checkpoints on test, one job per task per arm. See
[README.md](README.md).

Two arms over the same weights and the same rows:

- **ens_only** fixes the eval context at what the fine-tuning runs evaluated with
  and averages over context seeds. Nothing reads validation, so it waits on
  nothing and is the arm that answers first;
- **hpo_ens** ranks 66 context configurations on validation and ensembles the
  winner on test, in one job. It pays 24 passes, one per `lcs_bw_pl_grid`
  entry: the ctx sizes ride along on each as prefixes of the contexts already
  built.

Both score the *whole* test split, once per context seed -- nothing is
subsampled, so the metric is RelBench's own and each run writes a submission
directory. Only the tuning caps its rows: ranking configurations against each
other needs less of the split than the number being reported does.

`splits` does not gate the tuning: with a grid of more than one entry
`rt.eval.main` reads the val split whatever `splits` says, and `splits` decides
only whether the test phase runs at all. `["test"]` is therefore both phases.

**ens_only goes out first and takes the better slots**: its curve is the one the
paper needs, and hpo_ens only says how much tuning would add on top.

One config block for both: `main()` loops the two arm names over a single
`args` dict, and the four values the arms disagree on -- the context grid, the
val cap and the ensemble size -- sit inline at the arguments that take them.
An arm's wandb project, job name and log directory follow from its name.
"""

import functools
import json
import os
import shutil
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit

# a100 / b200 are unused while RESOURCES is blank, and imported so that
# filling it in is one line and not an import hunt.
from submit import a100, b200, ntest, targets_for  # noqa: F401

# The 21 RelBench forecast tasks, this experiment's own list rather than
# `submit.TASKS`: that one is whatever the fine-tuning sweep last submitted,
# which a partial resubmission narrows to a handful, and every task fine-tuning
# has reached belongs here.
TASKS = (
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "user-ltv"),
    ("rel-avito", "ad-ctr"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
    ("rel-hm", "item-sales"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "post-votes"),
    ("rel-stack", "user-badge"),
    ("rel-stack", "user-engagement"),
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-trial", "study-outcome"),
)

# Where `submit.py`'s fine-tuning runs land.
CKPT_ROOT = Path("/dfs/user/ranjanr/ckpts/rtv2/2026-08-11-fine_tune")

# Where the weights a job is going to load are copied to, out of reach of the
# training run that wrote them. 163M a task, shared by both arms, and deleted
# with the rest of the scratch.
PINNED = Path("/dfs/user/ranjanr/ckpts/rtv2/fine-tune-pinned")


@functools.cache
def run_dirs() -> dict[str, Path]:
    """`{db}/{task}` -> the output directory of its most recent run."""
    out: dict[str, list[Path]] = {}
    for d in sorted(CKPT_ROOT.iterdir()):
        name = json.loads((d / "params.json").read_text())["run_name"]
        out.setdefault(name, []).append(d)
    return {k: v[-1] for k, v in out.items()}


def task_metric(db: str, task: str) -> tuple[str, bool]:
    """The task's metric name and whether higher is better."""
    keys = targets_for(db, task)
    (metric,) = {k.split("/")[0] for k in keys if k.endswith(f"/{db}/{task}")}
    return metric, metric == "auroc"


def best_ckpt(db: str, task: str) -> Path:
    """Where that task's fine-tuning run publishes its best-on-val weights."""
    metric, _ = task_metric(db, task)
    return (
        run_dirs()[f"{db}/{task}"]
        / f"best_{'clf' if metric == 'auroc' else 'reg'}.safetensors"
    )


def ckpt_for(db: str, task: str) -> str:
    """The best-on-val checkpoint of that task's fine-tuning run.

    `rt.train` republishes `best_{clf,reg}.safetensors` at every eval that
    improves -- the better of the live net's winner and the SWA net's, over
    every segment the run was preempted into -- so a run still training has
    one under that name too, and it is whatever it was at the last eval.

    Copied, not pointed at: the run that wrote it is still going, and its next
    improvement replaces the file underneath a job that has not loaded it yet.
    A directory of its own, with the run's `config.json` beside the weights:
    `from_pretrained` reads the dims from a `config.json` sitting next to the
    file, under that exact name.
    """
    run = run_dirs()[f"{db}/{task}"]
    src = best_ckpt(db, task)
    assert src.exists(), f"{src} does not exist; has that run reached an eval?"
    pinned = PINNED / f"{db}__{task}" / src.name
    if not pinned.exists():
        pinned.parent.mkdir(parents=True, exist_ok=True)
        tmp = pinned.with_suffix(f".{os.getpid()}.tmp")
        shutil.copyfile(src, tmp)
        tmp.rename(pinned)
        shutil.copyfile(run / "config.json", pinned.parent / "config.json")
    return str(pinned)


def in_flight() -> set[str]:
    """The job names already queued or running, either arm.

    Resubmitting is how these sweeps pick up newly fine-tuned tasks, and a task
    whose job has not finished has no curve yet: without this it would be
    queued a second time on every rerun.
    """
    out = subprocess.run(
        ["squeue", "-h", "-o", "%j", "-u", os.environ["USER"]],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def ready(arm: str) -> list[tuple[str, str]]:
    """The tasks that arm can score now, largest test set first.

    An eval job's wall clock is the rows it reads, and nothing is subsampled:
    both arms score the whole test split, once per context seed. So the
    slowest job starts first and the small ones fill the slots behind it.

    A task whose fine-tuning run has not written a checkpoint yet -- it has not
    reached its first eval -- is skipped rather than aborting the submission:
    that run is still going, and this is rerun as they land.
    """
    started = [
        t
        for t in TASKS
        if best_ckpt(*t).exists() and f"{arm}-{t[0]}-{t[1]}" not in in_flight()
    ]
    return sorted(started, key=lambda t: -ntest()[f"{t[0]}/{t[1]}"])


# Which slot each job goes in, laid out by hand -- one line per job, keyed by
# `(arm, db, task)`. Commenting a line out is how a job is left out of a
# submission.
#
# NOT A DEFAULT TO INHERIT, and blank on purpose: whatever the last submission
# put here is a record of a different cluster and a different instruction. Work
# the assignment out again every time, following
# [Allocating a sweep](../README.md#allocating-a-sweep) -- read the cluster,
# subtract what your own jobs already hold, spend the tiers top down -- and
# write today's answer here, one line per job this submission sends:
#
#     ("ens_only", "rel-f1", "driver-dnf"): a100("il", "12:00:00"),
#
# The ens_only arm is the one to spend the high tiers on. A job with no line here
# stops the submission rather than taking a slot nobody chose for it.
RESOURCES: dict[tuple[str, str, str], Resources] = {}


def main() -> None:
    # ens_only first: it takes the slots while the budget is still unspent, and
    # a hpo_ens job that finds no line in RESOURCES stops a submission that has
    # already placed every ens_only job it was going to.
    for arm in ("ens_only", "hpo_ens"):
        for db, task in ready(arm):
            resources = RESOURCES[arm, db, task]
            name = f"{db}/{task}"
            print(
                f"  {arm:8s} {name:28s} "
                f"{resources.gpus} {resources.qos:15s} {resources.time}"
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
                    val_items_per_task=None if arm == "ens_only" else 2**12,
                    test_items_per_task=1_000_000_000,
                    mmap_populate=True,
                    shuffle_seed=0,
                    context_seed=0,
                    vector_db_path=None,
                    db_upto_test_timestamp=True,
                    ctx_size_list=[1024] if arm == "ens_only" else [512, 1024, 2048],
                    lcs_bw_pl_grid=(
                        [(1024, 256, False)]
                        if arm == "ens_only"
                        else [
                            (lcs, bw, pl)
                            for lcs in (128, 256, 512, 1024)
                            for bw in (16, 64, 256)
                            for pl in (False, True)
                        ]
                    ),
                    val_ensemble_size=1 if arm == "ens_only" else 4,
                    test_ensemble_size=8,
                    run_name=name,
                    targets=targets_for(db, task),
                    project="2026-08-11-eval",
                    entity="rtv2",
                    out_root="/dfs/user/ranjanr/ckpts",
                    wandb_disabled=False,
                ),
                resources=resources,
                name=f"{arm}-{db}-{task}",
                repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
                log_root=f"/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-{arm}",
                clone_root="/lfs/local/0/roach_clones",
                secrets_dir="/dfs/user/ranjanr/.secrets",
                run_id=None,
            )


if __name__ == "__main__":
    main()
