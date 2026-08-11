"""Ensembling on its own: score each task on test at the context the
fine-tuning runs evaluated with, averaged over context seeds, one job per task.
See [README.md](README.md).

No tuning, so nothing reads validation and this does not wait on
`submit_hpo_only.py`. `rt.eval` scores the running average after every seed, so one
job yields the whole test-metric-vs-ensemble-size curve, logged to wandb
against `ens_size` with each task's published target beside it.

Tasks go out largest train set first: the curve that takes longest starts
first, and the small ones fill the slots behind it.
"""

import functools
import json
import os
import shutil
import subprocess
from pathlib import Path

import wandb
from roach.slurm import Resources, submit

from ens_table import ENTITY, PROJECT, curves

# a100 / b200 are unused while RESOURCES is blank, and imported so that
# filling it in is one line and not an import hunt.
from submit import a100, b200, ntrain, targets_for  # noqa: F401

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

# Where `submit.py`'s fine-tuning runs land, and the wandb project they log to.
CKPT_ROOT = Path("/dfs/user/ranjanr/ckpts/rtv2/2026-08-08-fine_tune")
FINE_TUNE_PROJECT = "rtv2/2026-08-08-fine_tune"

# Where the weights a job is going to load are copied to, out of reach of the
# training run that wrote them. 163M a task, shared by every experiment that
# evaluates these checkpoints, and deleted with the rest of the scratch.
PINNED = Path("/dfs/user/ranjanr/ckpts/rtv2/fine-tune-pinned")


@functools.cache
def run_dirs() -> dict[str, Path]:
    """`{db}/{task}` -> the output directory of its most recent run."""
    out: dict[str, list[Path]] = {}
    for d in sorted(CKPT_ROOT.iterdir()):
        name = json.loads((d / "params.json").read_text())["run_name"]
        out.setdefault(name, []).append(d)
    return {k: v[-1] for k, v in out.items()}


def ckpt_for(db: str, task: str) -> str:
    """The best-on-val checkpoint of that task's fine-tuning run *so far*.

    `best_{clf,reg}.safetensors` is written only when a run reaches its final
    eval, so waiting for it would mean waiting for the whole sweep. Until then
    the same weights are already on disk under the name the periodic save gave
    them: `rt.train` prunes every checkpoint its running best does not point
    at, and keeps the live net's winner and the SWA net's.

    Which of the two is the better -- what `best_{tt}` would end up being -- is
    in the run's own val curve, so it is read from wandb: the highest AUROC or
    the lowest nMAE over both nets and every attempt, and the step it happened
    at names the file.
    """
    run = run_dirs()[f"{db}/{task}"]
    metric, higher = task_metric(db, task)
    key = f"{metric}/val/{db}/{task}"
    best = None  # (value, filename), better first
    for r in wandb.Api().runs(FINE_TUNE_PROJECT, {"config.run_name": f"{db}/{task}"}):
        for row in r.scan_history(keys=["step", key, f"swa/{key}"]):
            for k, name in ((key, "steps"), (f"swa/{key}", "swa_steps")):
                # A row carries a key only at the steps that logged it: evals
                # are every `eval_freq` steps, the rest of the history is the
                # per-step training curve.
                v = row.get(k)
                if v is None:
                    continue
                cand = (v, f"{name}={row['step']}.safetensors")
                if best is None or (cand[0] > best[0]) == higher:
                    best = cand
    assert best is not None, f"no val curve for {db}/{task} in {FINE_TUNE_PROJECT}"
    path = run / best[1]
    assert path.exists(), f"{path} was pruned; the val curve says it is the best"
    # Copied, not pointed at: the run that wrote it is still training, and its
    # next eval prunes every checkpoint its new best does not point at. A job
    # that starts after that finds nothing at the path -- and `load_rt_model`
    # reads a path that does not exist as a Hub repo id, so it fails on a 404
    # from the Hub rather than on the file that went missing.
    # A directory of its own: `from_pretrained` reads the dims from the
    # `config.json` sitting beside the weights file, under that exact name.
    pinned = PINNED / f"{db}__{task}" / path.name
    if not pinned.exists():
        pinned.parent.mkdir(parents=True, exist_ok=True)
        tmp = pinned.with_suffix(f".{os.getpid()}.tmp")
        shutil.copyfile(path, tmp)
        tmp.rename(pinned)
        shutil.copyfile(run / "config.json", pinned.parent / "config.json")
    return str(pinned)


def task_metric(db: str, task: str) -> tuple[str, bool]:
    """The task's metric name and whether higher is better."""
    keys = targets_for(db, task)
    (metric,) = {k.split("/")[0] for k in keys if k.endswith(f"/{db}/{task}")}
    return metric, metric == "auroc"


@functools.cache
def split_sizes() -> dict[str, dict[str, float]]:
    """`{db}/{task}` -> val and test row counts, from RelBench's task stats."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stats = pd.read_parquet(
        hf_hub_download(
            "stanford-star/relbench", "STATS/tasks.parquet", repo_type="dataset"
        )
    )
    return {
        f"{r.database}/{r.task}": {
            "val": float(r.num_rows_val),
            "test": float(r.num_rows_test),
        }
        for r in stats.itertuples()
    }


def nval() -> dict[str, float]:
    """Val-set size per `{db}/{task}`."""
    return {k: v["val"] for k, v in split_sizes().items()}


def ntest() -> dict[str, float]:
    """Test-set size per `{db}/{task}`."""
    return {k: v["test"] for k, v in split_sizes().items()}


def items_for(db: str, task: str) -> int:
    """How many test items one context seed scores.

    A seed is a whole pass over them and there are `test_ensemble_size` of
    them, so the largest splits are hours of wall clock for one curve. Above
    the largest rel-avito task the split is subsampled to 2**16; below it every
    row is scored.

    `shuffle_seed` fixes which rows a subsample takes, so the seeds being
    averaged score the same items, and the metric comes back named `nmae~` /
    `roc_auc~`: the same definition on part of the split, and no RelBench
    submission (`_emit_and_score` refuses to write one that does not cover the
    test set).
    """
    limit = max(ntest()[f"{d}/{t}"] for d, t in TASKS if d == "rel-avito")
    return 2**16 if ntest()[f"{db}/{task}"] > limit else 10_000_000


def job_name(db: str, task: str) -> str:
    """The slurm job name this experiment gives that task."""
    return f"ens-only-{db}-{task}"


def in_flight() -> set[str]:
    """The job names this experiment already has queued or running.

    Resubmitting is how this sweep picks up newly fine-tuned tasks, and a task
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


def ready() -> list[tuple[str, str]]:
    """The tasks that can be ensembled now, largest train set first.

    A task whose fine-tuning run has not written a checkpoint yet -- it has not
    reached its first eval -- is skipped rather than aborting the submission:
    that run is still going, and this is rerun as they land.
    """
    started = {
        t
        for t in TASKS
        if any(run_dirs()[f"{t[0]}/{t[1]}"].glob("*steps=*.safetensors"))
    }
    return sorted(started, key=lambda p: -ntrain()[f"{p[0]}/{p[1]}"])


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
    tasks = ready()
    # A task that already has a curve is not queued a second one. Comment this
    # out to re-ensemble them anyway -- worth it once a run has finished
    # training and the checkpoint under its curve has moved.
    scored = {tuple(name.split("/")) for name in curves(ENTITY, PROJECT)}
    tasks = [t for t in tasks if t not in scored and job_name(*t) not in in_flight()]
    # Re-scoring a task whose curve is stale: name them, and the filter above
    # is what would otherwise skip them. Leave commented in the normal case.
    # tasks = [("rel-amazon", "user-churn")]
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
                items_per_task={"test": items_for(db, task)},
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                db_upto_test_timestamp=True,
                ctx_size_list=[2048],
                lcs_bw_pl_grid=[(2048, 128, True)],
                val_ensemble_size=1,
                test_ensemble_size=16,
                run_name=name,
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune_ens_only",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=job_name(db, task),
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-ens-only",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
