"""Ensemble each fine-tuned checkpoint over context seeds, one job per task.
See [README.md](README.md).

The eval half of `submit.py`, at the context that script trains and evaluates
under -- `ctx_size` 1024, `(local_ctx_size, bfs_width, prefer_latest)` =
(1024, 128, False) -- swept over 8 context seeds, with the whole test split
scored after every seed. So one job yields the metric at every ensemble size up
to 8, and the last point is a RelBench-valid number over the full split rather
than the subsample the training curve carries.

No tuning: one configuration, so nothing reads validation -- which is what
`submit.py` trains on.

Each job waits on its own fine-tuning job, `afterok`, and loads the
`latest.safetensors` that run leaves behind. A task whose training has already
finished is submitted with no dependency at all.
"""

import dataclasses
import functools
import json
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from submit import TASKS, a100, b200, targets_for  # noqa: F401

# Where `submit.py`'s runs land, and the wandb project this one logs to.
CKPT_ROOT = Path("/dfs/user/ranjanr/ckpts/rtv2/2026-08-13-fine_tune")
PROJECT = "2026-08-13-ens"


@functools.cache
def run_dirs() -> dict[str, Path]:
    """`{db}/{task}` -> the output directory of its most recent run."""
    out: dict[str, list[Path]] = {}
    for d in sorted(CKPT_ROOT.iterdir()):
        name = json.loads((d / "params.json").read_text())["run_name"]
        out.setdefault(name, []).append(d)
    return {k: v[-1] for k, v in out.items()}


def ckpt_for(db: str, task: str) -> str:
    """The weights this task's fine-tuning run holds now.

    `latest.safetensors`, not `best_*`: `submit.py` trains on val, so nothing
    selects a checkpoint and the run keeps its last step. `rt.train` republishes
    the name at every eval, and the job that reads it waits for that run to
    finish, so what it opens is the fully decayed end of training. The run's
    `config.json` sits beside it, which is what `from_pretrained` reads the dims
    from.
    """
    return str(run_dirs()[f"{db}/{task}"] / "latest.safetensors")


@functools.cache
def train_jobs() -> dict[str, str]:
    """`{db}/{task}` -> the slurm id of its fine-tuning job, while it is queued.

    `submit.py` names a job `{db}-{task}`; a task missing here has already
    finished, and its ensembling job has nothing to wait for.
    """
    out = subprocess.run(
        ["squeue", "-h", "-u", "ranjanr", "-o", "%i %j"],
        capture_output=True,
        text=True,
        check=True,
    )
    jobs = {}
    for line in out.stdout.split("\n"):
        if not line.strip():
            continue
        job_id, name = line.split()
        for db, task in TASKS:
            if name == f"{db}-{task}":
                jobs[f"{db}/{task}"] = job_id
    return jobs


def dependency_for(db: str, task: str) -> str | None:
    """Wait for this task's fine-tuning job, or for nothing if it is done."""
    job = train_jobs().get(f"{db}/{task}")
    return None if job is None else f"afterok:{job}"


# Which slot each job goes in, laid out by hand -- one line per task.
# Commenting a line out is how a job is left out of a submission.
#
# NOT A DEFAULT TO INHERIT: whatever the last submission put here is a record of
# a different cluster and a different instruction. Work the assignment out again
# every time, following [Allocating a sweep](../README.md#allocating-a-sweep) --
# read the cluster, subtract what your own jobs already hold, spend the tiers
# top down.
#
# A task with no line here stops the submission rather than taking a slot
# nobody chose for it.
#
# 19:05: `il-lo` for all of them. Ensembling moved the metric by under 1.5
# points on the four tasks that have finished it, and never a rank, so it is
# the arm to run last: a high tier here would hold a card the 50-epoch runs and
# the remaining fine-tuning want. They resume from `ensemble_resume.pt`, so
# preemption costs one seed. No reservation: it expires at 2026-08-13T00:00 and
# most of these start after that.
RESOURCES: dict[tuple[str, str], Resources] = {
    ("rel-amazon", "user-churn"): a100("il-lo", "1-00:00:00"),
    ("rel-amazon", "user-ltv"): a100("il-lo", "1-00:00:00"),
    ("rel-stack", "user-badge"): a100("il-lo", "1-00:00:00"),
    ("rel-amazon", "item-ltv"): a100("il-lo", "1-00:00:00"),
    ("rel-amazon", "item-churn"): a100("il-lo", "1-00:00:00"),
    ("rel-stack", "post-votes"): a100("il-lo", "1-00:00:00"),
    ("rel-hm", "item-sales"): a100("il-lo", "1-00:00:00"),
    ("rel-stack", "user-engagement"): a100("il-lo", "1-00:00:00"),
    ("rel-hm", "user-churn"): a100("il-lo", "1-00:00:00"),
    ("rel-avito", "user-clicks"): a100("il-lo", "1-00:00:00"),
    ("rel-avito", "user-visits"): a100("il-lo", "1-00:00:00"),
    ("rel-trial", "site-success"): a100("il-lo", "1-00:00:00"),
    ("rel-trial", "study-adverse"): a100("il-lo", "1-00:00:00"),
    ("rel-event", "user-attendance"): a100("il-lo", "1-00:00:00"),
    ("rel-event", "user-ignore"): a100("il-lo", "1-00:00:00"),
    ("rel-avito", "ad-ctr"): a100("il-lo", "1-00:00:00"),
    ("rel-trial", "study-outcome"): a100("il-lo", "1-00:00:00"),
    ("rel-f1", "driver-position"): a100("il-lo", "1-00:00:00"),
    ("rel-f1", "driver-top3"): a100("il-lo", "1-00:00:00"),
    ("rel-f1", "driver-dnf"): a100("il-lo", "1-00:00:00"),
    ("rel-event", "user-repeat"): a100("il-lo", "1-00:00:00"),
}


def main() -> None:
    # `TASKS`' own order, which `submit.py` keeps shortest first.
    for db, task in TASKS:
        # The slot is the hand-laid choice above; the dependency is read off
        # the queue at submission time and set on a copy of it.
        resources = dataclasses.replace(
            RESOURCES[db, task], dependency=dependency_for(db, task)
        )
        name = f"{db}/{task}"
        print(
            f"  {name:28s} {resources.gpus} {resources.qos:15s} "
            f"{resources.dependency or 'no dependency'}"
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
                val_items_per_task=None,
                test_items_per_task=1_000_000_000,
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                db_cutoff="test",
                ctx_size_list=[1024],
                lcs_bw_pl_grid=[(1024, 128, False)],
                val_ensemble_size=1,
                test_ensemble_size=8,
                run_name=name,
                targets=targets_for(db, task),
                project=PROJECT,
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=f"ens-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-ens",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
