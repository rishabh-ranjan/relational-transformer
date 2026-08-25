import csv
import json
import subprocess
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

PROJECT = "2026-08-24-fine_tune"
ENTITY = "rtv2"
OUT_ROOT = "~/scratch/relational-transformer/fine_tune"

MODELS = (
    ("rt-plurel", "~/scratch/hf/stanford-star/rt-plurel"),
    ("rt", None),
    # ("rt-j", "~/scratch/hf/stanford-star/rt-j"),
)

TASKS = (
    ("rel-hm", "item-sales"),
    ("rel-stack", "user-engagement"),
    ("rel-amazon", "user-churn"),
    ("rel-trial", "study-adverse"),
    ("rel-hm", "user-churn"),
    ("rel-amazon", "item-churn"),
    ("rel-event", "user-attendance"),
    ("rel-amazon", "user-ltv"),
    ("rel-avito", "user-visits"),
    ("rel-stack", "user-badge"),
    ("rel-event", "user-ignore"),
    ("rel-amazon", "item-ltv"),
    ("rel-trial", "site-success"),
    ("rel-stack", "post-votes"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "ad-ctr"),
    ("rel-trial", "study-outcome"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-position"),
)


def paper() -> dict[str, dict[str, str]]:
    with open(HERE / "relarena_paper.csv", newline="") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def stds() -> dict[str, float]:
    path = Path("~/scratch/hf/stanford-star/relbench/regression_stds.json")
    return json.loads(path.expanduser().read_text())["stds"]


def targets_for(db: str, task: str) -> dict[str, float]:
    row = paper()[f"{db}/{task}"]
    if row["metric"] == "roc_auc":
        metric, scale, best = "auroc", 100.0, max
    else:
        metric, scale, best = "nmae", 100.0 / stds()[f"{db}/{task}"], min
    others = [
        float(v) for k, v in row.items() if k not in ("task", "metric", "rt-plurel")
    ]
    return {
        f"{metric}/test/{db}/{task}": float(row["rt-plurel"]) * scale,
        f"{metric}/test/{db}/{task}/best-baseline": best(others) * scale,
    }


def a100(qos: str, time: str) -> Resources:
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
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere7",
    )


def b200(qos: str, time: str) -> Resources:
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
        reservation=None,
        dependency=None,
    )


# 2026-08-24 22:35, read off the cluster: nothing of mine on il/il-interactive;
# the 8 b200 are all held by one user, 2 under il (30h left) and 6 under il-lo,
# which il/il-interactive preempt, so a high-tier b200 request starts within
# the grace period; ~30 a100 free across the amperes, ampere4 kept out (disk
# 99% full) and ampere7 (a job placed there finds its a100 with 16 MB free
# and dies in CUDA OOM at the first forward -- three jobs did on 2026-08-24;
# `sbatch --test-only` plans an il a100 job 8h out, which only a
# submission can confirm or refute). Tiers top down, the longest tasks (RelArena's measured wall
# clocks, item-sales and user-engagement first) on the fastest slots:
# il-interactive 2 x b200, il 2 x b200 + 8 x a100, then il-lo for the 30 left.
# TASKS is in that cost order, so the plan reads down the list.
RESOURCES: dict[tuple[str, str, str], Resources] = {
    ("rt-plurel", "rel-hm", "item-sales"): b200("il-interactive", "12:00:00"),
    ("rt", "rel-hm", "item-sales"): b200("il-interactive", "12:00:00"),
    ("rt-plurel", "rel-stack", "user-engagement"): b200("il", "7-00:00:00"),
    ("rt", "rel-stack", "user-engagement"): b200("il", "7-00:00:00"),
    ("rt-plurel", "rel-amazon", "user-churn"): a100("il", "7-00:00:00"),
    ("rt", "rel-amazon", "user-churn"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-trial", "study-adverse"): a100("il", "7-00:00:00"),
    ("rt", "rel-trial", "study-adverse"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-hm", "user-churn"): a100("il", "7-00:00:00"),
    ("rt", "rel-hm", "user-churn"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-amazon", "item-churn"): a100("il", "7-00:00:00"),
    ("rt", "rel-amazon", "item-churn"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-event", "user-attendance"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-event", "user-attendance"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-amazon", "user-ltv"): a100("il-lo", "3-00:00:00"),
    # 02:25: rt/user-engagement finished on its il b200 (3h25); the freed slot
    # goes to the pending task with the most left, which had not started.
    ("rt", "rel-amazon", "user-ltv"): b200("il", "7-00:00:00"),
    ("rt-plurel", "rel-avito", "user-visits"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-avito", "user-visits"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-stack", "user-badge"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-stack", "user-badge"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-event", "user-ignore"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-event", "user-ignore"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-amazon", "item-ltv"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-amazon", "item-ltv"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-trial", "site-success"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-trial", "site-success"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-stack", "post-votes"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-stack", "post-votes"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-avito", "user-clicks"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-avito", "user-clicks"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-avito", "ad-ctr"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-avito", "ad-ctr"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-trial", "study-outcome"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-trial", "study-outcome"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-event", "user-repeat"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-event", "user-repeat"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-f1", "driver-dnf"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-f1", "driver-dnf"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-f1", "driver-top3"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-f1", "driver-top3"): a100("il-lo", "3-00:00:00"),
    ("rt-plurel", "rel-f1", "driver-position"): a100("il-lo", "3-00:00:00"),
    ("rt", "rel-f1", "driver-position"): a100("il-lo", "3-00:00:00"),
}


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", "ranjanr", "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def main() -> None:
    busy = queued()
    for model, load_ckpt_root in MODELS:
        for db, task in TASKS:
            name = f"ft-{model}-{db}-{task}"
            if name in busy:
                print(f"  {name:44s} queued already")
                continue
            table = (
                Path(OUT_ROOT).expanduser()
                / ENTITY
                / PROJECT
                / f"{model}-{db}-{task}-test"
                / "eval_out"
                / f"{db}__{task}.csv"
            )
            if table.exists():
                print(f"  {name:44s} done already")
                continue
            resources = RESOURCES[model, db, task]
            print(f"  {name:44s} {resources.gpus} {resources.qos:15s} {resources.time}")
            submit(
                "expts.fine_tune.run:main",
                args=dict(
                    model=model,
                    db=db,
                    task=task,
                    load_ckpt_root=load_ckpt_root,
                    pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
                    tokens_per_gpu=2**18
                    if resources.gpus.startswith("b200")
                    else 2**17,
                    num_workers=resources.cpus_per_task,
                    eval_num_workers=2,
                    selection_steps=50_000,
                    patience_steps=10_000,
                    eval_freq=100,
                    eval_rows=1024,
                    selection_ensemble_size=4,
                    tune_rows=4096,
                    test_ensemble_size=8,
                    seed=0,
                    targets=targets_for(db, task),
                    project=PROJECT,
                    entity=ENTITY,
                    wandb_disabled=False,
                    out_root=OUT_ROOT,
                ),
                resources=resources,
                name=name,
                repo_root=str(HERE.parents[1]),
                cluster=ILC,
                job_env="expts/job_env.sh",
                log_root=f"{OUT_ROOT}/slurm-logs",
                clone_root="~/roach_clones",
                secrets_dir="~/scratch/.secrets",
            )


if __name__ == "__main__":
    main()
