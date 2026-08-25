import csv
import json
import subprocess
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

PROJECT = "2026-08-24-fine_tune-probe"
ENTITY = "rtv2"
OUT_ROOT = "~/scratch/relational-transformer/fine_tune"

MODELS = (
    ("rt-plurel", "~/scratch/hf/stanford-star/rt-plurel"),
    # ("rt", None),
    # ("rt-j", "~/scratch/hf/stanford-star/rt-j"),
)

TASKS = (
    ("rel-f1", "driver-dnf"),
    # ("rel-hm", "item-sales"),
    # ("rel-stack", "user-engagement"),
    # ("rel-amazon", "user-churn"),
    # ("rel-trial", "study-adverse"),
    # ("rel-hm", "user-churn"),
    # ("rel-amazon", "item-churn"),
    # ("rel-event", "user-attendance"),
    # ("rel-amazon", "user-ltv"),
    # ("rel-avito", "user-visits"),
    # ("rel-stack", "user-badge"),
    # ("rel-event", "user-ignore"),
    # ("rel-amazon", "item-ltv"),
    # ("rel-trial", "site-success"),
    # ("rel-stack", "post-votes"),
    # ("rel-avito", "user-clicks"),
    # ("rel-avito", "ad-ctr"),
    # ("rel-trial", "study-outcome"),
    # ("rel-event", "user-repeat"),
    # ("rel-f1", "driver-top3"),
    # ("rel-f1", "driver-position"),
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
        exclude="ampere4",
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


# 2026-08-24 22:20: nothing of mine on il/il-interactive; ~30 a100 free across
# the amperes; ampere4's local disk is 99% full, so it is kept out. One short
# probe, top priority, on an ampere.
RESOURCES: dict[tuple[str, str, str], Resources] = {
    ("rt-plurel", "rel-f1", "driver-dnf"): a100("il-interactive", "3:00:00"),
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
            assert name not in busy, (
                f"{name} is already queued or running; a second job would write "
                f"the same stage directories under {OUT_ROOT}"
            )
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
                    selection_steps=300,
                    patience_steps=200,
                    eval_freq=100,
                    eval_rows=256,
                    selection_ensemble_size=2,
                    tune_rows=512,
                    test_ensemble_size=2,
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
