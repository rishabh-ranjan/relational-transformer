from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

REPO_ROOT = "~/clones/rishabh-ranjan/relational-transformer-relarena"
SECRETS_DIR = "~/scratch/.secrets"
SHARE = "~/scratch/share/relarena"

CACHE_DIR = "~/.cache/relarena"

RESERVATION_WALL = "1-20:00:00"

EXPERIMENTS = (
    ("rt-plurel", "rel-amazon", "item-churn"),
    ("rt-plurel", "rel-amazon", "user-ltv"),
    ("rt-plurel", "rel-amazon", "item-ltv"),
    ("rt-plurel", "rel-amazon", "user-churn"),
    ("rt-plurel", "rel-hm", "item-sales"),
    ("rt-plurel", "rel-stack", "user-engagement"),
    ("rt-plurel", "rel-trial", "study-adverse"),
    ("rt-plurel", "rel-stack", "user-badge"),
    ("rt-plurel", "rel-hm", "user-churn"),
    ("rt-plurel", "rel-stack", "post-votes"),
    ("rt-plurel", "rel-avito", "user-clicks"),
    ("rt-plurel", "rel-avito", "user-visits"),
    ("rt-plurel", "rel-trial", "site-success"),
    ("rt-plurel", "rel-event", "user-ignore"),
    ("rt-plurel", "rel-f1", "driver-dnf"),
    ("rt-plurel", "rel-avito", "ad-ctr"),
    ("rt-plurel", "rel-event", "user-attendance"),
    ("rt-plurel", "rel-trial", "study-outcome"),
    ("rt-plurel", "rel-f1", "driver-top3"),
    ("rt-plurel", "rel-event", "user-repeat"),
    ("rt-plurel", "rel-f1", "driver-position"),
)

ZERO_SHOT: tuple = ()
_ZERO_SHOT_DONE = (
    ("rel-f1", "driver-top3", "val", True, 0),
    ("rel-f1", "driver-top3", "val", True, -1),
)


BAD_NODES = "ampere4,ampere7"


def relarena_setup() -> tuple[str, ...]:
    token = f'$(tr -d "[:space:]" < {SECRETS_DIR}/github)'
    url = f"git+https://x-access-token:{token}@github.com/rishabh-ranjan/relarena-alpha@main"
    return (
        f'pixi run uv pip install --no-deps "relarena @ {url}"',
        'pixi run uv pip install "configspace>=1.0" "jsonschema>=4.0"',
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


def reserved(time: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=time,
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem=None,
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation="ranjanr_deadline",
        dependency=None,
        exclude=BAD_NODES,
    )


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
        exclude=BAD_NODES,
    )


RESOURCES: dict[tuple[str, str, str], Resources] = {
    ("rt-plurel", "rel-amazon", "user-ltv"): b200("il", "7-00:00:00"),
    ("rt-plurel", "rel-amazon", "user-churn"): b200("il", "7-00:00:00"),
    ("rt-plurel", "rel-amazon", "item-churn"): b200("il-interactive", "12:00:00"),
    ("rt-plurel", "rel-amazon", "item-ltv"): b200("il-interactive", "12:00:00"),
    ("rt-plurel", "rel-stack", "post-votes"): b200("il-lo", "21-00:00:00"),
    ("rt-plurel", "rel-stack", "user-badge"): b200("il-lo", "21-00:00:00"),
    ("rt-plurel", "rel-trial", "study-adverse"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-hm", "item-sales"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-hm", "user-churn"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-trial", "site-success"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-avito", "user-visits"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-stack", "user-engagement"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-avito", "user-clicks"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-event", "user-ignore"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-event", "user-attendance"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-avito", "ad-ctr"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-trial", "study-outcome"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-f1", "driver-dnf"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-event", "user-repeat"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-f1", "driver-position"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-f1", "driver-top3"): reserved(RESERVATION_WALL),
}

ZERO_SHOT_RESOURCES: dict[tuple[str, str], Resources] = {
    ("rel-f1", "driver-top3"): a100("il", "1:00:00"),
}

BENCH: tuple[tuple[str, str], ...] = ()

RUN_IDS: dict[tuple[str, str, str], str] = {}


def main() -> None:
    for dataset, task, split, mask_labels, cutoff_offset in ZERO_SHOT:
        resources = ZERO_SHOT_RESOURCES[dataset, task]
        print(f"  zero-shot/{dataset}/{task}/{split} {resources.gpus} {resources.qos}")
        submit(
            "expts.relarena.zero_shot:main",
            args=dict(
                dataset=dataset,
                task=task,
                split=split,
                mask_labels=mask_labels,
                cutoff_offset=cutoff_offset,
                cache_dir=CACHE_DIR,
                out_dir=f"{SHARE}/results",
            ),
            resources=resources,
            name=f"relarena-zero-shot-{dataset}-{task}-{split}-off{cutoff_offset}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root="~/scratch/relational-transformer/relarena/slurm-logs",
            clone_root="~/roach_clones",
            secrets_dir=SECRETS_DIR,
        )

    for dataset, task in BENCH:
        submit(
            "expts.relarena.bench_compile:main",
            args=dict(dataset=dataset, task=task, cache_dir=CACHE_DIR),
            resources=reserved("2:00:00"),
            name=f"relarena-bench-compile-{dataset}-{task}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root="~/scratch/relational-transformer/relarena/slurm-logs",
            clone_root="~/roach_clones",
            secrets_dir=SECRETS_DIR,
        )
        print(f"  bench-compile/{dataset}/{task}")

    seed = 0
    for model, dataset, task in EXPERIMENTS:
        resources = RESOURCES[model, dataset, task]
        name = f"relarena/{model}/{dataset}/{task}"
        print(f"  {name:38s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit(
            "expts.relarena.run:main",
            args=dict(
                dataset=dataset,
                task=task,
                model=model,
                seed=seed,
                n_trials=1,
                cache_dir=CACHE_DIR,
                out_dir=f"{SHARE}/results",
            ),
            resources=resources,
            name=f"relarena-{model}-{dataset}-{task}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root="~/scratch/relational-transformer/relarena/slurm-logs",
            clone_root="~/roach_clones",
            secrets_dir=SECRETS_DIR,
            run_id=RUN_IDS.get((model, dataset, task)),
        )


if __name__ == "__main__":
    main()
