import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roach.slurm.clusters.ilc import ILC  # noqa: E402

from expts.relarena.submit import (  # noqa: E402
    CACHE_DIR,
    EXPERIMENTS,
    REPO_ROOT,
    RESERVATION_WALL,
    SECRETS_DIR,
    SHARE,
    a100,
    b200,
    relarena_setup,
    reserved,
)
from roach.slurm import submit  # noqa: E402

ORDER = [(d, t) for _m, d, t in EXPERIMENTS]


def plan(model: str) -> list:
    if model == "rt-j":
        tiers = (
            [b200("il-interactive", "12:00:00")] * 2
            + [b200("il", "7-00:00:00")] * 1
            + [reserved(RESERVATION_WALL)] * 8
            + [a100("il", "7-00:00:00")] * 5
            + [a100("il-lo", "21-00:00:00")] * 5
        )
    else:
        tiers = [a100("il-lo", "21-00:00:00")] * len(ORDER)
    return tiers[: len(ORDER)]


def main() -> None:
    models = sys.argv[1:] or ["rt-j", "rt"]
    for model in models:
        print(f"=== {model} ===")
        for (dataset, task), resources in zip(ORDER, plan(model)):
            job = submit(
                "expts.relarena.run:main",
                args=dict(
                    dataset=dataset,
                    task=task,
                    model=model,
                    seed=0,
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
            )
            print(f"  {model}/{dataset}/{task:22s} {resources.qos:15s} {job.id}")


if __name__ == "__main__":
    main()
