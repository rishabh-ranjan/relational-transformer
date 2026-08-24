import dataclasses
import functools
import json
import subprocess
from pathlib import Path

from roach.slurm.clusters.ilc import ILC
from submit import TASKS, a100, b200, targets_for  # noqa: F401

from roach.slurm import Resources, submit

CKPT_ROOT = Path("~/scratch/ckpts/rtv2/2026-08-13-fine_tune").expanduser()
PROJECT = "2026-08-13-ens"


@functools.cache
def run_dirs() -> dict[str, Path]:
    out: dict[str, list[Path]] = {}
    for d in sorted(CKPT_ROOT.iterdir()):
        name = json.loads((d / "params.json").read_text())["run_name"]
        out.setdefault(name, []).append(d)
    return {k: v[-1] for k, v in out.items()}


def ckpt_for(db: str, task: str) -> str:
    return str(run_dirs()[f"{db}/{task}"] / "latest.safetensors")


@functools.cache
def train_jobs() -> dict[str, str]:
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
    job = train_jobs().get(f"{db}/{task}")
    return None if job is None else f"afterok:{job}"


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
    for db, task in TASKS:
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
                pre_dir="~/scratch/share/stanford-star/relbench-preprocessed",
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
                out_root="~/scratch/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=f"ens-{db}-{task}",
            repo_root="~/clones/rishabh-ranjan/relational-transformer",
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root="~/scratch/relational-transformer/fine_tune/ens/slurm-logs",
            clone_root="~/roach_clones",
            secrets_dir="~/scratch/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
