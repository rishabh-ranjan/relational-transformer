"""Submit ensemble-curve jobs: one per (variant, task). See README.md."""

import json
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT_CLF,
    CKPT_REG,
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
)
from roach.slurm import Resources, submit

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = f"{OUT_ROOT}/repaper-enscurve"
LOG_ROOT = f"{LOG_ROOT}/repaper/enscurve/slurm-logs"

TASKS = [
    tuple(p)
    for p in json.loads(
        (Path(PRE_DIR).expanduser() / "db-task-lists" / "forecast.json").read_text()
    )
]

MEM = {
    "rel-amazon": "120G",
    "rel-avito": "48G",
    "rel-event": "48G",
    "rel-f1": "24G",
    "rel-hm": "64G",
    "rel-stack": "64G",
    "rel-trial": "48G",
}


def tuned_cfg(db: str, table: str) -> tuple[int, int, int, bool]:
    cfgs = json.loads(
        (REPO_ROOT / "expts" / "repaper" / "tune" / "tuned_configs.json").read_text()
    )
    ctx, lcs, bw, pl = cfgs[f"{db}/{table}"]["best_cfg"]
    return int(ctx), int(lcs), int(bw), bool(pl)


def resources(db: str, qos: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time="12:00:00" if qos == "il-interactive" else "2-00:00:00",
        gpus="a100:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem=MEM[db],
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
    )


def submit_variant(variant: str, qos: str, tasks=None) -> None:
    for db, table in tasks or TASKS:
        if variant == "default":
            ctx, lcs, bw, pl = 8192, 256, 32, True
        else:
            ctx, lcs, bw, pl = tuned_cfg(db, table)
        out_dir = f"{OUT_ROOT}/{variant}"
        if (Path(out_dir).expanduser() / f"{db}__{table}.json").exists():
            continue
        submit(
            "expts.repaper.enscurve.run:main",
            args=dict(
                variant=variant,
                db=db,
                table=table,
                ctx_size=ctx,
                local_ctx_size=lcs,
                bfs_width=bw,
                prefer_latest=pl,
                n_seeds=16,
                items_per_task=8192,
                split="test",
                pre_dir=PRE_DIR,
                out_dir=out_dir,
                num_walks=10_000,
                walk_length=20,
                shuffle_seed=0,
                context_seed=0,
                tokens_per_gpu=2**18,
                num_workers=8,
                prefetch_factor=2,
                mmap_populate=True,
                db_cutoff=None,
                ckpt_clf=CKPT_CLF,
                ckpt_reg=CKPT_REG,
            ),
            resources=resources(db, qos),
            name=f"ens-{variant}-{db}-{table}",
            repo_root=str(REPO_ROOT),
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=LOG_ROOT,
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
        )


if __name__ == "__main__":
    submit_variant("default", "il-lo")
    # After ../tune/tuned_configs.json is committed:
    # submit_variant("tuned", "il-lo")
