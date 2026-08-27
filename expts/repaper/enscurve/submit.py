import json
import os
import subprocess
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

TASKS = [
    tuple(p)
    for p in json.loads(
        (Path(PRE_DIR).expanduser() / "db-task-lists" / "forecast.json").read_text()
    )
]

SMALL = {
    ("rel-trial", "study-adverse"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-avito", "ad-ctr"),
    ("rel-trial", "study-outcome"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-dnf"),
    ("rel-event", "user-repeat"),
}


def cfg(variant: str, db: str, table: str) -> tuple[int, int, int, bool]:
    if variant == "default":
        return 8192, 256, 32, True
    cfgs = json.loads(
        (Path(__file__).parent.parent / "tune" / "tuned_configs.json").read_text()
    )
    ctx, lcs, bw, pl = cfgs[f"{db}/{table}"]["best_cfg"]
    return int(ctx), int(lcs), int(bw), bool(pl)


# 2026-08-27 00:30: the high tiers are spent on scaling's full-test passes
# (scaling/submit.py); a curve resumes per seed, so it takes il-lo with a
# limit that backfills -- 16 seeds over 8192 rows ran ~1h05 on an a100 in the
# 2026-08-19 round whatever the database, under 20 min on the tasks with fewer
# test rows than that.
def resources(db: str, table: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="1:00:00" if (db, table) in SMALL else "3:00:00",
        gpus="a100:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem={
            "rel-amazon": "120G",
            "rel-avito": "48G",
            "rel-event": "48G",
            "rel-f1": "24G",
            "rel-hm": "64G",
            "rel-stack": "64G",
            "rel-trial": "48G",
        }[db],
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere7",
    )


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()

for variant in ["default", "tuned"]:
    for db, table in TASKS:
        ctx, lcs, bw, pl = cfg(variant, db, table)
        out_dir = f"{OUT_ROOT}/repaper-enscurve/{variant}"
        name = f"repaper-ens-{variant}-{db}-{table}"
        if (Path(out_dir).expanduser() / f"{db}__{table}.json").exists():
            continue
        if name in busy:
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
            resources=resources(db, table),
            name=name,
            repo_root=str(Path(__file__).resolve().parents[3]),
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/enscurve/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            setup=("pixi run maturin develop --uv --release --features vecdb",),
        )
