import json
import os
import subprocess
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT,
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
)
from roach.slurm import Resources, submit


def resources(db: str, table: str, rank: int) -> Resources:
    # 2026-09-17 10:25: the il tier carries the fine_tune sweep, so the ICL
    # rerun rides il-lo -- the rel-amazon item pieces on the free blackwell
    # cards, everything else on the plentiful a100s.
    if db == "rel-amazon" and table in ("item-churn", "item-ltv"):
        return Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time="2-00:00:00",
            gpus="b200:1",
            cpus_per_task=8,
            ntasks=None,
            exclusive=False,
            mem="120G",
            mem_per_gpu=None,
            constraint=None,
            nodelist="blackwell1",
            reservation=None,
            dependency=None,
        )
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="2-00:00:00",
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

ckpt = "~/scratch/hf/stanford-star/rt-plurel"
subdir = "repaper-submit-rt-plurel"
prefix = "subp"
# ckpt = CKPT
# subdir = "repaper-submit"
# prefix = "sub"

cfgs = json.loads(
    (Path(__file__).parent.parent / "tune" / "tuned_configs.json").read_text()
)
for task_key, rec in sorted(cfgs.items()):
    db, table = task_key.split("/")
    for rank, (ctx, lcs, bw, pl) in enumerate(rec["top_cfgs"]):
        out_dir = f"{OUT_ROOT}/{subdir}/cfg{rank}"
        if (Path(out_dir).expanduser() / f"{db}__{table}.json").exists():
            continue
        if f"{prefix}-cfg{rank}-{db}-{table}" in busy:
            continue
        submit(
            "expts.repaper.enscurve.run:main",
            args=dict(
                variant=f"cfg{rank}",
                db=db,
                table=table,
                ctx_size=int(ctx),
                local_ctx_size=int(lcs),
                bfs_width=int(bw),
                prefer_latest=bool(pl),
                n_seeds=4,
                items_per_task=10_000_000,
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
                ckpt=ckpt,
            ),
            resources=resources(db, table, rank),
            name=f"{prefix}-cfg{rank}-{db}-{table}",
            repo_root=str(Path(__file__).resolve().parents[3]),
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/submit/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
        )
