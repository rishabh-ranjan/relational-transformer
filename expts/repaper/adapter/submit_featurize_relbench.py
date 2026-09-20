import json
import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT,
    CLONE_ROOT,
    LOG_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)

# The validation side of phase 1: the same raw unit-12 target-cell embedding as
# the Join dump, over RelBench's 21 forecast tasks.
#
# It has to be its own dump. features_rt-j is post-norm_out, and an RMSNorm's
# per-row rescaling is not something a linear adapter can undo, so it is not the
# quantity the Join dump holds; the rt_taps dump does hold raw unit 12 at the
# target slot, but only for 12 of the 21 tasks.
#
# row_rule="val+context" dumps every val row plus the exact 262,144-row draw
# GlobalContextRetriever makes at context_seed=0 -- 4.7M rows and 4.48 GiB,
# against 33.7 GiB for every row of every split. Test is not in it, so this dump
# cannot score the split we do not look at.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

# One job per db: the 21 tasks live in 7 of them, and a db is one mmap populate.
DBS = [
    "rel-amazon",
    "rel-stack",
    "rel-hm",
    "rel-trial",
    "rel-avito",
    "rel-event",
    "rel-f1",
]


def rows_for(db: str) -> int:
    pre = Path(PRE_DIR).expanduser()
    names = {
        n
        for d, n in json.loads((pre / "db-task-lists/forecast.json").read_text())
        if d == db
    }
    info = json.loads((pre / db / "table_info.json").read_text())
    return sum(
        info.get(f"{n}:Val", {}).get("num_nodes", 0)
        + min(info.get(f"{n}:Train", {}).get("num_nodes", 0), 262144)
        for n in names
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

# Read fresh every submission. 2026-09-20: the Join sweep is taking the whole
# b200 budget and il's 10, so these seven go to il-lo on ampere. They are half
# an hour each and the dump skips a task whose files exist, so a preemption
# costs the task in flight and nothing else.
for db in DBS:
    name = f"adapter-featurize-relbench-{db}"
    if name in busy:
        continue
    rows = rows_for(db)
    print(f"  {db}: {rows / 1e6:.2f}M rows")
    submit(
        "expts.repaper.adapter.featurize_target_cells:featurize_dbs",
        args=dict(
            dbs=[db],
            db_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features_relbench_u12",
            ckpt=CKPT,
            row_rule="val+context",
            # Identical to the Join dump, or the two corpora's embeddings are
            # not the same quantity and the adapter does not transfer.
            tap_unit=12,
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            batch_size=1024,
            min_rows=1,
            max_rows=0,
            context_rows=262144,
            expected_gib=rows * 1024 / 2**30 * 1.2,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time="12:00:00",
            gpus="a100:1",
            cpus_per_task=16,
            ntasks=None,
            exclusive=False,
            mem="120G",
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4,ampere6,ampere7,ampere9",
        ),
        name=name,
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
