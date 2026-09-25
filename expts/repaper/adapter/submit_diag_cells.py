import json
import sys
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, OUT_ROOT, PRE_DIR, SECRETS_DIR

REPO_ROOT = str(Path(__file__).resolve().parents[3])

POOL = "pool-v4-fp32-half-ctxnorm"
TASKS = [("rel-event", "user-ignore")]
if len(sys.argv) > 1:
    TASKS = [tuple(a.split("/")) for a in sys.argv[1:]]

for db, table in TASKS:
    submit(
        "expts.repaper.adapter.diag_pool:dump_cells",
        args=dict(
            db=db,
            table=table,
            pre_dir=PRE_DIR,
            ckpt=CKPT,
            pool_ckpt=f"{OUT_ROOT}/adapter/{POOL}/pool_final.pt",
            out_dir=f"{OUT_ROOT}/adapter-diag/{POOL}",
            n_rows=2**10,
            seed=0,
            compile=True,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time="2:00:00",
            gpus="a100:1",
            cpus_per_task=8,
            ntasks=None,
            exclusive=False,
            mem="120G" if db in ("rel-amazon", "rel-hm", "rel-stack", "rel-event") else "64G",
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4,ampere7",
        ),
        name=f"adapter-cells-{POOL}-{db}-{table}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
