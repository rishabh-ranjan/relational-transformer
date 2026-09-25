import json
import sys
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, OUT_ROOT, PRE_DIR, SECRETS_DIR

REPO_ROOT = str(Path(__file__).resolve().parents[3])

# RUNS = [("pool-v4-fp32-half-ctxnorm", "pool_final", "pool-v4-fp32-half-ctxnorm")]
# RUNS = [
#     ("pool-v4-fp32-half-ctxnorm", "pool_step0", "pool-v4-fp32-half-ctxnorm-step0"),
#     ("pool-v8-fp32-half-ctxnorm-colnorm", "pool_step0", "pool-v8-fp32-half-ctxnorm-colnorm-step0"),
# ]
RUNS = [("pool-v10-fp32-half-ctxnorm-colnorm-signsoftmax-t10", "pool_step0", "pool-v10-fp32-half-ctxnorm-colnorm-signsoftmax-t10-step0")]
# TASKS = [tuple(p) for p in json.loads(Path(f"{PRE_DIR}/db-task-lists/forecast.json").expanduser().read_text())]
TASKS = [("rel-event", "user-ignore")]
if len(sys.argv) > 1:
    TASKS = [tuple(a.split("/")) for a in sys.argv[1:]]

for (pool, ckpt_name, diag_name), (db, table) in [(r, t) for r in RUNS for t in TASKS]:
    submit(
        "expts.repaper.adapter.diag_pool:main",
        args=dict(
            db=db,
            table=table,
            pre_dir=PRE_DIR,
            ckpt=CKPT,
            pool_ckpt=f"{OUT_ROOT}/adapter/{pool}/{ckpt_name}.pt",
            out_dir=f"{OUT_ROOT}/adapter-diag/{diag_name}",
            n_rows=2**10,
            n_samples=8,
            n_depth=4,
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
        name=f"adapter-diag-{diag_name}-{db}-{table}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
