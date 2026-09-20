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

# Two checks before spending a gpu pass on the multi-tap, multi-cell dump:
#
# 1. Does the seed row's union of columns -- its own cells plus the cells of
#    the entity row its f2p link reaches -- actually appear in a 256-cell
#    sampled context? A column the sampler truncates away is a column the dump
#    would store as absent, and if coverage is poor then local_ctx_size or
#    bfs_width has to move before the sizing means anything.
# 2. Does norm_out over the block-12 hook reproduce forward()'s output, and
#    does its target cell reproduce the released features_rt-j blob? That
#    validates the hook, the canonical sort and the cell selection against a
#    known artifact.
#
# Three tasks spanning the union range: 8 columns, 29 columns (the widest), and
# the one task whose table carries two foreign keys rather than one.
DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
REPO_ROOT = str(Path(__file__).resolve().parents[4])

for db, table in [
    ("rel-f1", "driver-top3"),
    ("rel-trial", "study-outcome"),
    ("rel-event", "user-repeat"),
]:
    submit(
        "expts.repaper.baselines.probe_rt_taps:probe",
        args=dict(
            db=db,
            table=table,
            db_task_list=DB_TASK_LIST,
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features_rt-j",
            ckpt=CKPT,
            # Identical to the pass that built features_rt-j, or the
            # equivalence check compares two different things.
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            db_cutoff=None,
            batch_size=1024,
            max_rows=4096,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il",
            time="1:00:00",
            gpus="a100:1",
            cpus_per_task=8,
            ntasks=None,
            exclusive=False,
            mem="64G",
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4,ampere6,ampere7,ampere9",
        ),
        name=f"rel2tabv2-probe-taps-{db}-{table}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
