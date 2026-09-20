from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)

# features_rt-j is vectors and no labels, so the adapter's training loop has
# nothing to score a RelBench val metric against. This writes the labels once:
# every val row plus the exact context draw GlobalContextRetriever makes, so the
# 8192-row monitor and a later 2**18 run read the same file, and no test row is
# in it.
#
# Zero-gres, so il-cpu: no GPU is involved at ctx_size 1, and it keeps this off
# the cards the featurize sweep is holding. rambo has 288 cores and is set up.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.dump_relbench_labels:main",
    args=dict(
        db_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
        pre_dir=PRE_DIR,
        labels_root=f"{SHARE}/labels_relbench",
        context_rows=262144,
        context_seed=0,
        batch_size=8192,
    ),
    resources=Resources(
        partition="il-cpu",
        account="infolab",
        qos="il-cpu",
        time="12:00:00",
        gpus="0",
        cpus_per_task=64,
        ntasks=1,
        exclusive=False,
        mem="200G",
        mem_per_gpu=None,
        constraint=None,
        nodelist="rambo",
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name="adapter-dump-relbench-labels",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
