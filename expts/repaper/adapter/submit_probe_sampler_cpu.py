from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, PRE_DIR, SECRETS_DIR, SHARE

REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_sampler_cpu:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        join_pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        relbench_pre_dir=PRE_DIR,
        db_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
        ckpt=CKPT,
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-sampler-cpu",
        n_sample_tasks=200,
        rows_per_sample=64,
        populate_dbs=40,
        max_val_rows=4096,
        seed=0,
    ),
    resources=Resources(
        partition="il-cpu",
        account="infolab",
        qos="il-cpu",
        time="4:00:00",
        gpus="0",
        cpus_per_task=32,
        ntasks=1,
        exclusive=False,
        mem="300G",
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name="adapter-probe-sampler-cpu",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
