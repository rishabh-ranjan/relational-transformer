from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_rtj_forward_profile:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        ckpt=CKPT,
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-rtj-forward-profile",
        n_tasks=8,
        rows_per_task=512,
        batch_size=1024,
        reps=5,
        train_batch_size=512,
        seed=0,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il",
        time="1:30:00",
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem="200G",
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="ampere4",
    ),
    name="adapter-probe-rtj-forward-profile",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
