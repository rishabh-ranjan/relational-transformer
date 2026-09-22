from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

REPO_ROOT = str(Path(__file__).resolve().parents[3])

# il-lo: at submission (2026-09-22) the ten il gpus were all held by v10, v11
# and the entity-exaone round, and ampere8 had one a100 free. The probe is
# ~30 min and writes its report as it goes, so a preemption costs little.
submit(
    "expts.repaper.adapter.probe_pool_throughput:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        ckpt=CKPT,
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-pool-overlap",
        n_tasks=32,
        rows_per_task=1024,
        chunk_sizes=[256, 512, 1024, 2048],
        n_overlap_workers=[1, 2],
        overlap_chunk=1024,
        overlap_only=True,
        seed=0,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="3:00:00",
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
        exclude="ampere4,ampere6,ampere7,ampere9",
    ),
    name="adapter-probe-pool-throughput",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
