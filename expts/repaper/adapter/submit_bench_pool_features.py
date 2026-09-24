from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, PRE_DIR, SECRETS_DIR, SHARE

REPO_ROOT = str(Path(__file__).resolve().parents[3])

RUN = "bench-pool-features-smoke"
# RUN = "bench-pool-features"

submit(
    "expts.repaper.adapter.bench_pool_features:main",
    args=dict(
        relbench_pre_dir=PRE_DIR,
        relbench_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        out_dir=f"{LOG_ROOT}/repaper/adapter/{RUN}",
        relbench_n_ctx=2**16,
        relbench_n_query=2**14,
        modes=["target", "union"],
        tasks=["rel-event/user-repeat", "rel-amazon/item-churn"],
        # tasks=None,
        seed=0,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="3:00:00",
        # time="1-00:00:00",
        gpus="b200:1",
        cpus_per_task=16,
        ntasks=1,
        exclusive=False,
        mem="300G",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name=f"adapter-{RUN}",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
