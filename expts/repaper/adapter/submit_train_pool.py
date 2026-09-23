from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, OUT_ROOT, PRE_DIR, SECRETS_DIR, SHARE, project

REPO_ROOT = str(Path(__file__).resolve().parents[3])

RUN = "pool-v1-smoke"
# RUN = "pool-v1"

submit(
    "expts.repaper.adapter.train_pool:main",
    args=dict(
        join_pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        relbench_pre_dir=PRE_DIR,
        relbench_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
        task_list="pool_tasks.json",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        stats_path=f"{SHARE}/feature_stats_join_u12.npz",
        out_dir=f"{OUT_ROOT}/adapter/{RUN}",
        n_ctx=2**16,
        n_query=2**14,
        relbench_n_ctx=2**16,
        relbench_n_query=2**14,
        n_queries=64,
        head_chunk=2048,
        tabpfn_device="cuda:0",
        offload_cells=True,
        total_steps=5,
        # total_steps=2500,
        lr=3e-4,
        lr_min=1e-5,
        wd=0.0,
        warmup_steps=1,
        # warmup_steps=100,
        grad_norm_max=10.0,
        swa_momentum=0.9995,
        eval_every=2,
        # eval_every=250,
        save_every=2,
        # save_every=250,
        resume_save_mins=20.0,
        seed=0,
        run_name=f"adapter-{RUN}",
        project=project("adapter"),
        entity="rtv2",
        wandb_disabled=False,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="3:00:00",
        # time="5-00:00:00",
        gpus="b200:1",
        cpus_per_task=32,
        ntasks=1,
        exclusive=False,
        mem="400G",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name=f"adapter-{RUN}",
    run_id=None,
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
