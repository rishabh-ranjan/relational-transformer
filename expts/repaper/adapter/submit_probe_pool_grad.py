from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, OUT_ROOT, SECRETS_DIR, SHARE

REPO_ROOT = str(Path(__file__).resolve().parents[3])

MODE = "snr"

if MODE in ("ckpt", "fp"):
    target = "expts.repaper.adapter.probe_pool_grad:main"
    args = dict(
        join_pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        task_list="pool_tasks.json",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        ckpt_dir=f"{OUT_ROOT}/adapter/pool-v1",
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-pool-grad" + ("-fp" if MODE == "fp" else ""),
        steps=[100, 0, 62, 48, 46, 38, 127, 11, 56, 1, 16, 128, 24, 8, 50, 12],
        n_ctx=2**16,
        n_query=2**14,
        n_queries=64,
        head_chunk=2048,
        n_tasks_total=2608,
        jitter_frac=0.05,
        winsor_q=0.01,
        seed=0,
        n_fp_seeds=6 if MODE == "fp" else 0,
    )
elif MODE == "snr":
    target = "expts.repaper.adapter.probe_pool_grad:snr"
    args = dict(
        join_pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        task_list="pool_tasks.json",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        ckpt_dir=f"{OUT_ROOT}/adapter/pool-v1",
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-pool-grad-snr",
        steps=[100, 0, 62, 48, 46, 38, 127, 11, 56, 1, 16, 128, 24, 8, 50, 12],
        n_ctx=2**16,
        n_query=2**14,
        n_queries=64,
        head_chunk=2048,
        n_tasks_total=2608,
        sub_ctx=2**14,
        sub_q=2**12,
        seed=0,
    )
else:
    target = "expts.repaper.adapter.probe_pool_grad:replay"
    args = dict(
        join_pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        task_list="pool_tasks.json",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        ckpt_dir=f"{OUT_ROOT}/adapter/pool-v1",
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-pool-grad-replay",
        probe_steps=[0, 1, 8, 11, 12, 16, 24, 38, 46, 48, 50, 56, 62, 100, 127, 128],
        last_step=128,
        n_ctx=2**16,
        n_query=2**14,
        n_queries=64,
        head_chunk=2048,
        n_tasks_total=2608,
        lr=3e-4,
        lr_min=1e-5,
        warmup_steps=100,
        total_steps=2500,
        grad_norm_max=10.0,
        jitter_frac=0.05,
        winsor_q=0.01,
        seed=0,
    )

submit(
    target,
    args=args,
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="6:00:00",
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
    name=f"adapter-probe-pool-grad-{MODE}",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
