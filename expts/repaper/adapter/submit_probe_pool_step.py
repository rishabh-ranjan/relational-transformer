from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_pool_step:main",
    args=dict(
        pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-pool-step-n262144",
        db="join-met-museum",
        task="objects-is_public_domain",
        n_rows=2**18,
        local_ctx_size=256,
        embed_chunk=1024,
        head_chunk=2048,
        stats_rows=4096,
        n_queries=64,
        d_out=64,
        check_rows=2048,
        check_ctx=1536,
        check_chunk=512,
        tabpfn_modes=["fp32", "autocast_bf16"],
        tabpfn_rows=[2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18],
        lr=1e-3,
        seed=0,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="3:00:00",
        gpus="b200:2",
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
    name="adapter-probe-pool-step-n262144",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
