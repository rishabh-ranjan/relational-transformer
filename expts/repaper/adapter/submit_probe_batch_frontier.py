from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

# Run 1 (job 191915, d_out=512) answered the gate and then said the batch is
# not the lever at that width: fit_from_preprocessed DOES take the
# differentiable path and produce a finite adapter gradient, per-draw losses
# inside a B=2 batch match their B=1 values to 1e-7, but s/draw went
# 0.365 -> 0.348 -> 0.335 from B=1 to B=4 at n_ctx=256, i.e. 9%, and B=8 OOM'd
# at 75.6 GiB. Forced bf16 (inference_precision, not autocast) was worth far
# more on its own: 2.0-2.4x at B=1 across the ladder.
#
# Run 2 is the same probe at the width the v6 arm actually trains, d_out=64.
# TabPFN is cell-level, so the feature axis is ~8x narrower and the 51-63 GiB
# per draw of run 1 becomes ~6, which is what makes a real batch possible.
# fp32 first and bf16 second, because at 6 GiB memory is no longer the binding
# constraint and forcing bf16 costs the fingerprint feature. B runs to 32.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_batch_frontier:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        tabpfn_dir=f"{SHARE}/tabpfn",
        stats_path=f"{SHARE}/feature_stats_join_u12.npz",
        d_feat=512,
        # What v6 trains: the adapter projects to 64, so this is also the
        # feature count tabpfn sees (plus the fingerprint column).
        d_out=64,
        min_rows=512,
        n_query=256,
        # The v6 ladder's rungs, so the answer maps straight onto the run.
        sweep_n_ctx=[256, 512, 1024],
        sweep_batch=[1, 2, 4, 8, 12, 16, 24, 32],
        sweep_precision=["fp32", "bf16"],
        # False first: that is the arm a training loop would actually run, and
        # it is the baseline the checkpointed arm has to beat. save_peak_memory
        # _factor is deliberately left at the model default -- chunked_evaluate
        # asserts not x.requires_grad, so it cannot coexist with backprop.
        sweep_recompute=[False, True],
        n_tasks=8,
        warmup=2,
        iters=5,
        # ~23 min of sweep inside a 40 min job, leaving room for the clone, the
        # env, five checkpoint loads and the table. Every row prints as it is
        # measured, so a job cut short still reports what it got.
        budget_s=1400.0,
        seed=0,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il",
        time="40:00",
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem="120G",
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere7",
    ),
    name="adapter-probe-batch-frontier",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
