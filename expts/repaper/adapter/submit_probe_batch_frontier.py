from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

# The adapter loop runs one task draw per tabpfn forward, at 0.558 s/draw and
# 5.71 TFLOP/s -- 29% of the a100's fp32 peak, with attention shaped
# (1, 554, 16, 64) and 58 separate sdpa calls per forward. bf16 measured 0.98x
# at batch 1, which says the loop is shape-limited, not arithmetic-limited, so
# the only lever left is a bigger batch. fit_with_differentiable_input is
# batch-1 only; fit_from_preprocessed builds InferenceEngineBatchedNoPrepro-
# cessing with inference_mode=not differentiable_input, so gradients still
# flow and the dataset batch is the model's own batch dim.
#
# This measures the frontier: peak memory and s/draw over n_ctx x B x dtype x
# activation checkpointing, and -- first, because it gates everything -- whether
# the batched engine computes the same loss at B=1. It cannot: the batched
# engine skips preprocessing, and the live path adds a fingerprint column
# (FINGERPRINT_FEATURE, a per-row cpu hash). So the comparison is run three
# ways: legacy with the fingerprint (what trains today), legacy without it, and
# batched (which cannot have it). If the middle and the last agree, the
# fingerprint is the whole difference and it is a config flag.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_batch_frontier:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        tabpfn_dir=f"{SHARE}/tabpfn",
        stats_path=f"{SHARE}/feature_stats_join_u12.npz",
        d_feat=512,
        min_rows=512,
        n_query=256,
        # The v6 ladder's rungs, so the answer maps straight onto the run.
        sweep_n_ctx=[256, 512, 1024],
        sweep_batch=[1, 2, 4, 8, 16, 32],
        sweep_precision=["fp32", "bf16"],
        # False first: without checkpointing the frontier is low and the arm
        # OOMs out early, so it is nearly free and it is the baseline the
        # checkpointed arm is read against.
        sweep_recompute=[False, True],
        n_tasks=8,
        warmup=2,
        iters=5,
        # ~23 min of sweep inside a 40 min job, leaving room for the clone, the
        # env, five checkpoint loads and the table. Every row prints as it is
        # measured, so a job cut short still reports what it got.
        budget_s=1400,
        # The v6 arm projects to 64 dims, which shrinks the feature axis 8x and
        # moves the frontier with it. Measured last, only if the budget is left.
        probe_d_out=64,
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
        exclude="ampere4,ampere6,ampere7,ampere9",
    ),
    name="adapter-probe-batch-frontier",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
