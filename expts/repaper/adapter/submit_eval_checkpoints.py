from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    SECRETS_DIR,
    SHARE,
)

# Did the run learn its own objective at all? The per-window training loss cannot
# say: each window averages a different random sample of tasks, and that
# between-task variance swamps any plausible gain (the fitted slope over 10k
# steps was +0.029 +- 0.046 for clf, -0.054 +- 0.246 for reg -- both
# indistinguishable from zero). This scores a *fixed* sample of the training
# distribution -- same tasks, same context and query rows, same TabPFN config
# -- across every saved checkpoint, so the only thing that varies is the
# adapter. Identity is included as the reference the run started from.
#
# If the loss falls across checkpoints, the machinery works and v1 is a
# generalisation result. If it is flat, v1 never learned and the negative
# result says nothing about the method.
REPO_ROOT = str(Path(__file__).resolve().parents[3])
RUN = "join-v3-linear"

submit(
    "expts.repaper.adapter.eval_checkpoints:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        tabpfn_dir=f"{SHARE}/tabpfn",
        ckpt_dir=f"{OUT_ROOT}/adapter/{RUN}",
        out_json=f"{OUT_ROOT}/adapter/{RUN}/probe_curve.json",
        # 96 tasks is ~24 optimiser steps' worth of draws, enough that the
        # mean is not dominated by one hard task, and 21 checkpoints x 96
        # no-grad forwards is minutes rather than an hour.
        n_tasks=96,
        n_ctx=1024,
        n_query=256,
        d_feat=512,
        min_rows=512,
        # Not the training seed: these draws should not coincide with any
        # particular step's, though with replacement over 10k steps some
        # overlap is unavoidable and wanted -- this is the training
        # distribution, not a holdout.
        seed=1234,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il",
        time="2:00:00",
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
        exclude="ampere4",
    ),
    name=f"adapter-eval-ckpts-{RUN}",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
