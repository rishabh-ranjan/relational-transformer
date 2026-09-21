from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
    project,
)

# Phase 2: train the linear adapter between a frozen RT-J and a frozen TabPFN.
# Pure tensors off the phase-1 dump -- no rustler, no RT-J, one gpu.
#
# A step is tasks_per_step tasks accumulated, each drawn with
# P prop-to min(rows, 100_000): the mixture rt-j pretraining uses, taken off the
# true row count rather than the 4096 we capped the dump at, or the cap would
# flatten it to nearly uniform over tasks.
#
# n_ctx tops out at 2048 because the probe OOMs an 80G a100 above it with
# gradients on (2048x256 peaked at 33.3 GiB, 8192x256 died).
REPO_ROOT = str(Path(__file__).resolve().parents[3])
RUN = "join-v1"

submit(
    "expts.repaper.adapter.train_adapter:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        relbench_pre_dir=PRE_DIR,
        relbench_features_root=f"{SHARE}/features_rt-j",
        relbench_labels_root=f"{SHARE}/labels_relbench",
        # 8192 context rows for the in-loop metric; the dump also holds the
        # 2**18 draw, so checking that it generalises costs no refeaturizing.
        relbench_n_ctx=8192,
        # 1024 was sized off the probe's OOM at 8192x256, which was measured
        # with gradients on and does not apply to a no_grad eval. Query rows
        # attend to the context and not to each other, so this axis is linear
        # and chunkable; the context is the expensive one. 4096 roughly halves
        # the per-task standard error.
        relbench_n_query=4096,
        tabpfn_dir=f"{SHARE}/tabpfn",
        out_dir=f"{OUT_ROOT}/adapter/{RUN}",
        d_feat=512,
        min_rows=512,
        n_ctx_lo=256,
        # 2048 came from a probe measured under fp16 autocast; forcing fp32
        # (see train_adapter.tabpfn) roughly doubles activation memory, and
        # 2048x256 already peaked at 33 GiB of an 80 GiB card.
        n_ctx_hi=1024,
        n_query=256,
        tasks_per_step=4,
        total_steps=10_000,
        lr=1e-4,
        # Zero, deliberately. AdamW's decay pulls a weight toward 0, and this
        # weight starts at I -- decaying it is decaying the rt-j featurizer
        # away, not regularising the adapter toward it.
        wd=0.0,
        warmup_steps=100,
        grad_norm_max=1.0,
        # Step 0 is evaluated too, and at identity that is the un-adapted rt-j
        # featurizer: the control measured by the same code, for free.
        eval_every=250,
        save_every=500,
        seed=0,
        # The rt-j featurizer without an adapter, on the same 21 val tasks:
        # every val panel folds this in as its reference line.
        targets={"val/auroc": 0.7173, "val/nmae": 0.3584},
        run_name=f"adapter-{RUN}",
        project=project("adapter"),
        entity="rtv2",
        wandb_disabled=False,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il",
        time="12:00:00",
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
    name=f"adapter-train-{RUN}",
    # roach mints one and injects it into args; paste one here to relaunch an
    # existing run into the same wandb group.
    run_id=None,
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
