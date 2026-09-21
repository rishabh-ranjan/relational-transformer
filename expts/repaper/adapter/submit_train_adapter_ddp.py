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

# The batch, bought with hardware the way pretraining bought it. rt-j averaged
# total_bs=1024 independently drawn items per step over 8 a100s under DDP; the
# single-gpu adapter runs averaged 4 task draws, and rows sharing a context are
# strongly correlated, so 4 x 256 query rows is 4 samples against between-task
# variance -- the noise that made v1's loss curve unreadable.
#
# 4 ranks x 32 accumulation micro-steps x 4 tasks = 512 tasks per optimiser
# step, 128x the single-gpu run.
#
# COST: ~0.55 s per task draw measured on v1/v3 (2.19-2.30 s/step at 4 tasks),
# so 128 tasks per rank per step is ~70 s/step. 10,000 steps would be ~195 h
# = 8.1 days, over il's 7-day cap, so this runs 2,500 -- still 1.28M task
# draws, 32x the 40k of the whole v1 run, and it completes rather than being
# cut off mid-schedule. 8 ranks would fit 10k in ~4.1 days if this one says a
# bigger batch is what was missing.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

RUN = "join-v4-ddp-linear"

submit(
    "expts.repaper.adapter.train_adapter_ddp:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        relbench_pre_dir=PRE_DIR,
        relbench_features_root=f"{SHARE}/features_rt-j",
        relbench_labels_root=f"{SHARE}/labels_relbench",
        relbench_n_ctx=8192,
        relbench_n_query=4096,
        tabpfn_dir=f"{SHARE}/tabpfn",
        stats_path=f"{SHARE}/feature_stats_join_u12.npz",
        out_dir=f"{OUT_ROOT}/adapter/{RUN}",
        d_feat=512,
        min_rows=512,
        n_ctx_lo=256,
        n_ctx_hi=1024,
        n_query=256,
        # 4 ranks x 4 tasks x 32 accum. The script asserts the divisibility, so
        # changing the rank count without changing this fails at startup rather
        # than silently running a different batch.
        total_tasks_per_step=512,
        tasks_per_micro=4,
        total_steps=2_500,
        lr=1e-4,
        # Zero, deliberately. AdamW's decay pulls a weight toward 0, and this
        # weight starts at I -- decaying it is decaying the rt-j featurizer
        # away, not regularising the adapter toward it.
        wd=0.0,
        warmup_steps=100,
        # 10.0, the middle arm of the single-gpu clip sweep. A 512-task
        # gradient should also be far steadier than a 4-task one, so this
        # ought to bind on a small minority of steps; train/frac_clipped says.
        grad_norm_max=10.0,
        adapter_kind="linear",
        hidden_dim=0,
        # At ~70 s/step these are ~30 min and ~1 h of wall clock, not the
        # 10 min and 20 min they were on one gpu.
        eval_every=25,
        save_every=50,
        seed=0,
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
        # ~49 h at 70 s/step; 72 leaves room for a slow node.
        time="3-00:00:00",
        # One rank per gpu: roach maps SLURM_PROCID -> RANK and SLURM_NTASKS ->
        # WORLD_SIZE (roach/slurm/run.py:19), so ntasks=None gives 4 ranks.
        gpus="a100:4",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem="240G",
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere6,ampere7,ampere9",
    ),
    name=f"adapter-train-{RUN}",
    run_id=None,
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
