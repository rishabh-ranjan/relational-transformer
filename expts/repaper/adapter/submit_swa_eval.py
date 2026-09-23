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
)

# The per-step gradient sits at SNR ~0.95 even at 512 independent task draws,
# so the iterate is wandering in a noise ball: v7 moved dist_from_init to 1.156
# with no significant change in training loss and val auroc drifting -0.0020.
# Polyak-Ruppert averaging is the standard answer to exactly that, and it can
# be tested retrospectively on a run that is still going. If the averaged
# weights beat every individual checkpoint there is signal under the noise; if
# they do not, there probably is not.
#
# The averaging scheme is rt-j's own, rt.train.swa.SwaState, not a scheme
# invented here -- so a positive answer transfers straight into the training
# loop. Pretraining runs it at swa_momentum=0.9995 per optimiser step
# (pretrain/submit_ilc.py:57); these checkpoints are 200 steps apart, so the
# horizon-matched per-checkpoint momentum is 0.9995**200 = 0.90483 and the
# script derives it from ckpt_every rather than taking it as a constant.
REPO_ROOT = str(Path(__file__).resolve().parents[3])
RUN = "join-v8-ddp-proj64-cosine"

submit(
    "expts.repaper.adapter.swa_eval:main",
    args=dict(
        # Read-only. This is a live run (job 192079) and nothing here writes
        # into its output directory.
        ckpt_dir=f"{OUT_ROOT}/adapter/{RUN}",
        features_root=f"{SHARE}/features_join_u12",
        tabpfn_dir=f"{SHARE}/tabpfn",
        relbench_pre_dir=PRE_DIR,
        relbench_features_root=f"{SHARE}/features_rt-j",
        relbench_labels_root=f"{SHARE}/labels_relbench",
        # The training run's own val monitor, to the argument: fp32 ests,
        # 8192 ctx, 4096 val query, seed 0. Numbers are then directly
        # comparable with the run's val/auroc and val/nmae.
        relbench_n_ctx=8192,
        relbench_n_query=4096,
        seed=0,
        swa_momentum_per_step=0.9995,
        ckpt_every=200,
        # The ema is worth reading against the iterate at the same horizon,
        # not only at the end, and five snapshots cost five evaluations.
        ema_snapshot_steps=[2000, 4000, 6000, 8000],
        # eval_checkpoints.py's fixed training-distribution sample, same seed
        # and same draw order, so these numbers sit alongside its probe_curve.
        probe_n_tasks=96,
        probe_n_ctx=1024,
        probe_n_query=256,
        probe_min_rows=512,
        probe_seed=1234,
        d_feat=512,
        out_json=f"{LOG_ROOT}/repaper/adapter/swa-eval-v8/results.json",
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
    name=f"adapter-swa-eval-{RUN}",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
