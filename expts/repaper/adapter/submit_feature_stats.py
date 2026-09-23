from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

# The frozen standardiser the v3 adapter opens with. One pass over the 34 GiB
# dump, accumulating per-task moments and combining them under the training
# mixture, so (x - mu) / sigma standardises the exact distribution the
# optimiser draws from.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.feature_stats:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12_prop",
        out_npz=f"{SHARE}/feature_stats_join_u12_prop.npz",
        d_feat=512,
        # The training pool's filter, so the statistics describe the tasks the
        # adapter actually sees and nothing else.
        min_rows=512,
        chunk=8192,
        workers=24,
    ),
    resources=Resources(
        partition="il-cpu",
        account="infolab",
        qos="il-cpu",
        time="4:00:00",
        gpus="0",
        cpus_per_task=24,
        ntasks=1,
        exclusive=False,
        mem="120G",
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name="adapter-feature-stats-prop",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
