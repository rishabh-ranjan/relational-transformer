from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

# Before any of the adapter work is worth doing, three things have to be true
# of TabPFN's differentiable-input path, and none of them is documented:
#
# 1. a gradient reaches the adapter from both the context and the query
#    features, and no TabPFN parameter collects one -- we tune the adapter, not
#    the predictor;
# 2. an adapter placed in front of it can actually learn something, on a
#    synthetic task where the identity map is a bad featurizer;
# 3. what a step costs in seconds and GiB at the context sizes the training
#    loop would use. This is the number that sets the whole training budget and
#    it is the one thing the plan does not have.
#
# One a100: blackwell1 is fully allocated (8/8 b200) as of 2026-09-20 16:0x, so
# il-interactive would buy nothing, and ampere2 has a free card that il takes
# at once. Nothing else of mine holds a gpu.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_tabpfn_grad:main",
    args=dict(
        tabpfn_dir=f"{SHARE}/tabpfn",
        d_feat=512,
        shapes=[(512, 128), (2048, 256), (8192, 256)],
        sanity_steps=60,
        sanity_shape=(512, 128),
        sanity_signal_dims=8,
        lr=1e-3,
        seed=0,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il",
        time="2:00:00",
        gpus="a100:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem="64G",
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="ampere4",
    ),
    name="adapter-probe-tabpfn-grad",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
