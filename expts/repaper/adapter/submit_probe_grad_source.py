from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

# Where does the non-finite gradient come from? Three hypotheses have been
# wrong -- labels, degenerate feature dims, fp16 -- so this stops guessing and
# asks autograd, under both fp32 and autocast, on the tasks the training run
# named at step 0.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_grad_source:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        tabpfn_dir=f"{SHARE}/tabpfn",
        tasks=[
            # one that NaN'd and one that did not, so the forward traceback
            # can be read against a working case
            ("join-se-writing", "user-new-badge-adoption"),
            ("join-nhtsa-fars", "vehicles-first_mo"),
        ],
        n_ctx=512,
        n_query=256,
        d_feat=512,
        seed=0,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il",
        time="1:00:00",
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
    name="adapter-probe-grad-source",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
