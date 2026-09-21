from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

# The v2 run clips on 100% of steps: pre-clip norm 9.4-48.4 against
# grad_norm_max=1.0. This asks whether 20 is what the objective weighs -- rows
# accumulated into one outer-product sum -- or a blow-up in a particular layer,
# by replaying the real sampler and tracing every module of the tabpfn forward.
REPO_ROOT = str(Path(__file__).resolve().parents[3])

submit(
    "expts.repaper.adapter.probe_grad_scale:main",
    args=dict(
        features_root=f"{SHARE}/features_join_u12",
        tabpfn_dir=f"{SHARE}/tabpfn",
        d_feat=512,
        # The sampler arguments have to match submit_train_adapter exactly or
        # the batches replayed here are not the batches that clipped.
        min_rows=512,
        n_ctx_lo=256,
        n_ctx_hi=1024,
        n_query=256,
        tasks_per_step=4,
        n_steps=12,
        sweep_tasks=[
            ("join-nhtsa-fars", "vehicles-first_mo"),
            ("join-se-writing", "user-new-badge-adoption"),
        ],
        # 2048 is the run's old n_ctx_hi and the largest the dump's 4096 rows
        # allow beside a 512-row query block.
        sweep_n_ctx=[256, 512, 1024, 2048],
        sweep_n_query=[64, 128, 256, 512],
        trace_n_ctx=512,
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
    name="adapter-probe-grad-scale",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
