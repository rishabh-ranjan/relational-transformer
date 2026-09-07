import dataclasses
from pathlib import Path

from expts.pretrain.submit_marlowe import args
from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, project
from roach.slurm import submit
from roach.slurm.clusters import ilc

cluster = ilc.ILC
resources = dataclasses.replace(ilc.AMPERE_LO, nodes=2)

submit(
    "rt.train:main",
    args=args()
    | dict(run_name="base-rtj", db_task_list="expts/repaper/pretrain_abl/rt-j.json")
    | dict(
        stage_dir=None, tokens_per_gpu=2**17, num_workers=14, run_name="base-rtj-ilc"
    )
    # | dict(
    #     run_name="mask0-rtj",
    #     mask_prob_max=0.0,
    #     db_task_list="expts/repaper/pretrain_abl/rt-j.json",
    # )
    # | dict(run_name="base")
    # | dict(run_name="mask0", mask_prob_max=0.0)
    # | dict(run_name="mask25", mask_prob_max=0.25)
    # | dict(run_name="mask75", mask_prob_max=0.75)
    # | dict(
    #     run_name="mix-forecast",
    #     db_task_list="expts/repaper/pretrain_abl/cutoff-forecast.json",
    # )
    # | dict(
    #     run_name="mix-autocomplete",
    #     db_task_list="expts/repaper/pretrain_abl/cutoff-autocomplete.json",
    # )
    | dict(
        early_stop_after_steps=10_000,
        keep_all_ckpts=False,
        project=project("pretrain-abl"),
    ),
    resources=resources,
    name="pretrain-abl",
    run_id=None,
    inside=None,
    repo_root=str(Path(__file__).resolve().parents[3]),
    cluster=cluster,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/pretrain_abl/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
