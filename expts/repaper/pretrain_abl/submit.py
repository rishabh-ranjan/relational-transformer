from pathlib import Path

from expts.pretrain.submit_marlowe import args, cluster, resources
from expts.repaper.config import project
from roach.slurm import submit

submit(
    "rt.train:main",
    args=args()
    # | dict(run_name="mask0", mask_prob_max=0.0)
    # | dict(run_name="mask25", mask_prob_max=0.25)
    | dict(run_name="mask75", mask_prob_max=0.75)
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
    inside=447124,
    repo_root=str(Path(__file__).resolve().parents[3]),
    cluster=cluster,
    job_env="expts/job_env.sh",
    log_root="~/scratch/relational-transformer/repaper/pretrain_abl/slurm-logs",
    clone_root="~/roach_clones",
    secrets_dir="~/scratch/.secrets",
)
