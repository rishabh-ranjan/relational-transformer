from pathlib import Path

from expts.pretrain.submit_marlowe import args, cluster, resources
from expts.repaper.config import project
from roach.slurm import submit

ARMS = {
    "mask0": dict(mask_prob_max=0.0),
    "mask25": dict(mask_prob_max=0.25),
    "mask75": dict(mask_prob_max=0.75),
    "mix-forecast": dict(
        db_task_list="expts/repaper/pretrain_abl/cutoff-forecast.json"
    ),
    "mix-autocomplete": dict(
        db_task_list="expts/repaper/pretrain_abl/cutoff-autocomplete.json"
    ),
}

ARM = "mask0"
RUN_ID = None
INSIDE = None

submit(
    "rt.train:main",
    args=args()
    | ARMS[ARM]
    | dict(
        early_stop_after_steps=10_000,
        keep_all_ckpts=False,
        project=project("pretrain-abl"),
        run_name=ARM,
    ),
    resources=resources,
    name=f"pabl-{ARM}",
    run_id=RUN_ID,
    inside=INSIDE,
    repo_root=str(Path(__file__).resolve().parents[3]),
    cluster=cluster,
    job_env="expts/job_env.sh",
    log_root="~/scratch/relational-transformer/repaper/pretrain_abl/slurm-logs",
    clone_root="~/roach_clones",
    secrets_dir="~/scratch/.secrets",
)
