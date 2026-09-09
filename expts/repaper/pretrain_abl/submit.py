import dataclasses
from pathlib import Path

from expts.pretrain.submit_marlowe import args
from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR, project
from roach.slurm import submit
from roach.slurm.clusters import ilc

# cluster = ilc.ILC
#
# for arm, resources in [
#     (
#         dict(run_name="base-rtj"),
#         dataclasses.replace(ilc.AMPERE, nodes=1),
#     ),
#     (
#         dict(run_name="mask0-rtj", mask_prob_max=0.0),
#         dataclasses.replace(ilc.AMPERE_LO, nodes=1),
#     ),
#     (
#         dict(run_name="mask25-rtj", mask_prob_max=0.25, tokens_per_gpu=2**18),
#         dataclasses.replace(
#             ilc.BLACKWELL,
#             nodes=1,
#             gpus="b200:2",
#             qos="il",
#             time="7-00:00:00",
#             mem="1500000M",
#         ),
#     ),
# ]:

cluster = ilc.ILC
# resources = dataclasses.replace(ilc.AMPERE, nodes=1)
resources = dataclasses.replace(
    ilc.BLACKWELL, nodes=1, gpus="b200:2", qos="il", time="7-00:00:00", mem="1500000M"
)
arm = dict(
    run_name="mask0-plurel-filt",
    mask_prob_max=0.0,
    db_task_list="~/scratch/hf/stanford-star/plurel-preprocessed/db-task-lists/rt-plurel-train.json",
    pre_dir="~/scratch/hf/stanford-star/plurel-preprocessed",
    stage_dir=None,
    tokens_per_gpu=2**18,
    num_workers=resources.cpus_per_task,
    ctx_size_list=[1024, 2048, 4096, 8192],
    local_ctx_size_list=[512, 1024, 2048],
    bfs_width_list=[16, 32, 64, 128],
    prefer_latest_list=[False],
    early_stop_after_steps=None,
)
submit(
    "rt.train:main",
    args=args()
    # | dict(
    #     db_task_list="expts/repaper/pretrain_abl/rt-j.json",
    #     stage_dir=None,
    #     tokens_per_gpu=2**18,
    #     num_workers=resources.cpus_per_task,
    # )
    | arm
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
        keep_all_ckpts=False,
        project=project("pretrain-abl"),
    ),
    resources=resources,
    name=arm["run_name"],
    run_id="26-09-08_18-19-07_189275282",
    inside=None,
    repo_root=str(Path(__file__).resolve().parents[3]),
    cluster=cluster,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/pretrain_abl/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
