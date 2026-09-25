from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, OUT_ROOT, SECRETS_DIR, SHARE

REPO_ROOT = str(Path(__file__).resolve().parents[3])

RUN = "pool-v10-fp32-half-ctxnorm-colnorm-signsoftmax-t10"

submit(
    "expts.repaper.adapter.probe_pool_v10_spikes:main",
    args=dict(
        join_pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        task_list="pool_tasks.json",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        ckpt_dir=f"{OUT_ROOT}/adapter/{RUN}",
        out_dir=f"{LOG_ROOT}/repaper/adapter/probe-pool-v10-swiglu",
        spikes=[
            dict(step=32, task="join-open-power-system/weather_observations-pl_radiation_direct_horizontal", ckpt_step=50, logged_grad_norm=1.64, logged_loss=-2.6476),
            dict(step=101, task="join-reddit-hyperlinks-body/hyperlinks-liwc_sad", ckpt_step=100, logged_grad_norm=5.495, logged_loss=-2.7820),
            dict(step=142, task="join-se-workplace/post-long-horizon-vote-churn", ckpt_step=150, logged_grad_norm=0.9378, logged_loss=0.3239),
            dict(step=226, task="join-nhtsa-fars/vehicles-l_restri", ckpt_step=250, logged_grad_norm=3.345, logged_loss=-5.0551),
            dict(step=243, task="join-act-mooc/interactions-label", ckpt_step=250, logged_grad_norm=15.24, logged_loss=-5.2483),
            dict(step=256, task="join-nhtsa-fars/vehicles-haz_plac", ckpt_step=250, logged_grad_norm=3874.0, logged_loss=-4.7993),
            dict(step=38, task="join-se-workplace/users-down_votes", ckpt_step=50, logged_grad_norm=621.5, logged_loss=-4.6839),
            dict(step=48, task="join-bird-social-media/twitter-retweet_count", ckpt_step=50, logged_grad_norm=581.9, logged_loss=-3.2788),
            dict(step=128, task="join-se-raspberrypi/users-down_votes", ckpt_step=150, logged_grad_norm=265.0, logged_loss=-4.9007),
            dict(
                step=216,
                task="join-se-boardgames/user-post-score-mean-heavytail",
                ckpt_step=200,
                logged_grad_norm=190.1,
                logged_loss=-5.1934,
            ),
        ],
        n_ctx=2**15,
        n_query=2**13,
        n_queries=64,
        head_chunk=2048,
        n_tasks_total=2608,
        seed=0,
        dump=True,
        cell_dump=False,
        colnorm_tau=None,
        live_scale=False,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="1:00:00",
        gpus="b200:1",
        cpus_per_task=4,
        ntasks=1,
        exclusive=False,
        mem="120G",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name="adapter-probe-pool-v10-swiglu",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
