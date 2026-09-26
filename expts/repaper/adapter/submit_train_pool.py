from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, OUT_ROOT, PRE_DIR, SECRETS_DIR, SHARE, project

REPO_ROOT = str(Path(__file__).resolve().parents[3])

# RUN = "pool-v1-smoke"
# RUN = "pool-v1"
# RUN = "pool-v2-fp32-half"
# RUN = "pool-v3-fp32-half-ln"
# RUN = "pool-v4-fp32-half-ctxnorm"
# RUN = "pool-v5-bf16-ctxnorm"
# RUN = "pool-v6-fp32-half-ctxnorm-forecast"
# RUN = "pool-v7-fp32-half-ctxnorm-relbench"
# RUN = "pool-v8-fp32-half-ctxnorm-colnorm"
# RUN = "pool-v9-fp32-half-ctxnorm-colnorm-signsoftmax"
# RUN = "pool-v10-fp32-half-ctxnorm-colnorm-signsoftmax-t10"
# RUN = "pool-v11-fp32-half-ctxnorm-colnorm-signsoftmax-t10-tanh3-livescale"
# RUN = "pool-v12-fp32-half-ctxnorm-colnorm-signsoftmax-t10-tanh3-livescale-dedup"
# RUN = "pool-v13-fp32-half-ctxnorm-colnorm-signsoftmax-t10-tanh3-livescale-dedup-ssdim"
# RUN = "pool-v14-fp32-half-ctxnorm-colnorm-signsoftmax-t10-tanh3-livescale-dedup-ssdim-accum16-lr1e-3"
# RUN = "pool-smoke-b1-v13cfg"
# RUN = "pool-smoke-b8"
# RUN = "pool-v15-fp32-small-b8-accum16-dedup-ssdim-lr1e-3"
# RUN = "pool-v16-fp32-half-ctxnorm-colnorm-signsoftmax-t10-tanh3-livescale-dedup-ssdim-huber"
RUN = "pool-v17-fp32-half-ctxnorm-huber"

submit(
    "expts.repaper.adapter.train_pool:main",
    args=dict(
        join_pre_dir="~/scratch/hf/stanford-star/the-join-preprocessed",
        relbench_pre_dir=PRE_DIR,
        relbench_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
        # task_list="pool_tasks_forecast.json",
        task_list="pool_tasks.json",
        # train_source="relbench",
        train_source="join",
        ckpt=CKPT,
        tabpfn_dir=f"{SHARE}/tabpfn",
        stats_path=f"{SHARE}/feature_stats_join_u12.npz",
        out_dir=f"{OUT_ROOT}/adapter/{RUN}",
        n_ctx=2**15,
        # n_ctx=2**12,
        # n_ctx=2**16,
        n_query=2**13,
        # n_query=2**10,
        # n_query=2**14,
        relbench_n_ctx=2**16,
        relbench_n_query=2**14,
        n_queries=64,
        swiglu_norm="context",
        # input_norm="col_context_signsoftmax",
        signsoftmax_temp=10.0,
        # signsoftmax_temp=1.0,
        # colnorm_tau=3.0,
        colnorm_tau=None,
        # live_scale=True,
        # signsoftmax_scale_by_dim=True,
        signsoftmax_scale_by_dim=False,
        live_scale=False,
        # input_norm="col_context",
        input_norm="fixed",
        # swiglu_norm="layer",
        # swiglu_norm="none",
        head_chunk=2048,
        tabpfn_device="cuda:0",
        tabpfn_precision="fp32",
        # tabpfn_precision="bf16",
        offload_cells=True,
        # total_steps=5,
        total_steps=2500,
        # total_steps=157 * 16,
        # lr=3e-4,
        lr=3e-4,
        # lr=1e-3,
        lr_min=1e-5,
        wd=0.0,
        # warmup_steps=1,
        warmup_steps=100,
        grad_norm_max=10.0,
        # swa_momentum=0.9995,
        swa_momentum=0.9995,
        # swa_momentum=0.9995**128,
        # eval_every=2,
        eval_every=250,
        # save_every=2,
        save_every=50,
        resume_save_mins=5.0,
        spike_dump_gnorm=100.0,
        # spike_dump_gnorm=None,
        dedup_ctx=False,
        # dedup_ctx=True,
        grad_accum=1,
        # grad_accum=16,
        task_batch=1,
        # task_batch=8,
        reg_loss="huber",
        # reg_loss="nll",
        huber_delta=1.0,
        # grad_accum=1,

        seed=0,
        run_name=f"adapter-{RUN}",
        project=project("adapter"),
        entity="rtv2",
        wandb_disabled=False,
    ),
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        # time="3:00:00",
        time="5-00:00:00",
        gpus="b200:1",
        cpus_per_task=32,
        ntasks=1,
        exclusive=False,
        mem="400G",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name=f"adapter-{RUN}",
    run_id=None,
    # run_id="26-09-24_22-50-00_455118805",
    # run_id="26-09-23_11-53-16_522641285",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)
