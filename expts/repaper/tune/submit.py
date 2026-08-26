import json
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT_CLF,
    CKPT_REG,
    CKPT_ROOT,
    CLONE_ROOT,
    LOG_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    project,
)
from roach.slurm import Resources, submit

TASKS = [
    tuple(p)
    for p in json.loads(
        (Path(PRE_DIR).expanduser() / "db-task-lists" / "forecast.json").read_text()
    )
]

GRID = [
    (lcs, bw, pl)
    for lcs in (256, 512, 1024, 2048, 4096, 8192)
    for bw in (8, 32, 128)
    for pl in (True, False)
]

CLF = {
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "user-churn"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-badge"),
    ("rel-stack", "user-engagement"),
    ("rel-trial", "study-outcome"),
}


def resources(db: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="2-00:00:00",
        gpus="a100:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem={
            "rel-amazon": "120G",
            "rel-avito": "48G",
            "rel-event": "48G",
            "rel-f1": "24G",
            "rel-hm": "64G",
            "rel-stack": "64G",
            "rel-trial": "48G",
        }[db],
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
    )
    # return Resources(
    #     partition="il",
    #     account="infolab",
    #     qos="il-lo",
    #     time="2-00:00:00",
    #     gpus="b200:1",
    #     cpus_per_task=36,
    #     ntasks=None,
    #     exclusive=False,
    #     mem=None,
    #     mem_per_gpu=None,
    #     constraint=None,
    #     nodelist="blackwell1",
    #     reservation=None,
    #     dependency=None,
    # )


for db, table in TASKS:
    run_id = f"tune--{db}--{table}"
    if (
        Path(CKPT_ROOT).expanduser() / "rtv2" / project("tune") / run_id / "tuning.json"
    ).exists():
        continue
    submit(
        "rt.eval:main",
        args=dict(
            load_ckpt_path=CKPT_CLF if (db, table) in CLF else CKPT_REG,
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            splits=["val"],
            db_task_list=[(db, table)],
            pre_dir=PRE_DIR,
            tokens_per_gpu=2**18,
            num_workers=8,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            val_items_per_task=4096,
            test_items_per_task=None,
            ctx_size_list=[512, 1024, 2048, 4096, 8192],
            mmap_populate=True,
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            db_cutoff=None,
            lcs_bw_pl_grid=GRID,
            val_ensemble_size=4,
            test_ensemble_size=1,
            run_name=None,
            targets={},
            project=project("tune"),
            entity="rtv2",
            out_root=CKPT_ROOT,
            wandb_disabled=True,
        ),
        resources=resources(db),
        name=f"tune-{db}-{table}",
        run_id=run_id,
        repo_root=str(Path(__file__).resolve().parents[3]),
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/tune/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
