import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT_CLF,
    CKPT_REG,
    CLONE_ROOT,
    LOG_ROOT,
    PRE_DIR,
    RAW_DIR,
    SECRETS_DIR,
    SHARE,
)
from rt.data import get_tasks

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"

TASKS = get_tasks(PRE_DIR, DB_TASK_LIST, ("test",))
DBS = sorted({t.db_name for t in TASKS})
TABLES = {db: sorted(t.table_name for t in TASKS if t.db_name == db) for db in DBS}


def mem(db: str) -> str:
    return {
        "rel-amazon": "250G",
        "rel-avito": "64G",
        "rel-event": "400G",
        "rel-f1": "16G",
        "rel-hm": "150G",
        "rel-stack": "150G",
        "rel-trial": "64G",
    }[db]


def cpu_resources(mem: str, cpus: int) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="1-00:00:00",
        gpus="0",
        cpus_per_task=cpus,
        ntasks=1,
        exclusive=False,
        mem=mem,
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
    )


def featurized(db: str, subdir: str, tables: list[str]) -> bool:
    feat_dir = Path(SHARE).expanduser() / "features" / db / subdir
    return all((feat_dir / f"{table}_meta.json").exists() for table in tables)


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()


for db in DBS:
    if featurized(db, "sql_features", TABLES[db]) or f"repaper-feat-sql-{db}" in busy:
        continue
    submit(
        "expts.repaper.baselines.featurize_sql:featurize_db",
        args=dict(
            db=db,
            db_task_list=DB_TASK_LIST,
            pre_dir=PRE_DIR,
            raw_dir=RAW_DIR,
            features_root=f"{SHARE}/features",
            relbench_cache_dir=f"{SHARE}/relbench-cache",
        ),
        resources=cpu_resources(mem(db), 4),
        name=f"repaper-feat-sql-{db}",
        repo_root=str(Path(__file__).resolve().parents[3]),
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        pixi_env="featurize",
        setup=("pixi install -e featurize", "pixi run install-rdblearn"),
    )

for task in TASKS:
    db, table = task.db_name, task.table_name
    if (
        featurized(db, "rdblearn_features", [table])
        or f"repaper-feat-rdbl-{db}-{table}" in busy
    ):
        continue
    submit(
        "expts.repaper.baselines.featurize_rdblearn:featurize_table",
        args=dict(
            db=db,
            table=table,
            task_type=task.task_type,
            pre_dir=PRE_DIR,
            raw_dir=RAW_DIR,
            features_root=f"{SHARE}/features",
            relbench_cache_dir=f"{SHARE}/relbench-cache",
            max_depth=2,
            max_train_samples=1000,
        ),
        resources=cpu_resources(mem(db), 16),
        name=f"repaper-feat-rdbl-{db}-{table}",
        repo_root=str(Path(__file__).resolve().parents[3]),
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        pixi_env="featurize",
        setup=("pixi install -e featurize", "pixi run install-rdblearn"),
    )

for db in DBS:
    if featurized(db, "rt_features", TABLES[db]) or f"repaper-feat-rt-{db}" in busy:
        continue
    submit(
        "expts.repaper.baselines.featurize_rt:featurize_db",
        args=dict(
            db=db,
            db_task_list=DB_TASK_LIST,
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features",
            ckpt_clf=CKPT_CLF,
            ckpt_reg=CKPT_REG,
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            db_cutoff=None,
            batch_size=1024,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time="8:00:00" if db == "rel-amazon" else "3:00:00",
            gpus="a100:1",
            cpus_per_task=8,
            ntasks=None,
            exclusive=False,
            mem=min(mem(db), "240G", key=lambda m: int(m.rstrip("G"))),
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4,ampere7",
        ),
        name=f"repaper-feat-rt-{db}",
        repo_root=str(Path(__file__).resolve().parents[3]),
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        setup=("pixi run maturin develop --uv --release --features vecdb",),
    )

# for subdir, root in [
#     # ("rdblearn_features", f"{SHARE}/vector_db/rdblearn"),
#     ("rt_features", f"{SHARE}/vector_db/rt"),
# ]:
#     if all(
#         (Path(root).expanduser() / t.db_name / f"{t.table_name}.index").exists()
#         for t in TASKS
#     ):
#         continue
#     submit(
#         "expts.repaper.baselines.build_vector_db:build_all",
#         args=dict(
#             db_task_list=DB_TASK_LIST,
#             pre_dir=PRE_DIR,
#             features_root=f"{SHARE}/features",
#             features_subdir=subdir,
#             vector_db_root=root,
#             ivf_threshold=50_000,
#         ),
#         resources=cpu_resources("250G", 16),
#         name=f"repaper-vecdb-{subdir.removesuffix('_features')}",
#         repo_root=str(Path(__file__).resolve().parents[3]),
#         cluster=ILC,
#         job_env="expts/job_env.sh",
#         log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
#         clone_root=CLONE_ROOT,
#         secrets_dir=SECRETS_DIR,
#         setup=("pixi run maturin develop --uv --release --features vecdb",),
#     )
