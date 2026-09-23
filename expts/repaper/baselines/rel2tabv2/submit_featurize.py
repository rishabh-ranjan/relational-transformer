import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT,
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

# Two RT checkpoints, one feature root each, so both can write rt_features
# without colliding. featurize_rt resolves the whole db_task_list through
# get_tasks before filtering on db, which needed a curated one-task list while
# only rel-f1 was staged; all seven are staged now, so forecast.json is fine.
RT_CKPTS = {
    "rt-j": CKPT,
    "rt-plurel": "~/scratch/hf/stanford-star/rt-plurel",
}

# README measurements for one checkpoint: rel-amazon 2h16 and rel-hm 2h40 on a
# b200, rel-stack 1h01, rel-avito 7 min, rel-trial 5 min, rel-event and rel-f1
# about a minute. So three dbs are the pole and the rest are noise.
LONG = {"rel-amazon", "rel-hm", "rel-stack"}

# 2026-09-16: 8 b200 free on an unreserved blackwell1, 17 a100 free, nothing of
# mine on any gpu tier. il-interactive's 2 gpus and il's 2-b200 sub-cap go to
# the four longest jobs (rel-amazon and rel-hm, both checkpoints); every other
# rt job takes an ampere under il. That asks for 12 gpus under a 10-gpu cap, so
# two will sit on QOSMaxGRESPerUser and start as the short dbs drain -- minutes,
# since only rel-amazon, rel-hm and rel-stack are long. featurize_rt also skips
# a table whose blob exists, so nothing is redone if a job has to restart.
MEM = {
    "rel-amazon": "240G",
    "rel-avito": "64G",
    "rel-event": "240G",
    "rel-f1": "16G",
    "rel-hm": "150G",
    "rel-stack": "150G",
    "rel-trial": "64G",
}

RDBL_MEM = {
    "rel-amazon": "250G",
    "rel-avito": "64G",
    "rel-event": "400G",
    "rel-f1": "16G",
    "rel-hm": "150G",
    "rel-stack": "150G",
    "rel-trial": "64G",
}

EXCLUDE_CPU = (
    "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax2,"
    "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
)
REPO_ROOT = str(Path(__file__).resolve().parents[4])


# Same idiom as baselines/submit.py: a finished blob or an already-queued job is
# skipped, so this file can be rerun to fill in only what is missing. Without
# it a rerun would duplicate a job still writing its blob and redo the work.
def featurized(db: str, table: str) -> bool:
    meta = (
        Path(SHARE).expanduser() / f"features/{db}/rdblearn_features/{table}_meta.json"
    )
    return meta.exists()


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()


def smaller(a: str, b: str) -> str:
    return min(a, b, key=lambda m: int(m.rstrip("G")))


def rt_resources(db: str, card: str, qos: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time="12:00:00" if db in LONG else "2:00:00",
        gpus=f"{card}:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem=smaller(MEM[db], "240G"),
        mem_per_gpu=None,
        constraint="ampere" if card == "a100" else None,
        nodelist="blackwell1" if card == "b200" else None,
        reservation=None,
        dependency=None,
        exclude="ampere4" if card == "a100" else None,
    )


# The rt pass, submitted 2026-09-16 as 184077-184090 (both checkpoints, all
# seven dbs). Commented so this file can be rerun for rdblearn without
# duplicating jobs that are still running; featurize_rt skips a table whose
# blob exists, so uncommenting is safe to redo.
#
# b200_for = [(db, arm) for arm in RT_CKPTS for db in ("rel-amazon", "rel-hm")]
# qos_for = dict.fromkeys(b200_for[:2], "il-interactive")
#
# for arm, ckpt in RT_CKPTS.items():
#     for db in DBS:
#         card = "b200" if (db, arm) in b200_for else "a100"
#         submit(
#             "expts.repaper.baselines.featurize_rt:featurize_db",
#             args=dict(
#                 db=db, db_task_list=DB_TASK_LIST, pre_dir=PRE_DIR,
#                 features_root=f"{SHARE}/features_{arm}", ckpt=ckpt,
#                 local_ctx_size=256, bfs_width=32, shuffle_seed=0,
#                 context_seed=0, db_cutoff=None, batch_size=1024,
#             ),
#             resources=rt_resources(db, card, qos_for.get((db, arm), "il")),
#             name=f"rel2tabv2-feat-{arm}-{db}",
#             ...
#         )

# rdblearn: per task, not per db. Zero-gres on uncapped il-cpu, and the
# relbench cache is warm for all 21 tasks as of 2026-09-16, so no compute node
# needs internet for the dataset itself. rdblearn's own pins cannot solve into
# the featurize environment, hence the --no-deps install in setup.
for task in TASKS:
    name = f"rel2tabv2-feat-rdbl-{task.db_name}-{task.table_name}"
    if featurized(task.db_name, task.table_name) or name in busy:
        continue
    submit(
        "expts.repaper.baselines.featurize_rdblearn:featurize_table",
        args=dict(
            db=task.db_name,
            table=task.table_name,
            task_type=task.task_type,
            pre_dir=PRE_DIR,
            raw_dir=RAW_DIR,
            features_root=f"{SHARE}/features",
            relbench_cache_dir=f"{SHARE}/relbench-cache",
            max_depth=2,
            max_train_samples=1000,
        ),
        resources=Resources(
            partition="il-cpu",
            account="infolab",
            qos="il-cpu",
            time="1-00:00:00",
            gpus="0",
            cpus_per_task=16,
            ntasks=1,
            exclusive=False,
            mem=RDBL_MEM[task.db_name],
            mem_per_gpu=None,
            constraint=None,
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude=EXCLUDE_CPU,
        ),
        name=name,
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        pixi_env="featurize",
        setup=("pixi install -e featurize", "pixi run install-rdblearn"),
    )
