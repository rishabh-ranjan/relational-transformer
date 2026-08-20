"""Submit featurization jobs. See README.md for the order of operations."""

import json
from pathlib import Path

from roach.slurm import Resources, submit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"
SHARE = "/dfs/user/ranjanr/share/relational-transformer/repaper"
DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
RAW_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench"
LOG_ROOT = (
    "/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/"
    "expts/repaper_baselines"
)

PAIRS = [tuple(p) for p in json.loads(Path(DB_TASK_LIST).read_text())]
DBS = sorted({db for db, _ in PAIRS})

# Task types, needed by the rdblearn featurizer's base-estimator choice. Kept
# here rather than resolved from meta.json so this file submits without the
# default environment's data helpers.
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

# The whole database sits in DuckDB / DFS working memory, so memory follows
# the database, not the task.
MEM = {
    "rel-amazon": "250G",
    "rel-avito": "64G",
    # rel-event's DFS working set far outgrows its on-disk size (48G and 150G
    # jobs were OOM-killed): wide attendee tables fan out under depth-2
    # aggregation. 400G fits only the big-memory nodes (hyperturing/rambo
    # class), which zero-gres jobs reach.
    "rel-event": "400G",
    "rel-f1": "16G",
    "rel-hm": "150G",
    "rel-stack": "150G",
    "rel-trial": "64G",
}


# Zero-GPU jobs on the il partition under il-lo: the cpu-only partition
# (il-cpu) admits only the il-cpu-long QOS, whose whole 8-cpu per-user budget
# the standing dev-node allocation already holds. turing3/hyperion cpus are
# idle and their cards are too old for anything else, so taking cpus there
# starves nothing.
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


SETUP = (
    "pixi install -e featurize",
    "pixi run install-rdblearn",
)

COMMON = dict(
    repo_root=str(REPO_ROOT),
    log_root=LOG_ROOT,
    clone_root="/lfs/local/0/roach_clones",
    secrets_dir="/dfs/user/ranjanr/.secrets",
    pixi_env="featurize",
    setup=SETUP,
)


def submit_sql() -> None:
    for db in DBS:
        submit(
            "expts.repaper_baselines.featurize_sql:featurize_db",
            args=dict(
                db=db,
                db_task_list=DB_TASK_LIST,
                pre_dir=PRE_DIR,
                raw_dir=RAW_DIR,
                features_root=f"{SHARE}/features",
                relbench_cache_dir=f"{SHARE}/relbench-cache",
            ),
            resources=cpu_resources(MEM[db], 4),
            name=f"feat-sql-{db}",
            **COMMON,
        )


def submit_rdblearn() -> None:
    for db, table in PAIRS:
        submit(
            "expts.repaper_baselines.featurize_rdblearn:featurize_table",
            args=dict(
                db=db,
                table=table,
                task_type="clf" if (db, table) in CLF else "reg",
                pre_dir=PRE_DIR,
                raw_dir=RAW_DIR,
                features_root=f"{SHARE}/features",
                relbench_cache_dir=f"{SHARE}/relbench-cache",
                max_depth=2,
                max_train_samples=1000,
            ),
            resources=cpu_resources(MEM[db], 16),
            name=f"feat-rdbl-{db}-{table}",
            **COMMON,
        )


def submit_rt(dbs=None) -> None:
    # One GPU per db; needs the default environment (the RT model), not the
    # featurize one.
    for db in dbs or DBS:
        submit(
            "expts.repaper_baselines.featurize_rt:featurize_db",
            args=dict(
                db=db,
                db_task_list=DB_TASK_LIST,
                pre_dir=PRE_DIR,
                features_root=f"{SHARE}/features",
                ckpt_clf="/dfs/user/ranjanr/share/stanford-star/rt-j/classification",
                ckpt_reg="/dfs/user/ranjanr/share/stanford-star/rt-j/regression",
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
                time="1-00:00:00",
                gpus="a100:1",
                cpus_per_task=8,
                ntasks=None,
                exclusive=False,
                # the ampere job_submit plugin caps a 1-gpu job at 252154M
                mem=min(MEM[db], "240G", key=lambda m: int(m.rstrip("G"))),
                mem_per_gpu=None,
                constraint="ampere",
                nodelist=None,
                reservation=None,
                dependency=None,
            ),
            name=f"feat-rt-{db}",
            **{**COMMON, "pixi_env": "default", "setup": ()},
        )


def submit_vecdb(subdirs) -> None:
    # CPU index build over the finished feature blobs, one job per feature set.
    for subdir, root in [
        ("rdblearn_features", f"{SHARE}/vector_db/rdblearn"),
        ("rt_features", f"{SHARE}/vector_db/rt"),
    ]:
        if subdir not in subdirs:
            continue
        submit(
            "expts.repaper_baselines.build_vector_db:build_all",
            args=dict(
                db_task_list=DB_TASK_LIST,
                pre_dir=PRE_DIR,
                features_root=f"{SHARE}/features",
                features_subdir=subdir,
                vector_db_root=root,
                ivf_threshold=50_000,
                nprobe=0,
            ),
            resources=cpu_resources("250G", 16),
            name=f"vecdb-{subdir.removesuffix('_features')}",
            **{**COMMON, "pixi_env": "default", "setup": ()},
        )


if __name__ == "__main__":
    # All 21 rt-embedding tables are on disk: build their FAISS indices.
    submit_vecdb(["rt_features"])
    # submit_rt()
    # submit_rdblearn()
    # submit_sql()
    # submit_vecdb()   # after the feature blobs exist
