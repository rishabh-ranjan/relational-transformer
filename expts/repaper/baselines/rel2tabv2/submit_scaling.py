import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)

# In-context set size scaling for one fixed technique: rt-j features, TabPFN-3.5,
# the query-independent retriever. 2**10 is the size the 168-run sweep used, so
# it is here as the anchor rather than as new information.
#
# One row count per job. run.py refuses several for a query-independent
# retriever, because the evaluator keys the context by ITS context size and there
# is only one of those per run.
CONTEXT_ROWS = [1024]
#
# CONTEXT_ROWS = [1024, 16384, 262144]

ROUND = "rtj_tabpfn_ctx_scaling"
METHOD = "rt_tabpfn"
FEATURES_ROOT = f"{SHARE}/features_rt-j"
SUBDIR = "rt_features"

# Equivalence check first: rt-j + tabpfn at 1024 on this task is 0.7177 in
# n1024_4feat_2pred, which was run before the evaluator's context size was
# separated from the predictor's row count. Reproducing it here confirms the
# separation changed nothing about prediction.
DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
TASKS = [("rel-f1", "driver-dnf")]
#
# import json
# TASKS = [tuple(p) for p in json.loads(Path(DB_TASK_LIST).expanduser().read_text())]

# The evaluator's per-query context width, which this retriever discards. The
# rustler sampler pads every batch to it -- 3.2 GiB for eight queries at 262144,
# all of it thrown away -- so it stays at local_ctx_size and the predictor's row
# count rides separately. eval_bs = tokens_per_gpu // sampler_ctx_size.
SAMPLER_CTX_SIZE = 64
QUERY_BATCH = 256

BIG = {"rel-amazon", "rel-hm", "rel-stack"}
REPO_ROOT = str(Path(__file__).resolve().parents[4])


def blob_ready(db: str, table: str) -> bool:
    d = Path(FEATURES_ROOT).expanduser() / db / SUBDIR
    return (d / f"{table}_meta.json").is_file() and (
        d / f"{table}_vectors.bin"
    ).is_file()


def done(arm: str, db: str, table: str) -> bool:
    out = Path(OUT_ROOT).expanduser() / f"repaper-rel2tabv2/{ROUND}/{arm}"
    return (out / f"{db}__{table}.pkl").is_file()


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()

for n_rows in CONTEXT_ROWS:
    for db, table in TASKS:
        arm = f"rt-j-tabpfn-n{n_rows}"
        name = f"rel2tabv2-{ROUND}-{arm}-{db}-{table}"
        if name in busy or done(arm, db, table):
            continue
        assert blob_ready(db, table), f"no rt-j blob for {db}/{table}"
        submit(
            "expts.repaper.baselines.rel2tabv2.run:main",
            args=dict(
                method=METHOD,
                db=db,
                table=table,
                split="test",
                pre_dir=PRE_DIR,
                features_root=FEATURES_ROOT,
                out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}/{arm}",
                # Context ROWS for the predictor. A task with fewer train rows
                # gets all of them; the retriever clamps and records both the
                # requested and the effective count.
                ctx_size_list=[n_rows],
                sampler_ctx_size=SAMPLER_CTX_SIZE,
                items_per_task=10_000_000,
                local_ctx_size=SAMPLER_CTX_SIZE,
                bfs_width=32,
                prefer_latest=True,
                num_walks=0,
                walk_length=0,
                shuffle_seed=0,
                context_seed=0,
                tokens_per_gpu=SAMPLER_CTX_SIZE * QUERY_BATCH,
                num_workers=8,
                prefetch_factor=2,
                mmap_populate=True,
                db_cutoff=None,
                vector_db_path=None,
                tabicl_dir=f"{SHARE}/tabicl",
                tabicl_max_batch_size=1024,
                tabicl_min_bin_size=48,
                tabicl_softmax_temperature=0.9,
                lgbm_n_jobs=8,
                exaone_ensemble_count=8,
                tabfm_backend="pytorch",
                tabpfn_dir=f"{SHARE}/tabpfn",
                tabpfn_n_estimators="auto",
                tabpfn_fit_mode="fit_with_cache",
                retriever="global_train",
                context_sampler="uniform",
                context_split="train",
                tags=dict(
                    round=ROUND,
                    arm=arm,
                    featurizer="rt-j",
                    predictor="tabpfn",
                    n_context_rows=n_rows,
                    features_regenerable=True,
                ),
            ),
            resources=Resources(
                partition="il",
                account="infolab",
                qos="il",
                time="12:00:00" if db in BIG else "4:00:00",
                # a100 only: EXAONE has no sm_75 kernel and this keeps the
                # scaling curve on one card type, which matters because the
                # same config on two cards differed by 9.3e-05.
                gpus="a100:1",
                cpus_per_task=8,
                ntasks=None,
                exclusive=False,
                mem="120G" if db in BIG else "64G",
                mem_per_gpu=None,
                constraint="ampere",
                nodelist=None,
                reservation=None,
                dependency=None,
                exclude="ampere4,ampere6,ampere7,ampere9",
            ),
            name=name,
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
        )
