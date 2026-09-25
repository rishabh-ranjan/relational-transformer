import json
import os
import subprocess
import sys
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT,
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)

ROUND = "rtj_val_pool"
# POOL = "pool-v1"
# POOL = "pool-v4-fp32-half-ctxnorm"
POOL = "pool-v7-fp32-half-ctxnorm-relbench"
POOL_CKPT = f"{OUT_ROOT}/adapter/{POOL}/pool_final.pt"
# POOL_SHA256 = "57ca53507808f7dd28c57defc37e88b4ec82f1fdb3c18fb775a4a0b2ae769e11"
# POOL_SHA256 = "5893eeada989ef44d535c19ddcd0cdc677d7dfdf298403808c395f86dcacf23b"
POOL_SHA256 = "43a10b05677352a4dda9632d95e0f4fa4f3e69a042ab6cf93cdda7e197ccbded"
VARIANT = "head"
N_ESTIMATORS = 8
CONTEXT_ROWS = 262144
# CONTEXT_ROWS = 32768
# ARM = f"rtpool-{POOL}-{VARIANT}-tabpfn{N_ESTIMATORS}-n{CONTEXT_ROWS}"
ARM = f"rtpoolcat-{POOL}-{VARIANT}-tabpfn{N_ESTIMATORS}-n{CONTEXT_ROWS}"
# METHOD = f"rtpool{VARIANT}_tabpfn"
METHOD = f"rtpoolcat{VARIANT}_tabpfn"
FEATURES_ROOT = f"{SHARE}/features_rt-j-pool/{POOL}"
# FEATURES_ROOT = f"{SHARE}/features_rt-j-pool/{POOL}/n{CONTEXT_ROWS}"

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
TASKS = [tuple(p) for p in json.loads(Path(DB_TASK_LIST).expanduser().read_text())]
if len(sys.argv) > 1:
    TASKS = [tuple(a.split("/")) for a in sys.argv[1:]]

SAMPLER_CTX_SIZE = 64
QUERY_BATCH = 256

BIG = {"rel-amazon", "rel-hm", "rel-stack"}
MEM = {
    "rel-amazon": "240G",
    "rel-avito": "64G",
    "rel-event": "240G",
    "rel-f1": "64G",
    "rel-hm": "150G",
    "rel-stack": "150G",
    "rel-trial": "64G",
}
REPO_ROOT = str(Path(__file__).resolve().parents[4])

QOS = "il-lo"


def done(db: str, table: str) -> bool:
    out = Path(OUT_ROOT).expanduser() / f"repaper-rel2tabv2/{ROUND}/{ARM}"
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

for db, table in TASKS:
    name = f"rel2tabv2-{ROUND}-{ARM}-{db}-{table}"
    if name in busy or done(db, table):
        continue
    submit(
        "expts.repaper.baselines.rel2tabv2.pool_features:main",
        args=dict(
            featurize=dict(
                db=db,
                table=table,
                pre_dir=PRE_DIR,
                ckpt=CKPT,
                pool_ckpt=POOL_CKPT,
                features_root=FEATURES_ROOT,
                context_rows=CONTEXT_ROWS,
                context_seed=0,
                compile=True,
            ),
            run=dict(
                method=METHOD,
                db=db,
                table=table,
                split="val",
                pre_dir=PRE_DIR,
                features_root=FEATURES_ROOT,
                out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}/{ARM}",
                ctx_size_list=[CONTEXT_ROWS],
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
                tabpfn_n_estimators=N_ESTIMATORS,
                tabpfn_fit_mode="fit_with_cache",
                retriever="global_train",
                context_sampler="uniform",
                context_split="train",
                tags=dict(
                    round=ROUND,
                    arm=ARM,
                    featurizer="rt-j target cell + pool" if METHOD.startswith("rtpoolcat") else "rt-j+pool",
                    pool=POOL,
                    pool_variant=VARIANT,
                    pool_ckpt=POOL_CKPT,
                    pool_sha256=POOL_SHA256,
                    predictor="tabpfn",
                    split="val",
                    n_context_rows=CONTEXT_ROWS,
                    features_regenerable=True,
                ),
            ),
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos=QOS,
            time="12:00:00" if db in BIG else "4:00:00",
            gpus="a100:1",
            cpus_per_task=8,
            ntasks=None,
            exclusive=False,
            mem=MEM[db],
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4,ampere7",
        ),
        name=name,
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
