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

ROUND = "rtj_width_depth"
CONTEXT_ROWS = 32768
# BFS_WIDTHS = [32, 64, 128]
# LOCAL_CTXS = [256, 512, 1024, 2048]
# GRID = [(w, c) for w in BFS_WIDTHS for c in LOCAL_CTXS]
# GRID = [(128, 2048), (32, 256)]
GRID = [(32, 256), (64, 512), (128, 1024), (256, 2048), (32, 512), (32, 1024), (32, 2048)]

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
TASKS = [tuple(p) for p in json.loads(Path(DB_TASK_LIST).expanduser().read_text())]
if len(sys.argv) > 1:
    TASKS = [tuple(a.split("/")) for a in sys.argv[1:]]

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


def arm(w, c):
    return f"rt-j-w{w}-c{c}-tabpfn8-n{CONTEXT_ROWS}"


def done(w, c, db, table):
    return (Path(OUT_ROOT).expanduser() / f"repaper-rel2tabv2/{ROUND}/{arm(w, c)}/{db}__{table}.pkl").is_file()


def queued():
    out = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


busy = queued()

for (w, c), (db, table) in [(g, t) for g in GRID for t in TASKS]:
    name = f"rel2tabv2-{ROUND}-{arm(w, c)}-{db}-{table}"
    if name in busy or done(w, c, db, table):
        continue
    features_root = f"{SHARE}/features_rt-j-rows/w{w}-c{c}/n{CONTEXT_ROWS}"
    submit(
        "expts.repaper.baselines.rel2tabv2.rtj_rows:main",
        args=dict(
            featurize=dict(
                db=db,
                table=table,
                pre_dir=PRE_DIR,
                ckpt=CKPT,
                features_root=features_root,
                context_rows=CONTEXT_ROWS,
                context_seed=0,
                local_ctx=c,
                bfs_width=w,
                compile=True,
            ),
            run=dict(
                method="rtrows_tabpfn",
                db=db,
                table=table,
                split="val",
                pre_dir=PRE_DIR,
                features_root=features_root,
                out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}/{arm(w, c)}",
                ctx_size_list=[CONTEXT_ROWS],
                sampler_ctx_size=64,
                items_per_task=10_000_000,
                local_ctx_size=64,
                bfs_width=32,
                prefer_latest=True,
                num_walks=0,
                walk_length=0,
                shuffle_seed=0,
                context_seed=0,
                tokens_per_gpu=64 * 256,
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
                tabpfn_n_estimators=8,
                tabpfn_fit_mode="fit_with_cache",
                retriever="global_train",
                context_sampler="uniform",
                context_split="train",
                tags=dict(
                    round=ROUND,
                    arm=arm(w, c),
                    featurizer="rt-j",
                    bfs_width=w,
                    local_ctx=c,
                    depth=(c // w).bit_length() - 1,
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
            qos="il-lo",
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
