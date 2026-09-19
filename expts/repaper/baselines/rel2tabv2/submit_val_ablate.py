import json
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

# Sweep 2 of the token-count ablation. Identical to submit_val_aug.py in every
# respect but the feature root, which points at blobs rt-j produced while it was
# fed untransformed COPIES of the source cells instead of derived ones -- same
# token count, same positions, no new information.
#
# Three rounds now share everything except their features:
#   rtj_val_baseline  no extra tokens
#   rtj_val_ablate    extra tokens, untransformed values   <- this one
#   rtj_val_aug       extra tokens, derived values
# so aug-minus-ablate isolates the transforms and ablate-minus-baseline isolates
# the width.
#
# Sweep 1 (submit_featurize_ablate.py) must have written a db's blob first;
# blob_ready() below is the gate.
ROUND = "rtj_val_ablate"
ARM = "rt-j-tabpfn-n262144"
METHOD = "rt_tabpfn"
FEATURES_ROOT = f"{SHARE}/features_rt-j-ablate"
SUBDIR = "rt_features"

# Context ROWS for the predictor. 12 of the 21 tasks have fewer train rows than
# this and get all of them; the retriever records the requested and the
# effective count in every result. Only rel-amazon, rel-hm and rel-stack reach
# 2**18.
CONTEXT_ROWS = 262144

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
TASKS = [tuple(p) for p in json.loads(Path(DB_TASK_LIST).expanduser().read_text())]
#
# TASKS = [("rel-hm", "user-churn")]

# The evaluator's per-query context width, which the global retriever discards.
# The rustler sampler pads every batch to it -- 3.2 GiB for eight queries at
# 262144, all of it thrown away -- so it stays at local_ctx_size and the
# predictor's row count rides separately. eval_bs = tokens_per_gpu // it.
SAMPLER_CTX_SIZE = 64
QUERY_BATCH = 256

BIG = {"rel-amazon", "rel-hm", "rel-stack"}
REPO_ROOT = str(Path(__file__).resolve().parents[4])

# Sweep 1 holds most of the `il` gpu cap while it runs, so this goes out on
# `il-lo`: uncapped, so every ready task can sit in the queue and take whatever
# card frees rather than being held back by our own cap. Promote a batch to `il`
# once sweep 1 drains and the cap is free again.
#
# il-lo is preemptible and this run does not checkpoint, so a preemption costs
# one task's elapsed time; done() skips whatever finished, so resubmitting is
# the recovery.
# QOS = "il"
QOS = "il-lo"


def blob_ready(db: str, table: str) -> bool:
    d = Path(FEATURES_ROOT).expanduser() / db / SUBDIR
    return (d / f"{table}_meta.json").is_file() and (
        d / f"{table}_vectors.bin"
    ).is_file()


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
    # Sweep 1 writes these one table at a time, so a missing blob means "not
    # yet", not "broken". Skip it and resubmit as they land.
    if not blob_ready(db, table):
        print(f"{db}/{table}: no augmented blob yet, skipping", flush=True)
        continue
    submit(
        "expts.repaper.baselines.rel2tabv2.run:main",
        args=dict(
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
            tabpfn_n_estimators="auto",
            tabpfn_fit_mode="fit_with_cache",
            retriever="global_train",
            context_sampler="uniform",
            # The context stays in-sample whatever is being scored: val is the
            # query split, train is where the in-context rows come from.
            context_split="train",
            tags=dict(
                round=ROUND,
                arm=ARM,
                featurizer="rt-j-ablate",
                predictor="tabpfn",
                split="val",
                n_context_rows=CONTEXT_ROWS,
                features_regenerable=True,
                augment_transforms="identity+pair_left",
                augment_max_modal_share=0.5,
                augment_ablation=True,
            ),
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos=QOS,
            time="12:00:00" if db in BIG else "4:00:00",
            # a100 only: the same config on two different a100s differs by up to
            # 9.3e-05, and mixing card types would put a larger gap than that
            # between arms of this line of work.
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
