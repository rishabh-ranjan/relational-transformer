import json
import math
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

# The union-column arm: instead of the target cell alone, TabPFN is given the
# representation of every live column in the seed row's union set -- the task
# row's own cells plus the entity row its f2p link reaches -- at one
# relational unit. Raw residual stream, as dumped. Missing cells stay NaN,
# which is what TabPFN reads as missing (it declares allow_nan and validates X
# with ensure_all_finite=False).
#
# Measured against rtj_val_baseline: 0.7173 mean roc_auc over 12 clf tasks,
# 0.3584 mean nmae over 9 reg tasks, on the same val split at the same
# 2**18 context. Only the 12 small tasks were dumped, so this round covers a
# subset of those and the means are not directly comparable -- compare per
# task.
ROUND = "rtj_taps_val"
TAP_UNIT = 12
METHOD = f"rttaps{TAP_UNIT}_tabpfn"
FEATURES_ROOT = f"{SHARE}/features_rt-j"
ARM = f"rt-j-taps-u{TAP_UNIT}-tabpfn-n262144"

TASKS = [
    ("rel-avito", "ad-ctr"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-trial", "study-outcome"),
]

SAMPLER_CTX_SIZE = 64
QUERY_BATCH = 256
CONTEXT_ROWS = 262144
REPO_ROOT = str(Path(__file__).resolve().parents[4])


def n_estimators_for(db: str, table: str) -> int:
    """How many ensemble members full feature coverage would take.

    An estimator sees at most `max_features_per_estimator` (768) features, and
    `balanced` subsampling partitions a globally shuffled pool across members,
    so `ceil(n_features / 768)` covers everything exactly.

    This is NOT what runs -- `tabpfn_n_estimators` is `"auto"` below, which
    resolves to the checkpoint's N_ESTIMATORS of 8 and is then treated as
    explicit by scale_n_estimators_for_feature_coverage, so 8 x 768 = 6144
    features are sampled and the rest are not. It is recorded in the tags so a
    result says how much of its own feature space the run could see.
    """
    meta = json.loads(
        (
            Path(FEATURES_ROOT).expanduser() / db / "rt_taps" / f"{table}_meta.json"
        ).read_text()
    )
    live = meta["shape"][2] - len(meta["dead_slots"])
    n_features = live * meta["shape"][3]
    return max(8, math.ceil(n_features / 768))


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
BIG = {"rel-amazon", "rel-hm", "rel-stack"}

for db, table in TASKS:
    name = f"rel2tabv2-{ROUND}-{ARM}-{db}-{table}"
    if name in busy or done(db, table):
        continue
    n_est = n_estimators_for(db, table)
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
            # tabpfn_n_estimators=n_est,
            tabpfn_fit_mode="fit_with_cache",
            retriever="global_train",
            context_sampler="uniform",
            context_split="train",
            tags=dict(
                round=ROUND,
                arm=ARM,
                featurizer=f"rt-j-taps-u{TAP_UNIT}",
                predictor="tabpfn",
                split="val",
                tap_unit=TAP_UNIT,
                n_context_rows=CONTEXT_ROWS,
                n_estimators="auto",
                # What full coverage would have needed, against the 8 that
                # "auto" resolves to: above 8, the run saw 6144 of its
                # features and not the rest.
                n_estimators_for_full_coverage=n_est,
                features_regenerable=True,
            ),
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il",
            # Each estimator is its own kv cache, so the fit scales with the
            # count: up to 20 here against the baseline's 8.
            time="12:00:00" if db in BIG else "8:00:00",
            gpus="a100:1",
            cpus_per_task=8,
            ntasks=None,
            exclusive=False,
            # The context is live_slots x 512 float32 rather than 512, so the
            # host-side feature matrix is up to 30x the baseline's.
            mem="120G",
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4",
        ),
        name=name,
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
