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

# 4 relational units x 3 slot subsets x 12 tasks = 144 runs, all of them
# indexing the one dump that already exists. Nothing is re-featurized: the
# blob holds every unit and every slot, and an arm is a choice of which of
# them become the feature vector.
#
#   full   every live column of the seed row's union set
#   target the target cell alone -- what features_rt-j holds, modulo norm_out
#   other  the union set without the target cell
#
# target and other partition full, so the three answer: how much is in the
# target cell, how much is in the rest of the row, and whether together they
# beat either. Reading the same question across units 1/4/8/12 says at what
# depth rt-j has put it there.
#   proj   the full union set at unit 12, put through a fixed random
#          projection down to 512 dims -- the same width as `target`, so it
#          asks whether `full` wins on what it contains or merely on being
#          wider. One Xavier-initialised matrix per table, no gradients, the
#          same matrix for every row of that dataset.
ROUND = "rtj_taps_grid_val"
UNITS = [1, 4, 8, 12]
SUBSETS = ["full", "target", "other"]
# proj only at the final unit: it is a projection *of* the final
# representation, not a depth sweep of its own.
ARMS = [(u, s) for u in UNITS for s in SUBSETS] + [(12, "proj")]
FEATURES_ROOT = f"{SHARE}/features_rt-j"

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


def n_features(db: str, table: str, subset: str) -> int:
    meta = json.loads(
        (
            Path(FEATURES_ROOT).expanduser() / db / "rt_taps" / f"{table}_meta.json"
        ).read_text()
    )
    live = meta["shape"][2] - len(meta["dead_slots"])
    # proj reads every live slot and then projects to 512.
    n_slots = {"full": live, "proj": live, "target": 1, "other": live - 1}[subset]
    return 512 if subset == "proj" else n_slots * meta["shape"][3]


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

for unit, subset in ARMS:
    arm = f"u{unit}-{subset}"
    for db, table in TASKS:
        name = f"rel2tabv2-{ROUND}-{arm}-{db}-{table}"
        if name in busy or done(arm, db, table):
            continue
        nf = n_features(db, table, subset)
        submit(
            "expts.repaper.baselines.rel2tabv2.run:main",
            args=dict(
                method=f"rttaps{unit}{subset}_tabpfn",
                db=db,
                table=table,
                split="val",
                pre_dir=PRE_DIR,
                features_root=FEATURES_ROOT,
                out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}/{arm}",
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
                context_split="train",
                tags=dict(
                    round=ROUND,
                    arm=arm,
                    featurizer=f"rt-j-taps-u{unit}-{subset}",
                    predictor="tabpfn",
                    split="val",
                    tap_unit=unit,
                    tap_subset=subset,
                    n_features=nf,
                    # "auto" resolves to the checkpoint's 8 and an
                    # estimator sees at most 768 features, so anything
                    # over 6144 runs on part of its feature space.
                    n_estimators_for_full_coverage=math.ceil(nf / 768),
                    n_context_rows=CONTEXT_ROWS,
                    features_regenerable=True,
                ),
            ),
            resources=Resources(
                partition="il",
                account="infolab",
                qos="il",
                time="8:00:00",
                gpus="a100:1",
                cpus_per_task=8,
                ntasks=None,
                exclusive=False,
                # The context matrix is n_features x 262144 float32 at
                # worst, so memory tracks the arm rather than the task.
                mem="120G" if nf > 6144 else "64G",
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
