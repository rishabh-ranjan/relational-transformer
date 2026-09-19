import json
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
    SECRETS_DIR,
    SHARE,
)

# The token-count ablation for the augmented arm. Identical to
# submit_featurize_aug.py except `ablation: True`, which replaces every derived
# cell with an untransformed copy of its source -- same count, same positions,
# same column-name structure, but carrying a value the model already has.
#
# It answers the question the augmented arm on its own cannot: how much of any
# change comes from widening the row, and how much from the transforms.
#
# Verified token-matched on rel-f1: both plans emit 57 columns and 5048 derived
# cells at identical positions, sequence 256 -> 597 either way.
ROUND = "rtj_val_ablate"
FEATURES_ROOT = f"{SHARE}/features_rt-j-ablate"
STATS_DIR = f"{SHARE}/numeric_stats"

# All three transforms. On a real rel-f1 batch at these settings the sequence
# grows 2.33x (256 -> 597 cells), which is the cost being bought.
AUGMENT = {
    "stats_dir": STATS_DIR,
    # The transform list still selects WHICH cells get a copy: the ablation
    # mirrors the real plan, so it has to be built from the same selection.
    "transforms": ["ecdf", "signed_log1p", "pair_product"],
    "ablation": True,
    # A transform whose output piles onto one value is dropped: the ecdf of a
    # zero-inflated count column sends ~88% of rows to the same rank and still
    # costs a token.
    "max_modal_share": 0.5,
}

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
ALL_DBS = sorted(
    {db for db, _ in json.loads(Path(DB_TASK_LIST).expanduser().read_text())}
)
# One cheap db first, to measure how far the features move against the
# no-augment baseline before spending the rest.
DBS = ["rel-f1"]
# DBS = ALL_DBS

# The baseline pass ran 2 h on the small dbs and 12 h on rel-amazon, rel-hm and
# rel-stack at local_ctx_size=256. The augmentation more than doubles the
# sequence, and rt-j's masks are structured rather than dense so the true factor
# is between 2.3x and 5.4x; these walls take the pessimistic end.
LONG = {"rel-amazon", "rel-hm", "rel-stack"}
MEM = {
    "rel-amazon": "240G",
    "rel-hm": "240G",
    "rel-stack": "240G",
    "rel-event": "120G",
    "rel-trial": "120G",
    "rel-avito": "120G",
    "rel-f1": "64G",
}
REPO_ROOT = str(Path(__file__).resolve().parents[4])


def stats_ready(db: str) -> bool:
    d = Path(STATS_DIR).expanduser()
    return all(
        (d / name).is_file()
        for name in (f"{db}.json", f"{db}_names.npz", f"{db}_ablation_names.npz")
    )


def featurized(db: str) -> bool:
    d = Path(FEATURES_ROOT).expanduser() / db / "rt_features"
    if not d.is_dir():
        return False
    metas = list(d.glob("*_meta.json"))
    return bool(metas) and all(
        (m.with_name(m.name.replace("_meta.json", "_vectors.bin"))).is_file()
        for m in metas
    )


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
    name = f"rel2tabv2-{ROUND}-feat-{db}"
    if name in busy:
        continue
    assert stats_ready(db), (
        f"no numeric stats for {db} in {STATS_DIR}; run "
        f"expts.preprocess.numeric_stats and expts.preprocess.augment_names first"
    )
    submit(
        "expts.repaper.baselines.featurize_rt:featurize_db",
        args=dict(
            db=db,
            db_task_list=DB_TASK_LIST,
            pre_dir=PRE_DIR,
            features_root=FEATURES_ROOT,
            ckpt=CKPT,
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            db_cutoff=None,
            batch_size=1024,
            augment=AUGMENT,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            # 2026-09-19: the ctx-scaling sweep has finished and the il gpu cap
            # is entirely free, so all seven fit under its ten. Anything that
            # ends up pending on QOSMaxGRESPerUser moves to il-lo.
            qos="il",
            # qos="il-lo",
            time="1-00:00:00" if db in LONG else "8:00:00",
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
