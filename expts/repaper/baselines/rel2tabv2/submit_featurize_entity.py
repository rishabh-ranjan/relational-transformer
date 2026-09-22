import json
import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    PRE_DIR,
    RAW_DIR,
    SECRETS_DIR,
    SHARE,
)

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
REPO_ROOT = str(Path(__file__).resolve().parents[4])
FEATURES_ROOT = f"{SHARE}/features_entity"

# The zero-hop featurizer: each task row gets the entity table's own encoded row
# and nothing else. rel-f1 is already done locally (1509 features from `drivers`
# -- 5 GloVe text columns at 300 each, the timestamp's 7 parts, the entity id and
# the cutoff), and featurize_db skips a finished blob.
DBS = ["rel-trial", "rel-avito", "rel-event", "rel-stack", "rel-hm", "rel-amazon"]
#
# DBS = ["rel-f1"]

# il-cpu, not il: this featurizer has no forward pass to run. It loads the db
# into pandas, reads the graph cache the gnn pass already built, and gathers one
# row per task row. il-cpu is also uncapped on memory, which the alternative is
# not -- one a100 allows 252154M and rel-amazon's db alone is 23 M rows.
MEM = {
    "rel-amazon": "500G",
    "rel-avito": "192G",
    "rel-event": "240G",
    "rel-hm": "400G",
    "rel-stack": "240G",
    "rel-trial": "192G",
}
EXCLUDE_CPU = (
    "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax2,"
    "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
)


def featurized(db: str) -> bool:
    root = Path(FEATURES_ROOT).expanduser()
    tables = [
        t for d, t in json.loads(Path(DB_TASK_LIST).expanduser().read_text()) if d == db
    ]
    return all(
        (root / db / "entity_features" / f"{t}_meta.json").exists() for t in tables
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
    name = f"rel2tabv2-feat-entity-{db}"
    if name in busy or featurized(db):
        continue
    submit(
        "expts.repaper.baselines.featurize_entity:featurize_db",
        args=dict(
            db=db,
            db_task_list=DB_TASK_LIST,
            pre_dir=PRE_DIR,
            raw_dir=RAW_DIR,
            features_root=FEATURES_ROOT,
            # The same cache the gnn featurizer built and filled, read-only
            # here: this arm's inputs are that arm's inputs with the network
            # removed, so they must come from the same materialization.
            graph_cache_dir=f"{SHARE}/gnn-graph-cache",
            text_embedder_dir=f"{SHARE}/glove",
        ),
        resources=Resources(
            partition="il-cpu",
            account="infolab",
            qos="il-cpu",
            time="12:00:00",
            gpus="0",
            cpus_per_task=16,
            ntasks=1,
            exclusive=False,
            mem=MEM[db],
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
        pixi_env="gnn",
        setup=("pixi install -e gnn",),
    )
