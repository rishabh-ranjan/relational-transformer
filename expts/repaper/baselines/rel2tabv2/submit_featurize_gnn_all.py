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

# The GNN featurizer at one fixed configuration -- 512 channels, seed 0, no grid
# -- for every forecast task, feeding the 4 x 2 x 21 evaluation in submit_all.py.
# Separate from submit_featurize_gnn.py, which drives a channels x seed grid and
# is being edited concurrently; this file must not depend on that one's contents.
#
# It writes into that grid's (512, 0) cell, which is the same configuration by
# construction, so rel-f1 and rel-event -- already produced there -- are reused
# rather than recomputed. Both submitters skip a finished blob, so they
# cooperate; what neither survives is producing the *same* blob at the same time,
# so do not run this while the grid is submitting 512/seed-0 for these dbs.
CHANNELS = 512
SEED = 0
FEATURES_ROOT = f"{SHARE}/features_gnn-c{CHANNELS}-s{SEED}"

# One job per db, never per task: make_pkey_fkey_graph materializes the whole db
# into TensorFrames (text columns through GloVe) and caches it keyed on the db,
# so a db's tasks share one build -- and two concurrent jobs on a cold db race to
# write the same *.pt files. One job per db makes that race impossible among
# these, which matters because all five of these caches are cold.
# The two cheap dbs first, to measure what this path actually costs. An ampere
# caps one a100 at 14 CPUs and 252154M, so the big three cannot simply be given
# 400-600G -- buying more memory means asking for more GPUs, which is worth doing
# from a measurement rather than from a guess. rel-event's numbers do not
# extrapolate: 3.3 G of graph cache OOM-killed a 32 G job.
DBS = ["rel-trial", "rel-avito"]
#
# DBS = ["rel-stack", "rel-hm", "rel-amazon"]
#
# rel-f1 and rel-event are already done at this configuration.
# DBS = ["rel-f1", "rel-event"]

# Generous, and deliberately so. rel-event's cold job measured MaxRSS 3.3 G,
# 32 G was set from that, and it OOM-killed fifteen warm-cache jobs: the cold
# path peaks at one table's frames, a warm cache loads each cached .pt whole.
# These five all start cold and then go warm within the same job, so they have
# to survive the larger of the two peaks. The amperes have 2 T.
# 240G is the ceiling for a single a100 (the per-GPU cap is 252154M). The big
# three may need more than that, which means more GPUs per job; decide that from
# the MaxRSS these two report, not before.
MEM = {
    "rel-amazon": "240G",
    "rel-avito": "240G",
    "rel-hm": "240G",
    "rel-stack": "240G",
    "rel-trial": "240G",
}
GPUS = {
    "rel-amazon": 1,
    "rel-avito": 1,
    "rel-hm": 1,
    "rel-stack": 1,
    "rel-trial": 1,
}

# Sized off rel-f1 (74,063 db nodes, 24,058 task rows, 87 s at 512 wide) and the
# blob row counts: rel-amazon is ~17 M task rows across its four tasks, rel-hm
# ~10 M, rel-stack ~8 M. il allows 7 days, so these are loose on purpose -- a
# featurize job that dies on the wall clock wastes the graph build too.
TIME = {
    "rel-amazon": "3-00:00:00",
    "rel-avito": "12:00:00",
    "rel-hm": "2-00:00:00",
    "rel-stack": "2-00:00:00",
    "rel-trial": "12:00:00",
}


def featurized(db: str) -> bool:
    root = Path(FEATURES_ROOT).expanduser()
    tables = [
        t for d, t in json.loads(Path(DB_TASK_LIST).expanduser().read_text()) if d == db
    ]
    return all((root / db / "gnn_features" / f"{t}_meta.json").exists() for t in tables)


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
    name = f"rel2tabv2-featall-gnn-{db}"
    if name in busy or featurized(db):
        continue
    submit(
        "expts.repaper.baselines.featurize_gnn:featurize_db",
        args=dict(
            db=db,
            db_task_list=DB_TASK_LIST,
            pre_dir=PRE_DIR,
            raw_dir=RAW_DIR,
            features_root=FEATURES_ROOT,
            graph_cache_dir=f"{SHARE}/gnn-graph-cache",
            text_embedder_dir=f"{SHARE}/glove",
            channels=CHANNELS,
            num_layers=2,
            num_neighbors=128,
            aggr="sum",
            # "uniform": an unbiased sample of the eligible history, against
            # "last"'s most-recent-k, which is recency biased. Both honour the
            # cutoff, so neither leaks.
            temporal_strategy="uniform",
            # 14, not 1, which is a deliberate trade of reproducibility for
            # throughput on 33 M task rows: pyg-lib's per-thread sampler engines
            # make the blob depend on thread scheduling (see featurize_gnn). The
            # blob is therefore the artifact of record -- meta.json carries its
            # sha256 and `reproducible: false`, and collate_results.py reports
            # both. Never delete a blob that results were computed from.
            sampler_threads=14,
            batch_size=512,
            num_workers=0,
            seed=SEED,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il",
            time=TIME[db],
            # An ampere allows 14 CPUs and 252154M per a100, so a job needing
            # more memory has to hold more GPUs to be allowed it.
            gpus=f"a100:{GPUS[db]}",
            cpus_per_task=14 * GPUS[db],
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
        pixi_env="gnn",
        setup=("pixi install -e gnn",),
    )
