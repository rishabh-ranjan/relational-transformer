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

# One job per db, not per task: make_pkey_fkey_graph materializes every table of
# the db into TensorFrames (text columns through GloVe) and its cache is keyed on
# the dataset-level db, so the tasks of a db share it -- and two jobs on one db
# would race to write it.
# rel-event, whose user-ignore (clf) and user-attendance (reg) are the second
# round's pair: one db, one entity table, 1,958 test rows each. rel-f1 is done.
#
# The graph cache for this db is cold and it is 3.2 G of tables with text columns
# to push through GloVe, so the first (width, seed) is submitted alone to build
# it -- 16 at once would race to write the same *.pt files. Widen CHANNELS/SEEDS
# back out once it is warm; featurized() skips whatever is finished.
DBS = ["rel-event"]

# The width x init-seed grid feeding the ctx sweep in submit.py. Only `channels`
# and `seed` vary -- num_neighbors stays [128, 64] and num_layers stays 2 -- so a
# difference between two blobs is one of those two things. The root carries both
# because the blob filename inside it does not (<table>_vectors.bin).
CHANNELS = [8, 32, 128, 512]
SEEDS = list(range(4))
#
# The single warm-up job that built the graph cache, kept as the shape to
# rerun for the next db:
# CHANNELS = [8]
# SEEDS = [0]
#
# DBS = [
#     "rel-f1",
#     "rel-trial",
#     "rel-event",
#     "rel-avito",
#     "rel-stack",
#     "rel-hm",
#     "rel-amazon",
# ]

# rel-event's cold-cache job (184548) reported MaxRSS 3.3 G, so 32 G looked
# generous -- and OOM-killed all 15 warm-cache jobs. The cold path materializes
# table by table and peaks at one table's frames; a warm cache loads each cached
# .pt whole, which is a different and larger peak. Measure the path you are
# about to run, not the one that happened to run first.
MEM = {
    "rel-amazon": "400G",
    "rel-avito": "64G",
    "rel-event": "192G",
    "rel-f1": "32G",
    "rel-hm": "240G",
    "rel-stack": "240G",
    "rel-trial": "64G",
}
LONG = {"rel-amazon", "rel-hm", "rel-stack"}


# featurize_db skips a table whose blob exists, but that still costs a job, a
# clone and a pixi install: going from 3 seeds to 8 queued 12 no-op jobs ahead of
# the real ones under a 10-wide cap. Skip them here.
def featurized(db: str, channels: int, seed: int) -> bool:
    root = Path(SHARE).expanduser() / f"features_gnn-c{channels}-s{seed}"
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
    for channels in CHANNELS:
        for seed in SEEDS:
            name = f"rel2tabv2-feat-gnn-c{channels}-s{seed}-{db}"
            if name in busy or featurized(db, channels, seed):
                continue
            submit(
                "expts.repaper.baselines.featurize_gnn:featurize_db",
                args=dict(
                    db=db,
                    db_task_list=DB_TASK_LIST,
                    pre_dir=PRE_DIR,
                    raw_dir=RAW_DIR,
                    features_root=f"{SHARE}/features_gnn-c{channels}-s{seed}",
                    # Shared across widths: the materialized TensorFrames are the db's
                    # columns, which channels does not touch.
                    graph_cache_dir=f"{SHARE}/gnn-graph-cache",
                    text_embedder_dir=f"{SHARE}/glove",
                    # The example's defaults but for channels, whose sweep this is: 2
                    # layers, num_neighbors 128 halved per hop -> [128, 64].
                    # num_layers=2 is the 2-hop depth matching rdblearn's
                    # DFSConfig(max_depth=2).
                    channels=channels,
                    num_layers=2,
                    num_neighbors=128,
                    aggr="sum",
                    # "uniform": an unbiased sample of the eligible history, against
                    # "last"'s most-recent-k, which is recency biased. Both honour the
                    # cutoff, so neither leaks.
                    temporal_strategy="uniform",
                    # 1, and not for performance: see featurize_gnn. Anything above 1
                    # makes the blob differ run to run, which would confound every
                    # comparison between featurizer configurations.
                    sampler_threads=1,
                    batch_size=512,
                    # 0: the loader's workers would each hold a copy of the materialized
                    # graph, and rel-amazon's is the reason this job asks for 400G.
                    num_workers=0,
                    seed=seed,
                ),
                resources=Resources(
                    partition="il",
                    account="infolab",
                    qos="il",
                    time="12:00:00" if db in LONG else "4:00:00",
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
                pixi_env="gnn",
                setup=("pixi install -e gnn",),
            )
