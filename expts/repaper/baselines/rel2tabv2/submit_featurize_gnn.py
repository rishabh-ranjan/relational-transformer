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
# rel-f1 alone: this featurizer is new and rel-f1 is 9 tables and 74,063 nodes,
# so a whole width sweep costs a couple of minutes.
DBS = ["rel-f1"]

# The width sweep. Only `channels` varies -- num_neighbors stays [128, 64] and
# num_layers stays 2 -- so the three blobs differ in exactly one thing. The root
# carries the width because the blob filename inside it does not
# (<table>_vectors.bin), and the 128 root is rewritten rather than reused from
# the first run so that all three are produced by one code path.
CHANNELS = [32, 128, 512]
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

# The materialization holds the whole db as TensorFrames, text embeddings
# included, so this tracks db size rather than task size. Same ladder the rt
# featurize pass used, which is the closest measured thing.
MEM = {
    "rel-amazon": "400G",
    "rel-avito": "64G",
    "rel-event": "240G",
    "rel-f1": "32G",
    "rel-hm": "240G",
    "rel-stack": "240G",
    "rel-trial": "64G",
}
LONG = {"rel-amazon", "rel-hm", "rel-stack"}


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
        name = f"rel2tabv2-feat-gnn{channels}-{db}"
        if name in busy:
            continue
        submit(
            "expts.repaper.baselines.featurize_gnn:featurize_db",
            args=dict(
                db=db,
                db_task_list=DB_TASK_LIST,
                pre_dir=PRE_DIR,
                raw_dir=RAW_DIR,
                features_root=f"{SHARE}/features_gnn-{channels}",
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
                temporal_strategy="uniform",
                batch_size=512,
                # 0: the loader's workers would each hold a copy of the materialized
                # graph, and rel-amazon's is the reason this job asks for 400G.
                num_workers=0,
                seed=0,
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
