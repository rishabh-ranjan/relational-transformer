import json
import os
import subprocess
import sys
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

ROUND = "rtj_val_adapter"
ADAPTER = "join-v9-linear64-step10000"
ADAPTER_SOURCE = "~/scratch/ckpts/rtv2/adapter/join-v9-ddp-proj64-cosine1e-3/adapter_final.pt"
ADAPTER_SHA256 = "ce363001d8dee63f5a1128cb1b655b85e7cd844bf42fec7bb266c7132310a06f"
ARM = f"rtadapter-{ADAPTER}-tabpfn-n262144"
METHOD = f"rtadapter{ADAPTER}_tabpfn"
FEATURES_ROOT = f"{SHARE}/features_rt-j"
SUBDIR = "rt_features"
CONTEXT_ROWS = 262144

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
TASKS = [tuple(p) for p in json.loads(Path(DB_TASK_LIST).expanduser().read_text())]
if len(sys.argv) > 1:
    TASKS = [tuple(a.split("/")) for a in sys.argv[1:]]

SAMPLER_CTX_SIZE = 64
QUERY_BATCH = 256

BIG = {"rel-amazon", "rel-hm", "rel-stack"}
REPO_ROOT = str(Path(__file__).resolve().parents[4])

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


assert (Path(SHARE).expanduser() / "adapters" / f"{ADAPTER}.pt").is_file()
busy = queued()

for db, table in TASKS:
    name = f"rel2tabv2-{ROUND}-{ARM}-{db}-{table}"
    if name in busy or done(db, table):
        continue
    assert blob_ready(db, table), f"no rt-j blob for {db}/{table}"
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
            context_split="train",
            tags=dict(
                round=ROUND,
                arm=ARM,
                featurizer="rt-j+adapter",
                adapter=ADAPTER,
                adapter_source=ADAPTER_SOURCE,
                adapter_sha256=ADAPTER_SHA256,
                predictor="tabpfn",
                split="val",
                n_context_rows=CONTEXT_ROWS,
                features_regenerable=True,
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
            mem="120G" if db in BIG else "64G",
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
