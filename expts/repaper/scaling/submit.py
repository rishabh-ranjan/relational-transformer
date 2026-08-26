import json
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT_CLF,
    CKPT_REG,
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)
from roach.slurm import Resources, submit

TASKS = [
    tuple(p)
    for p in json.loads(
        (Path(PRE_DIR).expanduser() / "db-task-lists" / "forecast.json").read_text()
    )
]

RT_CTX = [256, 512, 1024, 2048, 4096, 8192]
BASELINE_CTX = RT_CTX + [16384, 32768, 65536, 131072]
FULL = 10_000_000

ARMS = {
    "fulltest/rt": ("rt", dict(ctx_size_list=RT_CTX, items_per_task=FULL)),
    "fulltest/rdblearn_tabicl": (
        "rdblearn_tabicl",
        dict(ctx_size_list=RT_CTX, items_per_task=FULL),
    ),
    "fulltest/sql_tabicl": (
        "sql_tabicl",
        dict(ctx_size_list=RT_CTX, items_per_task=FULL),
    ),
    "fulltest/rdblearn_lgbm": (
        "rdblearn_lgbm",
        dict(ctx_size_list=RT_CTX, items_per_task=FULL),
    ),
    "fulltest/sql_lgbm": ("sql_lgbm", dict(ctx_size_list=RT_CTX, items_per_task=FULL)),
    "subsampled/rt": ("rt", dict(ctx_size_list=RT_CTX, items_per_task=8192)),
    "subsampled/rdblearn_tabicl": (
        "rdblearn_tabicl",
        dict(ctx_size_list=BASELINE_CTX, items_per_task=8192),
    ),
    "subsampled/sql_tabicl": (
        "sql_tabicl",
        dict(ctx_size_list=BASELINE_CTX, items_per_task=8192),
    ),
    "subsampled/rdblearn_lgbm": (
        "rdblearn_lgbm",
        dict(ctx_size_list=BASELINE_CTX, items_per_task=8192),
    ),
    "subsampled/sql_lgbm": (
        "sql_lgbm",
        dict(ctx_size_list=BASELINE_CTX, items_per_task=8192),
    ),
    "abl/rand": (
        "rt",
        dict(
            ctx_size_list=RT_CTX, items_per_task=8192, num_walks=0, prefer_latest=False
        ),
    ),
    "abl/bfs32": (
        "rt",
        dict(
            ctx_size_list=RT_CTX, items_per_task=8192, local_ctx_size=8192, bfs_width=32
        ),
    ),
    "abl/bfs256": (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            local_ctx_size=8192,
            bfs_width=256,
        ),
    ),
    "abl/vdb_rdblearn": (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            vector_db_path=f"{SHARE}/vector_db/rdblearn",
        ),
    ),
    "abl/vdb_rt": (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            vector_db_path=f"{SHARE}/vector_db/rt",
        ),
    ),
    "abl/nosem": (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            pre_dir=f"{SHARE}/relbench-preprocessed-nosem",
        ),
    ),
}


def mem(db: str) -> str:
    return {
        "rel-amazon": "120G",
        "rel-avito": "48G",
        "rel-event": "48G",
        "rel-f1": "24G",
        "rel-hm": "64G",
        "rel-stack": "64G",
        "rel-trial": "48G",
    }[db]


def gpu_resources(db: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="2-00:00:00",
        gpus="a100:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem=mem(db),
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
    )


def cpu_resources(db: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="2-00:00:00",
        gpus="0",
        cpus_per_task=32,
        ntasks=1,
        exclusive=False,
        mem=mem(db),
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
    )


# submit(
#     "expts.repaper.scaling.make_nosem_data:main",
#     args=dict(
#         pre_dir=PRE_DIR,
#         out_dir=f"{SHARE}/relbench-preprocessed-nosem",
#         embedder="all-MiniLM-L12-v2",
#         d_text=384,
#         seed=0,
#     ),
#     resources=Resources(
#         partition="il",
#         account="infolab",
#         qos="il-lo",
#         time="4:00:00",
#         gpus="0",
#         cpus_per_task=4,
#         ntasks=1,
#         exclusive=False,
#         mem="32G",
#         mem_per_gpu=None,
#         constraint=None,
#         nodelist=None,
#         reservation=None,
#         dependency=None,
#     ),
#     name="nosem-data",
#     repo_root=str(Path(__file__).resolve().parents[3]),
#     cluster=ILC,
#     job_env="expts/job_env.sh",
#     log_root=f"{LOG_ROOT}/repaper/scaling/slurm-logs",
#     clone_root=CLONE_ROOT,
#     secrets_dir=SECRETS_DIR,
# )

for arm in ["fulltest/rt", "subsampled/rt", "abl/rand", "abl/bfs32", "abl/bfs256"]:
    # for arm in ["abl/vdb_rdblearn", "abl/vdb_rt", "abl/nosem"]:
    method, overrides = ARMS[arm]
    for db, table in TASKS:
        out_dir = f"{OUT_ROOT}/repaper-scaling/{arm}"
        if (Path(out_dir).expanduser() / f"{db}__{table}.json").exists():
            continue
        submit(
            "expts.repaper.scaling.run:main",
            args=dict(
                split="test",
                pre_dir=PRE_DIR,
                features_root=f"{SHARE}/features",
                local_ctx_size=256,
                bfs_width=32,
                prefer_latest=True,
                num_walks=10_000,
                walk_length=20,
                shuffle_seed=0,
                context_seed=0,
                tokens_per_gpu=2**18,
                num_workers=8,
                prefetch_factor=2,
                mmap_populate=True,
                db_cutoff=None,
                vector_db_path=None,
                ckpt_clf=CKPT_CLF,
                ckpt_reg=CKPT_REG,
                tabicl_dir=f"{SHARE}/tabicl",
                tabicl_max_batch_size=1024,
                tabicl_min_bin_size=48,
                tabicl_softmax_temperature=0.9,
                lgbm_n_jobs=32,
            )
            | overrides
            | dict(method=method, db=db, table=table, out_dir=out_dir),
            resources=gpu_resources(db),
            # resources=cpu_resources(db),
            name=f"scal-{arm.replace('/', '-')}-{db}-{table}",
            repo_root=str(Path(__file__).resolve().parents[3]),
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/scaling/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            setup=("pixi run maturin develop --uv --release --features vecdb",)
            if overrides.get("vector_db_path")
            else (),
        )
