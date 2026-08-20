"""Submit context-scaling eval jobs. See README.md.

One job per (arm, task). An arm is a method plus a context-sampler
configuration; ``ARMS`` below holds every arm the paper needs, and the
``__main__`` block picks what this submission sends. Jobs are idempotent (a
finished task's JSON makes its job a no-op), so resubmitting an arm fills in
only what is missing.
"""

import json
from pathlib import Path

from roach.slurm import Resources, submit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"
SHARE = "/dfs/user/ranjanr/share/relational-transformer/repaper"
OUT_ROOT = "/dfs/user/ranjanr/ckpts/rtv2/repaper-scaling"
LOG_ROOT = (
    "/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/"
    "expts/repaper_scaling"
)
CKPT_CLF = "/dfs/user/ranjanr/share/stanford-star/rt-j/classification"
CKPT_REG = "/dfs/user/ranjanr/share/stanford-star/rt-j/regression"

TASKS = [
    tuple(p)
    for p in json.loads((Path(PRE_DIR) / "db-task-lists" / "forecast.json").read_text())
]

# The paper's shared default context (lcs=256, bw=32, pl=1) and seeds; every
# arm inherits these and overrides only what it ablates.
DEFAULTS = dict(
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
    db_cutoff="test",
    vector_db_path=None,
    ckpt_clf=CKPT_CLF,
    ckpt_reg=CKPT_REG,
    tabicl_dir=f"{SHARE}/tabicl",
    tabicl_max_batch_size=1024,
    tabicl_min_bin_size=48,
    tabicl_softmax_temperature=0.9,
    lgbm_n_jobs=8,
)

RT_CTX = [256, 512, 1024, 2048, 4096, 8192]
BASELINE_CTX = RT_CTX + [16384, 32768, 65536, 131072]
FULL = 10_000_000

# arm name -> (method, overrides). Arm outputs land in <OUT_ROOT>/<arm>/.
ARMS = {
    # fig 2 + fig:baselines + per-task appendix: full official test splits.
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
    # appendix extended-context baselines: 8192-row test subsample, baselines
    # swept past RT's training context.
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
    # retriever ablation (fig:retriever), on the same 8192-row subsample; the
    # random-walk arm is subsampled/rt. `rand` is walk-free uniform sampling
    # (prefer_latest would otherwise order the fallback tier by recency);
    # `bfs*` is one BFS around the target (the original RT scheme).
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
    # schema-semantics ablation (fig:semantics): the semantics-on arm is
    # subsampled/rt; the ablated arm evaluates the same thing over the derived
    # pre_dir whose column-name embeddings are deranged (make_nosem_data.py).
    "abl/nosem": (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            pre_dir=f"{SHARE}/relbench-preprocessed-nosem",
        ),
    ),
}

# The whole database is mmapped and populated per job, so memory follows the
# db. rel-amazon is 33G on disk plus loader/context state.
MEM = {
    "rel-amazon": "120G",
    "rel-avito": "48G",
    "rel-event": "48G",
    "rel-f1": "24G",
    "rel-hm": "64G",
    "rel-stack": "64G",
    "rel-trial": "48G",
}


def gpu_resources(db: str, qos: str, gpu: str = "a100:1", nodelist=None) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time="12:00:00" if qos == "il-interactive" else "2-00:00:00",
        gpus=gpu,
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem=MEM[db],
        mem_per_gpu=None,
        constraint="ampere" if gpu.startswith("a100") else None,
        nodelist=nodelist,
        reservation=None,
        dependency=None,
    )


def cpu_resources(db: str) -> Resources:
    # Zero-GPU slots on the il partition (see repaper_baselines/submit.py for
    # why the cpu-only partition is unusable): LightGBM fits are the job.
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time="2-00:00:00",
        gpus="0",
        cpus_per_task=8,
        ntasks=1,
        exclusive=False,
        mem=MEM[db],
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
    )


def submit_arm(arm: str, tasks, resources_for) -> None:
    method, overrides = ARMS[arm]
    setup = ()
    if overrides.get("vector_db_path"):
        # The vecdb sampler is an opt-in cargo feature; rebuild the extension
        # in the job's clone once, under the clone lock.
        setup = ("pixi run maturin develop --uv --release --features vecdb",)
    for db, table in tasks:
        out_dir = f"{OUT_ROOT}/{arm}"
        if (Path(out_dir) / f"{db}__{table}.json").exists():
            continue
        submit(
            "expts.repaper_scaling.run:main",
            args={
                **DEFAULTS,
                **overrides,
                "method": method,
                "db": db,
                "table": table,
                "out_dir": out_dir,
            },
            resources=resources_for(db),
            name=f"scal-{arm.replace('/', '-')}-{db}-{table}",
            repo_root=str(REPO_ROOT),
            log_root=LOG_ROOT,
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            setup=setup,
        )


def submit_nosem_data() -> None:
    submit(
        "expts.repaper_scaling.make_nosem_data:main",
        args=dict(
            pre_dir=PRE_DIR,
            out_dir=f"{SHARE}/relbench-preprocessed-nosem",
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            seed=0,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time="4:00:00",
            gpus="0",
            cpus_per_task=4,
            ntasks=1,
            exclusive=False,
            mem="32G",
            mem_per_gpu=None,
            constraint=None,
            nodelist=None,
            reservation=None,
            dependency=None,
        ),
        name="nosem-data",
        repo_root=str(REPO_ROOT),
        log_root=LOG_ROOT,
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
    )


if __name__ == "__main__":
    # The SQL feature blobs are complete and the probes pass (TabICL handled
    # the heaviest 131k-context probe on an rtx8000 in ~40 min; LightGBM on 8
    # cpus swept it in 8 min): the SQL-featurizer arms go wide. TabICL takes
    # hyperturing2's ten idle rtx8000s (hyperturing1's cards throw ECC
    # errors); LightGBM takes zero-gres cpu slots. The rdblearn arms wait for
    # their features (the amazon DFS jobs run for hours).
    for arm in ["fulltest/sql_tabicl", "subsampled/sql_tabicl"]:
        submit_arm(
            arm,
            TASKS,
            lambda db: gpu_resources(
                db, "il-lo", gpu="rtx8000:1", nodelist="hyperturing2"
            ),
        )
    for arm in ["fulltest/sql_lgbm", "subsampled/sql_lgbm"]:
        submit_arm(arm, TASKS, cpu_resources)
    # The derived nosem data is in place and verified (bijective derangement,
    # no fixed points, all other rows byte-identical).
    submit_arm("abl/nosem", TASKS, lambda db: gpu_resources(db, "il-lo"))
    # The five RT arms (105 jobs) are in the queue from the first submission;
    # do not resubmit while they are pending -- the idempotence check reads
    # finished JSONs, not the queue.
    # for arm in ["fulltest/rt", "subsampled/rt", "abl/rand", "abl/bfs32", "abl/bfs256"]:
    #     submit_arm(arm, TASKS, lambda db: gpu_resources(db, "il-lo"))
    # After the featurize + vecdb jobs finish: the four baseline arms per
    # protocol, and abl/vdb_*.
