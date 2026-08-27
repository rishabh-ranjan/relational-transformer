import json
import os
import subprocess
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

TEST_ROWS = {
    "rel-amazon/user-churn": 351_885,
    "rel-amazon/user-ltv": 351_885,
    "rel-amazon/item-ltv": 178_334,
    "rel-amazon/item-churn": 166_842,
    "rel-stack/user-badge": 255_360,
    "rel-stack/post-votes": 160_903,
    "rel-hm/item-sales": 105_542,
    "rel-stack/user-engagement": 88_137,
    "rel-hm/user-churn": 74_575,
    "rel-avito/user-clicks": 47_996,
    "rel-avito/user-visits": 36_129,
    "rel-trial/site-success": 22_617,
    "rel-trial/study-adverse": 3_098,
    "rel-event/user-attendance": 1_958,
    "rel-event/user-ignore": 1_958,
    "rel-avito/ad-ctr": 1_816,
    "rel-trial/study-outcome": 825,
    "rel-f1/driver-position": 760,
    "rel-f1/driver-top3": 726,
    "rel-f1/driver-dnf": 702,
    "rel-event/user-repeat": 246,
}

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


# ampere7's seventh a100 comes up with no free memory (01:10: four RT passes
# died there at the first forward, as expts/icl saw on 2026-08-25) and ampere4's
# local disk was full on 2026-08-25; both stay out of every gpu job.
def a100(qos: str, hours: int, db: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=f"{hours}:00:00",
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
        exclude="ampere4,ampere7",
    )


def b200(qos: str, hours: int, db: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=f"{hours}:00:00",
        gpus="b200:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem=mem(db),
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
    )


# A zero-gres job lands on any node with the cpus, and turing1-3, hyperion1,
# hyperion3 and hyperturing2 have no node-local home for this user (2026-08-27
# 00:17: 63 LightGBM jobs and the nosem-data job died in under two seconds
# there, before roach could even open the log).
def cpu(hours: int, db: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=f"{hours}:00:00",
        gpus="0",
        cpus_per_task=16,
        ntasks=1,
        exclusive=False,
        mem=mem(db),
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="turing1,turing2,turing3,hyperion1,hyperion3,hyperturing2",
    )


def queued() -> dict[str, str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j %i"],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split() for line in out.stdout.splitlines())


# 2026-08-27 00:30, read off the live cluster: every a100 but ampere2's seven
# is allocated and those are planned for a whole-node il-lo job at 11:45, two
# b200s show free but are planned; il-interactive starts now on either card
# (sbatch --test-only), il and il-lo report a day and eleven days out, though
# my fairshare puts my jobs ahead of every other pending il-lo job. So: the
# two il-interactive slots and the il b200 sub-cap go to the four rel-amazon
# full-test RT passes (~6 h on an a100 by the 2026-08-19 round's rate, ~3 h on
# a b200), the il a100 slots to the next-longest non-resumable passes, and
# everything else is il-lo with a limit that fits ampere2's backfill window.
# ilc-icl holds the tenth il slot (a b200) until ~01:05.
HIGH = {
    ("fulltest/rt", "rel-amazon", "user-churn"): ("il-interactive", "b200", 12),
    ("fulltest/rt", "rel-amazon", "user-ltv"): ("il-interactive", "b200", 12),
    ("fulltest/rt", "rel-amazon", "item-ltv"): ("il", "b200", 12),
    ("fulltest/rt", "rel-amazon", "item-churn"): ("il", "a100", 12),
    ("fulltest/rt", "rel-stack", "user-badge"): ("il", "a100", 12),
    ("fulltest/rt", "rel-stack", "post-votes"): ("il", "a100", 8),
    ("fulltest/rt", "rel-hm", "item-sales"): ("il", "a100", 6),
    ("fulltest/rt", "rel-stack", "user-engagement"): ("il", "a100", 6),
    ("fulltest/rt", "rel-hm", "user-churn"): ("il", "a100", 4),
    ("subsampled/rdblearn_tabicl", "rel-stack", "user-badge"): ("il", "a100", 12),
    ("subsampled/sql_tabicl", "rel-stack", "user-badge"): ("il", "a100", 10),
    # 00:54: icl's il b200 ended and shows free; the longest pending pass
    # (~3 h on an a100) takes it under il.
    ("subsampled/rdblearn_tabicl", "rel-hm", "item-sales"): ("il", "b200", 6),
    # 01:46: rel-hm/user-churn's full-test pass handed an il a100 back; the
    # longest pass still queued (2h35 in the 2026-08-19 round) takes it.
    ("subsampled/rdblearn_tabicl", "rel-stack", "post-votes"): ("il", "a100", 6),
    # 01:52: rel-stack/user-engagement's full-test pass handed an il a100 back;
    # rel-hm/item-sales' other subsampled TabICL pass (~3 h) takes it.
    ("subsampled/sql_tabicl", "rel-hm", "item-sales"): ("il", "a100", 6),
    # 02:02: rel-amazon/item-ltv's full-test pass freed the il b200 (1h41 on
    # it); the longest pass still queued, post-votes' sql TabICL, takes it.
    ("subsampled/sql_tabicl", "rel-stack", "post-votes"): ("il", "b200", 6),
    # 02:10: rel-hm/item-sales' full-test pass freed an il a100; il-lo
    # preemptions have started (two subsampled RT passes requeued), so the
    # longest non-resumable pass still queued moves up.
    ("fulltest/rdblearn_tabicl", "rel-amazon", "item-churn"): ("il", "a100", 6),
    # 02:20: rel-hm/item-sales' rdblearn TabICL pass freed the il b200 (1h25
    # on it); rel-amazon/user-churn's subsampled sql TabICL pass takes it.
    ("subsampled/sql_tabicl", "rel-amazon", "user-churn"): ("il", "b200", 6),
}


def resources(arm: str, db: str, table: str) -> Resources:
    if (arm, db, table) in HIGH:
        qos, card, hours = HIGH[arm, db, table]
        return {"a100": a100, "b200": b200}[card](qos, hours, db)
    method = ARMS[arm][0]
    rows = TEST_ROWS[f"{db}/{table}"]
    full = arm.startswith("fulltest/")
    if method.endswith("_lgbm"):
        if full:
            return cpu(
                16
                if rows >= 150_000
                else 10
                if rows >= 50_000
                else 4
                if rows >= 10_000
                else 1,
                db,
            )
        return cpu(12 if rows >= 20_000 else 4, db)
    if method.endswith("_tabicl"):
        if full:
            hours = (
                6
                if rows >= 150_000
                else 4
                if rows >= 50_000
                else 2
                if rows >= 10_000
                else 1
            )
        else:
            hours = 6 if rows >= 100_000 else 3 if rows >= 20_000 else 2
        return a100("il-lo", hours, db)
    if full:
        return a100("il-lo", 3 if rows >= 30_000 else 2 if rows >= 10_000 else 1, db)
    return a100("il-lo", 2 if ARMS[arm][1].get("vector_db_path") else 1, db)


def nosem_ready() -> bool:
    root = Path(f"{SHARE}/relbench-preprocessed-nosem").expanduser()
    links = [root / "db-task-lists"] + [
        p for db in {db for db, _ in TASKS} for p in (root / db).glob("*")
    ]
    return bool(links) and all(p.exists() for p in links)


busy = queued()

nosem = (
    busy.get("repaper-nosem-data")
    or nosem_ready()
    or submit(
        "expts.repaper.scaling.make_nosem_data:main",
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
            time="2:00:00",
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
            exclude="turing1,turing2,turing3,hyperion1,hyperion3,hyperturing2",
        ),
        name="repaper-nosem-data",
        repo_root=str(Path(__file__).resolve().parents[3]),
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/scaling/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    ).id
)

# roach builds a clone once per commit per node, by whichever job lands first,
# and `setup` runs only then -- so a vecdb rebuild asked for by the vdb arms
# alone is skipped whenever another job of the same commit built the clone
# (01:46: all 21 vdb_rdblearn jobs died on "rustler was built without the
# 'vecdb' feature"). Every job of this round therefore builds the sampler
# with the feature; the vector-db path is only taken when vector_db_path is
# set, and enscurve/ and baselines/ ask for the same build.
for arm in [
    "fulltest/rt",
    "subsampled/rt",
    "abl/rand",
    "abl/bfs32",
    "abl/bfs256",
    "abl/vdb_rdblearn",
    "abl/nosem",
    "fulltest/rdblearn_tabicl",
    "fulltest/sql_tabicl",
    "subsampled/rdblearn_tabicl",
    "subsampled/sql_tabicl",
    "fulltest/rdblearn_lgbm",
    "fulltest/sql_lgbm",
    "subsampled/rdblearn_lgbm",
    "subsampled/sql_lgbm",
    # "abl/vdb_rt",
]:
    method, overrides = ARMS[arm]
    for db, table in TASKS:
        out_dir = f"{OUT_ROOT}/repaper-scaling/{arm}"
        name = f"repaper-scal-{arm.replace('/', '-')}-{db}-{table}"
        if (Path(out_dir).expanduser() / f"{db}__{table}.json").exists():
            continue
        if name in busy:
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
                lgbm_n_jobs=16,
            )
            | overrides
            | dict(method=method, db=db, table=table, out_dir=out_dir),
            resources=resources(arm, db, table),
            name=name,
            repo_root=str(Path(__file__).resolve().parents[3]),
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/scaling/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            setup=("pixi run maturin develop --uv --release --features vecdb",),
            after=nosem if arm == "abl/nosem" and nosem is not True else None,
        )
