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
from rt.data import get_tasks

TASKS = [
    tuple(p)
    for p in json.loads(
        (Path(PRE_DIR).expanduser() / "db-task-lists" / "forecast.json").read_text()
    )
]

REG_TASKS = [
    (t.db_name, t.table_name)
    for t in get_tasks(PRE_DIR, f"{PRE_DIR}/db-task-lists/forecast.json", ("test",))
    if t.task_type == "reg"
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
    # The intro figure's baseline curves run past RT-J's training context on
    # the full test splits (as the 2026-07-15 round's did): the TabICL arms
    # over the 9 regression tasks at 16k-131k cells, merged into the fulltest
    # runs by reduce.py. One job per context size (11:10: a single pass over
    # all four was 15-50 h on an a100, too long for il-interactive's 12 h and
    # one card per task; the 16k-64k pieces fit that tier and the 131k piece
    # is half the work), each writing <arm>/<ctx>/<db>__<table>.json.
    **{
        f"fulltest_ext/{method}/{ctx}": (
            method,
            dict(ctx_size_list=[ctx], items_per_task=FULL),
        )
        for method in ("rdblearn_tabicl", "sql_tabicl")
        for ctx in BASELINE_CTX[6:]
    },
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
# 2026-08-27 17:00: ampere6 stopped responding and slurm requeued the two 1-day
# sql ext pieces on it from scratch (5h40 each); it stays excluded until it
# has been back for a while.
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
        exclude="ampere4,ampere6,ampere7",
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
# there, before roach could even open the log). 15:05: LightGBM moves to the
# cpu-only partition (`il-cpu`, qos `il-cpu`: no cap, 31d, nothing else queues
# there) instead of saturating hyperturing1, the interactive node (252/252
# cpus at 14:50). Only rambo (288 cpus) and furiosa (144) are allowed: the
# other il-cpu nodes are the six above, hyperturing1 itself, and madmax*/
# trinity, which have never been set up (a first job would bootstrap the node
# before doing anything; trinity's local disk is also 94% full).
def cpu(hours: int, db: str) -> Resources:
    return Resources(
        partition="il-cpu",
        account="infolab",
        qos="il-cpu",
        time=f"{hours}:00:00",
        gpus="0",
        cpus_per_task=24,
        ntasks=1,
        exclusive=False,
        mem=mem(db),
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude=(
            "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax1,madmax2,"
            "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
        ),
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
# Per-(arm, db, table) placements that override the rule below. Emptied
# 2026-08-27 13:15: blackwell1 is DOWN and every arm rerun for the RDBLearn
# fix goes to il a100s; git holds the round's earlier b200 / il-interactive
# placements and the reasons for each. 16:20: blackwell1 is back (all eight
# cards on il-lo work, which il and il-interactive preempt: sbatch --test-only
# starts at once on either). A b200 counts against the same ten-gpu il total
# as an a100, so the sub-cap of two buys speed, not slots: it goes to the two
# longest pieces left (2d05h each on an a100 by the rate table, ~1d on a b200,
# 36h limit), moved to the top of my il queue with `scontrol top`; the two
# il-interactive slots take the two 14h pieces on a b200 within its 12h wall.
# 20:40: the two il-interactive b200 pieces finished in 2h25 and 1h31 against
# 14h a100 projections (5-9x), so a 2d05h piece fits the tier's 12h wall on a
# b200 with margin. Every ext piece projected at 13h or more on an a100 goes
# there, two at a time; the tier's a100 pieces (2-7h) move to il, where the
# main-result arms finish tonight and hand the slots over.
HIGH: dict[tuple[str, str, str], tuple[str, str, int]] = {
    ("fulltest_ext/rdblearn_tabicl/131072", "rel-amazon", "user-ltv"): (
        "il",
        "b200",
        36,
    ),
    ("fulltest_ext/rdblearn_tabicl/131072", "rel-hm", "item-sales"): ("il", "b200", 36),
    **{
        (f"fulltest_ext/{method}/{ctx}", db, table): ("il-interactive", "b200", 12)
        for method, ctx, db, table in [
            ("sql_tabicl", 131072, "rel-amazon", "user-ltv"),
            ("sql_tabicl", 131072, "rel-hm", "item-sales"),
            ("sql_tabicl", 131072, "rel-stack", "post-votes"),
            ("sql_tabicl", 131072, "rel-amazon", "item-ltv"),
            ("rdblearn_tabicl", 131072, "rel-stack", "post-votes"),
            ("rdblearn_tabicl", 65536, "rel-hm", "item-sales"),
            ("rdblearn_tabicl", 65536, "rel-stack", "post-votes"),
            ("rdblearn_tabicl", 32768, "rel-stack", "post-votes"),
            ("rdblearn_tabicl", 65536, "rel-amazon", "item-ltv"),
        ]
    },
}


# 03:35: three hours of ~50 concurrent jobs spent the fairshare that had put
# this sweep's il-lo jobs first (priority ~5700 at 00:15, ~730 now, under the
# ~1020 of the other il-lo work in the queue), so an il-lo job of mine no
# longer starts, and the ones running are being preempted. What is left --
# ~35 short RT passes, two full-test TabICL passes and 36 LightGBM fits --
# queues under il instead: the gpu jobs beyond the tier's ten wait on
# QOSMaxGRESPerUser behind my own and start as those finish, which keeps the
# tier full without a resubmission per slot; the zero-gres LightGBM jobs do
# not count against the gpu cap and start on free cores at il priority.
def resources(arm: str, db: str, table: str) -> Resources:
    if (arm, db, table) in HIGH:
        qos, card, hours = HIGH[arm, db, table]
        return {"a100": a100, "b200": b200}[card](qos, hours, db)
    method = ARMS[arm][0]
    rows = TEST_ROWS[f"{db}/{table}"]
    full = arm.startswith("fulltest/")
    if arm.startswith("fulltest_ext/"):
        # 11:10, measured on the single-pass attempt (batches/min on an a100 at
        # 16k-131k cells, all four sizes scored per batch): user-ltv and
        # item-ltv ~100, post-votes ~30, item-sales ~20, site-success ~155,
        # 2 rows per batch. A context size's share of the pass is proportional
        # to its cells, and the limit is twice the projection.
        rate = {
            "rel-amazon/user-ltv": 60,
            "rel-amazon/item-ltv": 95,
            "rel-stack/post-votes": 28,
            "rel-hm/item-sales": 18,
        }.get(f"{db}/{table}", 150)
        ctx = ARMS[arm][1]["ctx_size_list"][0]
        hours = max(2, int(2 * rows / 2 / rate / 60 * ctx / sum(BASELINE_CTX[6:]) + 1))
        return a100("il", hours, db)
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
        return a100("il", hours, db)
    if full:
        return a100("il", 3 if rows >= 30_000 else 2 if rows >= 10_000 else 1, db)
    return a100("il", 2 if ARMS[arm][1].get("vector_db_path") else 1, db)


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
    *[
        f"fulltest_ext/{m}/{c}"
        for c in reversed(BASELINE_CTX[6:])
        for m in ("rdblearn_tabicl", "sql_tabicl")
    ],
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
    "abl/vdb_rt",
]:
    method, overrides = ARMS[arm]
    for db, table in REG_TASKS if arm.startswith("fulltest_ext/") else TASKS:
        out_dir = f"{OUT_ROOT}/repaper-scaling/{arm}"
        name = f"repaper-scal-{arm.replace('/', '-')}-{db}-{table}"
        if (Path(out_dir).expanduser() / f"{db}__{table}.json").exists():
            continue
        if name in busy:
            continue
        if arm.startswith("fulltest_ext/"):
            # the single-pass layout of the first attempt: <arm>/<db>__<table>.json
            # holding every context size, and its job still running
            whole = Path(out_dir).expanduser().parent / f"{db}__{table}.json"
            if (
                whole.exists()
                and str(overrides["ctx_size_list"][0])
                in json.loads(whole.read_text())["per_ctx"]
            ):
                continue
            if (
                f"repaper-scal-{arm.rsplit('/', 1)[0].replace('/', '-')}-{db}-{table}"
                in busy
            ):
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
                lgbm_n_jobs=24,
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
