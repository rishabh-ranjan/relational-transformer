from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    RAW_DIR,
    SECRETS_DIR,
    SHARE,
)

DB = "rel-f1"
TABLE = "driver-dnf"
METHOD = "rdblearn_lgbm"

# 2026-09-15: both stages are zero-gres, so they take il-cpu (uncapped, no
# per-user cap to subtract). il-cpu is nearly empty -- rambo 280/288 cpus and
# hyperturing1/2 252 each idle -- and the exclude list keeps the job on the two
# nodes that are actually set up (rambo, furiosa): the madmaxes and turings
# were never bootstrapped and trinity's local disk is 94% full.
EXCLUDE = (
    "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax2,"
    "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
)


def cpu_resources(cpus: int, mem: str, time: str) -> Resources:
    return Resources(
        partition="il-cpu",
        account="infolab",
        qos="il-cpu",
        time=time,
        gpus="0",
        cpus_per_task=cpus,
        ntasks=1,
        exclusive=False,
        mem=mem,
        mem_per_gpu=None,
        constraint=None,
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude=EXCLUDE,
    )


REPO_ROOT = str(Path(__file__).resolve().parents[4])

feat = submit(
    "expts.repaper.baselines.featurize_rdblearn:featurize_table",
    args=dict(
        db=DB,
        table=TABLE,
        task_type="clf",
        pre_dir=PRE_DIR,
        raw_dir=RAW_DIR,
        features_root=f"{SHARE}/features",
        relbench_cache_dir=f"{SHARE}/relbench-cache",
        max_depth=2,
        max_train_samples=1000,
    ),
    resources=cpu_resources(16, "32G", "2:00:00"),
    name=f"rel2tabv2-feat-rdbl-{DB}-{TABLE}",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
    pixi_env="featurize",
    setup=("pixi install -e featurize", "pixi run install-rdblearn"),
)

# The requested arm: context == local context == 1024, so the query row's own
# BFS neighborhood takes the whole cell budget and the retriever is left with
# ~4 labelled rows per query (2026-09-15 probe on this task, 4 batches x bs 8:
# mean 4.3 train rows, min 0, max 9, 3.1% of queries get none at all, and the
# walks make no difference -- 4.0 with num_walks=0 -- because tier 1/2 never
# run). local_ctx_size=256 on the same 1024 cells gives 15.6 rows.
submit(
    "expts.repaper.baselines.rel2tabv2.run:main",
    args=dict(
        method=METHOD,
        db=DB,
        table=TABLE,
        split="test",
        pre_dir=PRE_DIR,
        features_root=f"{SHARE}/features",
        out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{METHOD}",
        ctx_size_list=[1024],
        items_per_task=10_000_000,
        local_ctx_size=1024,
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
        tabicl_dir=f"{SHARE}/tabicl",
        tabicl_max_batch_size=1024,
        tabicl_min_bin_size=48,
        tabicl_softmax_temperature=0.9,
        lgbm_n_jobs=8,
    ),
    resources=cpu_resources(8, "32G", "2:00:00"),
    name=f"rel2tabv2-{METHOD}-{DB}-{TABLE}",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
    after=feat.id,
)
