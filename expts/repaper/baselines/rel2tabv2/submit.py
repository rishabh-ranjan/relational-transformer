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

# The query-independent retriever: one seeded rng.choice draw of exactly 1024
# rows from the train split, the same context for every query, features
# precomputed once. Verified reproducible (same seed, same 1024 rows), spanning
# 1950-2004 at an 0.868 positive rate against train's 0.880.
#
# 1024 is a context ROW count, not cells, so it does not share an axis with the
# ctx=256..8192 curves; the per-query ctx=8192 arm averaged 336 labelled rows.
RETRIEVER = "global_train"
N_ROWS = [1024]
ROUND = "global_uniform_seed0_n1024"

# 2026-09-16: blackwell1 entirely free (8 b200) and not reserved, 8 a100 free
# across ampere1/2/3/5/8, nothing of mine on any gpu tier. il-interactive's 2
# gpus take b200 for the two 512-d TabFM arms -- the most expensive pair, TabFM
# being 6.2 GB of weights against EXAONE's 161 MB -- and il takes the rest,
# spending its 2-b200 sub-cap on the third TabFM arm and one 512-d EXAONE arm,
# with the two cheaper EXAONE arms on amperes.
ARMS = {
    "rt-j-tabfm": ("rt_tabfm", f"{SHARE}/features_rt-j", "il-interactive", "b200"),
    "rt-plurel-tabfm": (
        "rt_tabfm",
        f"{SHARE}/features_rt-plurel",
        "il-interactive",
        "b200",
    ),
    "rdblearn-tabfm": ("rdblearn_tabfm", f"{SHARE}/features", "il", "b200"),
    "rt-j-exaone": ("rt_exaone", f"{SHARE}/features_rt-j", "il", "b200"),
    "rt-plurel-exaone": ("rt_exaone", f"{SHARE}/features_rt-plurel", "il", "a100"),
    "rdblearn-exaone": ("rdblearn_exaone", f"{SHARE}/features", "il", "a100"),
}

REPO_ROOT = str(Path(__file__).resolve().parents[4])


def gpu_resources(qos: str, card: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time="4:00:00",
        gpus=f"{card}:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem="64G",
        mem_per_gpu=None,
        constraint="ampere" if card == "a100" else None,
        nodelist="blackwell1" if card == "b200" else None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere6,ampere7,ampere9" if card == "a100" else None,
    )


for arm, (method, features_root, qos, card) in ARMS.items():
    submit(
        "expts.repaper.baselines.rel2tabv2.run:main",
        args=dict(
            method=method,
            db=DB,
            table=TABLE,
            split="test",
            pre_dir=PRE_DIR,
            raw_dir=RAW_DIR,
            features_root=features_root,
            out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}/{arm}",
            ctx_size_list=N_ROWS,
            items_per_task=10_000_000,
            # the sampler only says which rows are queries and what their labels
            # are; its context is discarded, so num_walks=0 and a small
            # local_ctx_size keep it from doing work nobody reads
            local_ctx_size=64,
            bfs_width=32,
            prefer_latest=True,
            num_walks=0,
            walk_length=0,
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
            exaone_ensemble_count=8,
            tabfm_backend="pytorch",
            retriever=RETRIEVER,
            context_sampler="uniform",
            context_split="train",
        ),
        resources=gpu_resources(qos, card),
        name=f"rel2tabv2-{ROUND}-{arm}-{DB}-{TABLE}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )

# Per-query retriever rounds on this task, for reference. Those ctx numbers are
# CELLS; N_ROWS above is rows, and the two are not the same axis.
#
#   retriever="sampler", local_ctx_size=256, ctx_size_list=[8192]
#   rdblearn:  baserate 0.5657 | lgbm 0.8074 | tabicl 0.8102 | tabfm 0.8217
#              | exaone 0.8238
#   rt-j:      lgbm 0.7200 | tabicl 0.7627
#   rt-plurel: lgbm 0.6653 | tabicl 0.7220
#
# Global retriever, n=1024, rdblearn+exaone, before this round's layout:
#   0.7640560191284829, reproduced bit-identically across two runs; 662 of 702
#   predictions distinct, label=1 mean 0.8281 against label=0 mean 0.6961.
