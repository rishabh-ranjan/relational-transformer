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
METHOD = "rdblearn_exaone"

# The query-independent retriever: one uniform seeded draw from the *train*
# split, the same context for every query, its features precomputed once. The
# only thing that differs from the per-query arms is which rows are in the
# context -- given the same rows the computation is identical, which is what
# predict_shared's bit-exactness against predict_batch establishes.
#
# These numbers are context ROW counts, not cells, so they do not belong on the
# same axis as the ctx=256..8192 curves. 336 is here because it is what the
# per-query ctx=8192 arm averaged (mean_labels 336.3), making that one point a
# like-for-like comparison against its 0.8238.
RETRIEVER = "global_train"
N_ROWS = [64, 256, 336, 1024]
ROUND = f"global_uniform_seed0_{METHOD}"

# The sampler still runs, only to say which rows are queries and what their
# labels are; its context is discarded. num_walks=0 and a small local_ctx_size
# keep it from doing work nobody reads. max(N_ROWS) sets eval_bs
# (2**18 // 1024 = 256), so 702 test rows come in 3 batches and EXAONE fits
# 3 x 4 = 12 times rather than once per query.
REPO_ROOT = str(Path(__file__).resolve().parents[4])

# 2026-09-15: 4 of 8 b200 free, blackwell1 not reserved, nothing of mine on any
# gpu tier, so il-interactive takes a b200. EXAONE is 161 MB of weights with an
# 8-member ensemble, and the 1024-row contexts are the expensive end.
submit(
    "expts.repaper.baselines.rel2tabv2.run:main",
    args=dict(
        method=METHOD,
        db=DB,
        table=TABLE,
        split="test",
        pre_dir=PRE_DIR,
        raw_dir=RAW_DIR,
        features_root=f"{SHARE}/features",
        out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}",
        ctx_size_list=N_ROWS,
        items_per_task=10_000_000,
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
    resources=Resources(
        partition="il",
        account="infolab",
        qos="il-interactive",
        time="4:00:00",
        gpus="b200:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem="64G",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
        exclude=None,
    ),
    name=f"rel2tabv2-{ROUND}-{DB}-{TABLE}",
    repo_root=REPO_ROOT,
    cluster=ILC,
    job_env="expts/job_env.sh",
    log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
)

# Per-query retriever rounds on this task, for reference. Those ctx numbers are
# CELLS; the ones above are rows.
#
#   retriever="sampler", local_ctx_size=256, ctx_size_list=[8192]
#   baserate 0.5657 | lgbm 0.8074 | tabicl 0.8102 | tabfm 0.8217 | exaone 0.8238
#   (rdblearn 73-d features; rt-j 0.7627 and rt-plurel 0.7220 under tabicl)
