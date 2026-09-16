import os
import subprocess
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

# The two cheap rel-f1 tasks, one clf and one reg, where the other featurizers
# and predictors already have a number under this exact retriever and context,
# so a new arm is read against them rather than alone.
TASKS = [("rel-f1", "driver-dnf"), ("rel-f1", "driver-position")]
#
# The smoke wave that came before it, run as 184244-184249: rel-amazon/user-churn
# is the largest test split in v1 at 351,885 rows, ~1375 eval batches at
# 2**18/1024, and rel-f1/driver-position is the cheapest reg task at 760 rows,
# which together are the two paths the global retriever had never executed --
# scale and regression.
#
# TASKS = [("rel-amazon", "user-churn"), ("rel-f1", "driver-position")]

# The query-independent retriever: one seeded rng.choice draw of exactly 1024
# rows from the train split, the same context for every query, features
# precomputed once. Verified reproducible (same seed, same 1024 rows), spanning
# 1950-2004 at an 0.868 positive rate against train's 0.880 on rel-f1.
#
# 1024 is a context ROW count, not cells, so it does not share an axis with the
# ctx=256..8192 curves; the per-query ctx=8192 arm averaged 336 labelled rows.
RETRIEVER = "global_train"
# One job per (task, width, featurizer seed) covers all four sizes: the
# retriever draws once at 4096 and takes nested prefixes, so 64 is a subset of
# 256 of 1024 of 4096 and the curve over them varies only in how much context
# there is, not which rows. 4096 fits both tasks' train splits (11,411 for
# driver-dnf, 7,453 for driver-position).
N_ROWS = [64, 256, 1024, 4096]
ROUND = "gnn_width_x_ctx_x_seed"

# 4 widths x 4 featurizer init seeds x 4 context seeds, exaone throughout,
# crossed with N_ROWS above: every (width, ctx) cell is 16 evaluations, varying
# both the GNN's initialization and which rows the retriever drew. Context seed 0
# is already done and skipped by done() below, so this submits the other three.
# (gnn seeds 4-7 at cs0 also exist, from when this was an 8-seed single-context
# sweep; they are outside this grid, and the collation filters them out rather
# than letting them weight cs0.)
ARMS = {
    f"gnn-c{c}-s{seed}-cs{ctx_seed}-exaone": (
        "gnn_exaone",
        f"{SHARE}/features_gnn-c{c}-s{seed}",
        ctx_seed,
        {
            "featurizer": "gnn",
            "predictor": "exaone",
            "channels": c,
            "gnn_seed": seed,
            "context_seed": ctx_seed,
        },
    )
    for c in (8, 32, 128, 512)
    for seed in range(4)
    for ctx_seed in range(4)
}
#
# ARMS = {
#     "rdblearn-exaone": ("rdblearn_exaone", f"{SHARE}/features"),
#     "rdblearn-tabfm": ("rdblearn_tabfm", f"{SHARE}/features"),
#     "rdblearn-tabpfn": ("rdblearn_tabpfn", f"{SHARE}/features"),
# }
#
# The full sweep's six, for when the smoke wave lands:
#
# ARMS = {
#     "rdblearn-exaone": ("rdblearn_exaone", f"{SHARE}/features"),
#     "rdblearn-tabfm": ("rdblearn_tabfm", f"{SHARE}/features"),
#     "rt-j-exaone": ("rt_exaone", f"{SHARE}/features_rt-j"),
#     "rt-j-tabfm": ("rt_tabfm", f"{SHARE}/features_rt-j"),
#     "rt-plurel-exaone": ("rt_exaone", f"{SHARE}/features_rt-plurel"),
#     "rt-plurel-tabfm": ("rt_tabfm", f"{SHARE}/features_rt-plurel"),
# }

# 2026-09-16: 24 evals of 702 and 760 rows, a couple of minutes each now that
# each one sweeps four context sizes up to 4096 rows, so they go to a100 under
# il, whose cap is 10 -- the queue rolls through them. The b200 cap is 2 for the
# whole il partition, not 2 per qos: partition il carries QoS=il-part
# (gres/gpu:b200=2), and only il-lo is flagged OverPartQOS, so a b200 job counts
# against that 2 whichever qos it names -- which is what left the last round's
# rel-f1 pair pending on QOSMaxGRESPerUser behind its own siblings. Two rel-f1
# evals of 702 and 760 rows have no use for a b200 anyway, so both go to a100
# under il (cap 10, 80 G cards). il-interactive caps wall at 12 h.
QOS = {"rel-amazon": "il-interactive", "rel-f1": "il"}
CARD = {"rel-amazon": "b200", "rel-f1": "a100"}
TIME = {"rel-amazon": "12:00:00", "rel-f1": "2:00:00"}
# rel-amazon's preprocessed dir is 33 G and mmap_populate faults all of it in;
# featurize_rt needed 240 G on the same db. rel-f1 is 12 M.
MEM = {"rel-amazon": "240G", "rel-f1": "32G"}

REPO_ROOT = str(Path(__file__).resolve().parents[4])


# run.main also returns early on an existing json, but that still costs a job, a
# clone and a pixi install per finished arm -- 24 of them when this sweep went
# from 3 seeds to 8. Skip them here instead.
def done(arm: str, db: str, table: str) -> bool:
    out = Path(OUT_ROOT).expanduser() / "repaper-rel2tabv2" / ROUND / arm
    return (out / f"{db}__{table}.json").exists()


# Same guard as submit_featurize.py: a job still running has written no json yet
# and would be duplicated onto the same out_dir.
def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()


def gpu_resources(db: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=QOS[db],
        time=TIME[db],
        gpus=f"{CARD[db]}:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem=MEM[db],
        mem_per_gpu=None,
        constraint="ampere" if CARD[db] == "a100" else None,
        nodelist="blackwell1" if CARD[db] == "b200" else None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere6,ampere7,ampere9" if CARD[db] == "a100" else None,
    )


for db, table in TASKS:
    for arm, (method, features_root, ctx_seed, tags) in ARMS.items():
        name = f"rel2tabv2-{ROUND}-{arm}-{db}-{table}"
        if name in busy or done(arm, db, table):
            continue
        submit(
            "expts.repaper.baselines.rel2tabv2.run:main",
            args=dict(
                method=method,
                db=db,
                table=table,
                split="test",
                pre_dir=PRE_DIR,
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
                # The retriever's row draw. shuffle_seed stays 0 so the query
                # set and its order -- hence the labels -- are identical across
                # context seeds and the cells stay comparable.
                context_seed=ctx_seed,
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
                tabpfn_dir=f"{SHARE}/tabpfn",
                # "auto" rather than an integer: tabpfn raises n_estimators on a
                # dataset wider than max_features_per_estimator so that every
                # feature is seen by some ensemble member, and the three feature
                # roots here are 4 to 512 columns wide. A fixed count would
                # silently drop columns on the wide ones.
                tabpfn_n_estimators="auto",
                # The global retriever fits one context and then predicts on
                # every batch against it, which is exactly what fit_with_cache
                # is for: the train KV cache is built once in fit(). Switch this
                # to "fit_preprocessors" together with retriever="sampler",
                # where every query refits and a cache would never be reused.
                tabpfn_fit_mode="fit_with_cache",
                retriever=RETRIEVER,
                context_sampler="uniform",
                context_split="train",
                tags=tags,
            ),
            resources=gpu_resources(db),
            name=name,
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
        )

# Per-query retriever rounds on rel-f1/driver-dnf, for reference. Those ctx
# numbers are CELLS; N_ROWS above is rows, and the two are not the same axis.
#
#   retriever="sampler", local_ctx_size=256, ctx_size_list=[8192]
#   rdblearn:  baserate 0.5657 | lgbm 0.8074 | tabicl 0.8102 | tabfm 0.8217
#              | exaone 0.8238
#   rt-j:      lgbm 0.7200 | tabicl 0.7627
#   rt-plurel: lgbm 0.6653 | tabicl 0.7220
#
# Global retriever, n=1024, on the same task:
#   rdblearn:  exaone 0.7640 | tabfm 0.7740
#   rt-j:      exaone 0.7407 | tabfm 0.7416
#   rt-plurel: exaone 0.6758 | tabfm 0.6706
