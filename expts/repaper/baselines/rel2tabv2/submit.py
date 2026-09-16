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

# Smoke wave before the 21-task sweep. The global retriever has only ever run on
# rel-f1/driver-dnf: 702 test rows, clf. Two paths the sweep needs have never
# executed, and they are independent, so they are tested apart rather than
# paying for both at once. rel-amazon/user-churn is the largest test split in v1
# at 351,885 rows, ~1375 eval batches at 2**18/1024, and its rdblearn blob is 12
# features; rel-f1/driver-position is the cheapest reg task at 760 rows, which
# exercises PreprocessedLabels' normalised-label path and metric_for's reg
# branch for the price of a few minutes.
TASKS = [("rel-amazon", "user-churn"), ("rel-f1", "driver-position")]

# The query-independent retriever: one seeded rng.choice draw of exactly 1024
# rows from the train split, the same context for every query, features
# precomputed once. Verified reproducible (same seed, same 1024 rows), spanning
# 1950-2004 at an 0.868 positive rate against train's 0.880 on rel-f1.
#
# 1024 is a context ROW count, not cells, so it does not share an axis with the
# ctx=256..8192 curves; the per-query ctx=8192 arm averaged 336 labelled rows.
RETRIEVER = "global_train"
N_ROWS = [1024]
ROUND = "global_uniform_seed0_n1024"

# Only the rdblearn featurizer for the smoke wave: what is untested is the
# retriever/labels/evaluator path, which is shared by all three feature roots,
# and rdblearn is the one whose rel-f1 numbers are already known.
ARMS = {
    "rdblearn-exaone": ("rdblearn_exaone", f"{SHARE}/features"),
    "rdblearn-tabfm": ("rdblearn_tabfm", f"{SHARE}/features"),
}
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

# 2026-09-16: blackwell1 has 5 of 8 b200 free with 1.9 T of its 3 T memory free,
# and nothing of mine is on any gpu tier. il-interactive's 2 gpus go to the
# rel-amazon pair -- 351,885 queries against rel-f1's 760, so they are the whole
# wall clock -- and il's 2-b200 sub-cap takes the rel-f1 pair. Four b200 asked
# for, four free, nothing pends. il-interactive caps wall at 12 h.
QOS = {"rel-amazon": "il-interactive", "rel-f1": "il"}
TIME = {"rel-amazon": "12:00:00", "rel-f1": "2:00:00"}
# rel-amazon's preprocessed dir is 33 G and mmap_populate faults all of it in;
# featurize_rt needed 240 G on the same db. rel-f1 is 12 M.
MEM = {"rel-amazon": "240G", "rel-f1": "32G"}

REPO_ROOT = str(Path(__file__).resolve().parents[4])


def gpu_resources(db: str, card: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=QOS[db],
        time=TIME[db],
        gpus=f"{card}:1",
        cpus_per_task=8,
        ntasks=None,
        exclusive=False,
        mem=MEM[db],
        mem_per_gpu=None,
        constraint="ampere" if card == "a100" else None,
        nodelist="blackwell1" if card == "b200" else None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere6,ampere7,ampere9" if card == "a100" else None,
    )


for db, table in TASKS:
    for arm, (method, features_root) in ARMS.items():
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
            resources=gpu_resources(db, "b200"),
            name=f"rel2tabv2-{ROUND}-{arm}-{db}-{table}",
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
