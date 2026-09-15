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

DB = "rel-f1"
TABLE = "driver-dnf"

LOCAL_CTX_SIZE = 256
CTX_SIZE = 8192
ROUND = f"fm_lcs{LOCAL_CTX_SIZE}_ctx{CTX_SIZE}"

# The two tabular foundation models on the rdblearn blob, at the context the
# lgbm (0.8074) and tabicl (0.8102) arms used, so all four predictors are
# directly comparable on the same 73-d features and the same 336 labelled rows.
#
# Unlike TabICLBatchedPredictor, which packs thousands of contexts into one
# forward, both of these are sklearn-shaped: one fit+predict per query, so 702
# sequential model calls per arm (and EXAONE runs ensemble_count of them
# internally). Throughput is unmeasured, hence the generous wall clock.
ARMS = {
    "exaone": "rdblearn_exaone",
    "tabfm": "rdblearn_tabfm",
}

# 2026-09-15: 4 of 8 b200 free, blackwell1 not reserved, and nothing of mine
# holding any tier, so il-interactive's 2 gpus go to blackwell for both arms --
# TabFM is 6.2 GB of weights per forward, and the a100s had only 3 cards free
# across ampere2/ampere8.
REPO_ROOT = str(Path(__file__).resolve().parents[4])

for arm, method in ARMS.items():
    submit(
        "expts.repaper.baselines.rel2tabv2.run:main",
        args=dict(
            method=method,
            db=DB,
            table=TABLE,
            split="test",
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features",
            out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}/{arm}",
            ctx_size_list=[CTX_SIZE],
            items_per_task=10_000_000,
            local_ctx_size=LOCAL_CTX_SIZE,
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
            exaone_ensemble_count=8,
            tabfm_backend="pytorch",
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il-interactive",
            time="8:00:00",
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
        name=f"rel2tabv2-{ROUND}-{arm}-{DB}-{TABLE}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )

# Earlier rounds on this task, all at local_ctx 256 / ctx 8192 unless noted:
#
#   baserate  (183463 ctx=1024 -> 0.7252183672473527, the label base rate;
#              183464 ctx=8192 -> 0.5656907236617381, the floor at this context)
#   lgbm      (183457-9) rdblearn 0.8074, rt-j 0.7200, rt-plurel 0.6653
#   tabicl    (183460-2) rdblearn 0.8102, rt-j 0.7627, rt-plurel 0.7220
#
# ARMS = {
#     "rdblearn": ("rdblearn_lgbm", f"{SHARE}/features"),
#     "rt-j": ("rt_lgbm", f"{SHARE}/features_rt-j"),
#     "rt-plurel": ("rt_lgbm", f"{SHARE}/features_rt-plurel"),
# }
#
# The featurize stages, for when a blob has to be rebuilt. featurize_rt needs
# the curated one-task list: it resolves the whole db_task_list through
# get_tasks before filtering on db, and only rel-f1 is staged here, so
# forecast.json dies on rel-amazon/meta.json (183449/183451).
#
# for arm, ckpt in {
#     "rt-j": "~/scratch/hf/stanford-star/rt-j",
#     "rt-plurel": "~/scratch/hf/stanford-star/rt-plurel",
# }.items():
#     submit(
#         "expts.repaper.baselines.featurize_rt:featurize_db",
#         args=dict(
#             db=DB,
#             db_task_list="expts/repaper/baselines/rel2tabv2/rel-f1_driver-dnf.json",
#             pre_dir=PRE_DIR, features_root=f"{SHARE}/features_{arm}", ckpt=ckpt,
#             local_ctx_size=256, bfs_width=32, shuffle_seed=0, context_seed=0,
#             db_cutoff=None, batch_size=1024,
#         ),
#         resources=Resources(
#             partition="il", account="infolab", qos="il-interactive",
#             time="1:00:00", gpus="a100:1", cpus_per_task=8, ntasks=None,
#             exclusive=False, mem="32G", mem_per_gpu=None,
#             constraint="ampere", nodelist=None, reservation=None,
#             dependency=None, exclude="ampere4,ampere6,ampere7,ampere9",
#         ),
#         name=f"rel2tabv2-feat-{arm}-{DB}",
#     )
#
# And rdblearn (183395), which takes (db, table) directly and needs RAW_DIR
# plus the populated relbench cache:
#
# submit(
#     "expts.repaper.baselines.featurize_rdblearn:featurize_table",
#     args=dict(
#         db=DB, table=TABLE, task_type="clf", pre_dir=PRE_DIR, raw_dir=RAW_DIR,
#         features_root=f"{SHARE}/features",
#         relbench_cache_dir=f"{SHARE}/relbench-cache",
#         max_depth=2, max_train_samples=1000,
#     ),
#     pixi_env="featurize",
#     setup=("pixi install -e featurize", "pixi run install-rdblearn"),
# )
