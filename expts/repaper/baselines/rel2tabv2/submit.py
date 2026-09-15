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

# ctx == local_ctx == 1024 gave all three arms a bit-identical
# roc_auc=0.7252183672473527: with ~4.8 visible rows the context is always
# single-class, so lgbm._fit_predict returns float(y_int[0]) without ever
# touching X and the features cannot matter. local_ctx_size=256 at ctx=8192
# puts 337 labelled rows in front of the predictor instead (2026-09-15 probe),
# which is the first config that can actually separate the featurizers.
LOCAL_CTX_SIZE = 256
CTX_SIZE = 8192
ROUND = f"lcs{LOCAL_CTX_SIZE}_ctx{CTX_SIZE}"

# Every blob is already built and cached, so this round is the cpu stage only:
# rdblearn from 183395, rt-j from 183453, rt-plurel from 183455. The two rt
# arms share featurize_rt and differ only in the checkpoint's pretraining,
# which is why they need separate feature roots.
ARMS = {
    "rdblearn": ("rdblearn_lgbm", f"{SHARE}/features"),
    "rt-j": ("rt_lgbm", f"{SHARE}/features_rt-j"),
    "rt-plurel": ("rt_lgbm", f"{SHARE}/features_rt-plurel"),
}

EXCLUDE_CPU = (
    "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax2,"
    "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
)

REPO_ROOT = str(Path(__file__).resolve().parents[4])

for arm, (method, features_root) in ARMS.items():
    submit(
        "expts.repaper.baselines.rel2tabv2.run:main",
        args=dict(
            method=method,
            db=DB,
            table=TABLE,
            split="test",
            pre_dir=PRE_DIR,
            features_root=features_root,
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
            lgbm_n_jobs=16,
        ),
        # 337 rows x up to 512 features per fit and 702 fits, against ~5 rows
        # last round, so 16 cpus for the joblib fan-out. il-cpu is uncapped and
        # rambo alone had 280 free cpus, so all three arms run at once.
        resources=Resources(
            partition="il-cpu",
            account="infolab",
            qos="il-cpu",
            time="4:00:00",
            gpus="0",
            cpus_per_task=16,
            ntasks=1,
            exclusive=False,
            mem="32G",
            mem_per_gpu=None,
            constraint=None,
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude=EXCLUDE_CPU,
        ),
        name=f"rel2tabv2-{ROUND}-{arm}-{DB}-{TABLE}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )

# The featurize stages, kept for when a blob has to be rebuilt. The rt one
# needs the curated one-task list: featurize_rt resolves the whole
# db_task_list through get_tasks before filtering on db, and only rel-f1 is
# staged here, so forecast.json dies on rel-amazon/meta.json (183449/183451).
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
#             pre_dir=PRE_DIR,
#             features_root=f"{SHARE}/features_{arm}",
#             ckpt=ckpt,
#             local_ctx_size=256,
#             bfs_width=32,
#             shuffle_seed=0,
#             context_seed=0,
#             db_cutoff=None,
#             batch_size=1024,
#         ),
#         resources=Resources(
#             partition="il", account="infolab", qos="il-interactive",
#             time="1:00:00", gpus="a100:1", cpus_per_task=8, ntasks=None,
#             exclusive=False, mem="32G", mem_per_gpu=None,
#             constraint="ampere", nodelist=None, reservation=None,
#             dependency=None, exclude="ampere4,ampere6,ampere7,ampere9",
#         ),
#         name=f"rel2tabv2-feat-{arm}-{DB}",
#         repo_root=REPO_ROOT, cluster=ILC, job_env="expts/job_env.sh",
#         log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
#         clone_root=CLONE_ROOT, secrets_dir=SECRETS_DIR,
#     )
#
# And the rdblearn one (183395), which takes (db, table) directly and needs
# RAW_DIR plus the populated relbench cache:
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
