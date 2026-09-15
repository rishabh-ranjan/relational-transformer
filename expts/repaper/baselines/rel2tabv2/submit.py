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
ROUND = f"tabicl_lcs{LOCAL_CTX_SIZE}_ctx{CTX_SIZE}"

# The same three cached blobs the lgbm round used, now through TabICLv2.
# LightGBM at this context gave rdblearn 0.8074, rt-j 0.7200, rt-plurel
# 0.6653 -- both 512-d arms at or below the 0.7252 label-base-rate number,
# which is the regime (336 rows x 512 features) LightGBM handles worst and
# in-context tabular learning handles best, so this is the fairer predictor
# for the embedding featurizers.
#
# 2026-09-15: 2 of 8 b200 free and blackwell1 not reserved, so il-interactive
# (2 gpus, priority 1500, nothing of mine holding it) goes to both of them,
# spent on the two 512-d arms -- a CPU probe at n_train=336 put d=512 at
# 37.6s/4 items against 6.4s for d=73, so they are ~10x the work. rdblearn
# takes il on an ampere (ampere2 1 free, ampere8 2 free).
ARMS = {
    "rt-j": ("rt_tabicl", f"{SHARE}/features_rt-j", "b200"),
    "rt-plurel": ("rt_tabicl", f"{SHARE}/features_rt-plurel", "b200"),
    "rdblearn": ("rdblearn_tabicl", f"{SHARE}/features", "a100"),
}

REPO_ROOT = str(Path(__file__).resolve().parents[4])


def gpu_resources(card: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-interactive" if card == "b200" else "il",
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


for arm, (method, features_root, card) in ARMS.items():
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
            lgbm_n_jobs=8,
        ),
        resources=gpu_resources(card),
        name=f"rel2tabv2-{ROUND}-{arm}-{DB}-{TABLE}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )

# The lgbm round (183457-183459), zero-gres on il-cpu with 16 cpus:
#
# ARMS = {
#     "rdblearn": ("rdblearn_lgbm", f"{SHARE}/features"),
#     "rt-j": ("rt_lgbm", f"{SHARE}/features_rt-j"),
#     "rt-plurel": ("rt_lgbm", f"{SHARE}/features_rt-plurel"),
# }
# resources=Resources(
#     partition="il-cpu", account="infolab", qos="il-cpu", time="4:00:00",
#     gpus="0", cpus_per_task=16, ntasks=1, exclusive=False, mem="32G",
#     mem_per_gpu=None, constraint=None, nodelist=None, reservation=None,
#     dependency=None, exclude=(
#         "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax2,"
#         "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
#     ),
# )
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
#         resources=gpu_resources("a100"),
#         name=f"rel2tabv2-feat-{arm}-{DB}",
#         ...
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
