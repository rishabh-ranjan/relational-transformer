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

# What is 0.7252183672473527? Every arm scored it, bit-identical, at
# ctx == local_ctx == 1024, which proves the predictions cannot depend on the
# features but not what they *are*. The claim to check is that it is the
# visible-label base rate: with ~4.8 rows the context is single-class, so
# lgbm._fit_predict returns float(y_int[0]) and the prediction is the shared
# label of the query driver's recent races. BaseRatePredictor ignores X and
# returns mean(y_train > 0), through the identical sampler and seeds, so
# ctx=1024 reproducing that constant exactly settles it. ctx=8192 gives the
# label-only reference at the context the lgbm/tabicl rounds used.
CONFIGS = [(1024, 1024), (256, 8192)]

EXCLUDE_CPU = (
    "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax2,"
    "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
)

REPO_ROOT = str(Path(__file__).resolve().parents[4])

for local_ctx_size, ctx_size in CONFIGS:
    round_name = f"baserate_lcs{local_ctx_size}_ctx{ctx_size}"
    submit(
        "expts.repaper.baselines.rel2tabv2.run:main",
        args=dict(
            # the featurizer is loaded and then ignored, so any built blob
            # does; rdblearn's is the cheapest to read at 73 wide
            method="rdblearn_baserate",
            db=DB,
            table=TABLE,
            split="test",
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features",
            out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{round_name}",
            ctx_size_list=[ctx_size],
            items_per_task=10_000_000,
            local_ctx_size=local_ctx_size,
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
        # no fits at all, just a mean per query, so this is the cheapest arm
        # yet: zero-gres on uncapped il-cpu, 8 cpus for the sampler.
        resources=Resources(
            partition="il-cpu",
            account="infolab",
            qos="il-cpu",
            time="2:00:00",
            gpus="0",
            cpus_per_task=8,
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
        name=f"rel2tabv2-{round_name}-{DB}-{TABLE}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )

# The tabicl round (183460-183462): two 512-d arms on the free b200 via
# il-interactive, rdblearn on an ampere via il.
#
# ARMS = {
#     "rt-j": ("rt_tabicl", f"{SHARE}/features_rt-j", "b200"),
#     "rt-plurel": ("rt_tabicl", f"{SHARE}/features_rt-plurel", "b200"),
#     "rdblearn": ("rdblearn_tabicl", f"{SHARE}/features", "a100"),
# }
#
# The lgbm round (183457-183459), zero-gres on il-cpu with 16 cpus:
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
