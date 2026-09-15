from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT,
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)

DB = "rel-f1"
TABLE = "driver-dnf"

# Both released checkpoints are the same architecture and shape (config.json
# identical, d_model=512, 85.6M params), so the arms differ only in what the
# featurizer was pretrained on. rt-plurel's root safetensors is *not* the
# legacy PluRelTransformer -- those are its paper/*.pt files at d_model=256,
# which featurize_plurel takes.
ARMS = {
    "rt-j": CKPT,
    "rt-plurel": "~/scratch/hf/stanford-star/rt-plurel",
}

# 2026-09-15: the two featurize passes take il-interactive (2 gpus, priority
# 1500, and nothing of mine is holding it) on amperes rather than blackwell --
# all 8 b200 are held by other users' il-lo jobs at 0:10-2:48 elapsed against
# 2-3 day limits, so none frees inside the minute this job needs on an a100
# (README: rel-f1 RT featurize is about a minute). Free a100 at submit:
# ampere2 1, ampere8 2, the other amperes full; ampere4/6/7/9 stay excluded.
EXCLUDE_CPU = (
    "hyperion1,hyperion3,hyperturing1,hyperturing2,madmax2,"
    "madmax3,madmax4,madmax6,madmax7,trinity,turing1,turing2,turing3"
)

REPO_ROOT = str(Path(__file__).resolve().parents[4])


def gpu_resources(cpus: int, mem: str, time: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos="il-interactive",
        time=time,
        gpus="a100:1",
        cpus_per_task=cpus,
        ntasks=None,
        exclusive=False,
        mem=mem,
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere6,ampere7,ampere9",
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
        exclude=EXCLUDE_CPU,
    )


for arm, ckpt in ARMS.items():
    features_root = f"{SHARE}/features_{arm}"

    feat = submit(
        "expts.repaper.baselines.featurize_rt:featurize_db",
        args=dict(
            db=DB,
            # not forecast.json: featurize_rt resolves the whole list
            # through get_tasks before it filters on db, so every db in it
            # needs a staged meta.json, and only rel-f1 is staged here
            # (183449/183451 died on rel-amazon/meta.json). Repo-relative,
            # so it resolves from the clone root the ranks run in.
            db_task_list="expts/repaper/baselines/rel2tabv2/rel-f1_driver-dnf.json",
            pre_dir=PRE_DIR,
            features_root=features_root,
            ckpt=ckpt,
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            db_cutoff=None,
            batch_size=1024,
        ),
        resources=gpu_resources(8, "32G", "1:00:00"),
        name=f"rel2tabv2-feat-{arm}-{DB}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )

    # ctx == local_ctx == 1024 as in the rdblearn arm, so the three are
    # comparable: the query row's own BFS neighborhood takes the whole budget
    # and the retriever is left ~5 labelled rows (measured 4.8 on this task).
    submit(
        "expts.repaper.baselines.rel2tabv2.run:main",
        args=dict(
            method="rt_lgbm",
            db=DB,
            table=TABLE,
            split="test",
            pre_dir=PRE_DIR,
            features_root=features_root,
            out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/rt_lgbm-{arm}",
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
        name=f"rel2tabv2-rt_lgbm-{arm}-{DB}-{TABLE}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        after=feat.id,
    )

# The rdblearn arm, run 2026-09-15 as jobs 183395/183396 (roc_auc 0.7252):
#
# feat = submit(
#     "expts.repaper.baselines.featurize_rdblearn:featurize_table",
#     args=dict(
#         db=DB, table=TABLE, task_type="clf", pre_dir=PRE_DIR, raw_dir=RAW_DIR,
#         features_root=f"{SHARE}/features",
#         relbench_cache_dir=f"{SHARE}/relbench-cache",
#         max_depth=2, max_train_samples=1000,
#     ),
#     resources=cpu_resources(16, "32G", "2:00:00"),
#     name=f"rel2tabv2-feat-rdbl-{DB}-{TABLE}",
#     repo_root=REPO_ROOT, cluster=ILC, job_env="expts/job_env.sh",
#     log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
#     clone_root=CLONE_ROOT, secrets_dir=SECRETS_DIR,
#     pixi_env="featurize",
#     setup=("pixi install -e featurize", "pixi run install-rdblearn"),
# )
