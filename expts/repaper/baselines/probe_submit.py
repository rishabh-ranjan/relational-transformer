import os
import subprocess
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT_CLF,
    CKPT_REG,
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)
from roach.slurm import Resources, submit

TASKS = [
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "user-ltv"),
    ("rel-stack", "post-votes"),
]

# 2026-08-27 22:40: with the feature fix in, RDBLearn + TabICLv2 still gets
# worse with context on the heavy-tailed regression tasks (item-ltv 10.9 ->
# 23.0 raw MAE from 256 to 8192 cells, user-ltv 43.5 -> 75.1, post-votes
# 26.3 -> 45.4) while SQL + TabICLv2 improves on the same contexts and
# LightGBM on the same features is roughly flat. Same runner, 1024 query rows,
# three context sizes: the RDBLearn features as they are, clipped at |z|<=4,
# rank-gauss transformed, with the zero-heavy columns dropped, and the SQL
# features as a control.
ARMS = {
    "asis": ("rdblearn_tabicl", f"{SHARE}/features"),
    "clip4": ("rdblearn_tabicl", f"{SHARE}/features_probe/clip4"),
    "rankgauss": ("rdblearn_tabicl", f"{SHARE}/features_probe/rankgauss"),
    "dropzero": ("rdblearn_tabicl", f"{SHARE}/features_probe/dropzero"),
    "nocal": ("rdblearn_tabicl", f"{SHARE}/features_probe/nocal"),
    "noyear": ("rdblearn_tabicl", f"{SHARE}/features_probe/noyear"),
    "sql": ("sql_tabicl", f"{SHARE}/features"),
}


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()

for arm, (method, features_root) in ARMS.items():
    for db, table in TASKS:
        out_dir = f"{OUT_ROOT}/repaper-probe/{arm}"
        name = f"repaper-probe-{arm}-{db}-{table}"
        if (Path(out_dir).expanduser() / f"{db}__{table}.json").exists():
            continue
        if name in busy:
            continue
        if arm == "dropzero" and db == "rel-amazon":
            continue
        submit(
            "expts.repaper.scaling.run:main",
            args=dict(
                method=method,
                db=db,
                table=table,
                split="test",
                pre_dir=PRE_DIR,
                features_root=features_root,
                out_dir=out_dir,
                ctx_size_list=[256, 2048, 8192],
                items_per_task=1024,
                local_ctx_size=256,
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
                ckpt_clf=CKPT_CLF,
                ckpt_reg=CKPT_REG,
                tabicl_dir=f"{SHARE}/tabicl",
                tabicl_max_batch_size=1024,
                tabicl_min_bin_size=48,
                tabicl_softmax_temperature=0.9,
                lgbm_n_jobs=8,
            ),
            resources=Resources(
                partition="il",
                account="infolab",
                qos="il",
                time="2:00:00",
                gpus="a100:1",
                cpus_per_task=8,
                ntasks=None,
                exclusive=False,
                mem="120G" if db == "rel-amazon" else "64G",
                mem_per_gpu=None,
                constraint="ampere",
                nodelist=None,
                reservation=None,
                dependency=None,
                exclude="ampere4,ampere6,ampere7",
            ),
            name=name,
            repo_root=str(Path(__file__).resolve().parents[3]),
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            setup=("pixi run maturin develop --uv --release --features vecdb",),
        )
