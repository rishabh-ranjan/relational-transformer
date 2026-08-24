import json
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
)
from roach.slurm import Resources, submit

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = f"{OUT_ROOT}/repaper-valtest/tuned"
LOG_ROOT = f"{LOG_ROOT}/repaper/valtest/slurm-logs"

MEM = {
    "rel-amazon": "120G",
    "rel-avito": "48G",
    "rel-event": "48G",
    "rel-f1": "24G",
    "rel-hm": "64G",
    "rel-stack": "64G",
    "rel-trial": "48G",
}

if __name__ == "__main__":
    cfgs = json.loads(
        (REPO_ROOT / "expts" / "repaper" / "tune" / "tuned_configs.json").read_text()
    )
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        if (Path(OUT_DIR).expanduser() / f"{db}__{table}.json").exists():
            continue
        ctx, lcs, bw, pl = rec["best_cfg"]
        submit(
            "expts.repaper.scaling.run:main",
            args=dict(
                method="rt",
                db=db,
                table=table,
                split="test",
                pre_dir=PRE_DIR,
                features_root=None,
                out_dir=OUT_DIR,
                ctx_size_list=[int(ctx)],
                items_per_task=8192,
                local_ctx_size=int(lcs),
                bfs_width=int(bw),
                prefer_latest=bool(pl),
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
                tabicl_dir=None,
                tabicl_max_batch_size=1024,
                tabicl_min_bin_size=48,
                tabicl_softmax_temperature=0.9,
                lgbm_n_jobs=8,
            ),
            resources=Resources(
                partition="il",
                account="infolab",
                qos="il-lo",
                time="12:00:00",
                gpus="a100:1",
                cpus_per_task=8,
                ntasks=None,
                exclusive=False,
                mem=MEM[db],
                mem_per_gpu=None,
                constraint="ampere",
                nodelist=None,
                reservation=None,
                dependency=None,
            ),
            name=f"valtest-{db}-{table}",
            repo_root=str(REPO_ROOT),
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=LOG_ROOT,
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
        )
