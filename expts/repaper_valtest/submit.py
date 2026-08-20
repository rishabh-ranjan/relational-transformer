"""Submit the tuned arm of the default-vs-tuned table: one job per task.

The table compares the shared default context (8192, 256, 32, pl=1) against
each task's tuned configuration on the 8192-row test subsample at a single
context seed. The default column is the `subsampled/rt` arm of
``../repaper_scaling`` (identical protocol); only the tuned runs live here,
through the same runner.
"""

import json
from pathlib import Path

from roach.slurm import Resources, submit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"
OUT_DIR = "/dfs/user/ranjanr/ckpts/rtv2/repaper-valtest/tuned"
LOG_ROOT = (
    "/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/"
    "expts/repaper_valtest"
)

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
        (REPO_ROOT / "expts" / "repaper_tune" / "tuned_configs.json").read_text()
    )
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        if (Path(OUT_DIR) / f"{db}__{table}.json").exists():
            continue
        ctx, lcs, bw, pl = rec["best_cfg"]
        submit(
            "expts.repaper_scaling.run:main",
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
                ckpt_clf="/dfs/user/ranjanr/share/stanford-star/rt-j/classification",
                ckpt_reg="/dfs/user/ranjanr/share/stanford-star/rt-j/regression",
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
            log_root=LOG_ROOT,
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
        )
