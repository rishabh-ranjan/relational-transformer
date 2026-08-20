"""Submit the leaderboard-ensemble units: 21 tasks x top-4 configs.

Each unit reuses the ensemble runner (``expts.repaper_enscurve.run``) at one
of the task's top-4 validation configurations, 4 context seeds, on the FULL
official test split -- 16 raw predictions per task in total, which
``reduce.py`` averages into the RelBench submission and the paper's
tuned+ensembled table.
"""

import json
from pathlib import Path

from roach.slurm import Resources, submit

from expts.repaper_config import (
    CKPT_CLF,
    CKPT_REG,
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = f"{OUT_ROOT}/repaper-submit"
LOG_ROOT = f"{LOG_ROOT}/repaper_submit"

MEM = {
    "rel-amazon": "120G",
    "rel-avito": "48G",
    "rel-event": "48G",
    "rel-f1": "24G",
    "rel-hm": "64G",
    "rel-stack": "64G",
    "rel-trial": "48G",
}


def resources(db: str, qos: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time="12:00:00" if qos == "il-interactive" else "2-00:00:00",
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
    )


if __name__ == "__main__":
    cfgs = json.loads(
        (REPO_ROOT / "expts" / "repaper_tune" / "tuned_configs.json").read_text()
    )
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        for rank, (ctx, lcs, bw, pl) in enumerate(rec["top_cfgs"]):
            out_dir = f"{OUT_ROOT}/cfg{rank}"
            if (Path(out_dir) / f"{db}__{table}.json").exists():
                continue
            submit(
                "expts.repaper_enscurve.run:main",
                args=dict(
                    variant=f"cfg{rank}",
                    db=db,
                    table=table,
                    ctx_size=int(ctx),
                    local_ctx_size=int(lcs),
                    bfs_width=int(bw),
                    prefer_latest=bool(pl),
                    n_seeds=4,
                    items_per_task=10_000_000,
                    split="test",
                    pre_dir=PRE_DIR,
                    out_dir=out_dir,
                    num_walks=10_000,
                    walk_length=20,
                    shuffle_seed=0,
                    context_seed=0,
                    tokens_per_gpu=2**18,
                    num_workers=8,
                    prefetch_factor=2,
                    mmap_populate=True,
                    db_cutoff=None,
                    ckpt_clf=CKPT_CLF,
                    ckpt_reg=CKPT_REG,
                ),
                resources=resources(db, "il-lo"),
                name=f"sub-cfg{rank}-{db}-{table}",
                repo_root=str(REPO_ROOT),
                log_root=LOG_ROOT,
                clone_root=CLONE_ROOT,
                secrets_dir=SECRETS_DIR,
            )
