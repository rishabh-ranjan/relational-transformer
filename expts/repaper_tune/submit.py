"""Submit the context-hyperparameter grid: one tuning job per task.

Each job is ``rt.eval:main`` in tune-only mode: it scores every surviving
(ctx, lcs, bw, pl) configuration on the task's validation split -- 4096 rows,
each configuration's prediction averaged over 4 context seeds, the val split
evaluated under its own db cutoff -- and writes ``tuning.json`` with every
configuration's score. ``collect.py`` gathers the 21 files into the committed
``tuned_configs.json`` that the enscurve, val-vs-test and leaderboard
experiments read.

Grid: ctx {512,1024,2048,4096,8192} x lcs {256,...,8192 | lcs<=ctx} x
bw {16,64,256} x pl {T,F} = 120 configurations per task, evaluated as
36 evaluator passes (all ctx sizes of a (lcs,bw,pl) entry share one pass).
"""

import json
from pathlib import Path

from roach.slurm import Resources, submit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"
OUT_ROOT = "/dfs/user/ranjanr/ckpts"
PROJECT = "2026-08-19-repaper-tune"
LOG_ROOT = (
    "/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/"
    "expts/repaper_tune"
)

TASKS = [
    tuple(p)
    for p in json.loads((Path(PRE_DIR) / "db-task-lists" / "forecast.json").read_text())
]

GRID = [
    (lcs, bw, pl)
    for lcs in (256, 512, 1024, 2048, 4096, 8192)
    for bw in (16, 64, 256)
    for pl in (True, False)
]

MEM = {
    "rel-amazon": "120G",
    "rel-avito": "48G",
    "rel-event": "48G",
    "rel-f1": "24G",
    "rel-hm": "64G",
    "rel-stack": "64G",
    "rel-trial": "48G",
}


def resources(db: str, qos: str, gpu: str = "a100:1") -> Resources:
    b200 = gpu.startswith("b200")
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time="12:00:00" if qos == "il-interactive" else "2-00:00:00",
        gpus=gpu,
        cpus_per_task=36 if b200 else 8,
        ntasks=None,
        exclusive=False,
        # blackwell's DefMemPerGPU (240G) covers the biggest db; an explicit
        # --mem on ampere is capped by the per-gpu plugin instead.
        mem=None if b200 else MEM[db],
        mem_per_gpu=None,
        constraint=None if b200 else "ampere",
        nodelist="blackwell1" if b200 else None,
        reservation=None,
        dependency=None,
    )


def submit_task(db: str, table: str, qos: str, gpu: str = "a100:1") -> None:
    run_id = f"tune--{db}--{table}"
    tuning = Path(OUT_ROOT) / "rtv2" / PROJECT / run_id / "tuning.json"
    if tuning.exists():
        return
    submit(
        "rt.eval:main",
        args=dict(
            load_ckpt_path=None,  # filled below by task type
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            splits=["val"],
            db_task_list=[(db, table)],
            pre_dir=PRE_DIR,
            tokens_per_gpu=2**18,
            num_workers=8,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            val_items_per_task=4096,
            test_items_per_task=None,
            ctx_size_list=[512, 1024, 2048, 4096, 8192],
            mmap_populate=True,
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            # No db-level cutoff: per-row temporal masking is the only trim
            # (a val target sees history up to its own timestamp; future task
            # labels stay out via the sampler's timestamp and same-horizon
            # filters).
            db_cutoff=None,
            lcs_bw_pl_grid=GRID,
            val_ensemble_size=4,
            test_ensemble_size=1,
            run_name=None,
            targets={},
            project=PROJECT,
            entity="rtv2",
            out_root=OUT_ROOT,
            wandb_disabled=True,
        )
        | {
            "load_ckpt_path": (
                "/dfs/user/ranjanr/share/stanford-star/rt-j/classification"
                if (db, table) in CLF
                else "/dfs/user/ranjanr/share/stanford-star/rt-j/regression"
            )
        },
        resources=resources(db, qos, gpu),
        name=f"tune-{db}-{table}",
        run_id=run_id,
        repo_root=str(REPO_ROOT),
        log_root=LOG_ROOT,
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
    )


CLF = {
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "user-churn"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-badge"),
    ("rel-stack", "user-engagement"),
    ("rel-trial", "study-outcome"),
}

if __name__ == "__main__":
    # The grid is the critical path (the tuned enscurve arm, the
    # default-vs-tuned table and the leaderboard ensemble all wait on it), so
    # it gets the high tiers -- and blackwell1's eight b200s are all held by
    # il-lo jobs right now, so the high tiers preempt straight onto them:
    # 2 b200 through il (its b200 sub-cap) + 2 through il-interactive, on the
    # four rel-amazon tasks, the longest tuning jobs. This round moves those
    # four off the amperes they started on (a move costs at most one in-flight
    # grid entry; ensemble_resume.pt carries the finished ones).
    HIGH = {
        ("rel-amazon", "user-churn"): "il",
        ("rel-amazon", "user-ltv"): "il",
        ("rel-amazon", "item-churn"): "il-interactive",
        ("rel-amazon", "item-ltv"): "il-interactive",
    }
    # The amazon grids finished and handed the high tiers back; the four
    # biggest still-pending jobs take their b200 slots (yanay's refill is
    # il-lo again, so the tiers preempt straight on).
    PROMOTE = {
        ("rel-hm", "item-sales"): "il",
        ("rel-hm", "user-churn"): "il",
        ("rel-stack", "post-votes"): "il-interactive",
        ("rel-stack", "user-badge"): "il-interactive",
    }
    for (db, table), qos in PROMOTE.items():
        submit_task(db, table, qos, gpu="b200:1")
    # for db, table in TASKS:
    #     if (db, table) not in HIGH:
    #         submit_task(db, table, "il-lo")
