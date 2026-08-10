"""Score each task on test with the context `submit_hpo_only.py` picked on
validation, ensembled over context seeds, one job per task. See
[README.md](README.md)."""

import json
from pathlib import Path

from roach.slurm import Resources, submit

from submit import TASKS, ntrain
from submit_hpo_only import b200, ckpt_for

TUNING_ROOT = Path("/dfs/user/ranjanr/ckpts/rtv2/2026-08-10-fine_tune_hpo")


def cfg_for(db: str, task: str) -> tuple[int, int, int, bool]:
    """The `(ctx_size, local_ctx_size, bfs_width, prefer_latest)` this task's
    tuning run chose on validation.

    Read from the `tuning.json` every tuning run writes beside its `eval_out`,
    over every run under `TUNING_ROOT`: which run covered which task is not in
    the path, and one entry per task must match or the decision is ambiguous.
    """
    hits = [
        entry
        for p in TUNING_ROOT.glob("*/tuning.json")
        for key, entry in json.loads(p.read_text()).items()
        if key == f"{db}/{task}"
    ]
    assert len(hits) == 1, (
        f"{len(hits)} tuning entries for {db}/{task} in {TUNING_ROOT}"
    )
    ctx, lcs, bw, pl = hits[0]["best_cfg"]
    return ctx, lcs, bw, pl


def plan(n: int) -> list[Resources]:
    """The best n slots this cluster will give one-GPU jobs, best first.

    Its own count, not `submit_hpo_only.plan`'s: this submission goes out after the
    tuning one has finished, at whichever moment that is, and what was free
    then says nothing about now. `il-interactive` caps at 2 gpus of any type,
    `il` at 10 together with only 2 b200, `il-lo` is preemptible and
    uncapped. Blackwell throughout while blackwell1 has the cards: a test pass
    per context seed is the whole wall clock.

    Recount and rewrite this before every submission.

    An eval run does not checkpoint: a preemption restarts it from the top, so
    `il-lo` costs wall clock in whole runs rather than in minutes.
    """
    out = [b200("il-interactive", "12:00:00")] * min(n, 2)
    out += [b200("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    tasks = sorted(TASKS, key=lambda p: -ntrain()[f"{p[0]}/{p[1]}"])
    # Every checkpoint and winning config before any job: both assert, and a
    # task whose fine-tuning or tuning run has not got that far must abort the
    # submission rather than leave the tasks ahead of it queued and the rest
    # not.
    ckpts = {t: ckpt_for(*t) for t in tasks}
    cfgs = {t: cfg_for(*t) for t in tasks}
    for (db, task), resources in zip(tasks, plan(len(tasks)), strict=True):
        name = f"{db}/{task}"
        ctx, lcs, bw, pl = cfgs[db, task]
        print(f"  {name:28s} {(ctx, lcs, bw, pl)} {resources.qos:15s} {resources.time}")
        submit(
            "rt.eval:main",
            # Do not put comments inside this dict: it is a config block,
            # and reading it means scanning the values.
            args=dict(
                load_ckpt_path=ckpts[db, task],
                embedder="all-MiniLM-L12-v2",
                d_text=384,
                num_blocks=12,
                d_model=512,
                num_heads=8,
                d_ff=2048,
                splits=["test"],
                db_task_list=[(db, task)],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**18,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                num_walks=10_000,
                walk_length=20,
                items_per_task=10_000_000,
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                ctx_lcs_bw_pl_grid=[(ctx, lcs, bw, pl)],
                val_ensemble_size=1,
                test_ensemble_size=4,
                run_name=None,
                targets={},
                project="2026-08-10-fine_tune_ens",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=True,
            ),
            resources=resources,
            name=f"ens-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-ens",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
