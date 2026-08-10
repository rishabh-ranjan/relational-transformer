"""Tune the eval context per task on validation and score the winner on test,
both in one job per task, on the fine-tuned checkpoints. See
[README.md](README.md).

The undivided version of `submit_hpo_only.py` + `submit_ens.py`: nothing to
wait for and nothing to read back, at the cost of one `items_per_task` for both
phases.

`splits` does not gate the tuning: with a grid of more than one entry
`rt.eval.main` reads the val split whatever `splits` says, and `splits` decides
only whether the test phase runs at all. `["test"]` is therefore both phases.

The task list, the checkpoint each job loads and how much of a split a pass
covers are `submit_ens_only`'s -- one home each, and the two experiments score
the same weights on the same rows.
"""

from roach.slurm import Resources, submit
from submit import a100, b200, targets_for
from submit_ens_only import ckpt_for, in_flight, items_for, nval, ntest, ready


def plan(n: int) -> list[Resources]:
    """The qos budget spent top down, best slots first, per
    [../README.md](../README.md#allocating-a-sweep).

    Today: `il-interactive`'s 2 gpus are free and go to blackwells;
    `submit.py`'s fine-tuning sweep holds 6 of `il`'s 10, so 4 are left here,
    and 2 of those may be b200. That is 4 blackwell and 2 ampere at high
    priority; the rest is `il-lo`, which is preemptible and restarts an eval
    from the first configuration.

    12 hours covers a blackwell job here -- `len(grid)` val passes plus
    `test_ensemble_size` test passes, measured at ~20 min a pass on an ampere
    and a bit over twice that fast on a blackwell.

    Recount and rewrite this before every submission.
    """
    out = [b200("il-interactive", "12:00:00")] * min(n, 2)
    out += [b200("il", "7-00:00:00")] * min(n - len(out), 2)
    out += [a100("il", "7-00:00:00")] * min(n - len(out), 2)
    out += [a100("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def job_name(db: str, task: str) -> str:
    """The slurm job name this experiment gives that task."""
    return f"hpo-ens-{db}-{task}"


def cost(db: str, task: str) -> float:
    """Roughly what this task's job costs: the rows one pass scores.

    The slot order below is by this, not by train-set size: an eval job's wall
    clock is the split it reads, and `items_for` caps the biggest ones, so the
    two orders disagree.
    """
    cap = items_for(db, task)
    return min(nval()[f"{db}/{task}"], cap) + min(ntest()[f"{db}/{task}"], cap)


def main() -> None:
    tasks = [t for t in ready() if job_name(*t) not in in_flight()]
    # Slowest first, so the best slots go where they buy the most.
    tasks.sort(key=lambda t: -cost(*t))
    ckpts = {t: ckpt_for(*t) for t in tasks}
    for (db, task), resources in zip(tasks, plan(len(tasks)), strict=True):
        name = f"{db}/{task}"
        print(f"  {name:28s} {resources.gpus} {resources.qos:15s} {resources.time}")
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
                items_per_task=items_for(db, task),
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                ctx_size_list=[512, 1024, 2048],
                lcs_bw_pl_grid=[
                    (lcs, bw, pl)
                    for lcs in (512, 1024, 2048)
                    for bw in (64, 128, 256)
                    for pl in (True, False)
                ],
                val_ensemble_size=1,
                test_ensemble_size=4,
                run_name=name,
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune_hpo_ens",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=job_name(db, task),
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-hpo-ens",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
