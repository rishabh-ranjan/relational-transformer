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
from submit import a100, targets_for
from submit_ens_only import ckpt_for, in_flight, items_for, ready


def plan(n: int) -> list[Resources]:
    """One slot per task, largest train set first, all the same.

    Amperes: `submit.b200` pins `nodelist="blackwell1"`, whose 8 cards are
    mostly other people's long jobs, so a sweep this wide would sit at
    `ReqNodeNotAvail`. The a100s are 8 nodes behind no nodelist, and a queue
    this deep is bound by how often a card frees rather than by how fast one
    runs.

    `il-lo` throughout: `il`'s cap is 10 gpus of any kind together and
    `submit.py`'s fine-tuning sweep holds all ten, and `il-interactive`'s 12
    hours does not cover a job of this length. This one is `len(grid)` val
    passes plus `test_ensemble_size` test passes, and an eval run does not
    checkpoint -- a preemption restarts it at the first config.

    Recount and rewrite this before every submission.
    """
    return [a100("il-lo", "21-00:00:00")] * n


def job_name(db: str, task: str) -> str:
    """The slurm job name this experiment gives that task."""
    return f"hpo-ens-{db}-{task}"


def main() -> None:
    tasks = [t for t in ready() if job_name(*t) not in in_flight()]
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
                ctx_lcs_bw_pl_grid=[
                    (ctx, lcs, bw, pl)
                    for ctx in (512, 1024, 2048)
                    for lcs in (512, 1024, 2048)
                    for bw in (64, 128, 256)
                    for pl in (True, False)
                    if lcs <= ctx
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
