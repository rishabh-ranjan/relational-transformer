"""Tune the eval context per task on validation and score the winner on test,
both in one job per task, on the fine-tuned checkpoints. See
[README.md](README.md).

The undivided version of `submit_hpo_only.py` + `submit_ens.py`: nothing to
wait for and nothing to read back, at the cost of one `items_per_task` for
both phases -- the whole split, so the test number is the real one and every
grid entry pays a full validation pass.
"""

from roach.slurm import Resources, submit

from submit import TASKS, ntrain
from submit_hpo_only import b200, ckpt_for


def plan(n: int) -> list[Resources]:
    """The best n slots this cluster will give one-GPU jobs, best first.

    `il-interactive` caps at 2 gpus of any type, `il` at 10 together with only
    2 b200, `il-lo` is preemptible and uncapped. Blackwell throughout while
    blackwell1 has the cards: one val pass per grid entry and one test pass
    per context seed is the whole wall clock, and this script pays both.

    Recount and rewrite this before every submission.

    An eval run does not checkpoint: a preemption restarts it from the top, so
    `il-lo` costs wall clock in whole runs rather than in minutes.
    """
    out = [b200("il-interactive", "12:00:00")] * min(n, 2)
    out += [b200("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    tasks = sorted(TASKS, key=lambda p: -ntrain()[f"{p[0]}/{p[1]}"])
    # Every checkpoint before any job: `ckpt_for` asserts, and a task whose
    # fine-tuning run has not reached its first eval must abort the submission
    # rather than leave the tasks ahead of it queued and the rest not.
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
                items_per_task=10_000_000,
                ctx_size_list=[2048],
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                lcs_bw_pl_grid=[
                    (lcs, bw, pl)
                    for lcs in (512, 1024, 2048)
                    for bw in (64, 128, 256)
                    for pl in (True, False)
                ],
                val_ensemble_size=1,
                test_ensemble_size=4,
                project="2026-08-10-fine_tune_hpo_ens",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=True,
            ),
            resources=resources,
            name=f"hpo-ens-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-hpo-ens",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
