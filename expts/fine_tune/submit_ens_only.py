"""Ensembling on its own: score each task on test at the context the
fine-tuning runs evaluated with, averaged over context seeds, one job per task.
See [README.md](README.md).

No tuning, so nothing reads validation and this does not wait on
`submit_hpo_only.py`. `rt.eval` scores the running average after every seed, so one
job yields the whole test-metric-vs-ensemble-size curve, logged to wandb
against `ens_size` with each task's published target beside it.

Tasks go out largest train set first: the curve that takes longest starts
first, and the small ones fill the slots behind it.
"""

import json

from roach.slurm import Resources, submit

from submit import ntrain, published_best, targets_for
from submit_hpo_only import CKPT_ROOT, b200, ckpt_for


def ready() -> list[tuple[str, str]]:
    """The benchmark tasks that can be ensembled now, largest train set first.

    Not `submit.TASKS`: that is whatever `submit.py` last submitted, which a
    partial resubmission narrows to a handful, and this experiment is the whole
    benchmark. `published_best` is keyed by all 21 forecast tasks.

    A task whose fine-tuning run has not written a `best_*` yet is skipped
    rather than aborting the submission: those runs are still going, and this
    is rerun as they land.
    """
    tasks = {
        tuple(k.split("/")[2:]) for k in published_best() if not k.endswith("/mean")
    }
    done = {
        tuple(json.loads((d / "params.json").read_text())["run_name"].split("/"))
        for d in CKPT_ROOT.iterdir()
        if any(p.stem in ("best_clf", "best_reg") for p in d.glob("best_*.safetensors"))
    }
    return sorted(tasks & done, key=lambda p: -ntrain()[f"{p[0]}/{p[1]}"])


def plan(n: int) -> list[Resources]:
    """One slot per task, in the order `main` hands out tasks: largest train
    set first.

    An eval run does not checkpoint. A preemption or a wall limit restarts it
    from ensemble size 1, so what a slot is worth here is how sure it is to
    hold the whole run -- the opposite of `submit.plan`, where a short or
    preemptible slot costs minutes. So the safest slots go to the longest runs
    and the 12-hour ones to the shortest, rather than the best slots first.

    `il-lo` is preemptible and uncapped at 21 days, `il-interactive` is 2 gpus
    of any type but only 12 hours, and `il` is not used at all: its cap is 10
    gpus of any kind together, which `submit.py`'s amperes already hold, so an
    `il` job here waits on those rather than on a card. Blackwell throughout
    while blackwell1 has them -- a test pass per context seed is the whole wall
    clock, and there are `test_ensemble_size` of them.

    A whole 16-seed curve on a b200 is minutes, not hours, so 12 hours is no
    constraint on the two `il-interactive` slots and preemption on the rest
    costs one restart at worst.

    Recount and rewrite this before every submission.
    """
    out = [b200("il-interactive", "12:00:00")] * min(n, 2)
    out += [b200("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    tasks = ready()
    # Rerunning after more fine-tuning runs land: slice off the ones already
    # scored, so this does not queue a second curve for them. `plan` hands the
    # sliced list its first slots. Leave it commented to submit all that are
    # ready.
    # tasks = tasks[:2]
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
                lcs_bw_pl_grid=[(2048, 128, True)],
                val_ensemble_size=1,
                test_ensemble_size=16,
                run_name=name,
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune_ens_only",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=f"ens-only-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-ens-only",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
