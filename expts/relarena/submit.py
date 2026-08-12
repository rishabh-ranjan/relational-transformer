"""Submit one RelArena experiment per (model, dataset, task). See [run.py](run.py).

RelArena (github.com/rishabh-ranjan/relarena-alpha) owns the evaluation
protocol; this directory is only how its experiments reach slurm. The `rt` model
there wraps `rt.train` / `rt.eval`, so the job runs in *this* repo's environment
and RelArena is installed on top of it in `setup`.

`--no-deps` for relarena itself, then its own requirements by hand: a plain
install would resolve torch, and uv replacing this environment's CUDA build with
a wheel from PyPI is a long way to a broken job. Everything else relarena needs
that RT does not carry is listed explicitly.

The `rt` extra is deliberately *not* installed: it would pull
relational-transformer from git over the editable checkout this job is running.

**relbench is not installed here.** It is pinned to 2.1.2 in this branch's pixi
environment instead, because `pixi run` restores the environment to its lock
before the ranks start -- a `uv pip install` of a package the lock also names is
reverted between `setup` and the job. Anything relarena needs that the lock
carries has to be in the lock.
"""

from pathlib import Path

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

REPO_ROOT = "/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer-relarena"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"
SHARE = "/dfs/user/ranjanr/share/relarena"

# Node-local, and written by the job itself: the warm and the run are one job,
# so nothing has to be shared, and /dfs is slow enough to matter for a cache
# read on every context build.
CACHE_DIR = "/tmp/ranjanr/relarena-cache"

EXPERIMENTS = (
    # (model, dataset, task)
    ("rt", "rel-f1", "driver-top3"),
)

#: Zero-shot reads: the published checkpoint scored on test with no fine-tuning
#: and no selection arm (see zero_shot.py). Not protocol runs -- a shortcut for
#: reading a checkpoint's number on a task.
ZERO_SHOT = (
    # (dataset, task, split, quote_train_only, mask_labels)
    # With db_cutoff bounding contexts at the split horizon, val and test are
    # finally the same measurement: both re-run as the baseline the fine-tune
    # has to beat.
    ("rel-f1", "driver-top3", "val", True, True),
    ("rel-f1", "driver-top3", "test", True, False),
)


def relarena_setup() -> tuple[str, ...]:
    """Install relarena into the job's environment. See the module docstring."""
    token = f'$(tr -d "[:space:]" < {SECRETS_DIR}/github)'
    url = f"git+https://x-access-token:{token}@github.com/rishabh-ranjan/relarena-alpha@main"
    return (
        f'pixi run uv pip install --no-deps "relarena @ {url}"',
        # relarena's own dependencies, minus the ones this environment carries
        # already. relbench is one of those -- see the module docstring.
        'pixi run uv pip install "configspace>=1.0" "jsonschema>=4.0"',
    )


def a100(qos: str, time: str) -> Resources:
    """One A100. 14 cpus is what the site allows per gpu on a job that is not
    --exclusive; no --mem, so the partition's DefMemPerGPU applies, which is
    more than an explicit request would be given."""
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem=None,
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
    )


# One line per experiment, chosen against the cluster at submission time.
RESOURCES: dict[tuple[str, str, str], Resources] = {
    # rel-f1/driver-top3 is a few thousand rows: the 25k-step ceiling is the
    # whole cost, and early stopping usually ends it well before that.
    ("rt", "rel-f1", "driver-top3"): a100("il", "8:00:00"),
}

ZERO_SHOT_RESOURCES: dict[tuple[str, str], Resources] = {
    # No training at all -- one 8-seed pass over the test split.
    ("rel-f1", "driver-top3"): a100("il", "1:00:00"),
}

#: Relaunch an existing run instead of starting a new one.
RUN_IDS: dict[tuple[str, str, str], str] = {}


def main() -> None:
    for dataset, task, split, quote_train_only, mask_labels in ZERO_SHOT:
        resources = ZERO_SHOT_RESOURCES[dataset, task]
        print(
            f"  zero-shot/{dataset}/{task}/{split} "
            f"quote_train_only={quote_train_only}  {resources.gpus} {resources.qos}"
        )
        submit(
            "expts.relarena.zero_shot:main",
            args=dict(
                dataset=dataset,
                task=task,
                split=split,
                quote_train_only=quote_train_only,
                mask_labels=mask_labels,
                cache_dir=CACHE_DIR,
                out_dir=f"{SHARE}/results",
            ),
            resources=resources,
            name=f"relarena-zero-shot-{dataset}-{task}-{split}-masked{mask_labels}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            log_root=f"{SHARE}/slurm-logs",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir=SECRETS_DIR,
        )

    seed = 0
    for model, dataset, task in EXPERIMENTS:
        resources = RESOURCES[model, dataset, task]
        name = f"relarena/{model}/{dataset}/{task}"
        print(f"  {name:38s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit(
            "expts.relarena.run:main",
            # Do not put comments inside this dict: it is a config block,
            # and reading it means scanning the values.
            args=dict(
                dataset=dataset,
                task=task,
                model=model,
                seed=seed,
                n_trials=1,
                cache_dir=CACHE_DIR,
                out_dir=f"{SHARE}/results",
            ),
            resources=resources,
            name=f"relarena-{model}-{dataset}-{task}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            log_root=f"{SHARE}/slurm-logs",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir=SECRETS_DIR,
            run_id=RUN_IDS.get((model, dataset, task)),
        )


if __name__ == "__main__":
    main()
