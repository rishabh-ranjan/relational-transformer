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

# rt-norefit over the same 21 tasks: the reporting arm reports the selection
# arm's checkpoint instead of retraining on train+val, so the pair isolates what
# the refit is worth. Ordered by what the `rt` sweep measured, minus its refit
# term, fastest first -- so the answers land in that order and a card freed
# early takes the next job.
# The whole rt-norefit sweep again: it now selects across the live and SWA nets
# rather than fixing SWA, which changes what every one of its jobs reports, so
# the in-flight ones were cancelled rather than left to produce the old answer.
EXPERIMENTS = tuple(
    ("rt-norefit", db, task)
    for db, task in (
        ("rel-avito", "ad-ctr"),
        ("rel-event", "user-attendance"),
        ("rel-avito", "user-visits"),
        ("rel-trial", "study-outcome"),
        ("rel-f1", "driver-position"),
        ("rel-event", "user-repeat"),
        ("rel-f1", "driver-top3"),
        ("rel-trial", "site-success"),
        ("rel-f1", "driver-dnf"),
        ("rel-event", "user-ignore"),
        ("rel-avito", "user-clicks"),
        ("rel-stack", "post-votes"),
        ("rel-hm", "user-churn"),
        ("rel-trial", "study-adverse"),
        ("rel-hm", "item-sales"),
        ("rel-stack", "user-engagement"),
        ("rel-stack", "user-badge"),
        ("rel-amazon", "item-churn"),
        ("rel-amazon", "item-ltv"),
        ("rel-amazon", "user-churn"),
        ("rel-amazon", "user-ltv"),
    )
)

#: Zero-shot reads: the published checkpoint scored on test with no fine-tuning
#: and no selection arm (see zero_shot.py). Not protocol runs -- a shortcut for
#: reading a checkpoint's number on a task.
ZERO_SHOT: tuple = ()
_ZERO_SHOT_DONE = (
    # (dataset, task, split, quote_train_only, mask_labels, cutoff_offset)
    # Does the -1 in context_cutoff matter? rustler's bound is inclusive
    # (past_bound is `ts > bound`), so a cutoff landing exactly on the split's
    # first cohort should leave those rows quotable by every later seed --
    # 21 val rows at 2005-03-02 for this task. If that reading is right, the
    # offset-0 run scores higher; if it is wrong, the two agree.
    ("rel-f1", "driver-top3", "val", True, True, 0),
    ("rel-f1", "driver-top3", "val", True, True, -1),
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


def b200(qos: str, time: str) -> Resources:
    """One B200 on blackwell1. 36 cpus is its 288 cores split eight ways, and
    the memory that share of the node, under the site's MaxMemPerCPU."""
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="b200:1",
        cpus_per_task=36,
        ntasks=None,
        exclusive=False,
        mem="375000M",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
    )


def reserved(time: str) -> Resources:
    """One a100 on the node reserved for us. `il-lo` deliberately: a reserved
    card is ours whatever the tier asks for, so spending a capped tier there
    buys nothing (expts/README.md#a-reservation-is-il-lo-only)."""
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=time,
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem=None,
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation="ranjanr_deadline",
        dependency=None,
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


# One line per experiment, read off the cluster at 13:54 -- not reused from the
# 13:30 plan, which is what put four jobs on QOSMaxGRESPerUser last time: the
# caps count across all my jobs and my own second sweep had since filled them.
#
# Held by the rt sweep, which keeps running: 7 `il` and 1 `il-lo`. So 3 of
# `il`'s ten are free, both `il-interactive` slots are, and blackwell1 has 6 of
# 8 b200 free and is MIXED, not RESERVED.
#
# `ranjanr_deadline` (ampere8) runs to 2026-08-13T00:00, 10 hours out, and its
# cards are ours whatever tier asks -- so `il-lo` there, walled inside it.
#
# The scarce tiers go to the slowest jobs: a b200 saves hours on rel-amazon's
# ~5h of preprocessing and minutes on a twenty-minute task.
RESOURCES: dict[tuple[str, str, str], Resources] = {
    # il-interactive (2) + il's b200 sub-cap (2): the four rel-amazon jobs.
    ("rt-norefit", "rel-amazon", "item-churn"): b200("il-interactive", "12:00:00"),
    ("rt-norefit", "rel-amazon", "item-ltv"): b200("il-interactive", "12:00:00"),
    ("rt-norefit", "rel-amazon", "user-churn"): b200("il", "1-00:00:00"),
    ("rt-norefit", "rel-amazon", "user-ltv"): b200("il", "1-00:00:00"),
    # il's one remaining slot: the slowest of the rest.
    ("rt-norefit", "rel-hm", "user-churn"): a100("il", "1-00:00:00"),
    # The reservation's free cards: the fastest seven, all far inside a 9h wall.
    ("rt-norefit", "rel-avito", "ad-ctr"): reserved("9:00:00"),
    ("rt-norefit", "rel-event", "user-attendance"): reserved("9:00:00"),
    ("rt-norefit", "rel-avito", "user-visits"): reserved("9:00:00"),
    ("rt-norefit", "rel-trial", "study-outcome"): reserved("9:00:00"),
    ("rt-norefit", "rel-f1", "driver-position"): reserved("9:00:00"),
    ("rt-norefit", "rel-event", "user-repeat"): reserved("9:00:00"),
    ("rt-norefit", "rel-f1", "driver-top3"): reserved("9:00:00"),
    # Everything else on the uncapped tier.
    ("rt-norefit", "rel-trial", "site-success"): a100("il-lo", "1-00:00:00"),
    ("rt-norefit", "rel-f1", "driver-dnf"): a100("il-lo", "1-00:00:00"),
    ("rt-norefit", "rel-event", "user-ignore"): a100("il-lo", "1-00:00:00"),
    ("rt-norefit", "rel-avito", "user-clicks"): a100("il-lo", "1-00:00:00"),
    ("rt-norefit", "rel-stack", "post-votes"): a100("il-lo", "2-00:00:00"),
    ("rt-norefit", "rel-trial", "study-adverse"): a100("il-lo", "1-00:00:00"),
    ("rt-norefit", "rel-hm", "item-sales"): a100("il-lo", "2-00:00:00"),
    ("rt-norefit", "rel-stack", "user-engagement"): a100("il-lo", "2-00:00:00"),
    ("rt-norefit", "rel-stack", "user-badge"): a100("il-lo", "2-00:00:00"),
}

ZERO_SHOT_RESOURCES: dict[tuple[str, str], Resources] = {
    # No training at all -- one 8-seed pass over the test split.
    ("rel-f1", "driver-top3"): a100("il", "1:00:00"),
}

#: Relaunch an existing run instead of starting a new one.
RUN_IDS: dict[tuple[str, str, str], str] = {}


def main() -> None:
    for dataset, task, split, quote_train_only, mask_labels, cutoff_offset in ZERO_SHOT:
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
                cutoff_offset=cutoff_offset,
                cache_dir=CACHE_DIR,
                out_dir=f"{SHARE}/results",
            ),
            resources=resources,
            name=f"relarena-zero-shot-{dataset}-{task}-{split}-off{cutoff_offset}",
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
