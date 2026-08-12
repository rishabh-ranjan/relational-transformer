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

EXPERIMENTS = tuple(
    ("rt", db, task)
    for db, task in (
        # Smallest first, so the fastest answers land first.
        ("rel-f1", "driver-dnf"),
        ("rel-f1", "driver-position"),
        ("rel-event", "user-repeat"),
        ("rel-event", "user-ignore"),
        ("rel-event", "user-attendance"),
        ("rel-trial", "study-outcome"),
        ("rel-trial", "study-adverse"),
        ("rel-trial", "site-success"),
        ("rel-avito", "ad-ctr"),
        ("rel-avito", "user-clicks"),
        ("rel-avito", "user-visits"),
        ("rel-hm", "user-churn"),
        ("rel-hm", "item-sales"),
        ("rel-stack", "user-engagement"),
        ("rel-stack", "post-votes"),
        ("rel-stack", "user-badge"),
        ("rel-amazon", "user-churn"),
        ("rel-amazon", "user-ltv"),
        ("rel-amazon", "item-churn"),
        ("rel-amazon", "item-ltv"),
    )
    # rel-f1/driver-top3 is already running as job 129999.
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


# One line per experiment, worked out against the cluster at 07:05.
#
# Held already: three `il` a100 (129999 and the two cutoff probes), so 7 of
# `il`'s 10 are free and both `il-interactive` slots are.
#
# blackwell1 reads 6/8 b200 allocated and is MIXED, not RESERVED, so the two
# free cards are takeable. The b200 jobs currently running have 20h, 1d and 1d
# limits with an hour to six elapsed, so waiting for a third card is a day away
# -- the other high-tier work goes on amperes rather than queue for one.
#
# `ranjanr_deadline` holds all 8 of ampere8 until 2026-08-13T00:00, ~17h out.
# Those cards are ours whatever tier asks, so they take `il-lo` and the wall
# stays inside the reservation.
#
# Sizing: the selection arm is a flat 10k steps (~4.5h on an a100 at the 1.6
# s/step measured, less on a b200), early stopping usually ends it sooner; the
# refit adds the chosen step scaled by the row ratio; and the 8-seed test
# prediction scales with the test split, which is what makes the big tasks
# long. So the largest tasks take `il`'s 7-day wall and the smallest take the
# reservation.
RESOURCES: dict[tuple[str, str, str], Resources] = {
    # il-interactive (2, 12h wall): the two free b200. Medium tasks, so the
    # 12-hour cap is not the thing that ends them.
    ("rt", "rel-avito", "ad-ctr"): b200("il-interactive", "12:00:00"),
    ("rt", "rel-event", "user-attendance"): b200("il-interactive", "12:00:00"),
    # il (7 free of 10, 7d wall): the seven biggest, where a long wall matters
    # most -- their test splits are millions of rows and the ensemble is 8 of
    # them.
    ("rt", "rel-amazon", "user-churn"): a100("il", "2-00:00:00"),
    ("rt", "rel-amazon", "user-ltv"): a100("il", "2-00:00:00"),
    ("rt", "rel-amazon", "item-churn"): a100("il", "2-00:00:00"),
    ("rt", "rel-amazon", "item-ltv"): a100("il", "2-00:00:00"),
    ("rt", "rel-stack", "user-badge"): a100("il", "2-00:00:00"),
    ("rt", "rel-hm", "user-churn"): a100("il", "2-00:00:00"),
    ("rt", "rel-hm", "item-sales"): a100("il", "2-00:00:00"),
    # il-lo on the reservation (8 cards, ours, nothing preempts): the eight
    # smallest, walled inside the reservation's end.
    ("rt", "rel-f1", "driver-dnf"): reserved("12:00:00"),
    ("rt", "rel-f1", "driver-position"): reserved("12:00:00"),
    ("rt", "rel-event", "user-repeat"): reserved("12:00:00"),
    ("rt", "rel-event", "user-ignore"): reserved("12:00:00"),
    ("rt", "rel-trial", "study-outcome"): reserved("12:00:00"),
    ("rt", "rel-trial", "study-adverse"): reserved("12:00:00"),
    ("rt", "rel-trial", "site-success"): reserved("12:00:00"),
    ("rt", "rel-avito", "user-clicks"): reserved("12:00:00"),
    # il-lo in the general pool: preemptible, and these three resume.
    ("rt", "rel-avito", "user-visits"): a100("il-lo", "2-00:00:00"),
    ("rt", "rel-stack", "user-engagement"): a100("il-lo", "2-00:00:00"),
    ("rt", "rel-stack", "post-votes"): a100("il-lo", "2-00:00:00"),
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
