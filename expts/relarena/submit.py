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

import os
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

REPO_ROOT = os.path.expanduser(
    "~/clones/rishabh-ranjan/relational-transformer-relarena"
)
SECRETS_DIR = os.path.expanduser("~/scratch/.secrets")
SHARE = os.path.expanduser("~/scratch/share/relarena")

# Node-local, and written by the job itself: the warm and the run are one job,
# so nothing has to be shared, and ~/scratch is slow enough to matter for a cache
# read on every context build.
#
# On the node NVMe, not /tmp. /tmp is the root filesystem -- 438G on blackwell1,
# shared with the OS -- and one rel-amazon task's exports are ~85G, so four of
# them fill it and wedge every job on the node. The node-local home is 42-56T.
CACHE_DIR = os.path.expanduser("~/relarena-cache")

#: Wall for the jobs pinned to the `ranjanr_deadline` reservation. Not il-lo's
#: 21 days: a reservation job cannot outlive the reservation, so asking for
#: longer than the window that is left is asking for a job that cannot finish
#: -- and slurm rejects the request outright.
#:
#: The window ends 2026-08-15T00:00; the wall is set just inside what is left. Effectively no limit: the longest task placed here projects
#: at 2h42m, so this binds only if early stopping behaves nothing like it has.
#: That is the point -- the stop steps we have were all measured under a
#: different regime, so the safe wall is the one the reservation allows rather
#: than the one the projections suggest.
#:
#: Note slurm will not let a *user* raise TimeLimit on a running job
#: ("Access/permission denied"), only lower it, so getting this wrong means
#: resubmitting rather than adjusting.
RESERVATION_WALL = "1-20:00:00"

# The full `rt-plurel` sweep: all 21 RelBench entity tasks, one job each.
#
# One model now, not three. `rt` and `rt-norefit` are gone -- `rt-hpo`'s
# configuration won the three-task trial that compared them (rel-event/
# user-repeat 0.7983 at rank 1/11 against `rt`'s 0.7605 at 5/11;
# rel-trial/study-outcome 0.7274 at 2/11 against 0.7198 at 4/11), so it is now
# simply the model, renamed for the checkpoint every arm starts from.
#
# Ordered longest-projected-first, which is also the tier order below: the jobs
# that decide the makespan get the tier with no deadline on it.
EXPERIMENTS = (
    ("rt-plurel", "rel-amazon", "item-churn"),
    ("rt-plurel", "rel-amazon", "user-ltv"),
    ("rt-plurel", "rel-amazon", "item-ltv"),
    ("rt-plurel", "rel-amazon", "user-churn"),
    ("rt-plurel", "rel-hm", "item-sales"),
    ("rt-plurel", "rel-stack", "user-engagement"),
    ("rt-plurel", "rel-trial", "study-adverse"),
    ("rt-plurel", "rel-stack", "user-badge"),
    ("rt-plurel", "rel-hm", "user-churn"),
    ("rt-plurel", "rel-stack", "post-votes"),
    ("rt-plurel", "rel-avito", "user-clicks"),
    ("rt-plurel", "rel-avito", "user-visits"),
    ("rt-plurel", "rel-trial", "site-success"),
    ("rt-plurel", "rel-event", "user-ignore"),
    ("rt-plurel", "rel-f1", "driver-dnf"),
    ("rt-plurel", "rel-avito", "ad-ctr"),
    ("rt-plurel", "rel-event", "user-attendance"),
    ("rt-plurel", "rel-trial", "study-outcome"),
    ("rt-plurel", "rel-f1", "driver-top3"),
    ("rt-plurel", "rel-event", "user-repeat"),
    ("rt-plurel", "rel-f1", "driver-position"),
)

#: Zero-shot reads: the published checkpoint scored on test with no fine-tuning
#: and no selection arm (see zero_shot.py). Not protocol runs -- a shortcut for
#: reading a checkpoint's number on a task.
ZERO_SHOT: tuple = ()
_ZERO_SHOT_DONE = (
    # (dataset, task, split, mask_labels, cutoff_offset)
    # Does the -1 in context_cutoff matter? rustler's bound is inclusive
    # (past_bound is `ts > bound`), so a cutoff landing exactly on the split's
    # first cohort should leave those rows quotable by every later seed --
    # 21 val rows at 2005-03-02 for this task. If that reading is right, the
    # offset-0 run scores higher; if it is wrong, the two agree.
    ("rel-f1", "driver-top3", "val", True, 0),
    ("rel-f1", "driver-top3", "val", True, -1),
)


#: Nodes to keep off. Slurm calls a node with a full local disk healthy, so a
#: job placed there starts and then wedges partway through preprocessing.
#: ampere4's NVMe is at 99% (630G of 42T), and 37T of that is two other users --
#: nothing we delete wins it back. ampere7 hands out cards another tenant is
#: already holding: a job there died with 46GiB of someone else's process
#: resident on the same device. Every other gpu node is healthy.
BAD_NODES = "ampere4,ampere7"


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
        exclude=BAD_NODES,
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
        exclude=BAD_NODES,
    )


# One line per experiment. Placement optimizes **makespan** -- the time at which
# the last of the 21 results lands -- so every job starts now and the longest
# ones get the tier that cannot take their card away.
#
# The three tiers hold 21 concurrent jobs between them:
#   il-interactive  2 gpus, 12h wall   -- both spent on b200
#   il             10 gpus (<=2 b200), 7d wall
#   il-lo          uncapped, 21d, preemptible -- plus the ampere8 reservation,
#                  whose 8 a100 are ours outright until 2026-08-13T00:00
#
# **rel-amazon goes to b200.** Part of preprocessing is GPU work -- a
# sentence-transformer over 19.5M distinct strings -- and rel-amazon's is 1h46m
# of it on an a100 against 1h05m on a b200, on top of a 2.75x train and 2.3x
# eval speedup. Those four jobs are the makespan, so they take the four b200
# slots the two capped tiers allow between them: the two *shorter* ones on
# il-interactive, because its 12h wall is the binding one, and the two whose
# worst case is ~10h on `il`, where the wall is a week and cannot clip them.
#
# The other 17 are a100, sorted by projected worst case and cut at the tier
# boundaries so that **the jobs on the reservation are the ones least able to
# hit its wall**. The reservation's cards are free and ours, but its 7h window
# closes at 2026-08-13T00:00 and cannot be extended, so what goes there is the
# nine shortest -- 0h44m to 2h40m, every one of them with better than 2.5x
# headroom. The nine longest take `il`'s remaining slots, where the wall is a
# week and nothing can clip them, and rel-avito/user-clicks goes to plain il-lo,
# the only tier with no deadline of any kind.
#
# Duration is the right proxy for timeout risk here because every a100 task has
# a measured early-stop precedent -- the two that do not are both rel-amazon and
# both on b200. The one estimate that is genuinely shaky is the context search,
# which the model flattens to 1h09m for any task with >=4096 val rows; of the
# nine reservation jobs only rel-avito/user-visits carries that term, and it
# still lands at 2h40m against a 7h wall.
#
# Walls are each tier's maximum, per the request, except where the reservation
# is the tighter bound. The projections they cover are in
# /tmp/ranjanr/relarena-plan/timings.md -- worst case, and worst case assumes
# the full 50k steps for the two rel-amazon tasks with no early-stop precedent.
RESOURCES: dict[tuple[str, str, str], Resources] = {
    # -- b200 (6 of blackwell1's 8). rel-amazon takes four: part of preprocess
    # is a sentence-transformer over 19.5M strings, 1h46m of it on an a100
    # against 1h05m on a b200, on top of the train and eval speedups. The two
    # with the longer projections take `il`, whose week-long wall cannot clip
    # them; the two shorter take il-interactive, whose 12h wall is the binding
    # one and which they clear with room even in the worst case (requeue covers
    # the rest).
    ("rt-plurel", "rel-amazon", "user-ltv"): b200("il", "7-00:00:00"),
    ("rt-plurel", "rel-amazon", "user-churn"): b200("il", "7-00:00:00"),
    ("rt-plurel", "rel-amazon", "item-churn"): b200("il-interactive", "12:00:00"),
    ("rt-plurel", "rel-amazon", "item-ltv"): b200("il-interactive", "12:00:00"),
    # The two longest a100 jobs, moved onto the spare b200 through the only
    # tier that reaches them. il-lo is preemptible; these are 8h54 and 8h29 on
    # an a100 against 4h49 and 4h28 here, and taking them off the a100 pool is
    # what drops the makespan from 8h54 to 8h28.
    ("rt-plurel", "rel-stack", "post-votes"): b200("il-lo", "21-00:00:00"),
    ("rt-plurel", "rel-stack", "user-badge"): b200("il-lo", "21-00:00:00"),
    # -- a100 on `il`: the eight longest of the rest, week-long wall, no
    # deadline of any kind on them.
    ("rt-plurel", "rel-trial", "study-adverse"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-hm", "item-sales"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-hm", "user-churn"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-trial", "site-success"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-avito", "user-visits"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-stack", "user-engagement"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-avito", "user-clicks"): a100("il", "7-00:00:00"),
    ("rt-plurel", "rel-event", "user-ignore"): a100("il", "7-00:00:00"),
    # -- a100 on the reservation: the seven shortest. Its cards are ours
    # outright, but the window shuts at 2026-08-15T00:00 and cannot be
    # extended, so what goes here is what clears it by the widest margin --
    # 4h27 to 5h12 projected against a 1d20h wall.
    ("rt-plurel", "rel-event", "user-attendance"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-avito", "ad-ctr"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-trial", "study-outcome"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-f1", "driver-dnf"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-event", "user-repeat"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-f1", "driver-position"): reserved(RESERVATION_WALL),
    ("rt-plurel", "rel-f1", "driver-top3"): reserved(RESERVATION_WALL),
}

ZERO_SHOT_RESOURCES: dict[tuple[str, str], Resources] = {
    # No training at all -- one 8-seed pass over the test split.
    ("rel-f1", "driver-top3"): a100("il", "1:00:00"),
}

#: One-off measurement jobs, not protocol runs.
BENCH: tuple[tuple[str, str], ...] = ()

#: Relaunch an existing run instead of starting a new one.
RUN_IDS: dict[tuple[str, str, str], str] = {}


def main() -> None:
    for dataset, task, split, mask_labels, cutoff_offset in ZERO_SHOT:
        resources = ZERO_SHOT_RESOURCES[dataset, task]
        print(f"  zero-shot/{dataset}/{task}/{split} {resources.gpus} {resources.qos}")
        submit(
            "expts.relarena.zero_shot:main",
            args=dict(
                dataset=dataset,
                task=task,
                split=split,
                mask_labels=mask_labels,
                cutoff_offset=cutoff_offset,
                cache_dir=CACHE_DIR,
                out_dir=f"{SHARE}/results",
            ),
            resources=resources,
            name=f"relarena-zero-shot-{dataset}-{task}-{split}-off{cutoff_offset}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{SHARE}/slurm-logs",
            clone_root=os.path.expanduser("~/roach_clones"),
            secrets_dir=SECRETS_DIR,
        )

    for dataset, task in BENCH:
        submit(
            "expts.relarena.bench_compile:main",
            args=dict(dataset=dataset, task=task, cache_dir=CACHE_DIR),
            resources=reserved("2:00:00"),
            name=f"relarena-bench-compile-{dataset}-{task}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{SHARE}/slurm-logs",
            clone_root=os.path.expanduser("~/roach_clones"),
            secrets_dir=SECRETS_DIR,
        )
        print(f"  bench-compile/{dataset}/{task}")

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
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{SHARE}/slurm-logs",
            clone_root=os.path.expanduser("~/roach_clones"),
            secrets_dir=SECRETS_DIR,
            run_id=RUN_IDS.get((model, dataset, task)),
        )


if __name__ == "__main__":
    main()
