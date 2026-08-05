"""Submit the Join preprocessing sweep: one slurm job per database.

    pixi run python expts/preprocess/submit.py            # submit what is left
    pixi run python expts/preprocess/submit.py --dry-run  # just print the plan

One job per database rather than a job array over shards, because the
collection's work is lopsided -- the median database preprocesses to ~43 MiB and
the largest to 76 GiB -- so a fixed worker shape is either too thin for the
giants or wasteful for the 500-database tail. Per database, resources come from
what the previous build's output measured (`sizes.py`), and slurm places them.

Re-running is the resume: a database whose output is already complete is not
resubmitted, and one that is queued or running is not duplicated.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roach.slurm import Resources, submit  # noqa: E402

from expts.preprocess.preprocess import is_done  # noqa: E402
from expts.preprocess.sizes import load as load_sizes  # noqa: E402

REPO_ROOT = "/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer"
RAW_DIR = "/dfs/user/ranjanr/share/stanford-star/the-join"
OUT_DIR = "/dfs/user/ranjanr/share/stanford-star/the-join-preprocessed"
SOURCE_REPO = "stanford-star/the-join"
CLONE_ROOT = "/lfs/local/0/roach_clones"
LOG_ROOT = "/dfs/user/ranjanr/slurm-logs/preprocess"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"

EMBEDDER = "all-MiniLM-L12-v2"
BATCH_SIZE = 1024

# hyperturing1/2 have 2 TiB of memory and 252 cpus; the turings have 754 GiB
# (turing3 1.4 TiB) and 80. Everything can run on all five; the largest
# databases are held to the hyperturings, where one can have half a terabyte
# without leaving the node unable to run anything alongside it.
ALL_NODES = "hyperturing1,hyperturing2,turing1,turing2,turing3"
BIG_NODES = "hyperturing1,hyperturing2"

# (max expected output bytes, cpus, mem, wall clock, nodes). Memory is bounded
# by the site's MaxMemPerCPU of 10700M -- asking for more than cpus x 10700M is
# rejected, which is why the wide tiers are wide in cpus as well.
TIERS = (
    (1 << 30, 8, "80G", "2:00:00", ALL_NODES),
    (8 << 30, 16, "160G", "6:00:00", ALL_NODES),
    (25 << 30, 24, "250G", "12:00:00", ALL_NODES),
    (1 << 62, 48, "500G", "1-00:00:00", BIG_NODES),
)


def resources_for(expected_bytes: int) -> Resources:
    for limit, cpus, mem, walltime, nodes in TIERS:
        if expected_bytes < limit:
            break
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=walltime,
        # A bare count, not a type: these five nodes carry rtx8000s and 2080tis
        # and this job does not care which. Naming one would halve the pool.
        gpus="1",
        cpus_per_task=cpus,
        exclusive=False,
        mem=mem,
        constraint=None,
        nodelist=nodes,
    )


def queued_names() -> set[str]:
    """Databases already queued or running, so re-running submit.py is safe."""
    out = subprocess.run(
        [
            "squeue",
            "-h",
            "-u",
            subprocess.run(
                ["id", "-un"], capture_output=True, text=True
            ).stdout.strip(),
            "-o",
            "%j",
        ],
        capture_output=True,
        text=True,
    )
    return {n[4:] for n in out.stdout.split() if n.startswith("pre-")}


def datasets() -> list[str]:
    """Every database in the raw directory, largest expected output first.

    Largest first so the giants -- which alone set the makespan -- start before
    the tail rather than behind it.
    """
    sizes = load_sizes()
    names = sorted(p.parent.name for p in Path(RAW_DIR).glob("*/manifest.yaml"))
    return sorted(names, key=lambda n: -sizes.get(n, 0))


def main(dry_run: bool = False) -> None:
    sizes, out = load_sizes(), Path(OUT_DIR)
    names = datasets()
    running = queued_names()
    print(f"{len(names)} databases in {RAW_DIR}")

    todo, done, busy = [], 0, 0
    for name in names:
        if is_done(out / name, EMBEDDER):
            done += 1
        elif name in running:
            busy += 1
        else:
            todo.append(name)
    print(
        f"  {done} already preprocessed, {busy} queued or running, {len(todo)} to submit"
    )

    unsized = [n for n in todo if n not in sizes]
    if unsized:
        # New databases get the largest tier: unknown is not the same as small,
        # and guessing small is the failure that wastes a whole job.
        print(f"  {len(unsized)} not in sizes.json, submitted at the largest tier")

    for i, name in enumerate(todo, 1):
        expected = sizes.get(name, 1 << 62)
        resources = resources_for(expected)
        if dry_run:
            print(
                f"  [{i}/{len(todo)}] {name:42s} {expected / 2**30:8.2f} GiB  "
                f"{resources.cpus_per_task:3d} cpu  {resources.mem:>5s}  "
                f"{resources.time:>10s}  {resources.nodelist}"
            )
            continue
        submit(
            "expts.preprocess.preprocess:preprocess",
            args={
                "dataset": name,
                "raw_dir": RAW_DIR,
                "out_dir": OUT_DIR,
                "source_repo": SOURCE_REPO,
                "embedder": EMBEDDER,
                "batch_size": BATCH_SIZE,
            },
            resources=resources,
            name=f"pre-{name}",
            setup=("pixi run build-sampler",),
            repo_root=REPO_ROOT,
            log_root=LOG_ROOT,
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            # every job in the sweep is the same commit, so the clone is built
            # once per node and reused; a week outlives the sweep comfortably
            clone_ttl_days=7,
        )


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
