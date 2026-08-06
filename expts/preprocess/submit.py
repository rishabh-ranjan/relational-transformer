"""Submit the Join preprocessing sweep: one slurm job per database.

    pixi run python expts/preprocess/submit.py <collection>
    pixi run python expts/preprocess/submit.py <collection> --dry-run

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

from expts.preprocess.collection import Collection, pick  # noqa: E402
from expts.preprocess.preprocess import is_done, is_rustler_done  # noqa: E402
from expts.preprocess.sizes import load as load_sizes  # noqa: E402

REPO_ROOT = "/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer"
CLONE_ROOT = "/lfs/local/0/roach_clones"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"

EMBEDDER = "all-MiniLM-L12-v2"
BATCH_SIZE = 1024

# Every job runs `setup`, but the shared clone means it runs once per commit per
# node -- which is the right number of times to fetch the embedder. Left to the
# embed jobs, 50 of them start at once and each asks the Hub for the same model,
# and the Hub answers most of them with 429.
SETUP = (
    "pixi run build-sampler",
    f'pixi run python -c "from huggingface_hub import snapshot_download;'
    f" snapshot_download('sentence-transformers/{EMBEDDER}')\"",
)

# hyperturing1/2 have 2 TiB of memory and 252 cpus; the turings have 754 GiB
# (turing3 1.4 TiB) and 80. Everything can run on all five; the largest
# databases are held to the hyperturings, where one can have half a terabyte
# without leaving the node unable to run anything alongside it.
ALL_NODES = "hyperturing1,hyperturing2,turing1,turing2,turing3"
BIG_NODES = "hyperturing1,hyperturing2"
# hyperturing1's GPUs threw "uncorrectable ECC error" on 18 jobs; the cpu stage
# is unaffected, so only the GPU stage avoids it. Drop this back to ALL_NODES
# once the card is replaced or the node is drained.
EMBED_NODES = "hyperturing2,turing1,turing2,turing3"

# The rustler stage takes ONE cpu. Measured: TotalCPU equals Elapsed on every
# database, so it is single-threaded and a wider request buys nothing while
# costing a slot someone else's database could have had. How many run at once is
# then how many cpus the five nodes have -- which is the point.
#
# Only memory varies, and it comes from measurement too: MaxRSS ran to about
# twice a database's output, so this asks for three times with a floor. (The
# site's MaxMemPerCPU of 10700M is not enforced against a per-node --mem here;
# checked with sbatch --test-only, a 1-cpu job may ask for 200G.)
RUSTLER_CPUS = 1
MEM_FLOOR = 8 << 30
MEM_FACTOR = 3
# (max expected output bytes, wall clock, nodes)
TIERS = (
    (1 << 30, "2:00:00", ALL_NODES),
    (16 << 30, "8:00:00", ALL_NODES),
    (40 << 30, "16:00:00", ALL_NODES),
    (1 << 62, "1-00:00:00", BIG_NODES),
)

# The embedding stage takes SEVERAL GPUs in one job. sentence-transformers runs
# a worker per device, and measured on 2M texts that is 4.06x on six RTX8000s
# (68% efficiency) and 4.29x on six 2080Tis (71%) -- so the stage's own skew
# stops mattering: ten databases are 88% of it, and each of those now finishes
# in a quarter of the time instead of queueing behind one card.
#
# Card type barely matters: one RTX8000 does 929 texts/s against a 2080Ti's 849,
# a 9% gap, so there is nothing to route.
#
# Memory is per GPU, not per node, and that is not a detail: the partition sets
# DefMemPerGPU=240000M and applies it when deciding whether a job fits, so with
# --mem the most GPUs a job can hold is RealMemory/240000M -- 3 on a turing, 8
# on a hyperturing -- however little memory it asks for. --mem-per-gpu replaces
# that default and lifts the limit.
EMBED_GPUS = 6
EMBED_MEM_PER_GPU = "40G"
EMBED_CPUS = 24
# Walltime scales for the same reason memory does. join-overture-maps' 8 GiB of
# text was still embedding when a flat 4h cut it off, and a timeout costs the
# whole stage: the run has to start over from the first text.
EMBED_WALLTIMES = ((1 << 30, "2:00:00"), (8 << 30, "8:00:00"), (1 << 62, "1-00:00:00"))


def log_root(c: Collection) -> str:
    return f"/dfs/user/ranjanr/slurm-logs/preprocess-{c.name}"


def resources_for(expected_bytes: int) -> Resources:
    """The cpu-only rustler stage: one cpu, memory from the expected output."""
    for limit, walltime, nodes in TIERS:
        if expected_bytes < limit:
            break
    mem = max(MEM_FLOOR, MEM_FACTOR * expected_bytes)
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=walltime,
        # No GPU: this stage never touches one, and holding it would cap the
        # sweep at the 50 GPUs these five nodes have between them.
        gpus="0",
        cpus_per_task=RUSTLER_CPUS,
        ntasks=None,
        exclusive=False,
        mem=f"{mem // 2**30}G",
        mem_per_gpu=None,
        constraint=None,
        nodelist=nodes,
    )


def embed_resources(expected_bytes: int) -> Resources:
    """The GPU stage. A bare count, not a type: these nodes carry rtx8000s and
    2080tis and MiniLM does not care which."""
    for limit, walltime in EMBED_WALLTIMES:
        if expected_bytes < limit:
            break
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=walltime,
        gpus=str(EMBED_GPUS),
        cpus_per_task=EMBED_CPUS,
        # One rank with every GPU visible to it, not a rank per GPU:
        # sentence-transformers does the fan-out itself.
        ntasks=1,
        exclusive=False,
        mem=None,
        mem_per_gpu=EMBED_MEM_PER_GPU,
        constraint=None,
        nodelist=EMBED_NODES,
    )


def queued_names(prefix: str) -> set[str]:
    """Databases already queued or running in a stage, so re-running is safe."""
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
    return {n[len(prefix) :] for n in out.stdout.split() if n.startswith(prefix)}


def datasets(c: Collection) -> list[str]:
    """Every database in the raw directory, largest expected output first.

    Largest first so the giants -- which alone set the makespan -- start before
    the tail rather than behind it.
    """
    sizes = load_sizes(c)
    names = sorted(p.parent.name for p in Path(c.raw_dir).glob("*/manifest.yaml"))
    return sorted(names, key=lambda n: -sizes.get(n, 0))


def fully_downloaded(c: Collection, names: list[str]) -> set[str]:
    """Databases whose raw files are all present *and the right size*.

    A directory with a manifest.yaml in it is not a downloaded database: the raw
    collection is 28k files across 639 directories, so at any moment during a
    download most of them are partly there. Preprocessing one of those would
    quietly build a database short a few task tables and mark it finished, and
    nothing downstream would notice.

    Sizes, not just presence, because both ways of fetching this can leave a
    file that exists and is wrong: a git checkout holds LFS pointers (a couple
    of hundred bytes) until `git lfs pull` replaces them, and an interrupted
    download can leave a partial file. One metadata call answers it for the
    whole collection, and makes it safe to submit while the fetch is still
    running -- the cluster need not idle until the last byte lands.
    """
    from collections import defaultdict

    from huggingface_hub import HfApi

    remote: dict[str, dict[str, int]] = defaultdict(dict)
    info = HfApi().repo_info(c.source_repo, repo_type="dataset", files_metadata=True)
    for f in info.siblings:
        if "/" in f.rfilename:
            db, rest = f.rfilename.split("/", 1)
            remote[db][rest] = f.size or 0

    ready = set()
    for name in names:
        d = Path(c.raw_dir) / name
        want = remote.get(name)
        if not want:
            continue
        if all(
            (d / rel).is_file() and (d / rel).stat().st_size == size
            for rel, size in want.items()
        ):
            ready.add(name)
    return ready


def submit_legacy(c: Collection, name: str, expected_bytes: int):
    """The RT-v1 variant. One job: transform, rustler and embed together."""
    return submit(
        "expts.preprocess.preprocess:legacy",
        args={
            "dataset": name,
            "raw_dir": c.raw_dir,
            "out_dir": c.legacy_dir,
            "source_repo": c.source_repo,
            "embedder": EMBEDDER,
            "batch_size": BATCH_SIZE,
        },
        resources=embed_resources(expected_bytes),
        name=f"leg-{name}",
        setup=SETUP,
        repo_root=REPO_ROOT,
        log_root=log_root(c),
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        clone_ttl_days=7,
    )


def submit_embed(
    c: Collection, name: str, expected_bytes: int, after: str | None = None
):
    """The GPU stage for one database, optionally held until its rustler job
    succeeds -- which is how the two stages are submitted in one pass."""
    return submit(
        "expts.preprocess.preprocess:embed",
        args={
            "dataset": name,
            "out_dir": c.out_dir,
            "embedder": EMBEDDER,
            "batch_size": BATCH_SIZE,
        },
        resources=embed_resources(expected_bytes),
        name=f"emb-{name}",
        setup=SETUP,
        repo_root=REPO_ROOT,
        log_root=log_root(c),
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        clone_ttl_days=7,
        after=after,
    )


def check_tree_is_submittable() -> None:
    """Fail loudly, here, on the one thing that stops a sweep silently.

    `submit()` refuses a dirty or unpushed tree -- jobs clone the commit you
    submit from -- and raises once per database, deep in a loop, after printing
    a plan that says work is about to be submitted. Run unattended behind a log
    that trims output, that reads as a queue with nothing in it.
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "the working tree is dirty, so no job can be submitted "
            "(jobs clone the commit you submit from):\n" + dirty
        )


def main(c: Collection, dry_run: bool = False) -> None:
    """Submit whatever each database needs next.

    Two stages, one pass: a database with no rustler output gets a cpu-only
    rustler job, one that has it but no embeddings gets a GPU embed job. Because
    each is decided from what is on disk, re-running this is the resume for both
    stages at once, and the GPU queue fills behind the cpu one without any
    dependency to declare.
    """
    if not dry_run:
        check_tree_is_submittable()
    sizes, out = load_sizes(c), Path(c.out_dir)
    names = datasets(c)
    ready = fully_downloaded(c, names)
    pre_running, emb_running = queued_names("pre-"), queued_names("emb-")
    print(f"{len(names)} databases in {c.raw_dir}, {len(ready)} fully downloaded")

    leg_running = queued_names("leg-")
    legacy_out = Path(c.legacy_dir)
    to_rustle, to_embed, to_legacy, done, busy, waiting = [], [], [], 0, 0, 0
    for name in names:
        d = out / name
        if is_done(d, EMBEDDER):
            done += 1
        elif is_rustler_done(d):
            if name in emb_running:
                busy += 1
            else:
                to_embed.append(name)
        elif name in pre_running:
            busy += 1
        elif name not in ready:
            waiting += 1
        else:
            to_rustle.append(name)
        # The legacy tree is independent of the main build: it reads the same
        # raw data and writes its own directory, so it neither waits for nor
        # blocks the collection.
        if (
            c.legacy
            and name in ready
            and name not in leg_running
            and not is_done(legacy_out / name, EMBEDDER)
        ):
            to_legacy.append(name)
    print(f"  {done} complete, {busy} queued or running, {waiting} still downloading")
    print(
        f"  submitting {len(to_rustle)} rustler + {len(to_embed)} embed"
        + (f" + {len(to_legacy)} legacy" if c.legacy else "")
    )

    for name in to_embed:
        if dry_run:
            r = embed_resources(sizes.get(name, 1 << 35))
            print(
                f"  embed   {name:44s} {r.gpus} gpu  {r.cpus_per_task} cpu  "
                f"{r.mem_per_gpu}/gpu  {r.time}"
            )
            continue
        submit_embed(c, name, sizes.get(name, 1 << 35))

    for name in to_legacy:
        if dry_run:
            print(f"  legacy  {name:44s} -> {c.legacy_dir}")
            continue
        submit_legacy(c, name, sizes.get(name, 1 << 35))

    for name in to_rustle:
        expected = sizes.get(name, 1 << 62)  # unknown is not the same as small
        resources = resources_for(expected)
        if dry_run:
            print(
                f"  rustler {name:44s} {expected / 2**30:8.2f} GiB  "
                f"{resources.cpus_per_task:3d} cpu  {resources.mem:>5s}  "
                f"{resources.time:>10s}   -> embed on afterok"
            )
            continue
        job = submit(
            "expts.preprocess.preprocess:rustler",
            args={
                "dataset": name,
                "raw_dir": c.raw_dir,
                "out_dir": c.out_dir,
                "source_repo": c.source_repo,
            },
            resources=resources,
            name=f"pre-{name}",
            setup=SETUP,
            repo_root=REPO_ROOT,
            log_root=log_root(c),
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            clone_ttl_days=7,
        )
        # Queued now, held by slurm until its rustler stage succeeds: the GPU
        # queue fills itself behind the cpu one, with nothing to poll and no
        # second pass to remember to run.
        submit_embed(c, name, expected, after=job.id)


if __name__ == "__main__":
    main(pick(sys.argv), dry_run="--dry-run" in sys.argv)
