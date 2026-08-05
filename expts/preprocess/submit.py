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

from expts.preprocess.preprocess import is_done, is_rustler_done  # noqa: E402
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

# The embedding stage is uniform and short -- seconds to a couple of minutes --
# so one shape fits every database. It wants a GPU and almost nothing else.
EMBED = dict(cpus=4, mem="40G", walltime="2:00:00")


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
        exclusive=False,
        mem=f"{mem // 2**30}G",
        constraint=None,
        nodelist=nodes,
    )


def embed_resources() -> Resources:
    """The GPU stage. A bare count, not a type: these nodes carry rtx8000s and
    2080tis and MiniLM does not care which."""
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=EMBED["walltime"],
        gpus="1",
        cpus_per_task=EMBED["cpus"],
        exclusive=False,
        mem=EMBED["mem"],
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


def datasets() -> list[str]:
    """Every database in the raw directory, largest expected output first.

    Largest first so the giants -- which alone set the makespan -- start before
    the tail rather than behind it.
    """
    sizes = load_sizes()
    names = sorted(p.parent.name for p in Path(RAW_DIR).glob("*/manifest.yaml"))
    return sorted(names, key=lambda n: -sizes.get(n, 0))


def fully_downloaded(names: list[str]) -> set[str]:
    """Databases whose raw files are all present, per the Hub's own listing.

    A directory with a manifest.yaml in it is not the same as a downloaded
    database: `download.py` fetches 28k files across 639 directories, so at any
    moment most of them are partly there. Preprocessing one of those would
    quietly build a database short a few task tables and mark it finished, and
    nothing downstream would notice. One listing call is the only authority on
    what "all of it" means -- and checking makes this safe to run while the
    download is still going, which is the point: the cluster need not idle until
    the last byte lands.
    """
    from collections import defaultdict

    from huggingface_hub import HfApi

    remote: dict[str, set[str]] = defaultdict(set)
    for f in HfApi().list_repo_files(SOURCE_REPO, repo_type="dataset"):
        if "/" in f:
            db, rest = f.split("/", 1)
            remote[db].add(rest)

    ready = set()
    for name in names:
        d = Path(RAW_DIR) / name
        local = {
            str(p.relative_to(d))
            for p in d.rglob("*")
            if p.is_file() and ".cache" not in p.parts
        }
        if remote[name] and local >= remote[name]:
            ready.add(name)
    return ready


def submit_embed(name: str, after: str | None = None):
    return submit(
        "expts.preprocess.preprocess:embed",
        args={
            "dataset": name,
            "out_dir": OUT_DIR,
            "embedder": EMBEDDER,
            "batch_size": BATCH_SIZE,
        },
        resources=embed_resources(),
        name=f"emb-{name}",
        setup=SETUP,
        repo_root=REPO_ROOT,
        log_root=LOG_ROOT,
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
        clone_ttl_days=7,
        after=after,
    )


def main(dry_run: bool = False) -> None:
    """Submit whatever each database needs next.

    Two stages, one pass: a database with no rustler output gets a cpu-only
    rustler job, one that has it but no embeddings gets a GPU embed job. Because
    each is decided from what is on disk, re-running this is the resume for both
    stages at once, and the GPU queue fills behind the cpu one without any
    dependency to declare.
    """
    sizes, out = load_sizes(), Path(OUT_DIR)
    names = datasets()
    ready = fully_downloaded(names)
    pre_running, emb_running = queued_names("pre-"), queued_names("emb-")
    print(f"{len(names)} databases in {RAW_DIR}, {len(ready)} fully downloaded")

    to_rustle, to_embed, done, busy, waiting = [], [], 0, 0, 0
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
    print(f"  {done} complete, {busy} queued or running, {waiting} still downloading")
    print(f"  submitting {len(to_rustle)} rustler + {len(to_embed)} embed")

    for name in to_embed:
        if dry_run:
            print(f"  embed   {name:44s} 1 gpu  {EMBED['cpus']} cpu  {EMBED['mem']}")
            continue
        submit_embed(name)

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
                "raw_dir": RAW_DIR,
                "out_dir": OUT_DIR,
                "source_repo": SOURCE_REPO,
            },
            resources=resources,
            name=f"pre-{name}",
            setup=SETUP,
            repo_root=REPO_ROOT,
            log_root=LOG_ROOT,
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            clone_ttl_days=7,
        )
        # Queued now, held by slurm until its rustler stage succeeds: the GPU
        # queue fills itself behind the cpu one, with nothing to poll and no
        # second pass to remember to run.
        submit_embed(name, after=job.id)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
