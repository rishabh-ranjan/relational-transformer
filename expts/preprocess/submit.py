"""Submit the Join preprocessing sweep: one slurm job per database.

    pixi run python expts/preprocess/submit.py

Per database, resources come from what the previous build's output measured
(`sizes.py`), and slurm places them. Re-running is the resume: a database whose
output is already complete is not resubmitted, and one that is queued or running
is not duplicated. Why one job per database, and why two stages, is in
[README.md](README.md).
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roach.slurm import Resources, submit  # noqa: E402

from expts.preprocess.preprocess import is_done, is_rustler_done  # noqa: E402
from collections import defaultdict
from huggingface_hub import HfApi

# The two values every stage has to agree on, and the only ones kept out of
# their call sites: the embedder names the directory `is_done` looks for, so a
# copy that drifted from the one the jobs pass would report a finished database
# that was embedded with a different model. Everything else below is written
# where it is used.
EMBEDDER = "all-MiniLM-L12-v2"
# Every job runs `setup`, but the shared clone means it runs once per commit per
# node -- which is the right number of times to fetch the embedder. Left to the
# embed jobs, 50 of them start at once and each asks the Hub for the same model,
# and the Hub answers most of them with 429.
#
# The rustler extension is not built here: `pixi install` builds it, because the
# project is an editable dependency of its own environment.
SETUP = (
    f'pixi run python -c "from huggingface_hub import snapshot_download;'
    f" snapshot_download('sentence-transformers/{EMBEDDER}')\"",
)

# The collection. Edit it; the other sits below commented, as the record of what
# else has been run and what its quirks are.
NAME = "relbench"
SOURCE_REPO = "stanford-star/relbench"
TARGET_REPO = "stanford-star/relbench-preprocessed"
# A task list that cannot be derived from the build. None when every list the
# collection publishes can be recomputed from the metas.
CURATED = None
# Directories the published repo carries that are not a database of this build,
# and that the mirror upload must therefore not delete. legacy/ is RT-v1 boolean
# typing, which the released RT-v1 checkpoints read.
KEEP = ("db-task-lists", "legacy")
# Inside the collection's output, under the name it publishes at, so the whole
# replacement is one folder and one upload. It was outside once, to stop an
# unfinished tree riding along -- but finalize.py verifies both before it pushes
# anything, and a second upload call is a second thing that can fail after the
# first has already changed the repo. Which it did. None if there is none.
#
# The build directory is named separately from the repo it publishes to, so a
# rebuild can be staged beside the live one (`-preprocessed-new`) and promoted
# into place once it verifies. This is a rebuild, staged beside the live tree.
OUT_NAME = f"{NAME}-preprocessed-new"
LEGACY_DIR = f"/dfs/user/ranjanr/share/stanford-star/{OUT_NAME}/legacy"

# NAME = "the-join"
# SOURCE_REPO = "stanford-star/the-join"
# TARGET_REPO = "stanford-star/the-join-preprocessed"
# # rt-j is a whitelist of 475 databases, 126 excluded wholesale and none
# # partially: nothing in the build says which.
# CURATED = "rt-j-dbs.json"
# KEEP = ("db-task-lists",)
# OUT_NAME = f"{NAME}-preprocessed"
# LEGACY_DIR = None

RAW_DIR = f"/dfs/user/ranjanr/share/stanford-star/{NAME}"
OUT_DIR = f"/dfs/user/ranjanr/share/stanford-star/{OUT_NAME}"
LOG_ROOT = f"/dfs/user/ranjanr/slurm-logs/preprocess-{NAME}"
SIZES = Path(__file__).with_name(f"sizes-{NAME}.json")
REPO_ROOT = "/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer"
CLONE_ROOT = "/lfs/local/0/roach_clones"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"
BATCH_SIZE = 1024


def load_sizes() -> dict[str, dict[str, int]]:
    """database -> {"out": bytes, "text": bytes}, from the previous build.

    Two numbers because the stages scale on different things: rustler tracks
    output size, the embedding stage tracks text, and they are not proportional
    -- text is 12% of RelBench's output and 1.7% of the Join's. Written by
    sizes.py.
    """
    return json.loads(SIZES.read_text())


def out_bytes(sizes: dict, name: str, default: int) -> int:
    return sizes.get(name, {}).get("out", default)


def text_bytes(sizes: dict, name: str, default: int) -> int:
    return sizes.get(name, {}).get("text", default)


def resources_for(expected_bytes: int) -> Resources:
    """The cpu-only rustler stage: one cpu, memory from the expected output.

    hyperturing1/2 have 2 TiB of memory and 252 cpus; the turings have 754 GiB
    (turing3 1.4 TiB) and 80. Everything can run on all five; the largest
    databases are held to the hyperturings, where one can have half a terabyte
    without leaving the node unable to run anything alongside it.
    """
    all_nodes = "hyperturing1,hyperturing2,turing1,turing2,turing3"
    big_nodes = "hyperturing1,hyperturing2"
    # (max expected output bytes, wall clock, nodes)
    for limit, walltime, nodes in (
        (1 << 30, "2:00:00", all_nodes),
        (16 << 30, "8:00:00", all_nodes),
        (40 << 30, "16:00:00", all_nodes),
        (1 << 62, "1-00:00:00", big_nodes),
    ):
        if expected_bytes < limit:
            break
    # Memory comes from measurement: MaxRSS ran to about twice a database's
    # output, so this asks for three times with an 8 GiB floor. (The site's
    # MaxMemPerCPU of 10700M is not enforced against a per-node --mem here;
    # checked with sbatch --test-only, a 1-cpu job may ask for 200G.)
    mem = max(8 << 30, 3 * expected_bytes)
    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=walltime,
        # No GPU: this stage never touches one, and holding it would cap the
        # sweep at the 50 GPUs these five nodes have between them.
        gpus="0",
        # ONE cpu. Measured: TotalCPU equals Elapsed on every database, so the
        # stage is single-threaded and a wider request buys nothing while
        # costing a slot someone else's database could have had. How many run at
        # once is then how many cpus the five nodes have -- which is the point.
        cpus_per_task=1,
        ntasks=None,
        exclusive=False,
        mem=f"{mem // 2**30}G",
        mem_per_gpu=None,
        constraint=None,
        nodelist=nodes,
    )


# A database with this much text is the embedding stage's long pole, and gets a
# whole hyperturing instead of the usual six GPUs. Set above the second-largest
# in either collection (rel-stack 2.0 GiB, join-overture-maps 8.0 GiB is the one
# other database this catches) so it stays the exception it is meant to be.
BIG_TEXT_BYTES = 9 << 30
# rtx8000 nodes, ten cards each.
BIG_NODES = ("hyperturing1", "hyperturing2")
MAX_BIG_GPUS = 10


def biggest_free_gpu_node() -> tuple[str, int]:
    """The hyperturing with the most free GPUs right now, and how many.

    Read at submit time rather than fixed, because a request for ten cards on a
    node holding eight is a job that queues behind whatever is using the other
    two -- which for a database this size means the sweep waits on the scheduler
    instead of on the work. Asking for what is free starts now.

    Falls back to the full ten if slurm cannot be read or nothing is free: a job
    that queues is recoverable, a job asking for zero GPUs is not.
    """
    best = (BIG_NODES[0], MAX_BIG_GPUS)
    try:
        out = subprocess.run(
            [
                "sinfo",
                "-h",
                "-n",
                ",".join(BIG_NODES),
                "-O",
                "NodeHost:20,Gres:30,GresUsed:30",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return best

    free: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        node, gres, used = parts[0], parts[1], parts[2]
        try:
            # "gpu:rtx8000:10" and "gpu:rtx8000:4(IDX:0-3)"
            total = int(gres.split(":")[2].split("(")[0])
            taken = int(used.split(":")[2].split("(")[0])
        except (IndexError, ValueError):
            continue
        free[node] = max(0, min(MAX_BIG_GPUS, total) - taken)

    node = max(free, key=lambda n: free[n], default=None)
    if node is None or free[node] == 0:
        return best
    return node, free[node]


def embed_resources(text_bytes_: int) -> Resources:
    """The GPU stage. A bare count, not a type: these nodes carry rtx8000s and
    2080tis and MiniLM does not care which.

    Sized by text, not by total output: the stage's cost is the text it embeds,
    and the two are not proportional -- text is 12% of RelBench's output and
    1.7% of the Join's. Sizing embed off total output is what put
    join-overture-maps and rel-amazon in jobs too small for them.
    """
    # Walltime scales for the same reason memory does. join-overture-maps' 8 GiB
    # of text was still embedding when a flat 4h cut it off, and a timeout costs
    # the whole stage: the run has to start over from the first text. rel-amazon's
    # 11 GiB took hours; a database with a few hundred MiB takes minutes.
    for limit, walltime in (
        (1 << 28, "2:00:00"),
        (2 << 30, "8:00:00"),
        (1 << 62, "1-00:00:00"),
    ):
        if text_bytes_ < limit:
            break

    # The long pole gets a whole hyperturing. One database sets this stage's
    # makespan -- rel-amazon is 11 GiB of text against 2 GiB for the next one --
    # so it is the one worth giving every card on a node to, and the rtx8000s
    # are the better cards here. Everything else keeps the six-GPU shape below,
    # which is what lets the rest of the sweep run alongside it.
    if text_bytes_ >= BIG_TEXT_BYTES:
        node, gpus = biggest_free_gpu_node()
        return Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time=walltime,
            gpus=str(gpus),
            # Same four cpus per GPU the six-GPU shape uses: the dataloader
            # feeding each worker is what these are for.
            cpus_per_task=4 * gpus,
            ntasks=1,
            exclusive=False,
            mem=None,
            mem_per_gpu="40G",
            constraint=None,
            # Pinned to the node the count was measured on: "10 GPUs" is only
            # schedulable where 10 are actually free, and asking the pair would
            # let slurm pick the other one and hold the job until it drains.
            nodelist=node,
        )

    return Resources(
        partition="il",
        account="infolab",
        qos="il-lo",
        time=walltime,
        # SEVERAL GPUs in one job. sentence-transformers runs a worker per
        # device, and measured on 2M texts that is 4.06x on six RTX8000s (68%
        # efficiency) and 4.29x on six 2080Tis (71%) -- so the stage's own skew
        # stops mattering: ten databases are 88% of it, and each of those now
        # finishes in a quarter of the time instead of queueing behind one card.
        # Card type barely matters: one RTX8000 does 929 texts/s against a
        # 2080Ti's 849, a 9% gap, so there is nothing to route.
        gpus="6",
        cpus_per_task=24,
        # One rank with every GPU visible to it, not a rank per GPU:
        # sentence-transformers does the fan-out itself.
        ntasks=1,
        exclusive=False,
        mem=None,
        # Per GPU, not per node, and that is not a detail: the partition sets
        # DefMemPerGPU=240000M and applies it when deciding whether a job fits,
        # so with --mem the most GPUs a job can hold is RealMemory/240000M -- 3
        # on a turing, 8 on a hyperturing -- however little memory it asks for.
        # --mem-per-gpu replaces that default and lifts the limit.
        mem_per_gpu="40G",
        constraint=None,
        # All five rtx8000/2080Ti nodes. hyperturing1 was excluded for a while:
        # its GPUs threw "uncorrectable ECC error" on 18 jobs of the previous
        # build. The card has since been fixed, so it takes embed work again.
        nodelist="hyperturing1,hyperturing2,turing1,turing2,turing3",
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
    return sorted(names, key=lambda n: -out_bytes(sizes, n, 0))


def fully_downloaded(names: list[str]) -> set[str]:
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

    remote: dict[str, dict[str, int]] = defaultdict(dict)
    info = HfApi().repo_info(SOURCE_REPO, repo_type="dataset", files_metadata=True)
    for f in info.siblings:
        if "/" in f.rfilename:
            db, rest = f.rfilename.split("/", 1)
            remote[db][rest] = f.size or 0

    ready = set()
    for name in names:
        d = Path(RAW_DIR) / name
        want = remote.get(name)
        if not want:
            continue
        if all(
            (d / rel).is_file() and (d / rel).stat().st_size == size
            for rel, size in want.items()
        ):
            ready.add(name)
    return ready


def submit_legacy(name: str, expected_bytes: int):
    """The RT-v1 variant. One job: transform, rustler and embed together."""
    return submit(
        "expts.preprocess.preprocess:legacy",
        args={
            "dataset": name,
            "raw_dir": RAW_DIR,
            "out_dir": LEGACY_DIR,
            "source_repo": SOURCE_REPO,
            "embedder": EMBEDDER,
            "batch_size": 1024,
        },
        resources=embed_resources(expected_bytes),
        name=f"leg-{name}",
        setup=SETUP,
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        log_root=LOG_ROOT,
        # the node's own big disk, not /tmp (the 280G root filesystem): clones
        # are shared per commit and hold the pixi env, which pixi hardlinks from
        # the package cache only when the two are on the same filesystem
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
    )


def submit_embed(name: str, expected_bytes: int, after: str | None = None):
    """The GPU stage for one database, optionally held until its rustler job
    succeeds -- which is how the two stages are submitted in one pass."""
    return submit(
        "expts.preprocess.preprocess:embed",
        args={
            "dataset": name,
            "out_dir": OUT_DIR,
            "embedder": EMBEDDER,
            "batch_size": 1024,
        },
        resources=embed_resources(expected_bytes),
        name=f"emb-{name}",
        setup=SETUP,
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        log_root=LOG_ROOT,
        # the node's own big disk, not /tmp (the 280G root filesystem): clones
        # are shared per commit and hold the pixi env, which pixi hardlinks from
        # the package cache only when the two are on the same filesystem
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
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
        ["git", "status", "--porcelain"],
        cwd="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "the working tree is dirty, so no job can be submitted "
            "(jobs clone the commit you submit from):\n" + dirty
        )


def main() -> None:
    """Submit whatever each database needs next.

    Two stages, one pass: a database with no rustler output gets a cpu-only
    rustler job, one that has it but no embeddings gets a GPU embed job. Because
    each is decided from what is on disk, re-running this is the resume for both
    stages at once, and the GPU queue fills behind the cpu one without any
    dependency to declare.
    """
    check_tree_is_submittable()
    sizes, out = load_sizes(), Path(OUT_DIR)
    names = datasets()
    ready = fully_downloaded(names)
    pre_running, emb_running = queued_names("pre-"), queued_names("emb-")
    print(f"{len(names)} databases in {RAW_DIR}, {len(ready)} fully downloaded")

    leg_running = queued_names("leg-")
    legacy_out = Path(LEGACY_DIR)
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
            LEGACY_DIR
            and name in ready
            and name not in leg_running
            and not is_done(legacy_out / name, EMBEDDER)
        ):
            to_legacy.append(name)
    print(f"  {done} complete, {busy} queued or running, {waiting} still downloading")
    print(
        f"  submitting {len(to_rustle)} rustler + {len(to_embed)} embed"
        + (f" + {len(to_legacy)} legacy" if LEGACY_DIR else "")
    )

    for name in to_embed:
        submit_embed(name, text_bytes(sizes, name, 1 << 33))

    for name in to_legacy:
        submit_legacy(name, text_bytes(sizes, name, 1 << 33))

    for name in to_rustle:
        expected = out_bytes(sizes, name, 1 << 62)  # unknown is not small
        resources = resources_for(expected)
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
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root=LOG_ROOT,
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
        )
        # Queued now, held by slurm until its rustler stage succeeds: the GPU
        # queue fills itself behind the cpu one, with nothing to poll and no
        # second pass to remember to run.
        submit_embed(name, text_bytes(sizes, name, 1 << 33), after=job.id)


if __name__ == "__main__":
    main()
