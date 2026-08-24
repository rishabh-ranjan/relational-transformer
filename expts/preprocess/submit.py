import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import defaultdict

from huggingface_hub import HfApi
from roach.slurm.clusters.ilc import ILC  # noqa: E402

from expts.preprocess.preprocess import is_done, is_rustler_done  # noqa: E402
from roach.slurm import Resources, submit  # noqa: E402

EMBEDDER = "all-MiniLM-L12-v2"
SETUP = (
    f'pixi run python -c "from huggingface_hub import snapshot_download;'
    f" snapshot_download('sentence-transformers/{EMBEDDER}')\"",
)

NAME = "relbench"
SOURCE_REPO = "stanford-star/relbench"
TARGET_REPO = "stanford-star/relbench-preprocessed"
CURATED = None
BIG_TEXT_BYTES = 3 << 29
KEEP = ("db-task-lists", "legacy")
OUT_NAME = f"{NAME}-preprocessed"
LEGACY_DIR = f"~/scratch/share/stanford-star/{OUT_NAME}/legacy"


RAW_DIR = f"~/scratch/share/stanford-star/{NAME}"
OUT_DIR = f"~/scratch/share/stanford-star/{OUT_NAME}"
LOG_ROOT = f"~/scratch/relational-transformer/preprocess/{NAME}/slurm-logs"
SIZES = Path(__file__).with_name(f"sizes-{NAME}.json")
REPO_ROOT = "~/clones/rishabh-ranjan/relational-transformer"
CLONE_ROOT = "~/roach_clones"
SECRETS_DIR = "~/scratch/.secrets"
BATCH_SIZE = 1024


def load_sizes() -> dict[str, dict[str, int]]:
    return json.loads(SIZES.read_text())


def out_bytes(sizes: dict, name: str, default: int) -> int:
    return sizes.get(name, {}).get("out", default)


def text_bytes(sizes: dict, name: str, default: int) -> int:
    return sizes.get(name, {}).get("text", default)


QOS: str | None = None


def resources_for(expected_bytes: int) -> Resources:
    all_nodes = "hyperturing1,hyperturing2,turing1,turing2,turing3"
    big_nodes = "hyperturing1,hyperturing2"
    for limit, walltime, nodes in (
        (1 << 30, "2:00:00", all_nodes),
        (16 << 30, "8:00:00", all_nodes),
        (40 << 30, "16:00:00", all_nodes),
        (1 << 62, "1-00:00:00", big_nodes),
    ):
        if expected_bytes < limit:
            break
    mem = max(8 << 30, 3 * expected_bytes)
    return Resources(
        partition="il",
        account="infolab",
        qos=QOS,
        time=walltime,
        gpus="0",
        cpus_per_task=1,
        ntasks=None,
        exclusive=False,
        mem=f"{mem // 2**30}G",
        mem_per_gpu=None,
        constraint=None,
        nodelist=nodes,
    )


BIG_NODES = ("hyperturing2",)
BIG_GPUS = 10
SMALL_NODES = ("turing1", "turing2", "turing3")


def embed_resources(text_bytes_: int) -> Resources:
    for limit, walltime in (
        (1 << 28, "2:00:00"),
        (2 << 30, "8:00:00"),
        (1 << 62, "1-00:00:00"),
    ):
        if text_bytes_ < limit:
            break

    if text_bytes_ >= BIG_TEXT_BYTES:
        return Resources(
            partition="il",
            account="infolab",
            qos=QOS,
            time=walltime,
            gpus=str(BIG_GPUS),
            cpus_per_task=4 * BIG_GPUS,
            ntasks=1,
            exclusive=False,
            mem=None,
            mem_per_gpu="40G",
            constraint=None,
            nodelist=",".join(BIG_NODES),
        )

    return Resources(
        partition="il",
        account="infolab",
        qos=QOS,
        time=walltime,
        gpus="1",
        cpus_per_task=4,
        ntasks=1,
        exclusive=False,
        mem=None,
        mem_per_gpu="240G",
        constraint=None,
        nodelist=",".join(SMALL_NODES),
    )


def queued_names(prefix: str) -> set[str]:
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
    sizes = load_sizes()
    names = sorted(
        p.parent.name for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
    )
    return sorted(names, key=lambda n: -out_bytes(sizes, n, 0))


def fully_downloaded(names: list[str]) -> set[str]:
    remote: dict[str, dict[str, int]] = defaultdict(dict)
    info = HfApi().repo_info(SOURCE_REPO, repo_type="dataset", files_metadata=True)
    for f in info.siblings:
        if "/" in f.rfilename:
            db, rest = f.rfilename.split("/", 1)
            remote[db][rest] = f.size or 0

    ready = set()
    for name in names:
        d = Path(RAW_DIR).expanduser() / name
        want = remote.get(name)
        if not want:
            continue
        if all(
            (d / rel).is_file() and (d / rel).stat().st_size == size
            for rel, size in want.items()
        ):
            ready.add(name)
    return ready


def submit_legacy_rustler(name: str, expected_bytes: int):
    return submit(
        "expts.preprocess.preprocess:legacy_rustler",
        args={
            "dataset": name,
            "raw_dir": RAW_DIR,
            "out_dir": LEGACY_DIR,
        },
        resources=resources_for(expected_bytes),
        name=f"lpre-{name}",
        setup=SETUP,
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=LOG_ROOT,
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )


def submit_embed(
    name: str,
    expected_bytes: int,
    after: str | None = None,
    *,
    out_dir: str = OUT_DIR,
    prefix: str = "emb",
):
    return submit(
        "expts.preprocess.preprocess:embed",
        args={
            "dataset": name,
            "out_dir": out_dir,
            "embedder": EMBEDDER,
            "batch_size": 1024,
        },
        resources=embed_resources(expected_bytes),
        name=f"{prefix}-{name}",
        setup=SETUP,
        repo_root="~/clones/rishabh-ranjan/relational-transformer",
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=LOG_ROOT,
        clone_root="~/roach_clones",
        secrets_dir="~/scratch/.secrets",
        after=after,
    )


def check_tree_is_submittable() -> None:
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd="~/clones/rishabh-ranjan/relational-transformer",
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "the working tree is dirty, so no job can be submitted "
            "(jobs clone the commit you submit from):\n" + dirty
        )


def main() -> None:
    assert QOS is not None, (
        "QOS is blank: pick this submission's tier per "
        "../README.md#allocating-a-sweep and write it in"
    )
    check_tree_is_submittable()
    sizes, out = load_sizes(), Path(OUT_DIR).expanduser()
    names = datasets()
    ready = fully_downloaded(names)
    pre_running, emb_running = queued_names("pre-"), queued_names("emb-")
    print(f"{len(names)} databases in {RAW_DIR}, {len(ready)} fully downloaded")

    lpre_running, lemb_running = queued_names("lpre-"), queued_names("lemb-")
    legacy_out = Path(LEGACY_DIR).expanduser() if LEGACY_DIR else None
    to_rustle, to_embed, done, busy, waiting = [], [], 0, 0, 0
    to_legacy_rustle, to_legacy_embed = [], []
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
        if not LEGACY_DIR or name not in ready:
            continue
        ld = legacy_out / name
        if is_done(ld, EMBEDDER):
            continue
        if is_rustler_done(ld):
            if name not in lemb_running:
                to_legacy_embed.append(name)
        elif name not in lpre_running:
            to_legacy_rustle.append(name)
    print(f"  {done} complete, {busy} queued or running, {waiting} still downloading")
    print(
        f"  submitting {len(to_rustle)} rustler + {len(to_embed)} embed"
        + (
            f" + {len(to_legacy_rustle)} legacy rustler"
            f" + {len(to_legacy_embed)} legacy embed"
            if LEGACY_DIR
            else ""
        )
    )

    for name in to_embed:
        submit_embed(name, text_bytes(sizes, name, 1 << 33))

    for name in to_legacy_embed:
        submit_embed(
            name, text_bytes(sizes, name, 1 << 33), out_dir=LEGACY_DIR, prefix="lemb"
        )

    for name in to_legacy_rustle:
        job = submit_legacy_rustler(name, out_bytes(sizes, name, 1 << 62))
        submit_embed(
            name,
            text_bytes(sizes, name, 1 << 33),
            after=job.id,
            out_dir=LEGACY_DIR,
            prefix="lemb",
        )

    for name in to_rustle:
        expected = out_bytes(sizes, name, 1 << 62)
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
            repo_root="~/clones/rishabh-ranjan/relational-transformer",
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=LOG_ROOT,
            clone_root="~/roach_clones",
            secrets_dir="~/scratch/.secrets",
        )
        submit_embed(name, text_bytes(sizes, name, 1 << 33), after=job.id)


if __name__ == "__main__":
    main()
