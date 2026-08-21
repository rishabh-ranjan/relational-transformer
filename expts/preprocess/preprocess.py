"""One database of the Join, in two stages that slurm schedules separately.

Composed from `rt.preprocess`'s primitives rather than its `many` loop, for two
reasons that loop cannot express: the raw data is read from a directory
downloaded once, while `meta.json` still has to record the Hub spec the data
came from -- where it was read and what it is are different facts.

Both stages are idempotent, because a preempted job is requeued onto the same
arguments: work already finished is skipped, work interrupted part-way is redone
from the start.

Why they are two jobs is in [README.md](README.md).
"""

import json
import shutil
import time
from pathlib import Path

# Imported here rather than inside the targets: submit.py imports this module,
# so a name that has moved is a submit-time error instead of a job that queues,
# waits, starts and only then finds out.
from rt.preprocess import (
    dataset_name,
    embed_dataset,
    run_rustler_pre,
    update_meta_with_embeddings,
)
from rt.preprocess.legacy import rustler_one_legacy


def is_rustler_done(pre_dataset_dir: Path) -> bool:
    """True once rustler's artifacts are all there.

    `text.json` is written last, so its presence is what says the stage
    finished rather than died half-way through.
    """
    return all(
        (pre_dataset_dir / f).exists()
        for f in ("meta.json", "table_info.json", "column_index.json", "text.json")
    )


def is_done(pre_dataset_dir: Path, embedder: str) -> bool:
    """True once `meta.json` records this embedder's file and the file is there.

    Not "the directory exists": rustler writes its artifacts before the
    embedding stage runs, so a database between the two looks complete and is
    not.
    """
    meta_path = pre_dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        entry = (
            json.loads(meta_path.read_text()).get("text_embeddings", {}).get(embedder)
        )
    except json.JSONDecodeError:  # killed mid-write
        return False
    return bool(entry) and (pre_dataset_dir / entry["file"]).exists()


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


def rustler(
    *,
    dataset: str,
    raw_dir: str,
    out_dir: str,
    source_repo: str,
) -> None:
    """`<raw_dir>/<dataset>` -> `<out_dir>/<dataset>`. No GPU, one thread.

    `source_repo` is what `meta.json` records as the data's origin -- the Hub
    repo the raw directory was downloaded from, not the local path it was read
    through.
    """
    out_root, raw = Path(out_dir).expanduser(), Path(raw_dir).expanduser() / dataset
    pre_dataset_dir = out_root / dataset

    if is_rustler_done(pre_dataset_dir):
        print(f"= {dataset}: rustler already done", flush=True)
        return

    if not (raw / "manifest.yaml").is_file():
        raise FileNotFoundError(f"no manifest.yaml in {raw}")
    # rustler names the output directory from the manifest, not from the path,
    # so a mismatch would silently write to a name this sweep is not tracking.
    name = dataset_name(raw)
    if name != dataset:
        raise ValueError(f"{raw}/manifest.yaml is named {name!r}, not {dataset!r}")

    started = time.monotonic()
    run_rustler_pre(raw, out_root, source=f"{source_repo}/{dataset}", skip_tasks=False)
    print(
        f"= {dataset}: rustler {time.monotonic() - started:.0f}s  "
        f"{dir_bytes(pre_dataset_dir) / 2**30:.2f} GiB",
        flush=True,
    )


def embed(
    *,
    dataset: str,
    out_dir: str,
    embedder: str,
    batch_size: int,
) -> None:
    """Text embeddings for a database rustler has already written."""
    pre_dataset_dir = Path(out_dir).expanduser() / dataset

    if is_done(pre_dataset_dir, embedder):
        print(f"= {dataset}: embeddings already done", flush=True)
        return
    if not is_rustler_done(pre_dataset_dir):
        raise FileNotFoundError(f"{pre_dataset_dir} has no rustler output to embed")

    started = time.monotonic()
    d_text = embed_dataset(pre_dataset_dir, embedder, batch_size)
    update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)
    print(
        f"= {dataset}: embed {time.monotonic() - started:.0f}s  d_text {d_text}  "
        f"{dir_bytes(pre_dataset_dir) / 2**30:.2f} GiB total",
        flush=True,
    )


def legacy_rustler(
    *,
    dataset: str,
    raw_dir: str,
    out_dir: str,
) -> None:
    """The cpu stage of the RT-v1 variant: boolean transform, then rustler.

    The same shape as `rustler` above, and for the same reason -- it is one
    thread and no GPU, and running it inside the embedding job meant every
    legacy database held ten cards through a phase that never touched them,
    which is what kept the main build's long poles queued behind it.

    Its GPU stage is `embed`, unchanged: legacy writes rustler's usual artifacts
    into its own directory, so the stage that reads them does not care which
    tree it is pointed at.

    Writes beside the main build, not inside it, so an unfinished legacy tree
    cannot ride along with the upload that replaces the collection. The RT-v1
    checkpoints read this; half of it on the Hub is worse than the old one.
    """
    out_root = Path(out_dir).expanduser()
    pre_dataset_dir = out_root / dataset
    if is_rustler_done(pre_dataset_dir):
        print(f"= {dataset}: legacy rustler already done", flush=True)
        return

    started = time.monotonic()
    rustler_one_legacy(f"{raw_dir}/{dataset}", out_root)
    # rt.preprocess.legacy writes the boolean-cast copy of the raw database to
    # <out>/_transformed on its way through. It is scratch -- a relbench-format
    # copy of data that is already published elsewhere -- and this directory is
    # published, so it has to go -- 247 files and 5.9 GiB of it.
    shutil.rmtree(out_root / "_transformed" / dataset, ignore_errors=True)
    for leftover in (out_root / "_transformed",):
        if leftover.is_dir() and not any(leftover.iterdir()):
            leftover.rmdir()
    print(
        f"= {dataset}: legacy rustler {time.monotonic() - started:.0f}s  "
        f"{dir_bytes(pre_dataset_dir) / 2**30:.2f} GiB",
        flush=True,
    )
