"""One database of the Join, in two stages that slurm schedules separately.

Composed from `rt.preprocess`'s primitives rather than its `many` loop, for two
reasons this sweep needs and that loop cannot express: the raw data is read from
a directory downloaded once (639 databases fetched per job would be 639 Hub
round-trips, and the Hub rate-limits long before that), while `meta.json` still
has to record the Hub spec the data came from -- where it was read and what it
is are different facts.

The two stages are separate jobs because they want opposite things, and one job
doing both gets neither:

* `rustler` is **single-threaded and cpu-only** -- measured, `TotalCPU` equals
  `Elapsed` on every database -- so its concurrency should be bounded by memory
  and by nothing else. It is also ~70% of the work, and near-100% on the big
  databases.
* `embed` wants a GPU for a few seconds to a few minutes.

Run together, every rustler stage holds a GPU it is not using, and 50 GPUs
across five nodes caps the sweep at 50 concurrent databases no matter how much
memory is free. Apart, the cpu stage runs as wide as memory allows and the GPU
stage is a short queue behind it.

Both stages are idempotent, because a preempted job is requeued onto the same
arguments: work already finished is skipped, work interrupted part-way is redone
from the start.
"""

from __future__ import annotations

import json
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
from rt.preprocess.legacy import preprocess_one_legacy


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
    out_root, raw = Path(out_dir), Path(raw_dir) / dataset
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
    pre_dataset_dir = Path(out_dir) / dataset

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


def legacy(
    *,
    dataset: str,
    raw_dir: str,
    out_dir: str,
    source_repo: str,
    embedder: str,
    batch_size: int,
) -> None:
    """The RT-v1 variant of one database: boolean typing, then the usual pipeline.

    One job rather than two, unlike the main build. It is a handful of databases
    rather than 639, and `rt.preprocess.legacy` does the transform, rustler and
    the embedding in one call -- splitting it would mean reaching into that
    module for the sake of a stage that is not the long pole here.

    Writes beside the main build, not inside it, so an unfinished legacy tree
    cannot ride along with the upload that replaces the collection. The RT-v1
    checkpoints read this; half of it on the Hub is worse than the old one.
    """
    out_root = Path(out_dir)
    pre_dataset_dir = out_root / dataset
    if is_done(pre_dataset_dir, embedder):
        print(f"= {dataset}: legacy already done", flush=True)
        return

    started = time.monotonic()
    preprocess_one_legacy(
        f"{raw_dir}/{dataset}",
        out_root,
        embedder=embedder,
        batch_size=batch_size,
        upload_repo=None,  # published by finalize.py, once the tree is complete
        private=True,
        revision=None,
    )
    print(
        f"= {dataset}: legacy {time.monotonic() - started:.0f}s  "
        f"{dir_bytes(pre_dataset_dir) / 2**30:.2f} GiB",
        flush=True,
    )
