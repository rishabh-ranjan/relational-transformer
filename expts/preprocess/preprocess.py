"""Preprocess one database of the Join. One slurm job runs one of these.

Composed from `rt.preprocess`'s primitives rather than its `many` loop, for two
reasons this sweep needs and that loop cannot express: the raw data is read from
a directory downloaded once (639 databases fetched per job would be 639 Hub
round-trips, and the Hub rate-limits long before that), while `meta.json` still
has to record the Hub spec the data came from -- where it was read and what it
is are different facts. Slurm does the scheduling that `many`'s `--shard` would
otherwise be doing by hand, and doing badly: it splits round-robin, on a
collection where 20 of 639 databases are half the bytes.

Idempotent, because a preempted job is requeued onto the same arguments: a
database whose embeddings are already written is skipped, and one interrupted
part-way is redone from the start (rustler overwrites, and the marker of
completion is `meta.json` naming an embedding file that exists).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def is_done(pre_dataset_dir: Path, embedder: str) -> bool:
    """True once `meta.json` records this embedder's file and the file is there.

    Not "the directory exists": rustler writes its artifacts before the
    embedding step runs, so a job killed in between leaves a directory that
    looks complete and is not.
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


def preprocess(
    *,
    dataset: str,
    raw_dir: str,
    out_dir: str,
    source_repo: str,
    embedder: str,
    batch_size: int,
) -> None:
    """`<raw_dir>/<dataset>` -> `<out_dir>/<dataset>`, then text embeddings.

    `source_repo` is what `meta.json` records as the data's origin -- the Hub
    repo the raw directory was downloaded from, not the local path it was read
    through.
    """
    from rt.preprocess.main import (
        dataset_name,
        embed_dataset,
        run_rustler_pre,
        update_meta_with_embeddings,
    )

    out_root, raw = Path(out_dir), Path(raw_dir) / dataset
    pre_dataset_dir = out_root / dataset

    if is_done(pre_dataset_dir, embedder):
        print(f"= {dataset}: already done, nothing to do", flush=True)
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
    rustler_done = time.monotonic()

    d_text = embed_dataset(pre_dataset_dir, embedder, batch_size)
    update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)
    finished = time.monotonic()

    # One line per database, greppable out of the logs when a number is wanted
    # that the output directory no longer remembers.
    print(
        f"= {dataset}: {dir_bytes(pre_dataset_dir) / 2**30:.2f} GiB  "
        f"rustler {rustler_done - started:.0f}s  "
        f"embed {finished - rustler_done:.0f}s  "
        f"total {finished - started:.0f}s  d_text {d_text}",
        flush=True,
    )
