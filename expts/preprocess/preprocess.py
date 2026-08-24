import json
import shutil
import time
from pathlib import Path

from rt.preprocess import (
    dataset_name,
    embed_dataset,
    run_rustler_pre,
    update_meta_with_embeddings,
)
from rt.preprocess.legacy import rustler_one_legacy


def is_rustler_done(pre_dataset_dir: Path) -> bool:
    return all(
        (pre_dataset_dir / f).exists()
        for f in ("meta.json", "table_info.json", "column_index.json", "text.json")
    )


def is_done(pre_dataset_dir: Path, embedder: str) -> bool:
    meta_path = pre_dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        entry = (
            json.loads(meta_path.read_text()).get("text_embeddings", {}).get(embedder)
        )
    except json.JSONDecodeError:
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
    out_root, raw = Path(out_dir).expanduser(), Path(raw_dir).expanduser() / dataset
    pre_dataset_dir = out_root / dataset

    if is_rustler_done(pre_dataset_dir):
        print(f"= {dataset}: rustler already done", flush=True)
        return

    if not (raw / "manifest.yaml").is_file():
        raise FileNotFoundError(f"no manifest.yaml in {raw}")
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
    out_root = Path(out_dir).expanduser()
    pre_dataset_dir = out_root / dataset
    if is_rustler_done(pre_dataset_dir):
        print(f"= {dataset}: legacy rustler already done", flush=True)
        return

    started = time.monotonic()
    rustler_one_legacy(f"{raw_dir}/{dataset}", out_root)
    shutil.rmtree(out_root / "_transformed" / dataset, ignore_errors=True)
    for leftover in (out_root / "_transformed",):
        if leftover.is_dir() and not any(leftover.iterdir()):
            leftover.rmdir()
    print(
        f"= {dataset}: legacy rustler {time.monotonic() - started:.0f}s  "
        f"{dir_bytes(pre_dataset_dir) / 2**30:.2f} GiB",
        flush=True,
    )
