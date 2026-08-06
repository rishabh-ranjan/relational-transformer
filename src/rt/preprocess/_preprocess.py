#!/usr/bin/env python
"""Preprocess relbench-3.0.0 datasets into rustler's shareable on-disk format.

A dataset is addressed exactly like in the ``relbench`` loader: a local path, or
a HuggingFace Hub spec ``org/repo[/subdir]`` (e.g. ``stanford-star/the-join/join-act-mooc``
or ``stanford-star/relbench/rel-f1``). Hub datasets are downloaded (and cached) on
demand; local paths are used in place.

Pipeline per dataset:  download/resolve  ->  rustler `pre`  ->  text embeddings.
The result is a self-contained ``<out_dir>/<name>/`` directory (see ``meta.json``)
that can be used directly for training or uploaded to a Hub ``*-preprocessed`` repo
and consumed from there.

Subcommands::


Recommended sharing workflow for a large collection (e.g. the 650-dataset Join):
preprocess everything locally with ``many`` (skipping uploads), then push the whole
``out-dir`` in one resumable ``upload --bulk`` pass. ``--bulk`` uses
``upload_large_folder`` (batched commits, far fewer Hub API calls than per-dataset
``upload_folder``), which avoids the account rate limits that per-dataset uploads hit.

Build the preprocessor binary first: ``pixi run build-pre`` (or it is built
automatically by the ``preprocess`` pixi task).
"""

import json
import sys
from pathlib import Path


from huggingface_hub import HfApi, snapshot_download


# --------------------------------------------------------------------------- #
# Hub / local addressing  (mirrors relbench.hf so we need no relbench dep)
# --------------------------------------------------------------------------- #
def resolve_repo(spec: str) -> tuple[str, str]:
    """Split a Hub spec into ``(repo_id, subdir)``.

    ``"org/name"`` -> ``("org/name", "")``; ``"org/name/a/b"`` -> ``("org/name", "a/b")``.
    """
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"{spec!r} is not a Hub 'org/name[/subdir]' spec or a local path."
        )
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def resolve_dataset_dir(spec: str, revision: str | None = None) -> Path:
    """Return a local directory holding the dataset (manifest.yaml + db/ + tasks/).

    A local path with a ``manifest.yaml`` is used as-is; otherwise ``spec`` is a Hub
    ``org/repo[/subdir]`` and only that sub-path is downloaded (and cached).
    """
    p = Path(spec).expanduser()
    if (p / "manifest.yaml").exists():
        return p
    repo_id, subdir = resolve_repo(spec)
    if not subdir:
        return Path(
            snapshot_download(repo_id=repo_id, revision=revision, repo_type="dataset")
        )
    # One bulk snapshot scoped to ``subdir``. Downloading the files one by one
    # instead is what gets rate limited (HTTP 429): a dataset here is hundreds of
    # task dirs, and every hf_hub_download is its own HEAD + GET. snapshot_download
    # resolves the whole file list in one paginated listing and fetches in
    # parallel, so a dataset costs a couple of API calls rather than thousands.
    local_root = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns=f"{subdir}/*",
    )
    return Path(local_root) / subdir


def dataset_name(dataset_dir: Path) -> str:
    """Read the dataset name from its manifest (the output subdirectory name)."""
    import yaml  # PyYAML ships with huggingface_hub's deps; fall back to a tiny parse

    text = (dataset_dir / "manifest.yaml").read_text()
    try:
        return yaml.safe_load(text)["name"]
    except Exception:
        for line in text.splitlines():
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"no 'name' in {dataset_dir / 'manifest.yaml'}")


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def run_rustler_pre(
    dataset_dir: Path, out_dir: Path, source: str, skip_tasks: bool
) -> None:
    from rt.rustler import preprocess

    print(f"+ preprocess {dataset_dir} -> {out_dir}", flush=True)
    preprocess(str(dataset_dir), str(out_dir), source=source, skip_tasks=skip_tasks)


def embed_dataset(pre_dataset_dir: Path, embedder: str, batch_size: int) -> int:
    """Compute text embeddings for a preprocessed dataset; return d_text."""
    from rt.preprocess.embed import embed_texts

    # Lazy import so download/upload/list work without torch installed.

    out_root = pre_dataset_dir.parent
    name = pre_dataset_dir.name
    embed_texts(
        dataset_name=name,
        pre_dir=str(out_root),
        device=None,  # auto: all visible GPUs, else CPU
        batch_size=batch_size,
        embedder=embedder,
    )
    emb_path = pre_dataset_dir / f"text_emb_{embedder}.bin"
    num_text = len(json.loads((pre_dataset_dir / "text.json").read_text()))
    # bfloat16 -> 2 bytes/elem; the emb file is (num_text, d_text) row-major.
    d_text = emb_path.stat().st_size // (max(num_text, 1) * 2)
    return d_text


def _embeddings_done(pre_dataset_dir: Path) -> bool:
    """True once ``meta.json`` records its text-embedding files and they exist.
    Used by ``--skip-existing`` so a dataset whose embedding step was interrupted
    (meta.json present, but no ``.bin``) is reprocessed rather than skipped."""
    meta_path = pre_dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        embs = json.loads(meta_path.read_text()).get("text_embeddings", {})
    except Exception:
        return False
    return bool(embs) and all(
        (pre_dataset_dir / e["file"]).exists() for e in embs.values()
    )


def update_meta_with_embeddings(
    pre_dataset_dir: Path, embedder: str, d_text: int
) -> None:
    meta_path = pre_dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("text_embeddings", {})[embedder] = {
        "file": f"text_emb_{embedder}.bin",
        "d_text": d_text,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def upload_dataset(pre_dataset_dir: Path, repo: str, private: bool) -> None:
    """Upload ``<out_dir>/<name>/`` to ``repo`` under ``<name>/`` on the Hub."""
    name = pre_dataset_dir.name
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"uploading {pre_dataset_dir} -> {repo}/{name}", flush=True)
    api.upload_folder(
        folder_path=str(pre_dataset_dir),
        path_in_repo=name,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"add preprocessed {name}",
    )
    print(f"uploaded {repo}/{name}", flush=True)


def bulk_upload(out_dir: Path, repo: str, private: bool) -> None:
    """Upload an entire preprocessed ``out_dir`` (all ``<name>/`` subdirs) in one
    resumable pass with ``upload_large_folder``.

    This is the recommended path for sharing a whole collection (e.g. the 650-dataset
    Join): it batches files and commits in chunks, so it uses far fewer Hub API calls
    than uploading each dataset with ``upload_folder`` -- which trips account-level
    rate limits on big collections. It is resumable: re-running picks up where an
    interrupted upload left off. Workflow: preprocess locally, then bulk-upload.
    """
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"bulk-uploading {out_dir} -> {repo} (upload_large_folder)", flush=True)
    api.upload_large_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(out_dir),
    )
    print(f"bulk-uploaded {out_dir} -> {repo}", flush=True)


def preprocess_one(
    spec: str,
    out_dir: Path,
    *,
    embedder: str,
    batch_size: int,
    skip_tasks: bool,
    embed: bool = True,
    upload_repo: str | None,
    private: bool,
    revision: str | None,
) -> Path:
    dataset_dir = resolve_dataset_dir(spec, revision=revision)
    name = dataset_name(dataset_dir)
    pre_dataset_dir = out_dir / name
    print(f"=== preprocessing {name} ({spec}) -> {pre_dataset_dir} ===", flush=True)

    run_rustler_pre(dataset_dir, out_dir, source=spec, skip_tasks=skip_tasks)
    if embed:
        d_text = embed_dataset(pre_dataset_dir, embedder, batch_size)
        update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)
    if upload_repo:
        upload_dataset(pre_dataset_dir, upload_repo, private=private)
    return pre_dataset_dir


# --------------------------------------------------------------------------- #
# Listing a collection repo (e.g. the-join's join-*/ datasets)
# --------------------------------------------------------------------------- #
def list_datasets(repo: str, revision: str | None = None) -> list[str]:
    """Top-level dataset subdirectories of a Hub collection repo (those with a
    manifest.yaml), as ``org/repo/<subdir>`` specs."""
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    subdirs = sorted(
        {
            f.split("/", 1)[0]
            for f in files
            if f.endswith("/manifest.yaml") and f.count("/") == 1
        }
    )
    return [f"{repo}/{d}" for d in subdirs]


def one(
    *,
    dataset: str,
    out_dir: str,
    embedder: str,
    batch_size: int,
    skip_tasks: bool,
    embed: bool,
    upload_repo: str | None,
    public: bool,
    revision: str | None,
) -> None:
    """Preprocess a single dataset (a local path or ``org/repo[/subdir]``)."""
    preprocess_one(
        dataset,
        Path(out_dir).expanduser(),
        embedder=embedder,
        batch_size=batch_size,
        skip_tasks=skip_tasks,
        embed=embed,
        upload_repo=upload_repo,
        private=not public,
        revision=revision,
    )


def many(
    *,
    repo: str,
    out_dir: str,
    shard: int,
    num_shards: int,
    skip_existing: bool,
    embedder: str,
    batch_size: int,
    skip_tasks: bool,
    embed: bool,
    upload_repo: str | None,
    public: bool,
    revision: str | None,
) -> None:
    """Preprocess every dataset in a Hub collection, or this job's shard of them."""
    specs = list_datasets(repo, revision=revision)
    assert 0 <= shard < num_shards, (
        f"shard must be in [0, num_shards); got shard={shard} num_shards={num_shards}"
    )
    shard = specs[shard::num_shards]
    print(
        f"shard {shard}/{num_shards}: {len(shard)} of {len(specs)} datasets",
        flush=True,
    )
    out_dir = Path(out_dir).expanduser()
    failures = []
    for i, spec in enumerate(shard):
        name = spec.rsplit("/", 1)[-1]
        if skip_existing and _embeddings_done(out_dir / name):
            print(f"[{i + 1}/{len(shard)}] skip existing {name}", flush=True)
            continue
        print(f"[{i + 1}/{len(shard)}] {spec}", flush=True)
        try:
            preprocess_one(
                spec,
                out_dir,
                embedder=embedder,
                batch_size=batch_size,
                skip_tasks=skip_tasks,
                embed=embed,
                upload_repo=upload_repo,
                private=not public,
                revision=revision,
            )
        except Exception as e:  # one bad dataset shouldn't sink the shard
            print(
                f"  FAILED {spec}: {type(e).__name__}: {e}", file=sys.stderr, flush=True
            )
            failures.append(spec)
    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for s in failures:
            print(f"  {s}", file=sys.stderr)
        sys.exit(1)


def ls(*, repo: str, revision: str | None) -> None:
    """Print the dataset specs in a Hub collection."""
    for spec in list_datasets(repo, revision=revision):
        print(spec)


def upload(*, pre_dir: str, repo: str, bulk: bool, public: bool) -> None:
    """Upload one preprocessed dataset, or a whole collection with ``bulk``."""
    path = Path(pre_dir).expanduser()
    if bulk:
        bulk_upload(path, repo, private=not public)
    else:
        upload_dataset(path, repo, private=not public)
