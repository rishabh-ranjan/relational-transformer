"""pre_dir resolution: every ``pre_dir`` in rt accepts a local path or a Hub
repo ``org/repo[/subdir]``; local wins, Hub files download into the HF cache on
demand."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

CORE_FILES = (
    "meta.json",
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    "table_info.json",
    "column_index.json",
)

# Small per-dataset files sufficient to browse schema/tables/columns without
# pulling the (potentially large) node blobs or embeddings.
METADATA_FILES = ("meta.json", "table_info.json", "column_index.json")


def resolve_repo(spec: str) -> tuple[str, str]:
    """Split a Hub spec into ``(repo_id, subdir)``.

    ``"org/name"`` -> ``("org/name", "")``; ``"org/name/a/b"`` -> ``("org/name", "a/b")``.
    """
    parts = str(spec).strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"{spec!r} is neither an existing local path nor a Hub 'org/name[/subdir]' spec."
        )
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def is_local(pre_dir: str) -> bool:
    return Path(pre_dir).expanduser().exists()


def resolve_pre_dir(
    pre_dir: str,
    db_names,
    embedder: str,
    *,
    include_text: bool = False,
    metadata_only: bool = False,
    revision: str | None = None,
) -> str:
    """Return a local root directory containing ``<db>/`` subfolders for each db.

    If ``pre_dir`` is an existing local path it is returned as-is. Otherwise it is
    treated as a Hub ``org/repo[/subdir]`` and only the files needed for
    ``db_names`` (+ the chosen ``embedder``) are downloaded and cached.
    ``metadata_only`` fetches just the small schema files (no node blobs or
    embeddings) -- enough to browse tables/columns.
    """
    p = Path(pre_dir).expanduser()
    if p.exists():
        return str(p)

    repo_id, subdir = resolve_repo(pre_dir)
    prefix = f"{subdir}/" if subdir else ""
    file_set = METADATA_FILES if metadata_only else CORE_FILES
    patterns: list[str] = []
    for db in dict.fromkeys(db_names):  # dedup, preserve order
        base = f"{prefix}{db}"
        patterns += [f"{base}/{f}" for f in file_set]
        if not metadata_only:
            patterns.append(f"{base}/text_emb_{embedder}.bin")
        if include_text:
            patterns.append(f"{base}/text.json")

    # Every pattern here is a concrete path, so they double as the check that
    # the local cache actually holds this call's files.
    local = _snapshot(repo_id, revision, tuple(patterns), tuple(patterns))
    return str(Path(local) / subdir) if subdir else str(local)


@cache
def _snapshot(
    repo_id: str,
    revision: str | None,
    patterns: tuple[str, ...],
    require: tuple[str, ...] = (),
) -> str:
    """One bulk ``snapshot_download`` per (repo, revision, pattern set), cached.

    Every rank of a DDP job runs this independently, so anything issued per
    dataset here is multiplied by the world size and the Hub answers the burst
    with HTTP 429. Two rules keep the request count flat: fetch in bulk (one
    call for all the files, never one call per dataset), and try the local
    cache first so a warm node -- a requeue, or the other ranks on this node --
    makes no requests at all.

    ``require`` lists snapshot-relative paths that must all exist for the
    cached answer to count: a local-only snapshot happily returns a directory
    populated by some earlier, narrower download, so the directory existing
    proves nothing about the files this caller needs.
    """
    from huggingface_hub import snapshot_download

    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(patterns),
    )
    try:
        local = snapshot_download(**kwargs, local_files_only=True)
        if all((Path(local) / r).exists() for r in require):
            return local
    except Exception:
        pass
    return snapshot_download(**kwargs)


def _is_complete(dataset_dir: Path) -> bool:
    """A dataset is complete only once its text embeddings are written. The
    rustler step writes ``meta.json`` before embedding, so meta-presence alone
    would race a still-embedding dataset in a shared output dir."""
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        import json

        embs = json.loads(meta_path.read_text()).get("text_embeddings", {})
    except Exception:
        return False
    return bool(embs) and all(
        (dataset_dir / e["file"]).exists() for e in embs.values()
    )


def list_datasets(pre_dir: str, revision: str | None = None) -> list[str]:
    """Names of the preprocessed datasets under ``pre_dir`` (local dir or Hub repo)."""
    p = Path(pre_dir).expanduser()
    if p.exists():
        return sorted(d.name for d in p.iterdir() if _is_complete(d))

    from huggingface_hub import HfApi

    repo_id, subdir = resolve_repo(pre_dir)
    prefix = f"{subdir}/" if subdir else ""
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    out = set()
    for f in files:
        if f.startswith(prefix) and f.endswith("/meta.json"):
            rest = f[len(prefix):]
            if rest.count("/") == 1:  # <db>/meta.json
                out.add(rest.split("/", 1)[0])
    return sorted(out)


def read_meta(pre_dir: str, db: str, revision: str | None = None) -> dict:
    """Read one preprocessed dataset's ``meta.json`` (local or downloaded from Hub).

    Callers read metas for a whole db_task_list in a loop (see
    ``rt.data.tasks.get_tasks``), so this fetches *every* meta.json in the repo
    in one shot rather than one request per db: hundreds of dbs times the world
    size is exactly the request storm the Hub rate-limits.
    """
    p = Path(pre_dir).expanduser()
    if p.exists():
        return json.loads((p / db / "meta.json").read_text())

    repo_id, subdir = resolve_repo(pre_dir)
    prefix = f"{subdir}/" if subdir else ""
    rel = f"{prefix}{db}/meta.json"
    root = Path(_snapshot(repo_id, revision, (f"{prefix}*/meta.json",), (rel,)))
    return json.loads((root / rel).read_text())


# rustler's Sampler is an unpicklable Rust object, so any DataLoader over a
# RustlerDataset must use the 'fork' start method -- Python 3.14 defaults to
# 'forkserver'/'spawn', which pickle the worker's arguments and would fail with
# "cannot pickle 'builtins.Sampler'". We also share worker tensors via node-local
# files instead of /dev/shm (which dense multi-worker eval nodes, plus segments
# leaked by preempted jobs, exhaust -> "No space left on device"). Set both once,
# here, at import of the module that introduces the Sampler, so every entry point
# that touches rt.data (eval / baseline / scaling / training) is covered without
# each needing its own copy.
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass


@cache
def _load_column_index(db_name: str, pre_dir: str) -> dict:
    pre_dir = Path(pre_dir).expanduser()
    column_index_path = f"{pre_dir}/{db_name}/column_index.json"
    with open(column_index_path) as f:
        return json.load(f)


def get_column_index(
    column_name: str, table_name: str, db_name: str, pre_dir: str
) -> int:
    column_index = _load_column_index(db_name, pre_dir)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(
            f'Column "{target}" not found in {pre_dir}/{db_name}/column_index.json.'
        )

    return column_index[target]
