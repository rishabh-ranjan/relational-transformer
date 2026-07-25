"""pre_dir resolution: a ``pre_dir`` is a local directory of preprocessed
datasets (one subdirectory per db).

Preprocessed data is downloaded up front, not on demand -- see docs/train.md.
The datasets are hundreds of GiB and every rank and dataloader worker reads
them, so on-demand Hub fetches meant thousands of requests per run (rate-limited
with HTTP 429 even when the bytes were already cached, since each call still
revalidates over the network) and one copy per node. An explicit
``hf download`` into a path you can inspect is both faster and simpler.
"""

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

    Hub specs remain how released *checkpoints* are named (see
    ``rt.model.resolve_checkpoint``); data is local-only.
    """
    parts = str(spec).strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"{spec!r} is not a Hub 'org/name[/subdir]' spec.")
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def is_local(pre_dir: str) -> bool:
    return Path(pre_dir).expanduser().exists()


def resolve_pre_dir(pre_dir: str) -> str:
    """Validate a ``pre_dir`` and return it as a plain local path.

    Raises if it does not exist -- a missing pre_dir is a setup mistake to fix
    with ``hf download``, not something to paper over at runtime.
    """
    p = Path(pre_dir).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(
            f"pre_dir {pre_dir!r} does not exist. Preprocessed data is not "
            f"downloaded on demand; fetch it first, e.g.\n"
            f"  hf download stanford-star/the-join-preprocessed --repo-type dataset "
            f"--local-dir {pre_dir}\n"
            f"(see docs/train.md for fetching only the dbs you need)"
        )
    return str(p)


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
    return bool(embs) and all((dataset_dir / e["file"]).exists() for e in embs.values())


def list_datasets(pre_dir: str) -> list[str]:
    """Names of the complete preprocessed datasets under ``pre_dir``."""
    p = Path(resolve_pre_dir(pre_dir))
    return sorted(d.name for d in p.iterdir() if d.is_dir() and _is_complete(d))


def read_meta(pre_dir: str, db: str) -> dict:
    """Read one preprocessed dataset's ``meta.json`` from ``pre_dir``."""
    return json.loads((Path(pre_dir).expanduser() / db / "meta.json").read_text())


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

import torch  # noqa: E402

try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy("file_system")


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
