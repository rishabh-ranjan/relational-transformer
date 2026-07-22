"""Interactive web UI for inspecting rustler batch tensors.

Run:
    pixi run python -m rt.cli.ctx_viz
    # then open http://localhost:8765 in your browser

The server keeps a small LRU cache of `RustlerDataset` objects keyed by
the structural parameters that require a rebuild (db, table, target,
split, ctx sizes, bfs widths, walks, embedding model, shuffle seed, ...).
Per-request parameters that don't require a rebuild (context_seed,
mask_prob_max, item_idx, ctx_size within the configured cap) are applied
on the fly so iterating in the UI is fast.
"""

from __future__ import annotations

import errno
import json
import socket
import sys
import threading
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from rt.data import MAX_F2P_NBRS, RustlerDataset, get_column_index
from rt.data import resolve_pre_dir, resolve_repo
from rt.data import Task

SEM_TYPE_NAMES = ["number", "text", "datetime", "boolean"]
INT_MIN = np.iinfo(np.int32).min  # rustler uses i32::MIN as missing-timestamp sentinel



# ---------------------------------------------------------------------------
# Per-db assets (text vocab, table info, column index) cached in-process.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _load_text_json(db_dir: str) -> list[str]:
    with open(Path(db_dir) / "text.json") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _load_table_info(db_dir: str) -> dict:
    with open(Path(db_dir) / "table_info.json") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _load_column_index(db_dir: str) -> dict:
    with open(Path(db_dir) / "column_index.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sampler cache.
# ---------------------------------------------------------------------------
# Building a Sampler is expensive (it mmaps the rkyv blobs, computes table
# ranges, subsamples items). We rebuild only when something structural
# changes; mask_prob_max and context_seed are mutated on the cached object.

_DATASET_CACHE: "OrderedDict[tuple, RustlerDataset]" = OrderedDict()
_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_CACHE_MAX = 4


def _get_or_build_dataset(
    *,
    pre_root: str,
    db_name: str,
    table_name: str,
    target_column: str,
    split: str,
    columns_to_drop: tuple[str, ...],
    ctx_cap: int,
    local_ctx_size: int,
    bfs_width: int,
    num_walks: int,
    walk_length: int,
    prefer_latest: bool,
    embedding_model: str,
    d_text: int,
    items_per_task: int,
    bool_as_num: bool,
    skip_text_cols: bool,
    balance_labels: bool,
    ablate_schema_semantics: bool,
    shuffle_seed: int,
) -> RustlerDataset:
    key = (
        pre_root,
        db_name,
        table_name,
        target_column,
        split,
        columns_to_drop,
        ctx_cap,
        local_ctx_size,
        bfs_width,
        num_walks,
        walk_length,
        prefer_latest,
        embedding_model,
        d_text,
        items_per_task,
        bool_as_num,
        skip_text_cols,
        balance_labels,
        ablate_schema_semantics,
        shuffle_seed,
    )
    with _DATASET_CACHE_LOCK:
        if key in _DATASET_CACHE:
            _DATASET_CACHE.move_to_end(key)
            return _DATASET_CACHE[key]

    # local_ctx_sizes must contain values <= ctx_cap; we pass exactly one.
    local_ctx_sizes = [min(local_ctx_size, ctx_cap)]
    bfs_widths = [bfs_width]

    ds = RustlerDataset(
        tasks=[
            Task(
                db_name=db_name,
                table_name=table_name,
                target_column=target_column,
                # unused: context sampling never consults the task type.
                task_type="clf",
                split=split,
                leakage_columns=tuple(columns_to_drop),
            )
        ],
        pre_dir=pre_root,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_sizes=local_ctx_sizes,
        bfs_widths=bfs_widths,
        num_walks=num_walks,
        walk_length=walk_length,
        prefer_latest=[prefer_latest],
        mask_prob_max=0.0,  # mutated per-request via set_mask_prob_max_py
        embedding_model=embedding_model,
        d_text=d_text,
        shuffle_seed=shuffle_seed,
        context_seed=0,  # context_seed is folded into step at request time
        items_per_task=items_per_task,
        quiet=True,
        bool_as_num=bool_as_num,
        ignore_data_errors=False,
        skip_text_cols=skip_text_cols,
        mmap_populate=False,
        balance_labels=[balance_labels],
        timeout_per_item=3600.0,
        ablate_schema_semantics=ablate_schema_semantics,
        vector_db_path=None,
        train_only_fallback=False,
    )

    with _DATASET_CACHE_LOCK:
        _DATASET_CACHE[key] = ds
        _DATASET_CACHE.move_to_end(key)
        while len(_DATASET_CACHE) > _DATASET_CACHE_MAX:
            _DATASET_CACHE.popitem(last=False)
    return ds


# ---------------------------------------------------------------------------
# Context construction.
# ---------------------------------------------------------------------------


def _decode_value(sem_type: str, t: dict, text_vocab: list[str]) -> object:
    """Pull the human-readable value out of a token, in normalized form.

    The rustler-side number/datetime/boolean values are z-score normalized
    per column at preprocess time, so we display the raw normalized value;
    text cells we resolve back to the original string via text.json.
    """
    if sem_type == "text":
        idx = t["class_value_idx"]
        if 0 <= idx < len(text_vocab):
            return text_vocab[idx]
        return None
    if sem_type == "number":
        v = float(t["number_value"])
        return None if np.isnan(v) else v
    if sem_type == "datetime":
        v = float(t["datetime_value"])
        return None if np.isnan(v) else v
    if sem_type == "boolean":
        v = float(t["boolean_value"])
        return None if np.isnan(v) else v
    return None


def _format_value(sem_type: str, val: object) -> str:
    if val is None:
        return "—"
    if sem_type == "number":
        return f"{val:+.4g}"
    if sem_type == "boolean":
        return f"{val:+.4g}"
    if sem_type == "datetime":
        return f"{val:+.4g}"
    if sem_type == "text":
        s = str(val)
        return s if len(s) <= 80 else s[:77] + "…"
    return str(val)


def _build_context_payload(
    *,
    ds: RustlerDataset,
    db_name: str,
    db_dir: str,
    text_vocab: list[str],
    table_info: dict,
    column_index: dict,
    item_idx: int | None,
    bs: int,
    ctx_size: int,
    context_seed: int,
    mask_prob_max: float,
    target_column: str,
    target_column_idx: int,
    table_name: str,
    split: str,
) -> dict:
    """Generate one item's worth of context and return a JSON-serializable payload."""
    # Mutate seed/mask in place. The sampler's context_seed is folded
    # together with `step` (in eval-mode it's just used; in train-mode
    # it's combined with step). We treat eval-mode (item_idx given) as
    # canonical here, and use train-mode only when item_idx is None.
    ds.sampler.set_mask_prob_max_py(mask_prob_max)

    # For deterministic stepping, set the step explicitly when in train mode.
    if item_idx is None:
        ds.sampler.set_step_py(int(context_seed))
        ds.sampler.set_stride_py(0)
    # Note: eval-mode (item_idx given) doesn't use step/stride, but it
    # uses the sampler's `context_seed`. We can't trivially mutate that
    # without rebuilding, so eval-mode reproduces using a fixed seed
    # baked into the dataset. The "context_seed" knob primarily affects
    # train-mode here (item_idx == None).

    tup = ds.sampler.batch_py(item_idx, bs, ctx_size)
    batch = ds._process_batch(tup)

    # Pull row 0 of the batch.
    seq_len = ctx_size
    fields = {
        "node_idxs": batch["node_idxs"][0].tolist(),
        "is_padding": batch["is_padding"][0].tolist(),
        "is_task_nodes": batch["is_task_nodes"][0].tolist(),
        "is_targets": batch["is_targets"][0].tolist(),
        "sem_types": batch["sem_types"][0].tolist(),
        "col_name_idxs": batch["col_name_idxs"][0].tolist(),
        "table_name_idxs": batch["table_name_idxs"][0].tolist(),
        "class_value_idxs": batch["class_value_idxs"][0].tolist(),
        "timestamps": batch["timestamps"][0].tolist(),
        "f2p_nbr_idxs": batch["f2p_nbr_idxs"][0].tolist(),
        "number_values": batch["number_values"][0].squeeze(-1).float().tolist(),
        "datetime_values": batch["datetime_values"][0].squeeze(-1).float().tolist(),
        "boolean_values": batch["boolean_values"][0].squeeze(-1).float().tolist(),
        "seed_node_idxs": batch["seed_node_idxs"][0].tolist(),
        "bfs_depths": batch["bfs_depths"][0].tolist(),
        "batch_mask": batch["batch_mask"].tolist(),
    }

    # Build per-cell records.
    # Pre-compute table-name index → table-base-name lookup for legend coloring.
    table_offsets = sorted(
        (
            (info["node_idx_offset"], int(info["num_nodes"]), key)
            for key, info in table_info.items()
        ),
        key=lambda kv: kv[0],
    )

    def lookup_table_for_node(node_idx: int) -> tuple[str, str, bool]:
        """Return (display_name, full_key, is_task_table) for a node index.

        full_key is e.g. 'races:Db' or 'driver-position:Train'; display
        name is just 'races' / 'driver-position'.
        """
        prev_key = None
        for off, num, key in table_offsets:
            if node_idx >= off:
                prev_key = key
            else:
                break
        if prev_key is None:
            return "<unknown>", "", False
        base, _, ttype = prev_key.partition(":")
        return base, prev_key, (ttype != "Db")

    tokens = []
    real_token_count = 0
    primary_target_idx: int | None = None
    primary_target_token = None
    masked_feature_count = 0

    # f2p_nbr_idxs[0] is shape (S, MAX_F2P_NBRS) after `.tolist()` →
    # already a list of lists.
    f2p_arr = fields["f2p_nbr_idxs"]

    for i in range(seq_len):
        is_pad = bool(fields["is_padding"][i])
        node_idx = int(fields["node_idxs"][i])
        sem_type = SEM_TYPE_NAMES[int(fields["sem_types"][i])] if not is_pad else None
        col_name_idx = int(fields["col_name_idxs"][i])
        col_name = (
            text_vocab[col_name_idx]
            if 0 <= col_name_idx < len(text_vocab)
            else f"<idx:{col_name_idx}>"
        )
        # Strip the " of <table>" suffix for terse display.
        col_short = col_name.split(" of ", 1)[0] if " of " in col_name else col_name
        tbl_name_idx = int(fields["table_name_idxs"][i])
        tbl_text = (
            text_vocab[tbl_name_idx]
            if 0 <= tbl_name_idx < len(text_vocab)
            else f"<idx:{tbl_name_idx}>"
        )

        table_display, table_full, is_task_table = (
            lookup_table_for_node(node_idx) if not is_pad else ("", "", False)
        )

        f2p_list = [int(x) for x in f2p_arr[i] if int(x) != -1]

        ts = int(fields["timestamps"][i])
        ts_display = None if ts == INT_MIN else ts

        is_target = bool(fields["is_targets"][i])
        # Primary target = the cell whose node + column is the actual task
        # target (always emitted at position 0 by the rustler sampler). Any
        # other is_target flags came from `mask_prob` and are "masked
        # features": predicted alongside the primary target during training.
        is_primary = (
            is_target
            and not is_pad
            and col_name_idx == target_column_idx
            and (primary_target_idx is None)
        )

        seed_idx = int(fields["seed_node_idxs"][i])
        bfs_depth = int(fields["bfs_depths"][i])
        token: dict = {
            "i": i,
            "node_idx": node_idx,
            "is_padding": is_pad,
            "is_task_node": bool(fields["is_task_nodes"][i]),
            "is_target": is_target,
            "is_primary_target": is_primary,
            "is_masked_feature": is_target and not is_primary and not is_pad,
            "sem_type": sem_type,
            "col_name": col_name,
            "col_short": col_short,
            "col_name_idx": col_name_idx,
            "table_name_idx": tbl_name_idx,
            "table_text": tbl_text,
            "table_display": table_display,
            "table_full_key": table_full,
            "is_task_table": is_task_table,
            "class_value_idx": int(fields["class_value_idxs"][i]),
            "number_value": float(fields["number_values"][i]),
            "datetime_value": float(fields["datetime_values"][i]),
            "boolean_value": float(fields["boolean_values"][i]),
            "f2p_nbr_idxs": f2p_list,
            "timestamp": ts_display,
            "seed_node_idx": seed_idx if seed_idx != -1 else None,
            "bfs_depth": bfs_depth if bfs_depth != -1 else None,
        }
        if not is_pad:
            real_token_count += 1
            decoded = _decode_value(sem_type, token, text_vocab)
            token["value"] = decoded
            token["value_str"] = _format_value(sem_type, decoded)
            if is_primary:
                primary_target_idx = i
                primary_target_token = token
            elif token["is_masked_feature"]:
                masked_feature_count += 1
        else:
            token["value"] = None
            token["value_str"] = ""
        tokens.append(token)

    # Group cells by node for the row-view.
    by_node: "OrderedDict[int, list[int]]" = OrderedDict()
    for t in tokens:
        if t["is_padding"]:
            continue
        by_node.setdefault(t["node_idx"], []).append(t["i"])

    nodes_meta = []
    for node_idx, idxs in by_node.items():
        first = tokens[idxs[0]]
        # All cells of a node share the same seed/depth (added together
        # in one inner loop per BFS hit), so the first cell's values are
        # canonical for the whole node.
        nodes_meta.append(
            {
                "node_idx": node_idx,
                "table_display": first["table_display"],
                "table_full_key": first["table_full_key"],
                "is_task_table": first["is_task_table"],
                "is_target_node": any(tokens[k]["is_target"] for k in idxs),
                "is_task_node": any(tokens[k]["is_task_node"] for k in idxs),
                "is_primary_target_node": any(
                    tokens[k]["is_primary_target"] for k in idxs
                ),
                "is_seed": first["seed_node_idx"] == node_idx,
                "seed_node_idx": first["seed_node_idx"],
                "bfs_depth": first["bfs_depth"],
                "timestamp": first["timestamp"],
                "cell_idxs": idxs,
            }
        )

    # Per-seed aggregate. Used by the radial-shell graph layout: the target
    # seed sits at center, other seeds sit on a ring around it, and each
    # seed's BFS expansion sits in concentric shells around that seed.
    by_seed: "OrderedDict[int, list[int]]" = OrderedDict()
    for n in nodes_meta:
        s = n["seed_node_idx"]
        if s is None:
            continue
        by_seed.setdefault(s, []).append(n["node_idx"])
    primary_target_node_idx = (
        primary_target_token["node_idx"] if primary_target_token else None
    )
    seeds_meta: list[dict] = []
    for seed_idx, member_node_idxs in by_seed.items():
        seed_node_meta = next(
            (n for n in nodes_meta if n["node_idx"] == seed_idx), None
        )
        depths = [
            n["bfs_depth"]
            for n in nodes_meta
            if n["seed_node_idx"] == seed_idx and n["bfs_depth"] is not None
        ]
        seeds_meta.append(
            {
                "seed_node_idx": seed_idx,
                "is_target_seed": seed_idx == primary_target_node_idx,
                "table_display": (
                    seed_node_meta["table_display"] if seed_node_meta else None
                ),
                "is_task_table": (
                    seed_node_meta["is_task_table"] if seed_node_meta else False
                ),
                "max_depth": max(depths) if depths else 0,
                "node_count": len(member_node_idxs),
                "member_node_idxs": member_node_idxs,
            }
        )
    seeds_meta.sort(key=lambda s: (not s["is_target_seed"], -s["node_count"]))

    # Group cells by column key (table+col) for the column-view.
    by_col: "OrderedDict[tuple[str, str], list[int]]" = OrderedDict()
    for t in tokens:
        if t["is_padding"]:
            continue
        key = (t["table_display"], t["col_short"])
        by_col.setdefault(key, []).append(t["i"])
    cols_meta = []
    for (tbl, col), idxs in by_col.items():
        sem_types = sorted(
            {tokens[k]["sem_type"] for k in idxs if tokens[k]["sem_type"]}
        )
        cols_meta.append(
            {
                "table_display": tbl,
                "col_short": col,
                "sem_types": sem_types,
                "cell_idxs": idxs,
            }
        )

    # Build the graph payload (for the network view).
    node_set = {n["node_idx"] for n in nodes_meta}
    edges = []
    seen_edges: set[tuple[int, int]] = set()
    for t in tokens:
        if t["is_padding"]:
            continue
        for nbr in t["f2p_nbr_idxs"]:
            if nbr in node_set and nbr != t["node_idx"]:
                e = (t["node_idx"], nbr)
                if e not in seen_edges:
                    seen_edges.add(e)
                    edges.append({"source": t["node_idx"], "target": nbr})

    # Per-table summary (for legend coloring).
    table_summary: dict[str, dict] = {}
    for n in nodes_meta:
        s = table_summary.setdefault(
            n["table_display"],
            {
                "table_display": n["table_display"],
                "is_task": n["is_task_table"],
                "node_count": 0,
                "cell_count": 0,
            },
        )
        s["node_count"] += 1
        s["cell_count"] += len(n["cell_idxs"])

    # Sem-type counts for stats.
    sem_counts = {st: 0 for st in SEM_TYPE_NAMES}
    target_count = 0
    task_token_count = 0
    for t in tokens:
        if t["is_padding"]:
            continue
        if t["sem_type"]:
            sem_counts[t["sem_type"]] += 1
        if t["is_target"]:
            target_count += 1
        if t["is_task_node"]:
            task_token_count += 1

    payload = {
        "request": {
            "db_name": db_name,
            "table_name": table_name,
            "target_column": target_column,
            "target_column_idx": target_column_idx,
            "split": split,
            "ctx_size": ctx_size,
            "item_idx": item_idx,
            "context_seed": context_seed,
            "mask_prob_max": mask_prob_max,
        },
        "num_items": ds.sampler.num_items,
        "tokens": tokens,
        "nodes_meta": nodes_meta,
        "cols_meta": cols_meta,
        "seeds_meta": seeds_meta,
        "table_summary": list(table_summary.values()),
        "graph": {
            "node_ids": list(node_set),
            "edges": edges,
        },
        "stats": {
            "seq_len": seq_len,
            "real_tokens": real_token_count,
            "padding_tokens": seq_len - real_token_count,
            "padding_pct": round((seq_len - real_token_count) / seq_len * 100, 2),
            "num_nodes": len(nodes_meta),
            "num_columns": len(cols_meta),
            "num_tables": len(table_summary),
            "num_edges": len(edges),
            "num_seeds": len(seeds_meta),
            "sem_counts": sem_counts,
            "target_count": target_count,
            "task_token_count": task_token_count,
            "masked_feature_count": masked_feature_count,
        },
        "target_token_idx": primary_target_idx,
        "target_token": primary_target_token,
        "max_f2p_nbrs": MAX_F2P_NBRS,
    }
    return payload


# ---------------------------------------------------------------------------
# Helpers for browsing what's available on disk.
# ---------------------------------------------------------------------------


def _list_pre_roots(top_root: Path) -> list[str]:
    """Return immediate sub-dirs of `top_root` that look like 'pre roots'.

    A pre-root is a directory containing one or more dataset directories,
    each with a `nodes.rkyv` file. We also include `top_root` itself if it
    has an immediate dataset (e.g., when the pre root *is* a dataset dir).
    """
    roots: list[str] = []
    if not top_root.exists():
        return roots
    # Always offer top_root itself.
    if any(
        (top_root / d / "nodes.rkyv").exists()
        for d in _safe_iterdir(top_root)
        if (top_root / d).is_dir()
    ):
        roots.append(str(top_root))
    for d in _safe_iterdir(top_root):
        sub = top_root / d
        if not sub.is_dir():
            continue
        if (sub / "nodes.rkyv").exists():
            continue  # this is a dataset dir, not a root
        # Check whether sub has dataset children.
        for c in _safe_iterdir(sub):
            if (sub / c / "nodes.rkyv").exists():
                roots.append(str(sub))
                break
    return sorted(set(roots))


def _safe_iterdir(p: Path) -> list[str]:
    try:
        return [c.name for c in p.iterdir()]
    except (FileNotFoundError, PermissionError):
        return []


def _list_dbs_in_root(pre_root: Path) -> list[str]:
    out: list[str] = []
    if not pre_root.exists():
        return out
    for child in sorted(_safe_iterdir(pre_root)):
        full = pre_root / child
        if not full.is_dir():
            continue
        if (full / "nodes.rkyv").exists():
            out.append(child)
        else:
            for grand in sorted(_safe_iterdir(full)):
                gfull = full / grand
                if (gfull / "nodes.rkyv").exists():
                    out.append(f"{child}/{grand}")
    return out


def _root_is_hf(root_arg: str) -> bool:
    """A root spec is HuggingFace-hosted if it is not an existing local dir."""
    return not Path(root_arg).expanduser().exists()


def _list_dbs(root_arg: str) -> list[str]:
    """Databases under a root, whether a local directory or a HuggingFace repo."""
    if not _root_is_hf(root_arg):
        return _list_dbs_in_root(Path(root_arg).expanduser())
    from huggingface_hub import HfApi

    repo_id, subdir = resolve_repo(root_arg)
    prefix = f"{subdir}/" if subdir else ""
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        {
            f[len(prefix) :].split("/", 1)[0]
            for f in files
            if f.startswith(prefix) and f.endswith("/nodes.rkyv")
        }
    )


def _resolve_db_dir(
    root_arg: str,
    db: str,
    *,
    metadata_only: bool,
    embedding_model: str = "all-MiniLM-L12-v2",
) -> Path:
    """Return a LOCAL directory for `db`, downloading from HF on demand.

    For HF roots, `metadata_only=True` fetches just the schema files (fast
    browsing); `False` fetches the node blobs + embeddings needed to build
    contexts. Local roots are used in place.
    """
    if not _root_is_hf(root_arg):
        return Path(root_arg).expanduser() / db
    local_root = resolve_pre_dir(
        root_arg,
        [db],
        embedding_model,
        include_text=not metadata_only,
        metadata_only=metadata_only,
    )
    return Path(local_root) / db


def _list_tables(db_dir: Path, split: str) -> list[dict]:
    """Tables in this DB available for the given split (or 'auto' for any)."""
    info = _load_table_info(str(db_dir))
    by_base: dict[str, dict[str, dict]] = {}
    for full_key, meta in info.items():
        base, _, ttype = full_key.partition(":")
        by_base.setdefault(base, {})[ttype] = meta
    out = []
    for base, ttypes in sorted(by_base.items()):
        item = {
            "name": base,
            "splits": sorted(ttypes.keys()),
            "is_task_table": any(t != "Db" for t in ttypes.keys()),
            "num_nodes_by_split": {
                t: int(meta["num_nodes"]) for t, meta in ttypes.items()
            },
        }
        out.append(item)
    return out


def _columns_for_table(db_dir: Path, table_name: str) -> list[str]:
    """All columns belonging to `table_name`, parsed from column_index.json."""
    ci = _load_column_index(str(db_dir))
    out = []
    for key, idx in ci.items():
        col, _, tbl = key.partition(" of ")
        if tbl == table_name:
            out.append({"name": col, "idx": int(idx)})
    return sorted(out, key=lambda d: d["name"])


# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------


class CtxVizServer(ThreadingHTTPServer):
    # Re-bind without TIME_WAIT delay after a previous run on the same port
    # exits — the common cause of EADDRINUSE on quick restart cycles.
    allow_reuse_address = True


class CtxVizHandler(BaseHTTPRequestHandler):
    server_version = "ctx-viz/0.1"

    # Suppress per-request stderr logging (override of default).
    def log_message(self, format, *args):  # noqa: A002
        if self.server.quiet:
            return
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    # ---- routing ----

    def do_GET(self):  # noqa: N802
        try:
            url = urlparse(self.path)
            if url.path == "/":
                self._send_html(INDEX_HTML)
            elif url.path == "/api/list_pre_roots":
                root_arg = str(self.server.root_arg)
                if _root_is_hf(root_arg):
                    self._send_json({"default_root": root_arg, "pre_roots": [root_arg]})
                else:
                    pre_root = Path(root_arg).expanduser()
                    self._send_json(
                        {
                            "default_root": str(pre_root),
                            "pre_roots": _list_pre_roots(pre_root.parent)
                            + ([str(pre_root)] if pre_root.exists() else []),
                        }
                    )
            elif url.path == "/api/list_dbs":
                qs = parse_qs(url.query)
                root = qs.get("root", [str(self.server.root_arg)])[0]
                self._send_json({"dbs": _list_dbs(root)})
            elif url.path == "/api/list_tables":
                qs = parse_qs(url.query)
                root = qs.get("root", [str(self.server.root_arg)])[0]
                db = qs["db"][0]
                db_dir = _resolve_db_dir(root, db, metadata_only=True)
                tables = _list_tables(db_dir, split=qs.get("split", ["auto"])[0])
                self._send_json({"tables": tables})
            elif url.path == "/api/list_columns":
                qs = parse_qs(url.query)
                root = qs.get("root", [str(self.server.root_arg)])[0]
                db = qs["db"][0]
                tbl = qs["table"][0]
                db_dir = _resolve_db_dir(root, db, metadata_only=True)
                self._send_json({"columns": _columns_for_table(db_dir, tbl)})
            elif url.path == "/api/health":
                self._send_json({"ok": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown path: {url.path}")
        except BaseException as e:  # incl. pyo3 PanicException
            self._send_error(e)

    def do_POST(self):  # noqa: N802
        try:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body or b"{}")
            if url.path == "/api/build":
                payload = self._handle_build(req)
                self._send_json(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown path: {url.path}")
        except BaseException as e:  # incl. pyo3 PanicException
            self._send_error(e)

    # ---- handlers ----

    def _handle_build(self, req: dict) -> dict:
        root_arg = req.get("pre_root") or str(self.server.root_arg)
        db_name = req["db_name"]
        table_name = req["table_name"]
        target_column = req["target_column"]
        split = req.get("split", "val")
        columns_to_drop = tuple(req.get("columns_to_drop") or [])
        ctx_size = int(req.get("ctx_size", 256))
        local_ctx_size = int(req.get("local_ctx_size", ctx_size))
        bfs_width = int(req.get("bfs_width", 128))
        num_walks = int(req.get("num_walks", 0))
        walk_length = int(req.get("walk_length", 4))
        prefer_latest = bool(req.get("prefer_latest", False))
        embedding_model = req.get("embedding_model", "all-MiniLM-L12-v2")
        d_text = int(req.get("d_text", 384))
        items_per_task = int(req.get("items_per_task", -1))
        bool_as_num = bool(req.get("bool_as_num", False))
        skip_text_cols = bool(req.get("skip_text_cols", False))
        balance_labels = bool(req.get("balance_labels", False))
        ablate_schema_semantics = bool(req.get("ablate_schema_semantics", False))
        shuffle_seed = int(req.get("shuffle_seed", 0))
        context_seed = int(req.get("context_seed", 0))
        mask_prob_max = float(req.get("mask_prob_max", 0.0))
        item_idx = req.get("item_idx", None)
        if item_idx is not None and item_idx != "":
            item_idx = int(item_idx)
        else:
            item_idx = None

        db_dir = _resolve_db_dir(
            root_arg, db_name, metadata_only=False, embedding_model=embedding_model
        )
        if not db_dir.exists():
            raise FileNotFoundError(f"db dir does not exist: {db_dir}")
        pre_root = db_dir.parent

        ds = _get_or_build_dataset(
            pre_root=str(pre_root),
            db_name=db_name,
            table_name=table_name,
            target_column=target_column,
            split=split,
            columns_to_drop=columns_to_drop,
            ctx_cap=ctx_size,
            local_ctx_size=local_ctx_size,
            bfs_width=bfs_width,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest=prefer_latest,
            embedding_model=embedding_model,
            d_text=d_text,
            items_per_task=items_per_task,
            bool_as_num=bool_as_num,
            skip_text_cols=skip_text_cols,
            balance_labels=balance_labels,
            ablate_schema_semantics=ablate_schema_semantics,
            shuffle_seed=shuffle_seed,
        )

        target_column_idx = get_column_index(
            target_column, table_name, db_name, str(pre_root)
        )

        return _build_context_payload(
            ds=ds,
            db_name=db_name,
            db_dir=str(db_dir),
            text_vocab=_load_text_json(str(db_dir)),
            table_info=_load_table_info(str(db_dir)),
            column_index=_load_column_index(str(db_dir)),
            item_idx=item_idx,
            bs=1,
            ctx_size=ctx_size,
            context_seed=context_seed,
            mask_prob_max=mask_prob_max,
            target_column=target_column,
            target_column_idx=target_column_idx,
            table_name=table_name,
            split=split,
        )

    # ---- response helpers ----

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj) -> None:
        body = json.dumps(obj, default=_json_default, allow_nan=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, exc: BaseException) -> None:
        traceback.print_exc()
        body = json.dumps(
            {
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ).encode("utf-8")
        self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _json_default(o):
    """Coerce numpy scalars and convert non-finite floats to None."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _bind_with_fallback(host: str, port: int, max_tries: int) -> CtxVizServer:
    """Try `port`, then fall through to free ports if it's taken.

    On a shared box another user can hold the requested port; rather than
    erroring out, walk forward up to `max_tries` ports and finally fall
    back to OS-assigned (port=0). The actual port is read off the bound
    socket, so the printed URL reflects what was used.
    """
    last_err: OSError | None = None
    for offset in range(max_tries):
        try:
            return CtxVizServer((host, port + offset), CtxVizHandler)
        except OSError as e:
            if e.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            last_err = e
            continue
    # Final fallback: let the OS pick.
    try:
        return CtxVizServer((host, 0), CtxVizHandler)
    except OSError as e:
        raise (last_err or e)


@dataclass
class Config:
    """Context visualization server."""

    host: str
    """Interface to bind. 0.0.0.0 binds all interfaces so a browser on a
    laptop can hit a remote workstation directly; pass 127.0.0.1 for
    local-only."""
    port: int
    """Port to bind."""
    pre_root: str
    """Directory containing your pre-processed datasets."""
    quiet: bool
    """Suppress per-request HTTP logs."""
    port_fallback: bool
    """Fail loudly if --port is unavailable instead of trying nearby ports."""


def main(cfg: Config):
    pre_root = Path(cfg.pre_root).expanduser()
    if not cfg.port_fallback:
        server = CtxVizServer((cfg.host, cfg.port), CtxVizHandler)
    else:
        server = _bind_with_fallback(cfg.host, cfg.port, max_tries=20)
    server.pre_root = pre_root  # type: ignore[attr-defined]
    server.root_arg = cfg.pre_root  # raw spec: local dir or HF repo
    server.quiet = cfg.quiet  # type: ignore[attr-defined]

    actual_port = server.server_address[1]
    if actual_port != cfg.port:
        print(
            f"\n  \033[33m[note]\033[0m port {cfg.port} was taken; using {actual_port}",
            flush=True,
        )
    print(f"\n  ctx-viz running on {cfg.host}:{actual_port}", flush=True)
    if cfg.host in ("0.0.0.0", "::"):
        fqdn = socket.getfqdn()
        print(
            f"    local:   \033[1;36mhttp://127.0.0.1:{actual_port}\033[0m",
            flush=True,
        )
        print(
            f"    network: \033[1;36mhttp://{fqdn}:{actual_port}\033[0m",
            flush=True,
        )
    else:
        print(
            f"    open:    \033[1;36mhttp://{cfg.host}:{actual_port}\033[0m",
            flush=True,
        )
    print(f"  pre-root: {pre_root}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


# ---------------------------------------------------------------------------
# Frontend (single-page HTML, embedded for zero-deps deploy).
# ---------------------------------------------------------------------------

INDEX_HTML = (Path(__file__).parent / "index.html").read_text()
if __name__ == "__main__":
    main()
