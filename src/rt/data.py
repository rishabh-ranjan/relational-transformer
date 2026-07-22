"""Preprocessed-data access: pre_dir resolution (local or Hub), datasets, and
task enumeration.

* pre_dir resolution -- every ``pre_dir`` in rt accepts a local path or a Hub
  repo ``org/repo[/subdir]``; local wins, Hub files download into the HF cache
  on demand.
* datasets -- rustler-backed torch datasets for training and evaluation.
* tasks -- forecast tasks (explicit time-split task tables) and autocomplete
  tasks (schema-derived masked-column prediction), enumerated straight from the
  preprocessed data.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import ml_dtypes  # noqa: F401  # registers bfloat16 numpy dtype for rustler
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from rt.rustler import Sampler



# Files rustler's Sampler and rt.data read for each preprocessed dataset.
# `text.json` (the raw string vocabulary) is fetched only on demand -- it is not
# needed for training, only for tooling that resolves cells back to their
# original strings.
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
    embedding_model: str,
    *,
    include_text: bool = False,
    metadata_only: bool = False,
    revision: str | None = None,
) -> str:
    """Return a local root directory containing ``<db>/`` subfolders for each db.

    If ``pre_dir`` is an existing local path it is returned as-is. Otherwise it is
    treated as a Hub ``org/repo[/subdir]`` and only the files needed for
    ``db_names`` (+ the chosen ``embedding_model``) are downloaded and cached.
    ``metadata_only`` fetches just the small schema files (no node blobs or
    embeddings) -- enough to browse tables/columns.
    """
    p = Path(pre_dir).expanduser()
    if p.exists():
        return str(p)

    from huggingface_hub import snapshot_download

    repo_id, subdir = resolve_repo(pre_dir)
    prefix = f"{subdir}/" if subdir else ""
    file_set = METADATA_FILES if metadata_only else CORE_FILES
    patterns: list[str] = []
    for db in dict.fromkeys(db_names):  # dedup, preserve order
        base = f"{prefix}{db}"
        patterns += [f"{base}/{f}" for f in file_set]
        if not metadata_only:
            patterns.append(f"{base}/text_emb_{embedding_model}.bin")
        if include_text:
            patterns.append(f"{base}/text.json")

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
    )
    return str(Path(local) / subdir) if subdir else str(local)


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
    """Read one preprocessed dataset's ``meta.json`` (local or downloaded from Hub)."""
    p = Path(pre_dir).expanduser()
    if p.exists():
        return json.loads((p / db / "meta.json").read_text())

    from huggingface_hub import hf_hub_download

    repo_id, subdir = resolve_repo(pre_dir)
    filename = f"{subdir}/{db}/meta.json" if subdir else f"{db}/meta.json"
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision
    )
    return json.loads(Path(path).read_text())


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

MAX_F2P_NBRS = 5  # See fly.rs L32


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


SEM_TYPE_NUMBER = 0
SEM_TYPE_BOOLEAN = 3


def process_batch(tup, d_text, bool_as_num):
    out = dict(tup)
    seq_len = out.pop("seq_len")

    for k, v in out.items():
        if k in [
            "number_values",
            "datetime_values",
            "text_values",
            "col_name_values",
            "boolean_values",
        ]:
            out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
        else:
            out[k] = torch.from_numpy(v)

    out["node_idxs"] = out["node_idxs"].view(-1, seq_len)
    out["sem_types"] = out["sem_types"].view(-1, seq_len)
    out["is_targets"] = out["is_targets"].view(-1, seq_len)
    out["is_task_nodes"] = out["is_task_nodes"].view(-1, seq_len)
    out["is_padding"] = out["is_padding"].view(-1, seq_len)
    out["table_name_idxs"] = out["table_name_idxs"].view(-1, seq_len)
    out["col_name_idxs"] = out["col_name_idxs"].view(-1, seq_len)
    out["class_value_idxs"] = out["class_value_idxs"].view(-1, seq_len)
    out["timestamps"] = out["timestamps"].view(-1, seq_len)
    out["seed_node_idxs"] = out["seed_node_idxs"].view(-1, seq_len)
    out["bfs_depths"] = out["bfs_depths"].view(-1, seq_len)

    out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, seq_len, MAX_F2P_NBRS)
    out["number_values"] = out["number_values"].view(-1, seq_len, 1)
    out["datetime_values"] = out["datetime_values"].view(-1, seq_len, 1)
    out["boolean_values"] = out["boolean_values"].view(-1, seq_len, 1).bfloat16()
    out["text_values"] = out["text_values"].view(-1, seq_len, d_text)
    out["col_name_values"] = out["col_name_values"].view(-1, seq_len, d_text)

    if bool_as_num:
        bool_mask = out["sem_types"] == SEM_TYPE_BOOLEAN
        out["number_values"][bool_mask] = out["boolean_values"][bool_mask]
        out["boolean_values"][bool_mask] = 0
        out["sem_types"][bool_mask] = SEM_TYPE_NUMBER

    return out


class RustlerDataset:
    def __init__(
        self,
        tasks,
        pre_dir: str,
        global_rank,
        local_rank,
        world_size,
        local_ctx_sizes: list[int],
        bfs_widths: list[int],
        num_walks,
        walk_length,
        prefer_latest: list[bool],
        mask_prob_max,
        embedding_model,
        d_text,
        shuffle_seed,
        context_seed,
        items_per_task,
        quiet,
        bool_as_num,
        ignore_data_errors,
        skip_text_cols,
        mmap_populate,
        balance_labels: list[bool],
        timeout_per_item,
        ablate_schema_semantics,
        vector_db_path: str | None,
        train_only_fallback: bool,
    ):
        # `pre_dir` may be a local path or a HuggingFace repo spec; resolve to a
        # local root, downloading only the files needed for these databases.
        pre_dir = resolve_pre_dir(pre_dir, [t.db_name for t in tasks], embedding_model)
        if vector_db_path is not None:
            vector_db_path = str(Path(vector_db_path).expanduser())

        dataset_tuples = []
        target_column_indices = []
        drop_column_indices = []
        skipped_tasks = []

        for task in tasks:
            db_name = task.db_name
            table_name = task.table_name
            target_column = task.target_column
            split = task.split
            columns_to_drop = task.leakage_columns
            try:
                if split == "train":
                    split = "Train"
                elif split == "val":
                    split = "Val"
                elif split == "test":
                    split = "Test"

                table_info_path = f"{pre_dir}/{db_name}/table_info.json"
                with open(table_info_path) as f:
                    table_info = json.load(f)

                table_info_key = (
                    f"{table_name}:Db"
                    if f"{table_name}:Db" in table_info
                    else f"{table_name}:{split}"
                )
                info = table_info[table_info_key]
                node_idx_offset = info["node_idx_offset"]
                num_nodes = info["num_nodes"]

                target_idx = get_column_index(
                    target_column, table_name, db_name, pre_dir
                )
                target_column_indices.append(target_idx)

                drop_indices = []
                for col in columns_to_drop:
                    if col == target_column:
                        continue
                    try:
                        drop_indices.append(
                            get_column_index(col, table_name, db_name, pre_dir)
                        )
                    except ValueError:
                        pass  # skip_col not in task parquet; ignore
                drop_column_indices.append(drop_indices)

                dataset_tuples.append((db_name, table_name, node_idx_offset, num_nodes))
            except Exception as e:
                if not ignore_data_errors:
                    raise
                task_name = f"{db_name}/{table_name}/{target_column}"
                skipped_tasks.append((task_name, e))

        if skipped_tasks and local_rank == 0 and not quiet:
            print(
                f"\033[31mskipped {len(skipped_tasks)} task(s)\033[0m",
                flush=True,
            )
            for task_name, e in skipped_tasks:
                print(f"  \033[31mskipped {task_name}: {e}\033[0m", flush=True)

        self.world_size = world_size
        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            local_ctx_sizes=local_ctx_sizes,
            bfs_widths=bfs_widths,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest=prefer_latest,
            mask_prob_max=mask_prob_max,
            embedding_model=embedding_model,
            pre_dir=pre_dir,
            d_text=d_text,
            shuffle_seed=shuffle_seed,
            context_seed=context_seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
            items_per_task=items_per_task,
            quiet=quiet,
            ignore_data_errors=ignore_data_errors,
            num_prev_skipped=len(skipped_tasks),
            skip_text_cols=skip_text_cols,
            mmap_populate=mmap_populate,
            balance_labels=balance_labels,
            timeout_per_item=timeout_per_item,
            ablate_schema_semantics=ablate_schema_semantics,
            vector_db_path=vector_db_path,
            train_only_fallback=train_only_fallback,
        )
        self.num_items = self.sampler.num_items

        self.d_text = d_text
        self.bool_as_num = bool_as_num

    def _process_batch(self, tup):
        return process_batch(tup, self.d_text, self.bool_as_num)


class TrainDataset(RustlerDataset, IterableDataset):
    def __init__(
        self,
        tasks,
        pre_dir: str,
        train_ctx_sizes,
        train_tokens_per_gpu,
        total_bs,
        global_rank,
        local_rank,
        world_size,
        local_ctx_sizes: list[int],
        bfs_widths: list[int],
        num_walks,
        walk_length,
        prefer_latest: list[bool],
        mask_prob_max,
        embedding_model,
        d_text,
        seed,
        items_per_task,
        mask_prob_max_shared,
        bool_as_num,
        skip_text_cols,
        mmap_populate,
        balance_labels: list[bool],
        timeout_per_item,
        ablate_schema_semantics,
        vector_db_path: str | None,
        train_only_fallback: bool,
    ):
        # TrainDataset drives both shuffle and context construction from the
        # same seed — this matches prior single-seed behavior.
        RustlerDataset.__init__(
            self,
            tasks=tasks,
            pre_dir=pre_dir,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            local_ctx_sizes=local_ctx_sizes,
            bfs_widths=bfs_widths,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest=prefer_latest,
            mask_prob_max=mask_prob_max,
            embedding_model=embedding_model,
            d_text=d_text,
            shuffle_seed=seed,
            context_seed=seed,
            items_per_task=items_per_task,
            quiet=False,
            bool_as_num=bool_as_num,
            ignore_data_errors=True,
            skip_text_cols=skip_text_cols,
            mmap_populate=mmap_populate,
            balance_labels=balance_labels,
            timeout_per_item=timeout_per_item,
            ablate_schema_semantics=ablate_schema_semantics,
            vector_db_path=vector_db_path,
            train_only_fallback=train_only_fallback,
        )
        self.train_ctx_sizes = train_ctx_sizes
        self.seed = random.Random(seed).getrandbits(64)
        self.train_tokens_per_gpu = train_tokens_per_gpu
        self.total_bs = total_bs
        self.mask_prob_max_shared = mask_prob_max_shared
        # total_bs must split evenly into world_size * per_gpu_bs so the global
        # batch is exactly total_bs. Pick a GPU count that divides
        # total_bs / per_gpu_bs (the launcher does this); fail loudly otherwise.
        for c in train_ctx_sizes:
            train_bs = max(1, train_tokens_per_gpu // c)
            if total_bs < world_size * train_bs:
                assert total_bs % world_size == 0, (
                    f"total_bs={total_bs} not divisible by world_size={world_size}"
                    f" for ctx_size={c}"
                )
            else:
                assert total_bs % (world_size * train_bs) == 0, (
                    f"total_bs={total_bs} not divisible by world_size*bs_per_gpu="
                    f"{world_size * train_bs} for ctx_size={c}"
                )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.sampler.set_step_py(worker_info.id)
            self.sampler.set_stride_py(worker_info.num_workers)
            step = worker_info.id
            stride = worker_info.num_workers
        else:
            step = 0
            stride = 1
        # Single ctx_size: yield individual microbatches so workers prefetch
        # in parallel. List-yielding (multi-ctx case) blocks each worker for
        # grad_accum batches before yielding, which is unnecessary here since
        # all microbatches share the only ctx_size anyway.
        single_ctx = len(self.train_ctx_sizes) == 1
        while True:
            if self.mask_prob_max_shared is not None:
                self.sampler.set_mask_prob_max_py(self.mask_prob_max_shared.value)
            train_ctx_size = random.Random(self.seed + step).choice(
                self.train_ctx_sizes
            )
            train_bs = max(1, self.train_tokens_per_gpu // train_ctx_size)
            if self.total_bs < self.world_size * train_bs:
                train_bs = max(1, self.total_bs // self.world_size)
                grad_accum = 1
            else:
                grad_accum = self.total_bs // (self.world_size * train_bs)
            if single_ctx:
                tup = self.sampler.batch_py(None, train_bs, train_ctx_size)
                yield self._process_batch(tup)
            else:
                # Multi ctx_size: yield grad_accum batches atomically with
                # shared ctx_size to avoid worker-round-robin interleaving
                # ctx_sizes within an optimizer step.
                batches = []
                for _ in range(grad_accum):
                    tup = self.sampler.batch_py(None, train_bs, train_ctx_size)
                    batches.append(self._process_batch(tup))
                yield batches
            step += stride


class EvalDataset(Dataset):
    def __init__(
        self,
        rustler_dataset: RustlerDataset,
        eval_bs,
        eval_ctx_size,
    ):
        self.rustler_dataset = rustler_dataset
        self.eval_bs = eval_bs
        self.eval_ctx_size = eval_ctx_size

    def __len__(self):
        # Uniform across ranks: every rank iterates the same number of
        # batches. Higher-rank offsets on the last batch may legitimately
        # overshoot num_items; the rustler sampler fills those slots as
        # phantoms (batch_mask[i]=false) so the downstream fixed-size
        # gather is simple and correct.
        return math.ceil(
            self.rustler_dataset.num_items
            / (self.eval_bs * self.rustler_dataset.world_size)
        )

    def __getitem__(self, i):
        return self.rustler_dataset._process_batch(
            self.rustler_dataset.sampler.batch_py(i, self.eval_bs, self.eval_ctx_size)
        )


# relbench task_type -> RT task_type. Only node-level clf/reg tasks are modeled;
# link_prediction (recommendation) tasks are skipped.
_TASK_TYPE = {"binary_classification": "clf", "regression": "reg"}

# autocomplete: which sem-type becomes which task. Text/DateTime are not targets.
_SEM_TASK_TYPE = {"Boolean": "clf", "Number": "reg"}


@dataclass(frozen=True)
class Task:
    db_name: str
    table_name: str
    target_column: str
    task_type: str  # "clf" | "reg"
    split: str = ""  # "train" | "val" | "test"
    leakage_columns: tuple[str, ...] = ()


@cache
def _pkey_time_cols(pre_dir: str, db: str) -> frozenset[str]:
    """Primary-key / time columns to exclude as autocomplete targets, read from
    a ``manifest.yaml`` sitting next to the preprocessed db if one is present.
    Keyed as ``"<col> of <table>"`` to match :func:`rustler.column_sem_types`.

    This is a redundant safety net, not a hard dependency: relbench-3.0.0
    reindexes foreign *and* primary keys to row indices, and rustler ``pre``
    emits no cell for those, so :func:`rustler.column_sem_types` already excludes
    them; time columns surface as the ``DateTime`` sem-type, which is not a clf/
    reg target. We therefore never fetch over the network (that would hang the
    pretraining-task enumeration on compute nodes) -- only a local manifest is
    consulted, and its absence is fine.
    """
    from pathlib import Path

    p = Path(pre_dir).expanduser()
    manifest_path = p / db / "manifest.yaml"
    if not manifest_path.exists():
        return frozenset()
    try:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text())
        excl: set[str] = set()
        for table, info in (manifest.get("tables") or {}).items():
            for key in ("pkey", "time_col"):
                col = info.get(key)
                if col:
                    excl.add(f"{col} of {table}")
        return frozenset(excl)
    except Exception:
        return frozenset()


def _autocomplete_tasks(pre_dir: str, db: str, splits, task_types) -> list[Task]:
    """Schema-derived autocomplete tasks for one database.

    Emitted at the ``train`` split only (autocomplete is a pretraining-only
    signal; there is no held-out forecast horizon). The target column is masked
    and predicted from the rest of its row's context, so no leakage columns are
    needed.
    """
    if "train" not in splits:
        return []
    from rt.rustler import column_sem_types

    sem = column_sem_types(pre_dir, db)
    excluded = _pkey_time_cols(pre_dir, db)
    out: list[Task] = []
    for col_of_table, sem_type in sem.items():
        tt = _SEM_TASK_TYPE.get(sem_type)
        if tt is None or tt not in task_types or col_of_table in excluded:
            continue
        col, table = col_of_table.rsplit(" of ", 1)
        out.append(Task(db, table, col, tt, "train"))
    return out


def tasks_from_preprocessed(
    pre_dir: str,
    *,
    splits,
    task_types=("clf", "reg"),
    dbs=None,
) -> list[Task]:
    """Tasks across the preprocessed datasets under ``pre_dir`` for the given splits.

    A database with explicit forecast tasks (a non-empty ``meta.json`` ``tasks``
    list) contributes those; a database without any (DB-tables-only, the common
    ``the-join`` case) contributes schema-derived autocomplete tasks instead.
    """
    out: list[Task] = []
    for db in dbs if dbs is not None else list_datasets(pre_dir):
        meta = read_meta(pre_dir, db)
        explicit = [
            t
            for t in meta.get("tasks", [])
            if _TASK_TYPE.get(t.get("task_type")) and t.get("target_col")
        ]
        if explicit:
            for t in explicit:
                tt = _TASK_TYPE[t["task_type"]]
                if tt not in task_types:
                    continue
                for split in splits:
                    if split in t.get("splits", []):
                        out.append(Task(db, t["name"], t["target_col"], tt, split))
        else:
            out.extend(_autocomplete_tasks(pre_dir, db, splits, task_types))
    return out


def pretrain_tasks(pre_dir: str, *, dbs=None) -> list[Task]:
    """Train-split tasks across every preprocessed dataset (the pretraining mixture)."""
    return tasks_from_preprocessed(pre_dir, splits=("train",), dbs=dbs)


# The curated RelBench evaluation benchmark used by the released runs: 12 clf +
# 9 reg forecasting tasks (the "relbench_eval_w_event" set from the original
# repo). We evaluate on exactly these, not every explicit relbench-pre task, so
# the eval matches the released numbers and stays cheap (~21 vs ~34 tasks). Each
# entry is (db, task_table, target_column, task_type).
RELBENCH_EVAL_TASKS: tuple[tuple[str, str, str, str], ...] = (
    # clf
    ("rel-amazon", "user-churn", "churn", "clf"),
    ("rel-hm", "user-churn", "churn", "clf"),
    ("rel-stack", "user-badge", "WillGetBadge", "clf"),
    ("rel-amazon", "item-churn", "churn", "clf"),
    ("rel-stack", "user-engagement", "contribution", "clf"),
    ("rel-avito", "user-visits", "num_click", "clf"),
    ("rel-avito", "user-clicks", "num_click", "clf"),
    ("rel-event", "user-ignore", "target", "clf"),
    ("rel-trial", "study-outcome", "outcome", "clf"),
    ("rel-f1", "driver-dnf", "did_not_finish", "clf"),
    ("rel-event", "user-repeat", "target", "clf"),
    ("rel-f1", "driver-top3", "qualifying", "clf"),
    # reg
    ("rel-hm", "item-sales", "sales", "reg"),
    ("rel-amazon", "user-ltv", "ltv", "reg"),
    ("rel-amazon", "item-ltv", "ltv", "reg"),
    ("rel-stack", "post-votes", "popularity", "reg"),
    ("rel-trial", "site-success", "success_rate", "reg"),
    ("rel-trial", "study-adverse", "num_of_adverse_events", "reg"),
    ("rel-event", "user-attendance", "target", "reg"),
    ("rel-f1", "driver-position", "position", "reg"),
    ("rel-avito", "ad-ctr", "num_click", "reg"),
)


def eval_tasks(pre_dir: str, *, splits=("val", "test"), dbs=None) -> list[Task]:
    """The curated RelBench benchmark (:data:`RELBENCH_EVAL_TASKS`) at the given
    splits -- exactly the 21 forecasting tasks the released runs evaluated on,
    not an enumerate-all over the preprocessed eval datasets."""
    return [
        Task(db, table, col, tt, split)
        for (db, table, col, tt) in RELBENCH_EVAL_TASKS
        if dbs is None or db in dbs
        for split in splits
    ]


# -------------------------------------------------------------------------- #
# mlock: pin the preprocessed mixture in RAM across training restarts
# -------------------------------------------------------------------------- #
import ctypes
import ctypes.util
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mlock.restype = ctypes.c_int

_PROT_READ = 0x1
_MAP_SHARED = 0x01
_MAP_FAILED = ctypes.c_void_p(-1).value


def mlock_file(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            raise RuntimeError(f"empty file: {path}")
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr == _MAP_FAILED:
            err = ctypes.get_errno()
            raise OSError(err, f"mmap failed for {path}: {os.strerror(err)}")
    finally:
        os.close(fd)
    if _libc.mlock(addr, size) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mlock failed for {path}: {os.strerror(err)}")
    return size


def _included_dbs(include_dbs_file: str | None) -> set[str] | None:
    if not include_dbs_file:
        return None
    with open(include_dbs_file) as f:
        return {
            ln.strip()
            for ln in f
            if ln.strip() and not ln.lstrip().startswith("#")
        }


@dataclass
class MlockConfig:
    pre_dir: str

    include_dbs_file: str | None
    """restrict to the dbs in this file (e.g. docs/rt_j_dbs.txt); without
    it, every preprocessed db under --pre-dir is locked."""

    embedding_model_ref: str

    workers: int
    """parallel mlock workers; /dfs scales with concurrency (measured
    ~244MB/s single-stream vs ~1.2GB/s at 8+ parallel), so more workers
    saturate it faster."""


def mlock_main(cfg: MlockConfig) -> None:
    tasks = pretrain_tasks(cfg.pre_dir)
    db_names = sorted({t.db_name for t in tasks})
    include = _included_dbs(cfg.include_dbs_file)
    if include is not None:
        db_names = [d for d in db_names if d in include]
    print(f"mlock: {len(db_names)} unique dbs", flush=True)

    def db_paths(db: str) -> list[str]:
        base = os.path.join(cfg.pre_dir, db)
        return [
            os.path.join(base, "nodes.rkyv"),
            os.path.join(base, f"text_emb_{cfg.embedding_model_ref}.bin"),
            os.path.join(base, "p2f_adj.rkyv"),
        ]

    def fmt_size(n: int) -> str:
        return f"{n / 2**30:.2f} GiB"

    page_size = os.sysconf("SC_PAGESIZE")

    def allocated_size(p: str) -> int:
        return os.stat(p).st_blocks * 512

    def footprint_size(p: str) -> int:
        size = os.stat(p).st_size
        return ((size + page_size - 1) // page_size) * page_size

    db_sizes: dict[str, int] = {}
    db_footprints: dict[str, int] = {}
    size_errors: dict[str, str] = {}
    for db in db_names:
        try:
            paths = db_paths(db)
            db_sizes[db] = sum(allocated_size(p) for p in paths)
            db_footprints[db] = sum(footprint_size(p) for p in paths)
        except Exception as e:
            size_errors[db] = f"{type(e).__name__}: {e}"

    total_size = sum(db_sizes.values())
    width = max((len(fmt_size(s)) for s in db_sizes.values()), default=0)

    locked_files = 0
    total = 0
    skipped = 0

    for db in db_names:
        if db in size_errors:
            print(
                f"\x1b[31m[{'ERROR':>{width}}] {db}  {size_errors[db]}\x1b[0m",
                flush=True,
            )
            skipped += 1

    def lock_db(db: str) -> tuple[str, int, Exception | None]:
        n = 0
        try:
            for p in db_paths(db):
                mlock_file(p)
                n += 1
        except Exception as e:
            return db, n, e
        return db, n, None

    pending = [db for db in db_names if db not in size_errors]
    total_footprint = sum(db_footprints[db] for db in pending)
    import time

    t0 = time.time()
    pbar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024)
    pbar.set_postfix_str(f"footprint={fmt_size(total_footprint)}")
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(lock_db, db) for db in pending]
        for fut in as_completed(futures):
            db, n, err = fut.result()
            db_size = db_sizes[db]
            locked_files += n
            if err is not None:
                tqdm.write(
                    f"\x1b[31m[{fmt_size(db_size):>{width}}] {db}  "
                    f"ERROR: {type(err).__name__}: {err}\x1b[0m"
                )
                skipped += 1
                continue
            tqdm.write(f"[{fmt_size(db_size):>{width}}] {db}")
            total += db_size
            pbar.update(db_size)
    pbar.close()
    elapsed = time.time() - t0

    print(
        f"locked {locked_files} files, {fmt_size(total)} on disk, "
        f"{fmt_size(total_footprint)} memory footprint, "
        f"{skipped} dbs skipped, in {elapsed:.0f}s "
        f"({total / 2**30 / max(elapsed, 1e-9):.2f} GiB/s). "
        f"pid={os.getpid()}. sleeping until signaled.",
        flush=True,
    )

    def _fast_exit(signum: int, frame: object) -> None:
        # Proactively release all locked pages before exiting. Without this the
        # kernel reclaims ~1TB of mlocked memory lazily on process teardown,
        # which can exceed slurm's UnkillableStepTimeout on scancel and DRAIN the
        # node ("Kill task failed"). munlockall() makes teardown prompt.
        try:
            _libc.munlockall()
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGINT, _fast_exit)
    signal.signal(signal.SIGTERM, _fast_exit)
    signal.pause()
