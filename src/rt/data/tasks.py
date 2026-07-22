"""Task resolution from an explicit db-task list.

The set of tasks to train or evaluate on is always given explicitly as a
``db_task_list``: a list of ``(db_name, task_name)`` pairs, a local path to a
JSON file holding such a list, or a Hub path ``org/repo/path/to/list.json``
(only that file is downloaded). ``task_name`` is either a forecast task-table
name recorded in the db's ``meta.json`` or an autocomplete target spelled
``"<table>/<column>"``. There is no enumerate-everything fallback: the list is
the single source of truth for what runs.

Curated lists ship on the Hub, e.g.
``stanford-star/relbench/db-task-lists/forecast.json`` (the 21-task RelBench
benchmark) and ``stanford-star/the-join/db-task-lists/{forecast,rt-j}.json`` (the
pretraining mixtures).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rt.data.resolve import read_meta, resolve_pre_dir, resolve_repo

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


def resolve_db_task_list(db_task_list) -> list[tuple[str, str]]:
    """Materialize a db_task_list into ``[(db_name, task_name), ...]``.

    Accepts an in-memory list of pairs, a local JSON file path, or a Hub path
    ``org/repo/path/to/list.json`` (downloads only that file).
    """
    if isinstance(db_task_list, str):
        p = Path(db_task_list).expanduser()
        if p.exists():
            pairs = json.loads(p.read_text())
        else:
            from huggingface_hub import hf_hub_download

            repo_id, filename = resolve_repo(db_task_list)
            if not filename:
                raise ValueError(
                    f"{db_task_list!r}: expected a local file or a Hub path "
                    f"'org/repo/path/to/list.json'"
                )
            local = hf_hub_download(repo_id, filename, repo_type="dataset")
            pairs = json.loads(Path(local).read_text())
    else:
        pairs = db_task_list
    out = []
    for pair in pairs:
        db, name = pair
        out.append((str(db), str(name)))
    return out


def get_tasks(pre_dir, db_task_list, splits, *, embedding_model=None) -> list[Task]:
    """Build full :class:`Task` objects for a db_task_list at the given splits.

    Forecast task names are looked up in each db's ``meta.json`` (target column,
    task type, available splits). Autocomplete names (``"<table>/<column>"``)
    are emitted at the ``train`` split only; their clf/reg type is read off the
    preprocessed nodes via ``rustler.column_sem_types``, which requires the
    db's core files locally (``embedding_model`` selects the text-embedding
    file when ``pre_dir`` is a Hub repo and the data must be fetched -- the
    same files training/eval need anyway).
    """
    pairs = resolve_db_task_list(db_task_list)
    by_db: dict[str, list[str]] = {}
    for db, name in pairs:
        by_db.setdefault(db, []).append(name)

    out: list[Task] = []
    for db, names in by_db.items():
        meta = read_meta(pre_dir, db)
        explicit = {
            t["name"]: t
            for t in meta.get("tasks", [])
            if _TASK_TYPE.get(t.get("task_type")) and t.get("target_col")
        }
        sem = None
        for name in names:
            if name in explicit:
                t = explicit[name]
                tt = _TASK_TYPE[t["task_type"]]
                for split in splits:
                    if split in t.get("splits", []):
                        out.append(Task(db, name, t["target_col"], tt, split))
            elif "/" in name:
                if "train" not in splits:
                    continue  # autocomplete is a pretraining-only signal
                table, col = name.split("/", 1)
                if sem is None:
                    from rt.rustler import column_sem_types

                    if embedding_model is None:
                        raise ValueError(
                            f"resolving autocomplete task {db}/{name} needs "
                            f"embedding_model to fetch the db's core files"
                        )
                    local = resolve_pre_dir(pre_dir, [db], embedding_model)
                    sem = column_sem_types(local, db)
                tt = _SEM_TASK_TYPE.get(sem.get(f"{col} of {table}"))
                if tt is None:
                    raise ValueError(
                        f"{db}: autocomplete target {name!r} is not a "
                        f"Boolean/Number column of the preprocessed data"
                    )
                out.append(Task(db, table, col, tt, "train"))
            else:
                raise ValueError(
                    f"{db}: task {name!r} is neither a forecast task in "
                    f"meta.json ({sorted(explicit)}) nor an autocomplete "
                    f"'<table>/<column>' spec"
                )
    return out
