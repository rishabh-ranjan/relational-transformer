"""Task resolution from an explicit db-task list.

The set of tasks to train or evaluate on is always given explicitly as a
``db_task_list``: a list of ``(db_name, task_name)`` pairs, a local path to a
JSON file holding such a list. ``task_name`` is always a task recorded in the
db's ``meta.json``: a forecast/external task, or an autocomplete task
(``kind: autocomplete``, a manifest-only task dir whose target is a column of a
db table). There is no enumerate-everything fallback and no on-the-fly column
resolution: the list is the single source of truth.

Curated lists ship inside the preprocessed dataset repos, so they arrive with
the data they refer to: ``<pre_dir>/db-task-lists/forecast.json`` (for
relbench-preprocessed, the 21-task RelBench benchmark) and
``<pre_dir>/db-task-lists/{forecast,autocomplete,all,rt-j}.json`` (for
the-join-preprocessed, the pretraining mixtures).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rt.data.resolve import read_meta

# relbench task_type -> RT task_type. Only node-level clf/reg tasks are modeled;
# link_prediction (recommendation) tasks are skipped.
_TASK_TYPE = {"binary_classification": "clf", "regression": "reg"}


@dataclass(frozen=True)
class Task:
    db_name: str
    table_name: str
    target_column: str
    task_type: str  # "clf" | "reg"
    split: str = ""  # "train" | "val" | "test"
    # ``(table, column)`` pairs to keep out of this task's context because they
    # leak the target -- ``remove_columns`` in the task's manifest.yaml, carried
    # through to meta.json by the preprocessor. Empty for most tasks; it matters
    # for autocomplete, whose target is a real db column sitting in a row next
    # to columns trivially derivable from it.
    leakage_columns: tuple[tuple[str, str], ...] = ()


def resolve_db_task_list(db_task_list) -> list[tuple[str, str]]:
    """Materialize a db_task_list into ``[(db_name, task_name), ...]``.

    Accepts an in-memory list of pairs or a path to a JSON file of pairs. The
    released lists ship inside the preprocessed dataset repos, so they arrive
    with the data: ``<pre_dir>/db-task-lists/<name>.json``.
    """
    if isinstance(db_task_list, str):
        p = Path(db_task_list).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"db_task_list {db_task_list!r} does not exist. The released "
                f"lists ship with the preprocessed data, as "
                f"<pre_dir>/db-task-lists/<name>.json"
            )
        pairs = json.loads(p.read_text())
    else:
        pairs = db_task_list
    out = []
    for pair in pairs:
        db, name = pair
        out.append((str(db), str(name)))
    return out


def get_tasks(pre_dir, db_task_list, splits) -> list[Task]:
    """Build full :class:`Task` objects for a db_task_list at the given splits.

    Every name in the list must be an explicit task recorded in the db's
    ``meta.json`` -- there is no on-the-fly resolution of column specs.

    Forecast/external tasks carry a label table per split, so they are emitted
    once per requested split that the task actually ships. Autocomplete tasks
    (``kind: autocomplete``) have no label table at all: the target is a column
    of an existing db table, so they carry no splits and are emitted at the
    ``train`` split only.

    A task's ``remove_columns`` becomes :attr:`Task.leakage_columns`, which the
    dataset turns into column indices the sampler drops from every context.
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
        for name in names:
            t = explicit.get(name)
            if t is None:
                raise ValueError(
                    f"{db}: task {name!r} is not a task in meta.json "
                    f"({sorted(explicit)})"
                )
            tt = _TASK_TYPE[t["task_type"]]
            leaks = tuple(
                (str(tbl), str(col)) for tbl, col in t.get("remove_columns") or ()
            )
            # A non-autocomplete task with no splits can never yield a Task at
            # any split. Naming one is a mistake that would otherwise resolve to
            # nothing in silence -- the shape a the-join autocomplete task takes
            # if its `kind` goes missing from meta.json, since it ships no label
            # parquet. (A task that simply does not ship the *requested* split is
            # fine and stays quiet.)
            if t.get("kind") != "autocomplete" and not t.get("splits"):
                raise ValueError(
                    f"{db}: task {name!r} has kind={t.get('kind')!r} and no splits, "
                    f"so it resolves to no tasks; its meta.json entry is stale "
                    f"(re-preprocess the db)"
                )
            if t.get("kind") == "autocomplete":
                if "train" in splits:  # autocomplete is a pretraining-only signal
                    out.append(
                        Task(db, t["entity_table"], t["target_col"], tt, "train", leaks)
                    )
                continue
            for split in splits:
                if split in t.get("splits", []):
                    out.append(Task(db, name, t["target_col"], tt, split, leaks))
    return out
