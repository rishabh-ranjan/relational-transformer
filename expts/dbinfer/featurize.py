#!/usr/bin/env python
"""Stage 2: write depth-2 DFS feature matrices for the 4DBInfer tasks.

Runs :class:`rel2tab.RDBLearnFeaturizer` -- which loads the database through
``RDBDataset.from_4dbinfer`` (the upstream archives ``dbinfer_bench`` downloads)
and runs fastdfs depth-2 DFS -- and writes the result where
:class:`rel2tab.PrecomputedFeaturizer` reads it at eval time::

    <pre_dir>/<db>/rdblearn_features/<table>_vectors.bin   float32, (total_nodes, n_features)
    <pre_dir>/<db>/rdblearn_features/<table>_meta.json     {n_features, min_offset, total_nodes}

Splitting featurization from eval is what the method name says: the baseline is
``precomputed_rdblearn`` + ``tabicl_batched``, i.e. eval only ever does an index
lookup. It also isolates the dependency problem -- rdblearn pins ``relbench==1.1.0``
and pulls autogluon, which conflict with this repo's ``relbench-hf`` -- so this
stage runs in its own venv (see ``slurm_featurize.sh``) while eval runs in the
repo's pixi env.

**Row order is the contract.** ``PrecomputedFeaturizer.compute_features`` maps a
node to a feature row by ``node_idx - min_offset``, so the archive's task rows must
line up one-for-one with the preprocessed label tables. The featurizer asserts
equal row counts per split; ``--verify-rows <dir>`` additionally compares the label
*sequences* against the published dbinfer parquets, which is the check that would
actually catch a reordering.

    python expts/dbinfer/featurize.py --db dbinfer-retailrocket \\
        --pre-dir /dfs/user/$USER/pre/dbinfer-preprocessed \\
        --verify-rows /lfs/hyperturing1/0/$USER/dbinfer-build
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # local rel2tab/

MAX_DEPTH = 2
# 0 means "no cap" in RDBLearnFeaturizerConfig; DFS runs over every row because the
# feature matrix must cover train+val+test, not just a fitting subsample.
MAX_TRAIN_SAMPLES = 0
FEATURES_SUBDIR = "rdblearn_features"
SPLITS = ("train", "val", "test")
# Same mapping rt.data.tasks uses; only node-level clf/reg tasks are modelled.
TASK_TYPE = {"binary_classification": "clf", "regression": "reg"}


def verify_rows(build_dir: Path, db: str, table: str, split_labels: dict) -> None:
    """Compare archive-order label sequences against the published parquets.

    The published dbinfer task tables were written in archive order, so this should
    agree exactly. If it ever does not, features are silently misaligned with
    ``node_idx`` and every baseline number is wrong -- hence a hard failure rather
    than a warning.
    """
    import pandas as pd

    tdir = build_dir / db / "tasks" / table
    for split, got in split_labels.items():
        path = tdir / f"{split}.parquet"
        if not path.exists():
            print(
                f"  verify {db}/{table} {split}: no published parquet, skipped",
                flush=True,
            )
            continue
        published = pd.read_parquet(path)
        col = got.name
        if col not in published.columns:
            raise RuntimeError(
                f"{db}/{table} {split}: label column {col!r} not in {path} "
                f"(has {list(published.columns)})"
            )
        want = published[col]
        if len(want) != len(got):
            raise RuntimeError(
                f"{db}/{table} {split}: {len(got)} archive rows vs {len(want)} published"
            )
        # Labels differ in dtype across the two sources (the port coerces bool and
        # 't'/'f' to int8), so compare as strings.
        same = float((got.astype(str).values == want.astype(str).values).mean())
        if same < 1.0:
            raise RuntimeError(
                f"{db}/{table} {split}: label row-order agreement {same:.4f} != 1.0; "
                "the archive and the published task table are not in the same order, "
                "so DFS features cannot be indexed by node_idx"
            )
        print(
            f"  verify {db}/{table} {split}: row order agrees on {len(got)} rows",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--db", required=True, help="preprocessed db name, e.g. dbinfer-amazon"
    )
    ap.add_argument("--pre-dir", required=True)
    ap.add_argument(
        "--db-task-list", default=str(Path(__file__).resolve().parent / "tasks.json")
    )
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    ap.add_argument(
        "--verify-rows",
        default=None,
        help="dbinfer build dir (the published parquets) to check row order against",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pre_dir = os.path.expandvars(args.pre_dir)
    out_dir = Path(pre_dir).expanduser() / args.db / FEATURES_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        p for p in json.loads(Path(args.db_task_list).read_text()) if p[0] == args.db
    ]
    if not pairs:
        sys.exit(f"no tasks for {args.db!r} in {args.db_task_list}")
    tables = [t for _, t in pairs]
    pending = [
        t for t in tables if args.overwrite or not (out_dir / f"{t}_meta.json").exists()
    ]
    print(f"{args.db}: tasks={tables} pending={pending}", flush=True)
    if not pending:
        print("nothing to do.")
        return

    from rel2tab.featurizers import RDBLearnFeaturizer

    # clf/reg per task, straight out of the preprocessed meta.json -- the featurizer
    # only needs it to pick a base estimator. Reading three fields here is what lets
    # this stage import nothing from `rt`, whose package init pulls torch and the
    # compiled rustler extension that its env deliberately lacks.
    meta = json.loads((Path(pre_dir).expanduser() / args.db / "meta.json").read_text())
    by_name = {t["name"]: t for t in meta.get("tasks", [])}
    task_types = []
    for t in pending:
        if t not in by_name:
            sys.exit(f"{args.db}: task {t!r} not in meta.json ({sorted(by_name)})")
        tt = TASK_TYPE.get(by_name[t].get("task_type"))
        if tt is None:
            sys.exit(
                f"{args.db}/{t}: unmodelled task_type {by_name[t].get('task_type')!r}"
            )
        task_types.append((t, tt))
    print(f"task types: {task_types}", flush=True)

    # The featurizer keys its cache by (db, table) and precomputes everything at
    # init, so one construction covers this db's whole task list.
    featurizer = RDBLearnFeaturizer(
        pre_dir=pre_dir,
        db_name=args.db,
        tasks=task_types,
        max_depth=args.max_depth,
        max_train_samples=MAX_TRAIN_SAMPLES,
    )

    if args.verify_rows:
        build_dir = Path(os.path.expandvars(args.verify_rows)).expanduser()
        for (db, table), split_labels in featurizer._split_labels.items():
            verify_rows(build_dir, db, table, split_labels)

    for (db, table), (feats, min_offset) in featurizer._features.items():
        arr = feats.numpy().astype(np.float32, copy=False)
        (out_dir / f"{table}_vectors.bin").write_bytes(arr.tobytes())
        (out_dir / f"{table}_meta.json").write_text(
            json.dumps(
                {
                    "n_features": int(arr.shape[1]),
                    "min_offset": int(min_offset),
                    "total_nodes": int(arr.shape[0]),
                    "max_depth": args.max_depth,
                    "source": "from_4dbinfer",
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(
            f"wrote {out_dir}/{table}_vectors.bin  "
            f"({arr.shape[0]} rows x {arr.shape[1]} features, offset {min_offset})",
            flush=True,
        )


if __name__ == "__main__":
    main()
