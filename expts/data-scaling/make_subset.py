#!/usr/bin/env python
"""Derive a data-scaling subset of a db_task_list by sampling whole databases.

Sampling is at the database level, not the task level: a fraction of a db's
tasks would still expose the model to the whole db, which is not what a data
scaling curve is meant to vary. All of a sampled db's tasks are kept.

    python expts/data-scaling/make_subset.py \
        --src /dfs/user/ranjanr/pre/the-join-preprocessed/db-task-lists/rt-j.json \
        --out expts/data-scaling/10pct.json --frac 0.1 --seed 0
"""

import json
import random
from pathlib import Path

import tyro


def main(src: Path, out: Path, frac: float, seed: int = 0) -> None:
    pairs = [(str(db), str(task)) for db, task in json.loads(src.read_text())]
    dbs = sorted({db for db, _ in pairs})
    k = round(len(dbs) * frac)
    keep = set(random.Random(seed).sample(dbs, k))
    subset = [p for p in pairs if p[0] in keep]
    out.write_text(json.dumps(subset, indent=1) + "\n")
    print(f"{src}: {len(dbs)} dbs, {len(pairs)} tasks")
    print(
        f"{out}: {k} dbs ({k / len(dbs):.1%}), {len(subset)} tasks "
        f"({len(subset) / len(pairs):.1%}), seed={seed}"
    )


if __name__ == "__main__":
    tyro.cli(main)
