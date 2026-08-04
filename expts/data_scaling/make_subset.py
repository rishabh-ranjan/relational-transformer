#!/usr/bin/env python
"""Derive a data-scaling subset of a db_task_list by sampling whole databases.

Sampling is at the database level, not the task level: a fraction of a db's
tasks would still expose the model to the whole db, which is not what a data
scaling curve is meant to vary. All of a sampled db's tasks are kept.

    python expts/data-scaling/make_subset.py \
        --src /dfs/user/ranjanr/pre/the-join-preprocessed/db-task-lists/rt-j.json \
        --out expts/data-scaling/32pct.json --frac 0.3162 --seed 0 \
        --base expts/data-scaling/10pct.json
"""

import json
import random
from pathlib import Path

import tyro


def main(
    src: Path,
    out: Path,
    frac: float,
    seed: int = 0,
    base: tuple[Path, ...] = (),
) -> None:
    pairs = [(str(db), str(task)) for db, task in json.loads(src.read_text())]
    dbs = sorted({db for db, _ in pairs})
    k = round(len(dbs) * frac)
    fixed = [str(db) for b in base for db, _ in json.loads(b.read_text())]
    fixed = list(dict.fromkeys(fixed))
    assert set(fixed) <= set(dbs), f"--base dbs not in {src}"
    assert len(fixed) <= k, f"--base has {len(fixed)} dbs > {k} for frac={frac}"
    rest = [db for db in dbs if db not in set(fixed)]
    random.Random(seed).shuffle(rest)
    keep = set(fixed) | set(rest[: k - len(fixed)])
    subset = [p for p in pairs if p[0] in keep]
    out.write_text(json.dumps(subset, indent=1) + "\n")
    print(f"{src}: {len(dbs)} dbs, {len(pairs)} tasks")
    print(
        f"{out}: {k} dbs ({k / len(dbs):.1%}), {len(subset)} tasks "
        f"({len(subset) / len(pairs):.1%}), seed={seed}"
    )


if __name__ == "__main__":
    tyro.cli(main)
