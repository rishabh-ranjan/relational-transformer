"""The ensembling curve as a table: one row per task, the tuned context
configuration, one column per ensemble size, and the published baseline
beside them.

    pixi run python expts/fine_tune/ens_table.py [--project 2026-08-10-fine_tune_ens_only]

Reads the wandb history `rt.eval` logs per ensemble size, so it is a table of
whatever has been scored so far -- a task that has reached seed 5 fills five
columns and leaves the rest blank, and rerunning this later fills them in.
Columns stop at the largest size any task has reached.

`target` is `submit.published_best`, the bolded number in results.md, in the
same units as the curve (percent; nMAE for regression). It is the column every
other one is read against, so it comes first.

`cfg` is the `(ctx, lcs, bw, pl)` the run picked on validation, read from the
`tuning.json` it wrote beside its `eval_out`. A run whose grid held one entry
tuned nothing and has no `tuning.json`, so the column is blank for it.

A task interrupted and requeued has one wandb run per attempt, each replaying
the curve from size 1; the attempt that got furthest is the one shown.
"""

import argparse
import json
import math
from pathlib import Path

import wandb

from submit import ntrain, published_best

# Where a run's outputs land: <out_root>/<entity>/<project>/<run_id>.
OUT_ROOT = Path("/dfs/user/ranjanr/ckpts")

ENTITY = "rtv2"
PROJECT = "2026-08-10-fine_tune_ens_only"

# What a task is scored by, in the key family `rt.eval` logs and
# `published_best` keys: exactly one of these exists per task.
METRICS = ("auroc", "nmae")


def curves(entity: str, project: str) -> dict[str, dict[int, float]]:
    """`{db}/{task}` -> ensemble size -> test metric, best attempt per task.

    The metric key is derived from the run's own `run_name` rather than from
    the task type: a run logs exactly one `{metric}/test/{db}/{task}` key, and
    which one it is says what the task is.
    """
    out: dict[str, dict[int, float]] = {}
    for run in wandb.Api().runs(f"{entity}/{project}"):
        name = run.config["run_name"]
        keys = [
            f"{m}/test/{name}" for m in METRICS if f"{m}/test/{name}" in run.summary
        ]
        if not keys:
            continue
        (key,) = keys
        # `history`, not `scan_history`: the curve is `test_ensemble_size`
        # points, and the sampled endpoint returns them in one request where
        # the scanning one pages the run's whole `tune/` history to find them.
        got = {
            int(row["ens_size"]): float(row[key])
            for row in run.history(keys=["ens_size", key], pandas=False)
            if row.get("ens_size") is not None and row.get(key) is not None
        }
        # Attempts of the same task cover the same sizes from 1; the longest
        # one is the one that got furthest before it was interrupted.
        if len(got) > len(out.get(name, {})):
            out[name] = got
    return out


def tuned_cfg(entity: str, project: str) -> dict[str, str]:
    """`{db}/{task}` -> the context configuration its tuning run chose.

    Keyed off the run directories rather than wandb: the winner is written to
    `tuning.json` and never logged as a metric.
    """
    out: dict[str, str] = {}
    for path in (OUT_ROOT / entity / project).glob("*/tuning.json"):
        for name, entry in json.loads(path.read_text()).items():
            ctx, lcs, bw, pl = entry["best_cfg"]
            out[name] = f"{ctx}/{lcs}/{bw}/{'lat' if pl else 'frq'}"
    return out


def target_for(name: str) -> float:
    """The published best for the task's own metric, in the curve's units."""
    (v,) = [
        published_best()[f"{m}/test/{name}"]
        for m in METRICS
        if f"{m}/test/{name}" in published_best()
    ]
    return v


def table(rows: dict[str, dict[int, float]], cfgs: dict[str, str]) -> str:
    sizes = sorted({s for c in rows.values() for s in c})
    header = ["task", "target", "cfg", *(str(s) for s in sizes)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    # results.md's column order, so the two read the same way round.
    for name in sorted(rows, key=lambda n: ntrain().get(n, math.inf)):
        cells = [f"{rows[name][s]:.2f}" if s in rows[name] else "" for s in sizes]
        lines.append(
            f"| {name} | {target_for(name):.2f} | {cfgs.get(name, '')} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    a = p.parse_args()

    print(table(curves(a.entity, a.project), tuned_cfg(a.entity, a.project)))


if __name__ == "__main__":
    main()
