"""`submit_cutoff.py` as a table: each rel-f1 task with and without the
database trimmed to the test timestamp.

    pixi run python expts/fine_tune/cutoff_table.py

One row per task: the published baseline, the metric at the largest ensemble
size both arms have reached, and the difference. Both arms are read at the same
size so a task that is still ensembling compares like with like; the size is in
the `ens` column. A task only one arm has scored is left out.

Units are `ens_table.py`'s -- percent, nMAE for regression -- and the metric is
the task's own, so higher is better for AUROC and lower for nMAE.
"""

import math

from ens_table import ENTITY, curves, target_for
from submit import ntrain
from submit_cutoff import PROJECTS, TASKS


def table() -> str:
    arms = {upto: curves(ENTITY, project) for upto, project in PROJECTS.items()}
    header = ["task", "target", "ens", "whole db", "upto test ts", "delta"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for db, task in sorted(
        TASKS, key=lambda t: ntrain().get(f"{t[0]}/{t[1]}", math.inf)
    ):
        name = f"{db}/{task}"
        off, on = arms[False].get(name, {}), arms[True].get(name, {})
        common = sorted(set(off) & set(on))
        if not common:
            continue
        size = common[-1]
        lines.append(
            f"| {name} | {target_for(name):.2f} | {size} | {off[size]:.2f} | "
            f"{on[size]:.2f} | {on[size] - off[size]:+.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    print(table())


if __name__ == "__main__":
    main()
