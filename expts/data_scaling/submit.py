"""Submit the data-scaling arms.

The loop is the experiment: same target, same everything, one argument varying.
Run it from a clean, pushed checkout -- the jobs clone the commit you submit
from, so this file is also the record of what was run.

    pixi run python expts/data_scaling/submit.py
"""

from __future__ import annotations

from rt.slurm import submit

from expts.data_scaling.site import (
    AMPERE,
    BLACKWELL,
    EVAL_PRE_DIR,
    OUT_ROOT,
    PRE_DIR,
    SITE,
)

ARMS = {
    "10pct": "expts/data_scaling/10pct.json",
    "32pct": "expts/data_scaling/32pct.json",
    "all": "expts/data_scaling/all-usable.json",
    "rt-j": f"{PRE_DIR}/db-task-lists/rt-j.json",
}

BASE = dict(
    pre_dir=PRE_DIR,
    eval_pre_dir=EVAL_PRE_DIR,
    out_root=OUT_ROOT,
    project="data-scaling",
)


def main() -> None:
    for arm, task_list in ARMS.items():
        # The control arm gets the fast queue; the rest take what is free.
        resources = AMPERE if arm == "rt-j" else BLACKWELL
        submit(
            "expts.data_scaling.train:train",
            args={**BASE, "db_task_list": task_list},
            resources=resources,
            name=f"ds-{arm}",
            **SITE,
        )


if __name__ == "__main__":
    main()
