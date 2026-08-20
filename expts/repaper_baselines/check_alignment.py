"""Assert classic-relbench task tables match the raw HF-format parquets row
for row.

The preprocessed eval data was built from the HF-format raw collection
(``stanford-star/relbench``), so rustler's node index for a task row is that
parquet's row number. The featurize scripts instead load tasks through the
classic relbench 2.x package (which rdblearn requires) and write features
positionally. This check proves the two sources agree on row order -- entity
and timestamp columns equal, row by row, for every (task, split) -- so a
feature vector computed from the classic row lands on the right node index.

Runs in the featurize environment, on the login node (needs internet the
first time to populate the classic-relbench cache):

    RELBENCH_CACHE_DIR=<cache> pixi run -e featurize \
        python -m expts.repaper_baselines.check_alignment
"""

import json
from pathlib import Path

import pandas as pd

RAW_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench"
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"


def main() -> None:
    from relbench.tasks import get_task

    pairs = json.loads((Path(PRE_DIR) / "db-task-lists" / "forecast.json").read_text())
    for db, task_name in pairs:
        rb_task = get_task(db, task_name, download=True)
        entity_col = rb_task.entity_col
        time_col = rb_task.time_col
        for split in ("train", "val", "test"):
            classic = rb_task.get_table(split).df.reset_index(drop=True)
            raw = pd.read_parquet(
                Path(RAW_DIR) / db / "tasks" / task_name / f"{split}.parquet"
            ).reset_index(drop=True)
            assert len(classic) == len(raw), (
                f"{db}/{task_name}/{split}: {len(classic)} classic rows vs "
                f"{len(raw)} raw parquet rows"
            )
            for col in (entity_col, time_col):
                assert col in raw.columns, (
                    f"{db}/{task_name}/{split}: raw parquet lacks {col!r} "
                    f"(columns: {list(raw.columns)})"
                )
                a = classic[col].reset_index(drop=True)
                b = raw[col].reset_index(drop=True)
                if not a.astype(str).equals(b.astype(str)):
                    bad = (a.astype(str) != b.astype(str)).idxmax()
                    raise AssertionError(
                        f"{db}/{task_name}/{split}: column {col!r} differs at "
                        f"row {bad}: classic={a[bad]!r} raw={b[bad]!r}"
                    )
            print(f"OK {db}/{task_name}/{split}: {len(classic)} rows", flush=True)
    print("all task tables aligned", flush=True)


if __name__ == "__main__":
    main()
