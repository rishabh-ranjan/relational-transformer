"""Step 2 — Define the prediction task.

Copies your labeled split files (``config.TASK["splits"]``) into the dataset
directory written by step 1, plus a task manifest:

    <DATA_DIR>/<DB_NAME>/tasks/<task>/
      manifest.yaml     # entity table/column, time column, target, task type
      {train,val,test}.parquet

    pixi run python examples/inference/2_task_prep.py
"""

from pathlib import Path

import config
import pandas as pd
import yaml


def main():
    task = config.TASK
    if task["task_type"] not in ("binary_classification", "regression"):
        raise ValueError(f"task_type must be 'binary_classification' or 'regression', got {task['task_type']!r}")
    if "test" not in task["splits"]:
        raise ValueError("task 'splits' must include a 'test' entry (the labeled rows to score).")

    out = Path(config.DATA_DIR) / config.DB_NAME / "tasks" / task["name"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[step 2] task '{task['name']}' ({task['task_type']})")
    print(f"   predict '{task['target_col']}' for '{task['entity_table']}' at '{task['time_col']}'")
    for split, path in task["splits"].items():
        p = Path(path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df[task["time_col"]] = pd.to_datetime(df[task["time_col"]])
        df.to_parquet(out / f"{split}.parquet", index=False)
        print(f"   - {split}: {len(df)} labeled rows")

    yaml.safe_dump(
        {
            "entity_table": task["entity_table"],
            "entity_col": task["entity_col"],
            "target_col": task["target_col"],
            "task_type": task["task_type"],
            "time_col": task["time_col"],
        },
        open(out / "manifest.yaml", "w"),
        sort_keys=False,
    )
    print(f"[step 2] wrote task tables -> {out}")


if __name__ == "__main__":
    main()
