"""Populate the classic-relbench cache once, to a directory every node can read.

The featurize scripts load databases and task tables through the classic
relbench 2.x package, which downloads and materializes into
``RELBENCH_CACHE_DIR``. Compute nodes have no internet, so everything they
will touch -- every db (full, ``upto_test_timestamp=False``) and every task
split -- is materialized here first. Jobs then read the cache; nothing writes
to it concurrently. Run on the login node:

    RELBENCH_CACHE_DIR=/dfs/user/ranjanr/share/relational-transformer/repaper/relbench-cache \
        pixi run -e featurize python -m expts.repaper_baselines.populate_relbench_cache
"""

import json
import os
from pathlib import Path

PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"


def main() -> None:
    assert os.environ.get("RELBENCH_CACHE_DIR"), "set RELBENCH_CACHE_DIR"
    import relbench.base
    from rdblearn.datasets import RDBDataset
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from relbench.tasks import get_task_names

    pairs = json.loads((Path(PRE_DIR) / "db-task-lists" / "forecast.json").read_text())
    for db in sorted({db for db, _ in pairs}):
        ds = get_dataset(db, download=True)
        rb_db = ds.get_db(upto_test_timestamp=False)
        print(f"{db}: {len(rb_db.table_dict)} tables materialized", flush=True)
        # rdblearn walks the db's WHOLE task registry (recommendation and
        # autocomplete tasks included), and a task's zip extracts into the
        # cache on first table access -- extract every task of every db here,
        # once, so concurrent jobs never race the same extraction. A task that
        # cannot even build is reported and skipped: jobs never build it
        # either (rdblearn skips it the same way).
        for task_name in get_task_names(db):
            try:
                task = get_task(db, task_name, download=True)
                for split in ("train", "val", "test"):
                    task.get_table(split)
                print(f"{db}/{task_name}: extracted", flush=True)
            except Exception as e:
                print(
                    f"{db}/{task_name}: SKIPPED ({type(e).__name__}: {e})", flush=True
                )
        # rdblearn's own loader, so anything it materializes beyond the task
        # zips (db caches under its get_db patch) is also in place.
        _orig_get_db = relbench.base.Dataset.get_db

        def _full_get_db(self, *a, **kw):
            return _orig_get_db(self, upto_test_timestamp=False)

        _full_get_db.cache_clear = lambda: None
        relbench.base.Dataset.get_db = _full_get_db
        try:
            dataset = RDBDataset.from_relbench(db)
        finally:
            relbench.base.Dataset.get_db = _orig_get_db
        print(f"{db}: rdblearn tasks {sorted(dataset.tasks)}", flush=True)
    for db, task_name in pairs:
        task = get_task(db, task_name, download=True)
        for split in ("train", "val", "test"):
            n = len(task.get_table(split).df)
            print(f"{db}/{task_name}/{split}: {n} rows", flush=True)
    print("cache populated", flush=True)


if __name__ == "__main__":
    main()
