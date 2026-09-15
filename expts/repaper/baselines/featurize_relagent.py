import json
import os
from pathlib import Path


def featurize_table(
    *,
    db: str,
    table: str,
    program: str,
    pre_dir: str,
    raw_dir: str,
    features_root: str,
    features_subdir: str,
    relbench_cache_dir: str,
    upto_test_timestamp: bool,
    sql_query_timeout: float,
) -> None:
    os.environ["RELBENCH_CACHE_DIR"] = str(Path(relbench_cache_dir).expanduser())
    import duckdb
    import numpy as np
    import pandas as pd
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from expts.repaper.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        table_offset_and_len,
    )
    from expts.repaper.baselines.relagent_program import (
        build_feature_frame,
        load_program,
    )

    queries = load_program(program)

    # RelAgent reads these off the relbench task rather than carrying a table of
    # them (relbench_loader.py:398, main.py:183), and its queries are written
    # against whatever they are, so deriving them here is what keeps a program
    # runnable -- passing them in is a chance to pass the wrong pair.
    rb_task = get_task(db, table, download=True)
    entity_col, time_col = rb_task.entity_col, rb_task.time_col
    print(f"[{db}] {table}: entity_col={entity_col} time_col={time_col}", flush=True)

    out_dir = Path(features_root).expanduser() / db / features_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / f"{table}_vectors.bin"
    meta_path = out_dir / f"{table}_meta.json"
    if vectors_path.exists() and meta_path.exists():
        print(f"[{db}] {table}: already featurized, skipping", flush=True)
        return

    min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
    splits_info = get_table_splits(load_table_info(pre_dir, db), table)
    ordered = sorted(splits_info.items(), key=lambda kv: kv[1]["node_idx_offset"])
    task_rows = pd.concat(
        [
            pd.read_parquet(
                Path(raw_dir).expanduser() / db / "tasks" / table / f"{s}.parquet"
            ).reset_index(drop=True)
            for s, _ in ordered
        ],
        ignore_index=True,
    )
    assert len(task_rows) == total_nodes, (
        f"{db}/{table}: {len(task_rows)} task rows vs {total_nodes} nodes in "
        f"table_info.json -- the data the features are for is not the data "
        f"that was preprocessed"
    )
    for col in (entity_col, time_col):
        assert col in task_rows.columns, (
            f"{db}/{table}: task rows have no column {col!r}; got "
            f"{list(task_rows.columns)}"
        )

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")
    # upto_test_timestamp is the knob the leakage audit turns: the agent searched
    # against relbench's default (True, the db cut at test_timestamp), while a
    # blob covering the test rows needs the full db, since a 2010+ row's own
    # history runs past that cut. Featurizing one split twice at two cutoffs that
    # both postdate every row in it must give identical features; any difference
    # is a query reading past a row's own timestamp.
    rb_db = get_dataset(db, download=True).get_db(
        upto_test_timestamp=upto_test_timestamp
    )
    for name, tbl in rb_db.table_dict.items():
        con.register(name, tbl.df)
    print(
        f"[{db}] {len(rb_db.table_dict)} db tables in duckdb "
        f"(upto_test_timestamp={upto_test_timestamp}), "
        f"{len(queries)} feature queries",
        flush=True,
    )

    frame, diagnostics = build_feature_frame(
        con, queries, task_rows, entity_col, time_col, sql_query_timeout
    )
    for name, d in diagnostics.items():
        print(
            f"[{db}] {table}: {name}: {d['n_features']} features, "
            f"{d['missing_rate']:.1%} of rows missing, on {d['joined_on']}",
            flush=True,
        )

    arr = frame.to_numpy(dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    arr = np.nan_to_num((arr - mean) / std, nan=0.0)
    feats = arr.astype(np.float32)
    assert feats.shape[0] == total_nodes
    assert np.isfinite(feats).all()

    feats.tofile(vectors_path)
    meta_path.write_text(
        json.dumps(
            {
                "n_features": feats.shape[1],
                "min_offset": min_offset,
                "total_nodes": total_nodes,
            }
        )
    )
    (out_dir / f"{table}_columns.json").write_text(
        json.dumps(list(frame.columns), indent=2)
    )
    print(
        f"[{db}] {table}: {total_nodes} rows x {feats.shape[1]} features",
        flush=True,
    )
