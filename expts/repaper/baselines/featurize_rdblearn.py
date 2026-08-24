import json
import os
import time
from pathlib import Path


def featurize_table(
    *,
    db: str,
    table: str,
    task_type: str,
    pre_dir: str,
    raw_dir: str,
    features_root: str,
    relbench_cache_dir: str,
    max_depth: int,
    max_train_samples: int,
) -> None:
    os.environ["RELBENCH_CACHE_DIR"] = relbench_cache_dir
    import fastdfs
    import numpy as np
    import pandas as pd
    import relbench.base
    from fastdfs import DFSConfig
    from rdblearn.config import RDBLearnConfig
    from rdblearn.datasets import RDBDataset
    from rdblearn.estimator import RDBLearnEstimator
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline

    from expts.repaper.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        table_offset_and_len,
    )

    out_dir = Path(features_root).expanduser() / db / "rdblearn_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / f"{table}_vectors.bin"
    meta_path = out_dir / f"{table}_meta.json"
    if vectors_path.exists() and meta_path.exists():
        print(f"[{db}] {table}: already featurized, skipping", flush=True)
        return

    config = RDBLearnConfig(
        dfs=DFSConfig(
            max_depth=max_depth,
            agg_primitives=["max", "min", "mean", "count", "mode", "std"],
            engine="dfs2sql",
        ),
        enable_target_augmentation=False,
        max_train_samples=(max_train_samples if max_train_samples > 0 else 10**9),
        predict_batch_size=5000,
    )

    tic = time.time()
    _orig_get_db = relbench.base.Dataset.get_db

    def _full_get_db(self, *a, **kw):
        return _orig_get_db(self, upto_test_timestamp=False)

    _full_get_db.cache_clear = lambda: None
    relbench.base.Dataset.get_db = _full_get_db
    try:
        dataset = RDBDataset.from_relbench(db)
    finally:
        relbench.base.Dataset.get_db = _orig_get_db

    assert table in dataset.tasks, (
        f"no rdblearn task {table!r} in {db!r}; available: {list(dataset.tasks)}"
    )
    rdb_task = dataset.tasks[table]
    target_col = rdb_task.metadata.target_col

    min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
    splits_info = get_table_splits(load_table_info(pre_dir, db), table)
    ordered = sorted(splits_info.items(), key=lambda kv: kv[1]["node_idx_offset"])
    combined = pd.concat(
        [
            pd.read_parquet(
                Path(raw_dir) / db / "tasks" / table / f"{s}.parquet"
            ).reset_index(drop=True)
            for s, _ in ordered
        ],
        ignore_index=True,
    )
    assert len(combined) == total_nodes, (
        f"{db}/{table}: {len(combined)} task rows vs {total_nodes} nodes in "
        f"table_info.json -- the data the features are for is not the data "
        f"that was preprocessed"
    )

    X = combined.drop(columns=[target_col])
    y = combined[target_col]

    base_model = LogisticRegression() if task_type == "clf" else Ridge()
    estimator = RDBLearnEstimator(
        base_estimator=make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0), base_model
        ),
        config=config,
    )
    estimator.fit(
        X=X,
        y=y,
        rdb=dataset.rdb,
        key_mappings=rdb_task.metadata.key_mappings,
        cutoff_time_column=rdb_task.metadata.time_col,
    )

    X_copy = X.copy()
    estimator._ensure_keys_are_strings(X_copy, estimator.key_mappings_)
    X_dfs = fastdfs.compute_dfs_features(
        estimator.rdb_,
        X_copy,
        key_mappings=estimator.key_mappings_,
        cutoff_time_column=estimator.cutoff_time_column_,
        config=estimator.config.dfs or DFSConfig(),
    )
    feats = estimator.preprocessor_.transform(X_dfs).fillna(0).values.astype(np.float32)
    assert feats.shape[0] == total_nodes

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
    print(
        f"[{db}] {table}: {total_nodes} rows x {feats.shape[1]} features "
        f"in {time.time() - tic:.0f}s",
        flush=True,
    )
