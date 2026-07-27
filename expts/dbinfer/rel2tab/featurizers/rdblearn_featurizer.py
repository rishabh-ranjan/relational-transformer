from dataclasses import dataclass

import numpy as np
import torch

from rel2tab.featurizer import Featurizer


@dataclass
class RDBLearnFeaturizerConfig:
    """Config for RDBLearnFeaturizer.

    DFS features are precomputed once at init for all rows per (db, table).
    ``compute_features`` just does a lookup by node index.

    This copy loads databases through ``RDBDataset.from_4dbinfer`` (the original
    upstream 4DBInfer archives that ``dbinfer_bench`` downloads), not
    ``from_relbench``. Set ``DBB_DATASET_HOME`` to a directory of extracted
    archives to avoid re-downloading them.
    """

    pre_dir: str
    db_task_list: list[tuple[str, str]] | str
    eval_splits: list[str]
    max_depth: int
    max_train_samples: int

    def build(self, device):
        return RDBLearnFeaturizer(
            pre_dir=self.pre_dir,
            db_task_list=self.db_task_list,
            eval_splits=self.eval_splits,
            max_depth=self.max_depth,
            max_train_samples=self.max_train_samples,
            db=None,
        )


class RDBLearnFeaturizer(Featurizer):
    """Precompute rdblearn DFS features at init, look up at eval time.

    At init, loads each relbench dataset, fits an RDBLearnEstimator on
    all rows, extracts DFS features, and stores them as a tensor indexed
    by node_idx.
    """

    def __init__(
        self, pre_dir, db_task_list, eval_splits, max_depth, max_train_samples, db
    ):
        import time

        import fastdfs
        from fastdfs import DFSConfig
        from rdblearn.config import RDBLearnConfig
        from rdblearn.datasets import RDBDataset
        from rdblearn.estimator import RDBLearnEstimator
        from rt.data import get_tasks
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline

        from rel2tab.featurizer import load_table_info

        # (db, table) -> (precomputed_features_tensor, min_offset)
        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}
        # (db, table) -> {split: label Series}, for row-order verification only
        self._split_labels: dict[tuple[str, str], dict] = {}

        dfs_cfg = DFSConfig(
            max_depth=max_depth,
            agg_primitives=["max", "min", "mean", "count", "mode", "std"],
            engine="dfs2sql",
            # Strict `<` against the task cutoff: DFS aggregates only rows before
            # the target's own timestamp, never the target's row itself. This is
            # fastdfs's default, pinned here so an upstream flip to `<=` cannot
            # silently leak the label of a task whose target is a database column
            # (retailrocket/cvr <- View.added_to_cart, amazon/rating <-
            # Review.rating). It is also what RT's sampler does: drop the leakage
            # column on rows sharing the target's timestamp, keep strictly-past ones.
            include_cutoff_time=False,
        )
        config = RDBLearnConfig(
            dfs=dfs_cfg,
            enable_target_augmentation=False,
            max_train_samples=(max_train_samples if max_train_samples > 0 else 10**9),
            predict_batch_size=5000,
        )

        all_tasks = get_tasks(pre_dir, db_task_list, tuple(eval_splits))
        if db is not None:
            all_tasks = [t for t in all_tasks if db in t.db_name]

        # Deduplicate eval tasks by (db_name, table_name).
        seen: set[tuple[str, str]] = set()
        for task in all_tasks:
            key = (task.db_name, task.table_name)
            if key in seen:
                continue
            seen.add(key)
            db_name, table_name = key

            tic = time.time()

            # fastdfs's DBInferAdapter takes the bare 4DBInfer name ("amazon",
            # "diginetica", ...) and reads the upstream archive via
            # dbinfer_bench.load_rdb_data. It filters each table to the columns
            # metadata.yaml declares, which is exactly the column set the
            # stanford-star/dbinfer port publishes -- so this featurizer and RT see
            # the same schema, and neither sees e.g. Posts.Score.
            rdb_name = db_name.removeprefix("dbinfer-")
            dataset = RDBDataset.from_4dbinfer(rdb_name)
            table_info = load_table_info(db_name, pre_dir)

            # Find the rdblearn task by name (matches our table_name).
            if table_name not in dataset.tasks:
                raise ValueError(
                    f"No rdblearn task '{table_name}' in db '{db_name}'."
                    f" Available: {list(dataset.tasks.keys())}"
                )
            rdb_task = dataset.tasks[table_name]

            target_col = rdb_task.metadata.target_col
            key_mappings = rdb_task.metadata.key_mappings
            cutoff_time_column = rdb_task.metadata.time_col

            # Build combined DataFrame aligned to node_idx offsets.
            splits_info: dict[str, dict] = {}
            for split in ["train", "val", "test"]:
                info_key = f"{table_name}:{split.capitalize()}"
                if info_key in table_info:
                    splits_info[split] = table_info[info_key]

            if not splits_info:
                raise KeyError(
                    f"Table '{table_name}' not in table_info.json for db "
                    f"'{db_name}'. Keys: {list(table_info.keys())}"
                )

            min_offset = min(info["node_idx_offset"] for info in splits_info.values())

            split_dfs = {
                "train": rdb_task.train_df,
                "val": rdb_task.val_df,
                "test": rdb_task.test_df,
            }
            ordered = sorted(splits_info.items(), key=lambda x: x[1]["node_idx_offset"])

            # Row order is the contract: ``compute_features`` maps node_idx ->
            # combined_df row by ``node_idx - min_offset``, so the archive's split
            # frames must line up one-for-one with the preprocessed label tables.
            # Equal lengths per split is the cheap half of that check; the label
            # sequences themselves are compared by ``featurize.py --verify-rows``.
            for split, info in ordered:
                got, want = len(split_dfs[split]), info["num_nodes"]
                if got != want:
                    raise RuntimeError(
                        f"{db_name}/{table_name} {split}: 4DBInfer archive has {got} "
                        f"rows but the preprocessed table has {want}; features would "
                        f"be misaligned with node_idx"
                    )
            parts = [split_dfs[s].reset_index(drop=True) for s, _ in ordered]
            import pandas as pd

            combined_df = pd.concat(parts, ignore_index=True)

            X = combined_df.drop(columns=[target_col])
            y = combined_df[target_col]

            # Fit estimator on all rows and extract DFS features.
            base_model = LogisticRegression() if task.task_type == "clf" else Ridge()
            estimator = RDBLearnEstimator(
                base_estimator=make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=0),
                    base_model,
                ),
                config=config,
            )
            estimator.fit(
                X=X,
                y=y,
                rdb=dataset.rdb,
                key_mappings=key_mappings,
                cutoff_time_column=cutoff_time_column,
            )

            # Extract features for all rows.
            X_copy = X.copy()
            estimator._ensure_keys_are_strings(X_copy, estimator.key_mappings_)
            X_dfs = fastdfs.compute_dfs_features(
                estimator.rdb_,
                X_copy,
                key_mappings=estimator.key_mappings_,
                cutoff_time_column=estimator.cutoff_time_column_,
                config=estimator.config.dfs or DFSConfig(),
            )
            X_transformed = estimator.preprocessor_.transform(X_dfs)
            feats = X_transformed.fillna(0).values.astype(np.float32)
            feats_tensor = torch.from_numpy(feats)

            elapsed = time.time() - tic
            self._features[key] = (feats_tensor, min_offset)
            # Keep just the label column per split so the caller can verify row
            # order against the published task tables (featurize.py --verify-rows).
            # Only the labels, because the frames themselves are large.
            self._split_labels[key] = {
                split: split_dfs[split][target_col].reset_index(drop=True)
                for split, _info in ordered
            }
            print(
                f"  RDBLearnFeaturizer: precomputed {db_name}/{table_name}"
                f" ({len(combined_df)} rows, {feats.shape[1]} features,"
                f" {elapsed:.1f}s)"
            )

    def compute_features(self, task, node_idxs, device, batch_size):
        key = (task.db_name, task.table_name)
        feats_tensor, min_offset = self._features[key]
        local_idxs = node_idxs.cpu() - min_offset
        return feats_tensor[local_idxs].to(device)

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        return train_feats, train_labels, test_feat
