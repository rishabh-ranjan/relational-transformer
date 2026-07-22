"""rel2tab featurizers: turn a task context into a flat feature table."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
import torch
from torch import nn
from rt.data import process_batch
import numpy as np
import time


# -------------------------------------------------------------------------- #
# featurizer
# -------------------------------------------------------------------------- #
def load_table_info(db: str, pre_dir: str) -> dict:
    """Load table metadata from ``pre_dir/db/table_info.json``."""
    path = Path(pre_dir).expanduser() / db / "table_info.json"
    with open(path) as f:
        return json.load(f)


def get_table_splits(table_info: dict, table_name: str) -> dict[str, dict]:
    """Return ``{split: {node_idx_offset, num_nodes}}`` for available splits."""
    splits = {}
    for split in ["train", "val", "test", "db"]:
        key = f"{table_name}:{split.capitalize()}"
        if key in table_info:
            splits[split] = table_info[key]
    return splits


def validate_contiguous(splits_info: dict[str, dict], db: str, table_name: str):
    """Raise if node indices are not contiguous across splits."""
    sorted_offsets = sorted(
        (info["node_idx_offset"], info["num_nodes"]) for info in splits_info.values()
    )
    for i in range(len(sorted_offsets) - 1):
        end = sorted_offsets[i][0] + sorted_offsets[i][1]
        nxt = sorted_offsets[i + 1][0]
        if end != nxt:
            raise ValueError(
                f"Non-contiguous node_idxs across splits for {db}/{table_name}. "
                f"Offsets: {sorted_offsets}"
            )


class Featurizer(ABC):
    """Controls how task-node rows are transformed into features for the predictor.

    Lifecycle within ``Rel2TabModel.predict``:
      1. ``compute_features`` is called once per batch with all N task-node
         indices.  Use this for expensive bulk work (e.g. building local
         contexts and running a neural encoder).
      2. ``featurize`` is called once per (batch-item, context-size) with the
         visible train rows for a single target.  Use this for row selection
         or transformation (e.g. filtering to same-entity rows).

    """

    @abstractmethod
    def compute_features(self, task, node_idxs, device, batch_size):
        """Compute per-node features in bulk.

        Args:
            task: Eval Task namedtuple (has .db_name, .table_name, .split,
                .task_type, etc.).
            node_idxs: 1-D LongTensor of length N (node indices in the graph).
            device: torch device for computation.
            batch_size: Suggested micro-batch size for chunked inference.

        Returns:
            (N, d_feat) Tensor of per-node features, or None if no features
            are produced.
        """

    @abstractmethod
    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        """Select/transform visible train rows for one target before prediction.

        Args:
            train_labels: 1-D float Tensor of visible train labels.
            train_f2ps: (num_train, F) LongTensor of f2p_nbr_idxs per train row.
            target_f2p: (F,) LongTensor, f2p_nbr_idxs of the target row.
            train_feats: (num_train, d_feat) Tensor or None (from compute_features).
            test_feat: (d_feat,) Tensor or None (from compute_features).

        Returns:
            3-tuple (train_feats, train_labels, test_feat) to pass to the
            predictor.  Any element may be None.
        """


# -------------------------------------------------------------------------- #
# global_featurizer
# -------------------------------------------------------------------------- #
@dataclass
class GlobalFeaturizerConfig:
    """Config for GlobalFeaturizer (no fields needed)."""

    def build(self, device):
        return GlobalFeaturizer()


class GlobalFeaturizer(Featurizer):
    """Pass all train rows through with no features (for global mean baseline)."""

    def compute_features(self, task, node_idxs, device, batch_size):
        return None

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        return None, train_labels, None


# -------------------------------------------------------------------------- #
# entity_featurizer
# -------------------------------------------------------------------------- #
@dataclass
class EntityFeaturizerConfig:
    """Config for EntityFeaturizer (no fields needed)."""

    def build(self, device):
        return EntityFeaturizer()


class EntityFeaturizer(Featurizer):
    """Pass only train rows sharing the target's foreign key entity."""

    def compute_features(self, task, node_idxs, device, batch_size):
        return None

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        match = (train_f2ps == target_f2p).all(dim=-1)
        matched_labels = train_labels[match]
        if len(matched_labels) == 0:
            return None, train_labels, None
        return None, matched_labels, None


# -------------------------------------------------------------------------- #
# rt_featurizer
# -------------------------------------------------------------------------- #
@dataclass
class RTFeaturizerConfig:
    """Config for RTFeaturizer.

    Fully self-contained: builds its own RT model, loads checkpoint, and
    creates samplers for all eval tasks.
    """

    # RT model params
    embedding_model: str
    d_text: int
    num_blocks: int
    d_model: int
    num_heads: int
    d_ff: int
    compile: bool
    materialize_attn_masks: bool
    load_ckpt_path: str | None

    # Sampler params
    ctx_size: int
    bfs_width: int
    eval_splits: list[str]
    pre_dir: str
    shuffle_seed: int
    context_seed: int
    # See rt.config.TrainConfig.vector_db_path.
    vector_db_path: str | None

    def build(self, device):
        return RTFeaturizer(
            embedding_model=self.embedding_model,
            d_text=self.d_text,
            num_blocks=self.num_blocks,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            compile=self.compile,
            materialize_attn_masks=self.materialize_attn_masks,
            load_ckpt_path=self.load_ckpt_path,
            device=device,
            eval_splits=self.eval_splits,
            pre_dir=self.pre_dir,
            ctx_size=self.ctx_size,
            bfs_width=self.bfs_width,
            shuffle_seed=self.shuffle_seed,
            context_seed=self.context_seed,
            vector_db_path=self.vector_db_path,
            db=None,
        )


class RTFeaturizer(Featurizer, nn.Module):
    """Build local contexts and produce masked-token embeddings from a RelationalTransformer.

    Fully self-contained: creates its own RT model, loads checkpoint, and
    builds samplers for all eval tasks.
    """

    def __init__(
        self,
        embedding_model,
        d_text,
        num_blocks,
        d_model,
        num_heads,
        d_ff,
        compile,
        materialize_attn_masks,
        load_ckpt_path,
        device,
        eval_splits,
        pre_dir,
        ctx_size,
        bfs_width,
        shuffle_seed,
        context_seed,
        vector_db_path,
        db,
    ):
        super().__init__()

        from rt.model import RelationalTransformer

        self.rt_model = RelationalTransformer(
            num_blocks=num_blocks,
            d_model=d_model,
            d_text=d_text,
            num_heads=num_heads,
            d_ff=d_ff,
            compile=compile,
            materialize_attn_masks=materialize_attn_masks,
        )
        if load_ckpt_path is not None:
            from rt.model import load_model

            raw = load_model(Path(load_ckpt_path).expanduser())
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}
            self.rt_model.load_state_dict(state_dict)
        self.rt_model.to(device).to(torch.bfloat16)
        self.rt_model.requires_grad_(False)
        self.rt_model.eval()

        from rt.data import RustlerDataset
        from rt.data import eval_tasks

        all_tasks = eval_tasks(pre_dir, splits=tuple(eval_splits))
        if db is not None:
            all_tasks = [t for t in all_tasks if db in t.db_name]

        self._samplers = {}
        for task in all_tasks:
            ds = RustlerDataset(
                tasks=[
                    (
                        task.db_name,
                        task.table_name,
                        task.target_column,
                        task.split,
                        task.leakage_columns,
                    )
                ],
                pre_dir=pre_dir,
                global_rank=0,
                local_rank=0,
                world_size=1,
                local_ctx_sizes=[ctx_size],
                bfs_widths=[bfs_width],
                num_walks=0,
                walk_length=0,
                prefer_latest=False,
                mask_prob_max=0.0,
                embedding_model=embedding_model,
                d_text=d_text,
                shuffle_seed=shuffle_seed,
                context_seed=context_seed,
                items_per_task=0,
                quiet=True,
                bool_as_num=True,
                ignore_data_errors=False,
                skip_text_cols=False,
                mmap_populate=False,
                balance_labels=False,
                timeout_per_item=3600.0,
                ablate_schema_semantics=False,
                vector_db_path=vector_db_path,
                train_only_fallback=False,
            )
            self._samplers[task] = ds.sampler

    def compute_features(self, task, node_idxs, device, batch_size):
        sampler = self._samplers[task]
        lc_tup = sampler.batch_for_nodes_py(
            node_idxs=node_idxs.cpu().tolist(),
            dataset_idx=0,
            ctx_size=sampler.local_ctx_size,
        )
        lc_batch = process_batch(lc_tup, sampler.d_text)
        lc_batch.pop("batch_mask", None)

        N = node_idxs.shape[0]
        chunks = []
        with torch.inference_mode():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                real_bs = end - start
                chunk = {
                    k: v[start:end].to(device, non_blocking=True)
                    for k, v in lc_batch.items()
                }
                # Pad last chunk to full batch_size to avoid recompilation
                if real_bs < batch_size:
                    pad_size = batch_size - real_bs
                    chunk = {
                        k: torch.cat(
                            [
                                v,
                                torch.zeros(
                                    pad_size,
                                    *v.shape[1:],
                                    dtype=v.dtype,
                                    device=v.device,
                                ),
                            ]
                        )
                        for k, v in chunk.items()
                    }
                embeddings = self.rt_model(
                    chunk, return_embeddings=True
                )  # (batch_size, S_max, d_model)
                is_targets = chunk["is_targets"][:real_bs]  # (real_bs, S_max)
                chunks.append(embeddings[:real_bs][is_targets])  # (real_bs, d_model)
        return torch.cat(chunks, dim=0)

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        return train_feats, train_labels, test_feat


# -------------------------------------------------------------------------- #
# rdblearn_featurizer
# -------------------------------------------------------------------------- #
@dataclass
class RDBLearnFeaturizerConfig:
    """Config for RDBLearnFeaturizer.

    DFS features are precomputed once at init for all rows per (db, table).
    ``compute_features`` just does a lookup by node index.
    """

    pre_dir: str
    eval_splits: list[str]
    max_depth: int
    max_train_samples: int

    def build(self, device):
        return RDBLearnFeaturizer(
            pre_dir=self.pre_dir,
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

    def __init__(self, pre_dir, eval_splits, max_depth, max_train_samples, db):
        import time

        import fastdfs
        import relbench.base
        from fastdfs import DFSConfig
        from rdblearn.config import RDBLearnConfig
        from rdblearn.datasets import RDBDataset
        from rdblearn.estimator import RDBLearnEstimator
        from rt.data import eval_tasks
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline

        from rt.rel2tab.featurizer import load_table_info

        # (db, table) -> (precomputed_features_tensor, min_offset)
        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}

        dfs_cfg = DFSConfig(
            max_depth=max_depth,
            agg_primitives=["max", "min", "mean", "count", "mode", "std"],
            engine="dfs2sql",
        )
        config = RDBLearnConfig(
            dfs=dfs_cfg,
            enable_target_augmentation=False,
            max_train_samples=(max_train_samples if max_train_samples > 0 else 10**9),
            predict_batch_size=5000,
        )

        all_tasks = eval_tasks(pre_dir, splits=tuple(eval_splits))
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

            # rdblearn expects bare dataset name (e.g. "rel-avito")
            rdb_name = db_name.removeprefix("relbench/")
            # fastdfs's RelBenchAdapter calls get_db() with no args; we need
            # the full db (not truncated at the test timestamp), so patch it
            # just for this call.
            # this allows up-to-date rows in the context window, which matters for rel-f1
            _orig_get_db = relbench.base.Dataset.get_db
            relbench.base.Dataset.get_db = lambda self, *args, **kwargs: _orig_get_db(
                self, upto_test_timestamp=False
            )
            dataset = RDBDataset.from_relbench(rdb_name)
            relbench.base.Dataset.get_db = _orig_get_db
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
            parts = [split_dfs[s].reset_index(drop=True) for s, _ in ordered]
            
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


# -------------------------------------------------------------------------- #
# precomputed_featurizer
# -------------------------------------------------------------------------- #
@dataclass
class PrecomputedFeaturizerConfig:
    """Config for PrecomputedFeaturizer.

    Reads features pre-computed by ``rt.rel2tab.featurize`` from disk.
    ``compute_features`` loads the binary vectors and does index lookup.
    """

    pre_dir: str
    eval_splits: list[str]
    features_subdir: str

    def build(self, device):
        return PrecomputedFeaturizer(
            pre_dir=self.pre_dir,
            eval_splits=self.eval_splits,
            features_subdir=self.features_subdir,
        )


class PrecomputedFeaturizer(Featurizer):
    """Load pre-computed feature vectors saved by ``rt.rel2tab.featurize``.

    At init, eagerly loads ``{table}_vectors.bin`` and ``{table}_meta.json``
    for every (db, table) pair referenced by the eval task set.  At eval time,
    ``compute_features`` does a fast index lookup.
    """

    def __init__(self, pre_dir, eval_splits, features_subdir):
        from rt.data import eval_tasks

        # (db, table) -> (features_tensor, min_offset)
        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}

        seen: set[tuple[str, str]] = set()
        for task in eval_tasks(pre_dir, splits=tuple(eval_splits)):
            key = (task.db_name, task.table_name)
            if key in seen:
                continue
            seen.add(key)
            db_name, table_name = key

            feat_dir = Path(pre_dir).expanduser() / db_name / features_subdir
            vectors_path = feat_dir / f"{table_name}_vectors.bin"
            meta_path = feat_dir / f"{table_name}_meta.json"

            import json

            with open(meta_path) as f:
                meta = json.load(f)

            n_features = meta["n_features"]
            min_offset = meta["min_offset"]
            total_nodes = meta["total_nodes"]

            vectors = np.fromfile(str(vectors_path), dtype=np.float32).reshape(
                total_nodes, n_features
            )
            feats_tensor = torch.from_numpy(vectors)

            self._features[key] = (feats_tensor, min_offset)
            print(
                f"  PrecomputedFeaturizer: loaded {db_name}/{table_name}"
                f" ({total_nodes} rows, {n_features} features)"
            )

    def compute_features(self, task, node_idxs, device, batch_size):
        key = (task.db_name, task.table_name)
        feats_tensor, min_offset = self._features[key]
        local_idxs = node_idxs.cpu() - min_offset
        return feats_tensor[local_idxs].to(device)

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        return train_feats, train_labels, test_feat


# -------------------------------------------------------------------------- #
# sql_featurizer: SQL-based featurizer for relbench tasks.
# -------------------------------------------------------------------------- #
@dataclass
class SQLFeaturizerConfig:
    """Config for SQLFeaturizer.

    Args:
        pre_dir: Directory containing preprocessed table_info.json files.
        eval_splits: task splits to load data for.
    """

    pre_dir: str
    eval_splits: list[str]

    def build(self, device):
        return SQLFeaturizer(
            pre_dir=self.pre_dir,
            eval_splits=self.eval_splits,
            db=None,
        )


def _run_sql_features(con, sql, task_df, key_cols):
    """Run feature SQL on task_df, return aligned feature array."""
    con.register("task_table", task_df)
    feats_df = con.execute(sql).df()
    feats_df = feats_df.drop_duplicates(subset=key_cols, keep="first")
    merged = task_df.merge(feats_df, on=key_cols, how="left")

    feat_cols = [c for c in feats_df.columns if c not in key_cols]
    if not feat_cols:
        return None, []

    arr = merged[feat_cols].values.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    np.nan_to_num(arr, copy=False, nan=0.0)
    return arr, feat_cols


class SQLFeaturizer(Featurizer):
    """Pre-computed SQL features with entity-aware row filtering.

    At init:
      1. Loads relbench datasets into DuckDB.
      2. Runs feature SQL once for all rows per task.
      3. Z-score normalizes globally and stores as a contiguous tensor
         indexed by ``node_idx - min_offset``.

    In ``compute_features``: simple index lookup (no SQL).

    In ``featurize``: filters train rows to those sharing the target's
    foreign-key entity, falling back to all train rows if no matches.
    """

    def __init__(self, pre_dir, eval_splits, db):
        import duckdb
        import pandas as pd
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task

        from rt.data import eval_tasks

        pre_dir = str(Path(pre_dir).expanduser())

        # Per-dataset DuckDB connections.
        connections: dict[str, duckdb.DuckDBPyConnection] = {}

        # (rt_db_name, rt_table_name) -> (features_tensor, min_offset)
        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}

        all_tasks = eval_tasks(pre_dir, splits=tuple(eval_splits))
        if db is not None:
            all_tasks = [t for t in all_tasks if db in t.db_name]

        seen: set[tuple[str, str]] = set()
        for task in all_tasks:
            key = (task.db_name, task.table_name)
            if key in seen:
                continue
            seen.add(key)

            ds = task.db_name.split("/")[-1]
            sql_key = (ds, task.table_name)
            if sql_key not in SQL_REGISTRY:
                raise ValueError(
                    f"No SQL features for {sql_key}."
                    f" Available: {list(SQL_REGISTRY.keys())}"
                )

            entry = SQL_REGISTRY[sql_key]
            sql = entry["sql"]
            entity_col = entry["entity_col"]
            time_col = entry["time_col"]

            # Load DB tables into DuckDB (once per dataset).
            if ds not in connections:
                con = duckdb.connect()
                con.execute("SET preserve_insertion_order=false")
                con.execute("SET threads=4")
                rb_dataset = get_dataset(ds, download=True)
                # this allows up-to-date rows in the context window, which matters for rel-f1
                rb_db = rb_dataset.get_db(upto_test_timestamp=False)
                for tbl_name, tbl in rb_db.table_dict.items():
                    con.register(tbl_name, tbl.df)
                connections[ds] = con
                print(
                    f"  SQLFeaturizer: loaded DB '{ds}' into DuckDB "
                    f"({len(rb_db.table_dict)} tables)"
                )

            con = connections[ds]

            # Build combined DataFrame aligned to node_idx offsets.
            table_info = load_table_info(task.db_name, pre_dir)
            splits_info: dict[str, dict] = {}
            for split in ["train", "val", "test"]:
                info_key = f"{task.table_name}:{split.capitalize()}"
                if info_key in table_info:
                    splits_info[split] = table_info[info_key]

            if not splits_info:
                raise KeyError(
                    f"Table '{task.table_name}' not in table_info.json for db "
                    f"'{task.db_name}'. Keys: {list(table_info.keys())}"
                )

            min_offset = min(info["node_idx_offset"] for info in splits_info.values())

            rb_task = get_task(ds, task.table_name, download=True)
            split_dfs = {
                "train": rb_task.get_table("train").df,
                "val": rb_task.get_table("val").df,
                "test": rb_task.get_table("test").df,
            }
            ordered = sorted(splits_info.items(), key=lambda x: x[1]["node_idx_offset"])
            parts = [split_dfs[s].reset_index(drop=True) for s, _ in ordered]
            combined_df = pd.concat(parts, ignore_index=True)

            # Pre-compute features for ALL rows.
            tic = time.time()
            key_cols = [time_col, entity_col]
            feat_array, feat_cols = _run_sql_features(con, sql, combined_df, key_cols)

            if feat_array is None:
                raise RuntimeError(
                    f"SQLFeaturizer: {ds}/{task.table_name} produced no features"
                )

            # Global z-score normalization.
            mean = feat_array.mean(axis=0, keepdims=True)
            std = feat_array.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            feat_array = (feat_array - mean) / std

            feat_tensor = torch.from_numpy(feat_array)
            self._features[key] = (feat_tensor, min_offset)

            elapsed = time.time() - tic
            print(
                f"  SQLFeaturizer: precomputed {ds}/{task.table_name} "
                f"({len(combined_df)} rows, {len(feat_cols)} features, "
                f"{elapsed:.1f}s)"
            )

        # Free DuckDB connections (no longer needed after precomputation).
        for con in connections.values():
            con.close()

    def compute_features(self, task, node_idxs, device, batch_size):
        key = (task.db_name, task.table_name)
        feat_tensor, min_offset = self._features[key]
        local_idxs = node_idxs.cpu() - min_offset
        return feat_tensor[local_idxs].to(device)

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        """Append per-row context features to the SQL features.

        For each row (train or test), computes:
          - same_entity: 1.0 if this row shares the target's FK entity, else 0.0
          - entity_mean: mean label of same-entity train rows
          - entity_count: log-scaled count of same-entity train rows
          - global_mean: mean label of all visible train rows
        """
        n_train = len(train_labels)
        if test_feat is None and (train_feats is None or n_train == 0):
            return train_feats, train_labels, test_feat

        # Everything runs on CPU (model.py moves tensors to CPU before
        # calling featurize / predict).
        global_mean = train_labels.mean().item() if n_train > 0 else 0.0

        # Entity match mask for train rows.
        target_match = (train_f2ps == target_f2p).all(dim=-1)
        n_match = target_match.sum().item()
        entity_mean = (
            train_labels[target_match].mean().item() if n_match > 0 else global_mean
        )
        log_entity_count = float(np.log1p(n_match))

        if train_feats is not None and n_train > 0:
            # Per-row: same_entity flag differs per train row.
            same_entity = target_match.float().unsqueeze(-1)
            shared = (
                torch.tensor(
                    [entity_mean, log_entity_count, global_mean],
                )
                .unsqueeze(0)
                .expand(n_train, -1)
            )
            train_feats = torch.cat([train_feats.cpu(), same_entity, shared], dim=-1)

        if test_feat is not None:
            test_ctx = torch.tensor(
                [1.0, entity_mean, log_entity_count, global_mean],
            )
            test_feat = torch.cat([test_feat.cpu(), test_ctx], dim=-1)

        return train_feats, train_labels, test_feat
