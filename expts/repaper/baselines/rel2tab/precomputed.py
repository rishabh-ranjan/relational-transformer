"""Eval-time lookup of features precomputed by the featurize scripts."""

import json
from pathlib import Path

import numpy as np
import torch

from expts.repaper.baselines.rel2tab.featurizer import Featurizer


class PrecomputedFeaturizer(Featurizer):
    """Load per-table feature blobs written by the featurize scripts.

    Reads ``<features_root>/<db>/<subdir>/<table>_vectors.bin`` (row-major
    float32) + ``<table>_meta.json`` for every (db, table) pair, eagerly at
    init; ``compute_features`` is then an index lookup by
    ``node_idx - min_offset``. ``featurize`` passes rows through unchanged --
    the paper's precomputed baselines use the on-disk features as-is.
    """

    def __init__(self, features_root, features_subdir, db_tables):
        # (db, table) -> (features_tensor, min_offset)
        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}
        for db, table in sorted(set(db_tables)):
            feat_dir = Path(features_root).expanduser() / db / features_subdir
            with open(feat_dir / f"{table}_meta.json") as f:
                meta = json.load(f)
            vectors = np.fromfile(
                feat_dir / f"{table}_vectors.bin", dtype=np.float32
            ).reshape(meta["total_nodes"], meta["n_features"])
            self._features[db, table] = (
                torch.from_numpy(vectors),
                meta["min_offset"],
            )
            print(
                f"PrecomputedFeaturizer: {db}/{table} "
                f"({meta['total_nodes']} rows, {meta['n_features']} features)",
                flush=True,
            )

    def compute_features(self, task, node_idxs, device):
        feats, min_offset = self._features[task.db_name, task.table_name]
        return feats[node_idxs.cpu() - min_offset].to(device)

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        return train_feats, train_labels, test_feat
