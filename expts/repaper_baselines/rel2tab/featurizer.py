"""Featurizer interface + table-metadata helpers shared by the featurize
scripts and the eval-time feature lookup."""

import json
from abc import ABC, abstractmethod
from pathlib import Path


def load_table_info(pre_dir: str, db: str) -> dict:
    """Load ``<pre_dir>/<db>/table_info.json``."""
    path = Path(pre_dir).expanduser() / db / "table_info.json"
    with open(path) as f:
        return json.load(f)


def get_table_splits(table_info: dict, table_name: str) -> dict[str, dict]:
    """``{split: {node_idx_offset, num_nodes}}`` for the splits the table has."""
    splits = {}
    for split in ["train", "val", "test", "db"]:
        key = f"{table_name}:{split.capitalize()}"
        if key in table_info:
            splits[split] = table_info[key]
    return splits


def table_offset_and_len(pre_dir: str, db: str, table_name: str) -> tuple[int, int]:
    """(min node_idx offset, total rows) of a task table across its splits.

    Feature files are indexed ``node_idx - min_offset``, so the splits must be
    contiguous in node-index space; assert it rather than produce misaligned
    features.
    """
    splits_info = get_table_splits(load_table_info(pre_dir, db), table_name)
    assert splits_info, f"{db}/{table_name} not in table_info.json"
    sorted_offsets = sorted(
        (info["node_idx_offset"], info["num_nodes"]) for info in splits_info.values()
    )
    for (off, n), (nxt, _) in zip(sorted_offsets, sorted_offsets[1:]):
        assert off + n == nxt, (
            f"non-contiguous node_idxs across splits for {db}/{table_name}: "
            f"{sorted_offsets}"
        )
    min_offset = sorted_offsets[0][0]
    total = sum(n for _, n in sorted_offsets)
    return min_offset, total


class Featurizer(ABC):
    """How task-table rows become feature vectors for the predictor.

    Lifecycle within ``Rel2TabModel.predict``:

    1. ``compute_features`` once per batch with all task-node indices seen in
       the batch's contexts (bulk lookup or bulk computation).
    2. ``featurize`` once per (batch item, ctx size) with the train rows
       visible at that prefix, to select/transform rows for one target.
    """

    @abstractmethod
    def compute_features(self, task, node_idxs, device):
        """(N, d_feat) features for ``node_idxs`` (1-D LongTensor), or None."""

    @abstractmethod
    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        """(train_feats, train_labels, test_feat) to hand the predictor."""
