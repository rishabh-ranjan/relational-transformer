from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rt.rel2tab.featurizer import Featurizer


@dataclass
class PrecomputedFeaturizerConfig:
    """Config for PrecomputedFeaturizer.

    Reads features pre-computed by ``rt.rel2tab.featurize`` from disk.
    ``compute_features`` loads the binary vectors and does index lookup.
    """

    pre_dir: str
    eval_recipe: str
    features_subdir: str

    def build(self, device):
        return PrecomputedFeaturizer(
            pre_dir=self.pre_dir,
            eval_recipe=self.eval_recipe,
            features_subdir=self.features_subdir,
        )


class PrecomputedFeaturizer(Featurizer):
    """Load pre-computed feature vectors saved by ``rt.rel2tab.featurize``.

    At init, eagerly loads ``{table}_vectors.bin`` and ``{table}_meta.json``
    for every (db, table) pair referenced by the eval recipe.  At eval time,
    ``compute_features`` does a fast index lookup.
    """

    def __init__(self, pre_dir, eval_recipe, features_subdir):
        from rt.recipes import get_tasks

        # (db, table) -> (features_tensor, min_offset)
        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}

        seen: set[tuple[str, str]] = set()
        for task in get_tasks(eval_recipe, pre_dir):
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
