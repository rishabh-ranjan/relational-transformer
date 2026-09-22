import json
from pathlib import Path

import numpy as np
import torch


class AdapterFeaturizer:
    def __init__(self, features_root, db_tables, adapter_name, adapter_root,
                 chunk_rows=65536):
        from expts.repaper.adapter.train_adapter import build_adapter

        self.name = adapter_name
        if adapter_name == "identity":
            adapter, step = None, None
        else:
            path = Path(adapter_root).expanduser() / f"{adapter_name}.pt"
            assert path.is_file(), f"no adapter checkpoint at {path}"
            ck = torch.load(path, map_location="cpu", weights_only=True)
            step = ck["step"]
            adapter = build_adapter(
                ck["adapter_kind"],
                ck["state_dict"]["0.mean"].shape[0],
                ck["d_out"],
                ck["hidden_dim"],
                ck["stats_path"],
                "cpu",
            )
            adapter.load_state_dict(ck["state_dict"], strict=True)
            adapter.eval()

        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}
        for db, table in sorted(set(db_tables)):
            d = Path(features_root).expanduser() / db / "rt_features"
            meta = json.loads((d / f"{table}_meta.json").read_text())
            vectors = torch.from_numpy(
                np.fromfile(d / f"{table}_vectors.bin", dtype=np.float32).reshape(
                    meta["total_nodes"], meta["n_features"]
                )
            )
            if adapter is not None:
                with torch.no_grad():
                    vectors = torch.cat(
                        [
                            adapter(vectors[i : i + chunk_rows])
                            for i in range(0, vectors.shape[0], chunk_rows)
                        ]
                    ).contiguous()
                assert torch.isfinite(vectors).all(), f"{db}/{table}: non-finite adapter output"
            self._features[db, table] = (vectors, meta["min_offset"])
            print(
                f"AdapterFeaturizer[{adapter_name}"
                f"{'' if step is None else f' step {step}'}]: {db}/{table} "
                f"({meta['total_nodes']} rows, {meta['n_features']} -> {vectors.shape[1]})",
                flush=True,
            )

    def compute_features(self, task, node_idxs, device):
        feats, min_offset = self._features[task.db_name, task.table_name]
        return feats[node_idxs.cpu() - min_offset].to(device)
