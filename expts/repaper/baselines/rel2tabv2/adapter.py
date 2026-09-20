import json
from pathlib import Path

import numpy as np
import torch


class AdapterFeaturizer:
    def __init__(self, features_root, db_tables, adapter_name, adapter_root,
                 chunk_rows=65536):
        # rt_features is the post-norm_out unit-12 target cell, which is exactly
        # what the Join dump the adapter trained on holds. "identity" is the
        # control arm: the same code path with W = I, so any difference from the
        # rt-j baseline is the predictor's preprocessing and not the adapter.
        self.chunk_rows = chunk_rows
        self.name = adapter_name
        if adapter_name == "identity":
            self.w, self.b, step = None, None, None
        else:
            path = Path(adapter_root).expanduser() / f"{adapter_name}.pt"
            assert path.is_file(), f"no adapter checkpoint at {path}"
            ck = torch.load(path, map_location="cpu", weights_only=True)
            sd = ck["state_dict"]
            # nn.Linear stores (out, in) and applies x @ W.T; transposing once
            # here keeps the per-batch matmul a plain (rows, in) @ (in, out).
            self.w = sd["weight"].float().t().contiguous()
            self.b = sd["bias"].float()
            step = ck.get("step")

        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}
        for db, table in sorted(set(db_tables)):
            d = Path(features_root).expanduser() / db / "rt_features"
            meta = json.loads((d / f"{table}_meta.json").read_text())
            vectors = np.fromfile(
                d / f"{table}_vectors.bin", dtype=np.float32
            ).reshape(meta["total_nodes"], meta["n_features"])
            if self.w is not None:
                assert meta["n_features"] == self.w.shape[0], (
                    f"{db}/{table} has {meta['n_features']} features but the "
                    f"adapter takes {self.w.shape[0]}"
                )
            self._features[db, table] = (
                torch.from_numpy(vectors),
                meta["min_offset"],
            )
            n_out = meta["n_features"] if self.w is None else self.w.shape[1]
            print(
                f"AdapterFeaturizer[{adapter_name}"
                f"{'' if step is None else f' step {step}'}]: {db}/{table} "
                f"({meta['total_nodes']} rows, {meta['n_features']} -> {n_out})",
                flush=True,
            )

    def compute_features(self, task, node_idxs, device):
        feats, min_offset = self._features[task.db_name, task.table_name]
        rows = feats[node_idxs.cpu() - min_offset]
        if self.w is None:
            return rows.to(device)
        out = torch.empty(rows.shape[0], self.w.shape[1], dtype=torch.float32)
        for i in range(0, rows.shape[0], self.chunk_rows):
            j = i + self.chunk_rows
            torch.addmm(self.b, rows[i:j], self.w, out=out[i:j])
        return out.to(device)
