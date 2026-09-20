import json
from pathlib import Path

import ml_dtypes
import numpy as np
import torch


class TapsFeaturizer:
    def __init__(self, features_root, db_tables, tap_unit, chunk_rows=8192):
        self.tap_unit = tap_unit
        self.chunk_rows = chunk_rows
        self._f: dict[tuple[str, str], tuple] = {}
        for db, table in sorted(set(db_tables)):
            d = Path(features_root).expanduser() / db / "rt_taps"
            meta = json.loads((d / f"{table}_meta.json").read_text())
            arr = np.memmap(
                d / f"{table}_taps.bin",
                dtype=ml_dtypes.bfloat16,
                mode="r",
                shape=tuple(meta["shape"]),
            )
            ti = meta["taps"].index(tap_unit)
            dead = set(meta["dead_slots"])
            # Dead slots are key columns -- rt carries a key as an edge, never
            # as a value cell, so those slots are NaN on every row. Dropping
            # them here keeps them out of tabpfn's feature budget, which is
            # what max_features_per_estimator is spent against.
            live = np.array([s for s in range(meta["shape"][2]) if s not in dead])
            self._f[db, table] = (arr, ti, live, meta["min_offset"])
            print(
                f"TapsFeaturizer: {db}/{table} unit {tap_unit} "
                f"({meta['shape'][0]} rows, {len(live)} live of "
                f"{meta['shape'][2]} slots, {len(live) * meta['shape'][3]} "
                f"features), dropped {sorted(meta['slots'][s] for s in dead)}",
                flush=True,
            )

    def compute_features(self, task, node_idxs, device):
        arr, ti, live, min_offset = self._f[task.db_name, task.table_name]
        idx = node_idxs.cpu().numpy().astype(np.int64) - min_offset
        n, d_model = len(idx), arr.shape[3]
        out = np.empty((n, len(live) * d_model), dtype=np.float32)
        # Chunked: a fancy index over the whole blob would materialise every
        # tap and every slot for the context, which is tens of GiB at a large
        # context size. NaN is carried through -- it is what marks a cell the
        # database does not have.
        for lo in range(0, n, self.chunk_rows):
            rows = idx[lo : lo + self.chunk_rows]
            block = np.asarray(arr[rows, ti][:, live, :]).astype(np.float32)
            out[lo : lo + len(rows)] = block.reshape(len(rows), -1)
        return torch.from_numpy(out).to(device)
