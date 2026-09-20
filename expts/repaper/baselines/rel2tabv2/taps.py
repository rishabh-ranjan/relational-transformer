import json
from pathlib import Path

import ml_dtypes
import numpy as np
import torch


class TapsFeaturizer:
    def __init__(self, features_root, db_tables, tap_unit, subset, chunk_rows=8192):
        assert subset in ("full", "target", "other"), f"unknown subset {subset!r}"
        self.tap_unit = tap_unit
        self.subset = subset
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
            live = [s for s in range(meta["shape"][2]) if s not in dead]
            tgt = meta["target_slot"]
            assert tgt in live, f"{db}/{table}: the target slot is dead"
            # full  = every live column of the union set
            # target= the target cell alone, which is what features_rt-j holds
            #         (modulo norm_out) -- the control for "does the rest of
            #         the row add anything"
            # other = the union set without the target cell, which asks the
            #         complement: how much is in the row around it
            keep = {
                "full": live,
                "target": [tgt],
                "other": [s for s in live if s != tgt],
            }[subset]
            assert keep, f"{db}/{table}: subset {subset!r} selects no slots"
            sel = np.array(keep)
            self._f[db, table] = (arr, ti, sel, meta["min_offset"])
            print(
                f"TapsFeaturizer: {db}/{table} unit {tap_unit} subset {subset} "
                f"({meta['shape'][0]} rows, {len(sel)} of {len(live)} live "
                f"slots, {len(sel) * meta['shape'][3]} features), "
                f"target slot {tgt}, dead {sorted(meta['slots'][s] for s in dead)}",
                flush=True,
            )

    def compute_features(self, task, node_idxs, device):
        arr, ti, sel, min_offset = self._f[task.db_name, task.table_name]
        idx = node_idxs.cpu().numpy().astype(np.int64) - min_offset
        n, d_model = len(idx), arr.shape[3]
        out = np.empty((n, len(sel) * d_model), dtype=np.float32)
        # Chunked: a fancy index over the whole blob would materialise every
        # tap and every slot for the context, which is tens of GiB at a large
        # context size. NaN is carried through -- it is what marks a cell the
        # database does not have.
        for lo in range(0, n, self.chunk_rows):
            rows = idx[lo : lo + self.chunk_rows]
            block = np.asarray(arr[rows, ti][:, sel, :]).astype(np.float32)
            out[lo : lo + len(rows)] = block.reshape(len(rows), -1)
        return torch.from_numpy(out).to(device)
