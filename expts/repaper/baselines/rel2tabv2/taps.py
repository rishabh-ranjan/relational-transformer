import json
from pathlib import Path

import ml_dtypes
import numpy as np
import torch


class TapsFeaturizer:
    def __init__(
        self, features_root, db_tables, tap_unit, subset, proj_dim=512, proj_seed=0,
        chunk_rows=8192,
    ):
        assert subset in ("full", "target", "other", "proj"), (
            f"unknown subset {subset!r}"
        )
        self.tap_unit = tap_unit
        self.subset = subset
        self.proj_dim = proj_dim
        self.proj_seed = proj_seed
        self.chunk_rows = chunk_rows
        self._proj: dict[tuple[str, str], torch.Tensor] = {}
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
                "proj": live,
                "target": [tgt],
                "other": [s for s in live if s != tgt],
            }[subset]
            assert keep, f"{db}/{table}: subset {subset!r} selects no slots"
            sel = np.array(keep)
            self._f[db, table] = (arr, ti, sel, meta["min_offset"])
            if subset == "proj":
                # One Xavier-initialised matrix per table, drawn from a fixed
                # seed and never trained: the same projection for every row of
                # that dataset, and reproducible from the seed alone rather
                # than being an artifact to keep.
                n_in = len(sel) * meta["shape"][3]
                g = torch.Generator().manual_seed(proj_seed)
                w = torch.empty(n_in, proj_dim)
                torch.nn.init.xavier_uniform_(w, generator=g)
                self._proj[db, table] = w.requires_grad_(False)
                print(
                    f"TapsFeaturizer: {db}/{table} random projection "
                    f"{n_in} -> {proj_dim}, xavier_uniform seed {proj_seed}, "
                    f"bound {(6.0 / (n_in + proj_dim)) ** 0.5:.5f}",
                    flush=True,
                )
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
        x = torch.from_numpy(out)
        if self.subset == "proj":
            # A dense matmul cannot carry a missing-value mask: one NaN in a
            # row makes every one of the 512 outputs NaN, which on a table
            # like study-outcome (42% of cells absent) would be almost every
            # row. So missing cells become 0 before the projection, and this
            # arm alone gives up the NaN-as-missing signal the others keep.
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x @ self._proj[task.db_name, task.table_name]
        return x.to(device)
