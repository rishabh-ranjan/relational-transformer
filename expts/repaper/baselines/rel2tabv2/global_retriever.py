from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from expts.repaper.baselines.rel2tab.featurizer import (
    get_table_splits,
    load_table_info,
    table_offset_and_len,
)

BATCH_KEYS = ("is_targets", "node_idxs", "number_values")


@dataclass
class SharedContext:
    num_queries: int
    query_features: torch.Tensor
    per_ctx: dict[int, tuple[torch.Tensor, torch.Tensor]]
    visible: torch.Tensor


class GlobalContextRetriever:
    query_independent = True

    def __init__(
        self,
        sampler,
        featurizer,
        db,
        table,
        pre_dir,
        raw_dir,
        context_split,
        n_rows_list,
    ):
        self.featurizer = featurizer
        self.db = db
        self.table = table

        min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
        splits = get_table_splits(load_table_info(pre_dir, db), table)
        assert context_split in splits, (
            f"{db}/{table} has no {context_split!r} split; got {sorted(splits)}"
        )
        info = splits[context_split]
        lo = info["node_idx_offset"]
        candidates = np.arange(lo, lo + info["num_nodes"])

        ordered = sorted(splits.items(), key=lambda kv: kv[1]["node_idx_offset"])
        rows = pd.concat(
            [
                pd.read_parquet(
                    Path(raw_dir).expanduser() / db / "tasks" / table / f"{s}.parquet"
                ).reset_index(drop=True)
                for s, _ in ordered
            ],
            ignore_index=True,
        )
        assert len(rows) == total_nodes, (
            f"{db}/{table}: {len(rows)} task rows vs {total_nodes} nodes in "
            f"table_info.json"
        )
        self._min_offset = min_offset
        self._labels_by_row = rows
        self._candidates = candidates

        # One draw at the largest size, then prefixes, so the sizes are nested:
        # the n=64 context is a subset of the n=256 one and a curve over them
        # varies only in how much context there is, not in which rows it is.
        self.n_rows_list = sorted(n_rows_list)
        drawn = sampler.sample(candidates, self.n_rows_list[-1])

        self._ctx_node_idxs = {n: np.sort(drawn[:n]) for n in self.n_rows_list}
        self._ctx_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _label_column(self, task):
        target = task.target_column
        assert target in self._labels_by_row.columns, (
            f"{self.db}/{self.table}: task rows have no target column "
            f"{target!r}; got {list(self._labels_by_row.columns)}"
        )
        return self._labels_by_row[target]

    def prepare(self, task):
        # Raw 0/1 labels from the task parquet binarise identically to the
        # sampler's z-scored ones under the predictors' `> 0` test, so clf is
        # exact. Regression is not: the arms fit on normalised targets and
        # rt.eval.relbench denormalises with the train statistics, so raw
        # targets here would put the predictions in the wrong space.
        assert task.task_type == "clf", (
            f"{self.db}/{self.table} is {task.task_type}; the global retriever "
            f"reads context labels from the task parquet, which is only in the "
            f"same space as the sampler's labels for binary classification"
        )
        labels = self._label_column(task)
        for n, node_idxs in self._ctx_node_idxs.items():
            if n in self._ctx_cache:
                continue
            feats = self.featurizer.compute_features(
                task, torch.from_numpy(node_idxs.astype(np.int64)), "cpu"
            )
            y = torch.from_numpy(
                labels.to_numpy()[node_idxs - self._min_offset].astype(np.float32)
            )
            self._ctx_cache[n] = (feats, y)
            print(
                f"GlobalContextRetriever: {self.db}/{self.table} n={n} "
                f"context rows, {feats.shape[1]} features, "
                f"positive rate {float((y > 0).float().mean()):.3f}",
                flush=True,
            )

    def retrieve(self, batch, ctx_size_list, task):
        if not self._ctx_cache:
            self.prepare(task)
        # subset, not equality: evaluator.mem_guard probes predict() with just
        # [max(ctx_size_list)].
        unknown = sorted(set(ctx_size_list) - set(self.n_rows_list))
        assert not unknown, (
            f"this retriever drew contexts for {self.n_rows_list} but the "
            f"evaluator asked for {unknown}; for the global retriever these are "
            f"row counts, not cells"
        )

        cpu = {k: batch[k].cpu() for k in BATCH_KEYS}
        is_targets = cpu["is_targets"]
        num_queries = int(is_targets.any(dim=1).sum().item())
        b_idxs, s_idxs = is_targets.nonzero(as_tuple=True)

        query_nodes = torch.full((is_targets.shape[0],), -1, dtype=torch.long)
        query_nodes[b_idxs] = cpu["node_idxs"][b_idxs, s_idxs].long()
        visible = query_nodes >= 0

        feats = self.featurizer.compute_features(
            task, query_nodes.clamp_min(self._min_offset), "cpu"
        )
        return SharedContext(
            num_queries=num_queries,
            query_features=feats,
            per_ctx={n: self._ctx_cache[n] for n in self.n_rows_list},
            visible=visible,
        )
