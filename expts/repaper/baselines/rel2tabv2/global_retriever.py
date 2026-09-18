from dataclasses import dataclass

import numpy as np
import torch

from expts.repaper.baselines.rel2tab.featurizer import (
    get_table_splits,
    load_table_info,
    table_offset_and_len,
)

BATCH_KEYS = ("is_targets", "node_idxs")


@dataclass
class QueryRows:
    num_queries: int
    node_idxs: torch.Tensor
    visible: torch.Tensor


class GlobalContextRetriever:
    query_independent = True

    def __init__(
        self, sampler, db, table, pre_dir, context_split, n_rows_list, ctx_keys
    ):
        self.db = db
        self.table = table

        min_offset, _total = table_offset_and_len(pre_dir, db, table)
        splits = get_table_splits(load_table_info(pre_dir, db), table)
        assert context_split in splits, (
            f"{db}/{table} has no {context_split!r} split; got {sorted(splits)}"
        )
        info = splits[context_split]
        lo = info["node_idx_offset"]
        candidates = np.arange(lo, lo + info["num_nodes"])

        self.min_offset = min_offset

        # ctx_keys are what the *evaluator* calls these contexts, which is not
        # the same number as how many rows they hold. The evaluator's context
        # size is a per-query cell count that this retriever discards, and the
        # rustler sampler pads every batch to it -- 3.2 GiB for eight queries at
        # 262144 -- so a run asks the evaluator for something small and keys the
        # real row count under it. For the per-query retriever the two coincide.
        assert len(ctx_keys) == len(n_rows_list), (
            f"{len(ctx_keys)} ctx_keys for {len(n_rows_list)} row counts"
        )
        assert len(set(ctx_keys)) == len(ctx_keys), f"duplicate ctx_keys {ctx_keys}"
        pairs = sorted(zip(ctx_keys, n_rows_list), key=lambda kv: kv[1])
        self.n_rows_list = [k for k, _ in pairs]

        # A task with fewer train rows than the requested context gets all of
        # them: asking for 2**18 rows of rel-f1/driver-top3, which has 1353, is
        # a smaller experiment, not an error. The sampler itself still refuses
        # to over-draw, so the clamp is here where the pool is known.
        self.requested_rows = dict(pairs)
        self.effective_rows = {k: min(n, len(candidates)) for k, n in pairs}
        # One draw at the largest size, then prefixes, so the sizes are nested:
        # the n=64 context is a subset of the n=256 one and a curve over them
        # varies only in how much context there is, not in which rows. Sorting
        # is per level, after the prefix -- sorting the draw first would make
        # every smaller level the lowest node indices, which is the earliest
        # rows in time rather than a sample of them.
        drawn = sampler.sample(candidates, max(self.effective_rows.values()))
        self._context = {k: np.sort(drawn[: self.effective_rows[k]]) for k, _ in pairs}
        for k, _ in pairs:
            req, eff = self.requested_rows[k], self.effective_rows[k]
            note = (
                "" if req == eff else f"  (clamped from {req}: only {eff} train rows)"
            )
            print(
                f"    GlobalContextRetriever: ctx key {k} -> {eff} rows{note}",
                flush=True,
            )

    def context_node_idxs(self):
        return self._context

    def queries(self, batch, ctx_size_list):
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

        node_idxs = torch.full((is_targets.shape[0],), -1, dtype=torch.long)
        node_idxs[b_idxs] = cpu["node_idxs"][b_idxs, s_idxs].long()
        return QueryRows(
            num_queries=num_queries,
            node_idxs=node_idxs.clamp_min(self.min_offset),
            visible=node_idxs >= 0,
        )
