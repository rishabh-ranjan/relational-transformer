"""Rel2TabModel: a (featurizer, predictor) pair behind the ``rt.eval``
evaluator's model contract."""

import time

import torch
from torch import nn

from rt.model.net import SEM_TYPE_BOOLEAN


def _fmt(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


class Rel2TabModel(nn.Module):
    """Label-matched tabular baseline over RT's eval contexts.

    ``predict(batch, eval_ctx_size_list, device, task)`` matches
    ``rt.model.net.RelationalTransformer.predict``, so
    ``rt.eval.evaluator.Evaluator.evaluate_raw`` drives it unchanged. For each
    row it takes the task-table rows visible in the row's context at each
    requested ctx-size prefix as the train set, features them, and predicts the
    target -- so the baseline consumes exactly the labels RT saw.
    """

    def __init__(self, featurizer, predictor):
        super().__init__()
        self.featurizer = featurizer
        self.predictor = predictor

    def _extract_task_nodes(self, batch):
        """Unique task-node label cells across the batch.

        Returns (item_idxs, positions, node_idxs, labels, is_target, f2ps), all
        length-N 1-D tensors: one entry per cell of the target column on a task
        node (the target cell itself included, flagged by ``is_target``).
        """
        B = batch["is_targets"].shape[0]
        node_idxs = batch["node_idxs"]
        col_name_idxs = batch["col_name_idxs"]

        b_idxs, s_idxs = batch["is_targets"].nonzero(as_tuple=True)
        if len(b_idxs) == 0:
            empty_long = torch.empty(0, dtype=torch.long)
            return (
                empty_long,
                empty_long,
                empty_long,
                torch.empty(0, dtype=torch.float),
                torch.empty(0, dtype=torch.bool),
                torch.empty((0, batch["f2p_nbr_idxs"].shape[-1]), dtype=torch.long),
            )
        target_col = col_name_idxs[b_idxs[0], s_idxs[0]]
        target_node_per_b = torch.full((B,), -1, dtype=node_idxs.dtype)
        target_node_per_b[b_idxs] = node_idxs[b_idxs, s_idxs]

        is_label_cell = (
            batch["is_task_nodes"]
            & ~batch["is_padding"]
            & (col_name_idxs == target_col)
        )
        lc_b, lc_s = is_label_cell.nonzero(as_tuple=True)
        # The label lives in the channel its semantic type names (same rule as
        # Evaluator.evaluate_raw's label read).
        vals = torch.where(
            (batch["sem_types"] == SEM_TYPE_BOOLEAN).unsqueeze(-1),
            batch["boolean_values"],
            batch["number_values"],
        ).squeeze(-1)
        return (
            lc_b,
            lc_s,
            node_idxs[lc_b, lc_s],
            vals[lc_b, lc_s].float(),
            node_idxs[lc_b, lc_s] == target_node_per_b[lc_b],
            batch["f2p_nbr_idxs"][lc_b, lc_s],
        )

    def _predict_per_ctx(
        self,
        eval_ctx_sizes,
        true_bs,
        task_type,
        item_idxs,
        positions,
        labels,
        is_target,
        f2ps,
        features,
    ):
        """{ctx_size: (true_bs,) predictions}. One work item per (row, ctx)."""
        preds = {ctx: torch.zeros(true_bs) for ctx in eval_ctx_sizes}
        default = 0.5 if task_type == "clf" else 0.0
        has_feats = features is not None

        work_items = []  # (ctx, b, train_feats, train_labels, test_feat)
        for b in range(true_bs):
            b_mask = item_idxs == b
            if not b_mask.any():
                for ctx in eval_ctx_sizes:
                    preds[ctx][b] = default
                continue
            b_positions = positions[b_mask]
            b_labels = labels[b_mask]
            b_is_target = is_target[b_mask]
            b_f2ps = f2ps[b_mask]
            b_feats = features[b_mask] if has_feats else None

            target_idx = b_is_target.nonzero(as_tuple=True)[0]
            if len(target_idx) == 0:
                for ctx in eval_ctx_sizes:
                    preds[ctx][b] = default
                continue
            target_idx = target_idx[0].item()
            target_f2p = b_f2ps[target_idx]
            test_feat = b_feats[target_idx] if has_feats else None

            for ctx in eval_ctx_sizes:
                visible = b_positions < ctx
                if not visible[target_idx]:
                    preds[ctx][b] = default
                    continue
                train_mask = visible & ~b_is_target
                train_feats = (
                    b_feats[train_mask] if has_feats and train_mask.any() else None
                )
                ft, lt, tt = self.featurizer.featurize(
                    b_labels[train_mask],
                    b_f2ps[train_mask],
                    target_f2p,
                    train_feats,
                    test_feat,
                )
                work_items.append((ctx, b, ft, lt, tt))

        if not work_items:
            return preds

        if hasattr(self.predictor, "predict_batch"):
            results = self.predictor.predict_batch(
                [(ft, lt, tt, task_type) for _ctx, _b, ft, lt, tt in work_items]
            )
            for (ctx, b, _ft, _lt, _tt), pred_val in zip(work_items, results):
                preds[ctx][b] = pred_val
        else:
            for ctx, b, ft, lt, tt in work_items:
                preds[ctx][b] = self.predictor.predict(ft, lt, tt, task_type)
        return preds

    def predict(self, batch, eval_ctx_size_list, device, task):
        """Predictions at every requested ctx size, ``(bs,)`` each.

        Rustler lays real rows at indices 0..true_bs-1 and pads the rest as
        phantoms; phantom slots stay 0 and the caller drops them by
        ``batch_mask``.
        """
        bs = batch["is_targets"].size(0)
        true_bs = int(batch["is_targets"].any(dim=1).sum().item())
        cpu_batch = {
            k: batch[k].cpu()
            for k in (
                "is_targets",
                "node_idxs",
                "col_name_idxs",
                "is_task_nodes",
                "is_padding",
                "sem_types",
                "boolean_values",
                "number_values",
                "f2p_nbr_idxs",
            )
        }

        tic = time.time()
        item_idxs, positions, node_idxs, labels, is_target, f2ps = (
            self._extract_task_nodes(cpu_batch)
        )
        t_extract = time.time() - tic

        if item_idxs.shape[0] == 0:
            return {ctx: torch.zeros(bs) for ctx in eval_ctx_size_list}

        tic = time.time()
        features = self.featurizer.compute_features(task, node_idxs, "cpu")
        t_featurize = time.time() - tic

        tic = time.time()
        dense = self._predict_per_ctx(
            eval_ctx_size_list,
            true_bs,
            task.task_type,
            item_idxs,
            positions,
            labels,
            is_target,
            f2ps,
            features,
        )
        preds = {}
        for ctx, p in dense.items():
            full = torch.zeros(bs)
            full[:true_bs] = p
            preds[ctx] = full
        t_predict = time.time() - tic

        print(
            f"    rel2tab: {item_idxs.shape[0]} task cells |"
            f" extract {_fmt(t_extract)}"
            f" featurize {_fmt(t_featurize)}"
            f" predict {_fmt(t_predict)}",
            flush=True,
        )
        return preds

    def forward(self, batch, return_embeddings):
        raise NotImplementedError("Rel2TabModel is eval-only; use predict().")
