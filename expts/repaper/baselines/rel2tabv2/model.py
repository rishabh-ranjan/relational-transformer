import time

import torch
from torch import nn


def _fmt(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


class Rel2TabModel(nn.Module):
    def __init__(self, retriever, featurizer, predictor, labels=None):
        super().__init__()
        self.retriever = retriever
        self.featurizer = featurizer
        self.predictor = predictor
        # Only a query-independent retriever needs this: it returns indices for
        # rows that are not in the batch, so their labels have to be fetched.
        # The per-query retriever's context rows are in the batch already.
        self.labels = labels
        self._context = None

    def _shared_context(self, task):
        # Built once per task: the retriever's indices do not vary across
        # batches, so neither do the features and labels for them.
        if self._context is None:
            assert self.labels is not None, (
                "a query-independent retriever needs a label source"
            )
            self._context = {}
            for n, node_idxs in self.retriever.context_node_idxs().items():
                idx = torch.from_numpy(node_idxs.astype("int64"))
                feats = self.featurizer.compute_features(task, idx, "cpu")
                y = self.labels.labels_for(task, node_idxs)
                self._context[n] = (feats, y)
                print(
                    f"    rel2tab context: {task.db_name}/{task.table_name} "
                    f"n={n} rows, {feats.shape[1]} features, "
                    f"label mean {float(y.mean()):.4f} "
                    f"frac>0 {float((y > 0).float().mean()):.4f}",
                    flush=True,
                )
        return self._context

    def _predict_shared(self, batch, eval_ctx_size_list, task):
        bs = batch["is_targets"].size(0)

        tic = time.time()
        context = self._shared_context(task)
        queries = self.retriever.queries(batch, eval_ctx_size_list)
        query_features = self.featurizer.compute_features(
            task, queries.node_idxs, "cpu"
        )
        t_retrieve = time.time() - tic

        default = 0.5 if task.task_type == "clf" else 0.0
        preds = {}
        visible = queries.visible
        n_visible = int(visible.sum().item())

        tic = time.time()
        for c in eval_ctx_size_list:
            out = torch.full((bs,), default)
            out[queries.num_queries :] = 0.0
            if n_visible:
                train_features, train_labels = context[c]
                values = self.predictor.predict_shared(
                    train_features,
                    train_labels,
                    query_features[visible],
                    task.task_type,
                )
                assert len(values) == n_visible, (
                    f"predict_shared returned {len(values)} values for "
                    f"{n_visible} queries"
                )
                out[visible] = torch.tensor(values, dtype=out.dtype)
            preds[c] = out
        t_predict = time.time() - tic

        print(
            f"    rel2tab: {n_visible} queries, one shared context |"
            f" retrieve {_fmt(t_retrieve)}"
            f" predict {_fmt(t_predict)}",
            flush=True,
        )
        return preds

    def predict(self, batch, eval_ctx_size_list, device, task):
        # A query-independent retriever hands back one context for the whole
        # batch, so the predictor fits once instead of once per query and the
        # context's features were computed when the retriever was built.
        if getattr(self.retriever, "query_independent", False):
            return self._predict_shared(batch, eval_ctx_size_list, task)

        bs = batch["is_targets"].size(0)

        tic = time.time()
        ctx = self.retriever.retrieve(batch, eval_ctx_size_list, task)
        t_retrieve = time.time() - tic

        if ctx.node_idxs.numel() == 0:
            return {c: torch.zeros(bs) for c in eval_ctx_size_list}

        tic = time.time()
        features = self.featurizer.compute_features(task, ctx.node_idxs, "cpu")
        t_featurize = time.time() - tic

        tic = time.time()
        default = 0.5 if task.task_type == "clf" else 0.0
        preds = {c: torch.full((bs,), default) for c in eval_ctx_size_list}
        for c in eval_ctx_size_list:
            preds[c][ctx.num_queries :] = 0.0

        work_items = []
        slots = []
        for c in eval_ctx_size_list:
            for q, selected in enumerate(ctx.per_ctx[c]):
                if selected is None:
                    continue
                train_pos, query_pos = selected
                work_items.append(
                    (
                        features[train_pos],
                        ctx.labels[train_pos],
                        features[query_pos],
                        task.task_type,
                    )
                )
                slots.append((c, q))

        if work_items:
            for (c, q), value in zip(slots, self.predictor.predict_batch(work_items)):
                preds[c][q] = value
        t_predict = time.time() - tic

        print(
            f"    rel2tab: {ctx.node_idxs.shape[0]} task cells |"
            f" retrieve {_fmt(t_retrieve)}"
            f" featurize {_fmt(t_featurize)}"
            f" predict {_fmt(t_predict)}",
            flush=True,
        )
        return preds

    def forward(self, batch, return_embeddings):
        raise NotImplementedError("Rel2TabModel is eval-only; use predict().")
