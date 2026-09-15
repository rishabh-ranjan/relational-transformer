import time

import torch
from torch import nn


def _fmt(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


class Rel2TabModel(nn.Module):
    def __init__(self, retriever, featurizer, predictor):
        super().__init__()
        self.retriever = retriever
        self.featurizer = featurizer
        self.predictor = predictor

    def predict(self, batch, eval_ctx_size_list, device, task):
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
