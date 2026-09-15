from dataclasses import dataclass

import torch

BATCH_KEYS = (
    "is_targets",
    "node_idxs",
    "col_name_idxs",
    "is_task_nodes",
    "is_padding",
    "number_values",
)


@dataclass
class Context:
    num_queries: int
    node_idxs: torch.Tensor
    labels: torch.Tensor
    per_ctx: dict[int, list[tuple[torch.Tensor, int] | None]]


class SamplerRetriever:
    def retrieve(self, batch, ctx_size_list, task):
        cpu = {k: batch[k].cpu() for k in BATCH_KEYS}
        is_targets = cpu["is_targets"]
        num_queries = int(is_targets.any(dim=1).sum().item())

        b_idxs, s_idxs = is_targets.nonzero(as_tuple=True)
        if len(b_idxs) == 0:
            return Context(
                num_queries=num_queries,
                node_idxs=torch.empty(0, dtype=torch.long),
                labels=torch.empty(0, dtype=torch.float),
                per_ctx={ctx: [None] * num_queries for ctx in ctx_size_list},
            )

        node_idxs = cpu["node_idxs"]
        col_name_idxs = cpu["col_name_idxs"]
        target_col = col_name_idxs[b_idxs[0], s_idxs[0]]
        target_node_per_b = torch.full(
            (is_targets.shape[0],), -1, dtype=node_idxs.dtype
        )
        target_node_per_b[b_idxs] = node_idxs[b_idxs, s_idxs]

        is_label_cell = (
            cpu["is_task_nodes"] & ~cpu["is_padding"] & (col_name_idxs == target_col)
        )
        lc_b, lc_s = is_label_cell.nonzero(as_tuple=True)
        vals = cpu["number_values"].squeeze(-1)

        item_idxs = lc_b
        positions = lc_s
        row_node_idxs = node_idxs[lc_b, lc_s]
        labels = vals[lc_b, lc_s].float()
        is_query = row_node_idxs == target_node_per_b[lc_b]

        per_ctx = {}
        for ctx in ctx_size_list:
            selected = []
            for q in range(num_queries):
                in_item = (item_idxs == q).nonzero(as_tuple=True)[0]
                if in_item.numel() == 0:
                    selected.append(None)
                    continue
                visible = in_item[positions[in_item] < ctx]
                query_pos = visible[is_query[visible]]
                if query_pos.numel() == 0:
                    selected.append(None)
                    continue
                train_pos = visible[~is_query[visible]]
                selected.append((train_pos, int(query_pos[0].item())))
            per_ctx[ctx] = selected

        return Context(
            num_queries=num_queries,
            node_idxs=row_node_idxs,
            labels=labels,
            per_ctx=per_ctx,
        )
