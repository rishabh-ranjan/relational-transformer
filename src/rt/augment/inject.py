import torch

from rt.augment.plan import NUMBER_SEM_TYPE, AugmentPlan

COPIED_FROM_PARENT = (
    "node_idxs",
    "table_name_idxs",
    "timestamps",
    "seed_node_idxs",
    "bfs_depths",
    "is_task_nodes",
    "f2p_nbr_idxs",
)
ZEROED = ("text_values", "datetime_values")


def _source_cells(
    batch: dict, col_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # A derived cell of a masked target would hand the model the answer it is
    # being asked to reconstruct, so a target cell never produces one.
    mask = (
        (batch["col_name_idxs"] == col_idx)
        & (batch["sem_types"] == NUMBER_SEM_TYPE)
        & (~batch["is_padding"])
        & (~batch["is_targets"])
    )
    rows, slots = mask.nonzero(as_tuple=True)
    return rows, slots


def _match_pairs(
    rows_a: torch.Tensor,
    slots_a: torch.Tensor,
    rows_b: torch.Tensor,
    slots_b: torch.Tensor,
    node_idxs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rows_a.numel() == 0 or rows_b.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=rows_a.device)
        return empty, empty, empty
    stride = int(node_idxs.max().item()) + 1
    key_a = rows_a.to(torch.int64) * stride + node_idxs[rows_a, slots_a].to(torch.int64)
    key_b = rows_b.to(torch.int64) * stride + node_idxs[rows_b, slots_b].to(torch.int64)
    order = torch.argsort(key_b)
    sorted_b = key_b[order]
    pos = torch.searchsorted(sorted_b, key_a).clamp(max=sorted_b.numel() - 1)
    hit = sorted_b[pos] == key_a
    matched_b = order[pos[hit]]
    return rows_a[hit], slots_a[hit], slots_b[matched_b]


def augment_batch(batch: dict, plan: AugmentPlan) -> dict:
    if not plan.columns:
        return batch
    node_idxs = batch["node_idxs"]
    device = node_idxs.device
    batch_size, seq_len = node_idxs.shape
    values = batch["number_values"].squeeze(-1)

    em_rows: list[torch.Tensor] = []
    em_parent: list[torch.Tensor] = []
    em_value: list[torch.Tensor] = []
    em_col: list[torch.Tensor] = []
    em_name_row: list[torch.Tensor] = []

    for planned in plan.unary:
        rows, slots = _source_cells(batch, planned.source_col_idxs[0])
        if rows.numel() == 0:
            continue
        em_rows.append(rows)
        em_parent.append(slots)
        em_value.append(planned.derived.values(values[rows, slots].float()))
        em_col.append(
            torch.full_like(rows, planned.synthetic_col_idx, dtype=torch.long)
        )
        em_name_row.append(
            torch.full_like(rows, planned.name_embedding_row, dtype=torch.long)
        )

    for planned in plan.pairs:
        left_idx, right_idx = planned.source_col_idxs
        rows_a, slots_a = _source_cells(batch, left_idx)
        rows_b, slots_b = _source_cells(batch, right_idx)
        rows, slots_left, slots_right = _match_pairs(
            rows_a, slots_a, rows_b, slots_b, node_idxs
        )
        if rows.numel() == 0:
            continue
        em_rows.append(rows)
        em_parent.append(slots_left)
        em_value.append(
            planned.derived.values(
                values[rows, slots_left].float(), values[rows, slots_right].float()
            )
        )
        em_col.append(
            torch.full_like(rows, planned.synthetic_col_idx, dtype=torch.long)
        )
        em_name_row.append(
            torch.full_like(rows, planned.name_embedding_row, dtype=torch.long)
        )

    if not em_rows:
        return batch

    rows = torch.cat(em_rows)
    parent = torch.cat(em_parent)
    value = torch.cat(em_value)
    col = torch.cat(em_col)
    name_row = torch.cat(em_name_row)

    order = torch.argsort(rows, stable=True)
    rows, parent, value, col, name_row = (
        rows[order],
        parent[order],
        value[order],
        col[order],
        name_row[order],
    )
    counts = torch.bincount(rows, minlength=batch_size)
    extra = int(counts.max().item())
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(rows.numel(), device=device) - starts[rows]
    dest = seq_len + rank

    out_len = seq_len + extra
    # Anything that is not one entry per cell -- batch_mask, and whatever a
    # caller has already popped or added -- rides through untouched.
    out: dict = {
        k: v
        for k, v in batch.items()
        if v.dim() < 2 or v.shape[1] != seq_len
    }

    def grow(key: str, fill) -> torch.Tensor:
        src = batch[key]
        shape = (batch_size, out_len, *src.shape[2:])
        big = torch.full(shape, fill, dtype=src.dtype, device=device)
        big[:, :seq_len] = src
        return big

    for key in COPIED_FROM_PARENT:
        big = grow(key, 0)
        big[rows, dest] = batch[key][rows, parent]
        out[key] = big

    for key in ZEROED:
        out[key] = grow(key, 0)

    is_padding = grow("is_padding", True)
    is_padding[rows, dest] = False
    out["is_padding"] = is_padding

    is_targets = grow("is_targets", False)
    is_targets[rows, dest] = False
    out["is_targets"] = is_targets

    sem_types = grow("sem_types", NUMBER_SEM_TYPE)
    sem_types[rows, dest] = NUMBER_SEM_TYPE
    out["sem_types"] = sem_types

    class_value_idxs = grow("class_value_idxs", -1)
    class_value_idxs[rows, dest] = -1
    out["class_value_idxs"] = class_value_idxs

    col_name_idxs = grow("col_name_idxs", 0)
    col_name_idxs[rows, dest] = col.to(col_name_idxs.dtype)
    out["col_name_idxs"] = col_name_idxs

    col_name_values = grow("col_name_values", 0)
    col_name_values[rows, dest] = plan.name_embeddings[name_row].to(
        col_name_values.dtype
    )
    out["col_name_values"] = col_name_values

    number_values = grow("number_values", 0)
    number_values[rows, dest, 0] = value.to(number_values.dtype)
    out["number_values"] = number_values

    missing = set(batch) - set(out)
    if missing:
        raise KeyError(
            f"augment_batch did not rebuild {sorted(missing)}; every batch "
            f"tensor must be widened or the model will see mismatched shapes"
        )
    return out
