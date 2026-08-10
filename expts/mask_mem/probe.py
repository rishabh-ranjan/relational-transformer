"""Peak memory and time of the three attention-mask constructions, at the
pretraining shape. See [README.md](README.md)."""

import time

import torch
from torch.nn.attention.flex_attention import create_block_mask

from rt.progress import log


def _inputs(batch_size, seq_len, max_f2p_nbrs, num_nodes, device):
    """Shaped like a real batch: node ids repeat (a node owns several feature
    cells, which is what makes `same_node` a band rather than a diagonal), f2p
    neighbours point at other nodes, no padding."""
    g = torch.Generator(device="cpu").manual_seed(0)
    randint = lambda hi, *shape: torch.randint(  # noqa: E731
        0, hi, shape, generator=g
    ).to(device)
    return dict(
        node_idxs=randint(num_nodes, batch_size, seq_len).to(torch.int32),
        f2p_nbr_idxs=randint(num_nodes, batch_size, seq_len, max_f2p_nbrs).to(
            torch.int32
        ),
        col_name_idxs=randint(64, batch_size, seq_len).to(torch.int32),
        table_name_idxs=randint(8, batch_size, seq_len).to(torch.int32),
        is_padding=torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device),
    )


def _block_mask(mask, batch_size, seq_len, device):
    """``rt.model.net._make_block_mask``, inlined so the variants below differ
    only in what they hand it."""
    return create_block_mask(
        mask_mod=lambda b, h, q, kv: mask[b, q, kv],
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )


def all_at_once(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    """What `net.py` does today: build all three dense masks, then convert all
    three. Every mask is alive while the last one is built."""
    batch_size, seq_len = node_idxs.shape
    device = node_idxs.device
    pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
    same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
    kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
    q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
    same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
        table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
    )
    attn_masks = {
        "feat": (same_node | kv_in_f2p) & pad,
        "nbr": q_in_f2p & pad,
        "col": same_col_table & pad,
    }
    kv_sizes = {
        k: v.sum(dim=-1, keepdim=True).bfloat16() for k, v in attn_masks.items()
    }
    attn_masks = {k: v.contiguous() for k, v in attn_masks.items()}
    block_masks = {
        k: _block_mask(v, batch_size, seq_len, device) for k, v in attn_masks.items()
    }
    return block_masks, kv_sizes


def one_at_a_time(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    """Build one dense mask, convert it, drop it, then the next. A BlockMask is
    a block-granularity index (S/128 squared), so holding three of those and one
    dense mask costs far less than holding three dense masks."""
    batch_size, seq_len = node_idxs.shape
    device = node_idxs.device
    pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])

    def feat():
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
        return (same_node | kv_in_f2p) & pad

    def nbr():
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
        return q_in_f2p & pad

    def col():
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )
        return same_col_table & pad

    block_masks, kv_sizes = {}, {}
    for attn_type, build in (("feat", feat), ("nbr", nbr), ("col", col)):
        mask = build()
        kv_sizes[attn_type] = mask.sum(dim=-1, keepdim=True).bfloat16()
        block_masks[attn_type] = _block_mask(
            mask.contiguous(), batch_size, seq_len, device
        )
        del mask
    return block_masks, kv_sizes


def mask_mod(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    """`materialize_attn_masks=False`: `create_block_mask` samples the predicate
    at block granularity and no (B, S, S) tensor is ever built. Included as the
    floor -- what the dense path is paying for."""
    batch_size, seq_len = node_idxs.shape
    device = node_idxs.device

    def feat_mod(b, h, q, kv):
        not_pad = (~is_padding[b, q]) & (~is_padding[b, kv])
        return (
            (node_idxs[b, q] == node_idxs[b, kv])
            | (f2p_nbr_idxs[b, q] == node_idxs[b, kv]).any(dim=-1)
        ) & not_pad

    def nbr_mod(b, h, q, kv):
        not_pad = (~is_padding[b, q]) & (~is_padding[b, kv])
        return (f2p_nbr_idxs[b, kv] == node_idxs[b, q]).any(dim=-1) & not_pad

    def col_mod(b, h, q, kv):
        not_pad = (~is_padding[b, q]) & (~is_padding[b, kv])
        return (
            (col_name_idxs[b, q] == col_name_idxs[b, kv])
            & (table_name_idxs[b, q] == table_name_idxs[b, kv])
        ) & not_pad

    return {
        name: create_block_mask(
            mask_mod=mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            _compile=True,
        )
        for name, mod in (("feat", feat_mod), ("nbr", nbr_mod), ("col", col_mod))
    }, None


def main(
    *,
    batch_size: int,
    seq_len: int,
    max_f2p_nbrs: int,
    num_nodes: int,
    repeats: int,
) -> None:
    """One line per variant: peak allocated over the call, and its time.

    Every argument is required; the arguments are the record of the measurement.
    """
    device = "cuda"
    inputs = _inputs(batch_size, seq_len, max_f2p_nbrs, num_nodes, device)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()

    for variant in (all_at_once, one_at_a_time, mask_mod):
        fn = torch.compile(variant, dynamic=False)
        # Warm up: the first call compiles, and compilation allocates.
        out = fn(**inputs)
        del out
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(repeats):
            out = fn(**inputs)
            del out
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / repeats

        log(
            variant=variant.__name__,
            peak_gib=f"{(torch.cuda.max_memory_allocated() - resident) / 1024**3:.2f}",
            ms=f"{elapsed * 1000:.1f}",
        )
        torch.cuda.empty_cache()
