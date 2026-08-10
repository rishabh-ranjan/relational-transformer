"""Peak memory and time of the three attention-mask constructions, at the
pretraining shape. See [README.md](README.md)."""

import time

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from rt.model.net import _kv_sizes
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

    block_masks = {
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
    }
    # The dense path gets kv_sizes for free by summing a mask it already has;
    # this path has to derive them, and that cost belongs in the comparison.
    kv_sizes = _kv_sizes(
        node_idxs.contiguous(),
        f2p_nbr_idxs.contiguous(),
        col_name_idxs.contiguous(),
        table_name_idxs.contiguous(),
        is_padding.contiguous(),
    )
    return block_masks, kv_sizes


def pad_inlined(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    """`all_at_once`, minus the materialized `pad`. Three masks consume it, so it
    is the one intermediate inductor has to keep a whole (B, S, S) buffer for;
    written as two broadcasts inside each mask it fuses into the mask's own
    kernel instead."""
    batch_size, seq_len = node_idxs.shape
    device = node_idxs.device
    not_q = ~is_padding[:, :, None]
    not_kv = ~is_padding[:, None, :]
    same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
    kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
    q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
    same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
        table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
    )
    attn_masks = {
        "feat": (same_node | kv_in_f2p) & not_q & not_kv,
        "nbr": q_in_f2p & not_q & not_kv,
        "col": same_col_table & not_q & not_kv,
    }
    kv_sizes = {
        k: v.sum(dim=-1, keepdim=True).bfloat16() for k, v in attn_masks.items()
    }
    block_masks = {
        k: _block_mask(v.contiguous(), batch_size, seq_len, device)
        for k, v in attn_masks.items()
    }
    return block_masks, kv_sizes


def _packed_block_mask(mask, batch_size, seq_len, device):
    """Pack the kv axis 8-to-a-byte and index the bits from the mask_mod. Still a
    lookup per partial-block element -- the property the dense path exists for --
    but the tensor flex_attention reads is an eighth the size."""
    packed = torch.zeros(
        batch_size, seq_len, seq_len // 8, dtype=torch.uint8, device=device
    )
    for bit in range(8):
        packed |= mask[:, :, bit::8].to(torch.uint8) << bit
    return (
        create_block_mask(
            mask_mod=lambda b, h, q, kv: (
                (packed[b, q, kv // 8] >> (kv % 8)) & 1
            ).bool(),
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            _compile=True,
        ),
        packed,
    )


def packed_bits(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    """`all_at_once`, but each dense mask is packed to bits and the bool dropped,
    so only one (B, S, S) bool is ever live and what survives per mask is an
    eighth of one."""
    batch_size, seq_len = node_idxs.shape
    device = node_idxs.device
    not_q = ~is_padding[:, :, None]
    not_kv = ~is_padding[:, None, :]

    def feat():
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
        return (same_node | kv_in_f2p) & not_q & not_kv

    def nbr():
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
        return q_in_f2p & not_q & not_kv

    def col():
        same = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )
        return same & not_q & not_kv

    block_masks, kv_sizes, keep = {}, {}, []
    for attn_type, build in (("feat", feat), ("nbr", nbr), ("col", col)):
        mask = build()
        kv_sizes[attn_type] = mask.sum(dim=-1, keepdim=True).bfloat16()
        bm, packed = _packed_block_mask(mask, batch_size, seq_len, device)
        block_masks[attn_type] = bm
        keep.append(packed)  # flex_attention reads it, so it has to outlive this
        del mask
    return block_masks, (kv_sizes, keep)


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
    # Attention is what the masks are for, and what the dense path is buying:
    # flex_attention evaluates the mask_mod for every partial block, so a mask
    # that is a lookup and a mask that recomputes a predicate cost differently
    # *here*, not where they are built.
    num_heads, head_dim = 8, 64
    qkv = [
        torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    flex = torch.compile(flex_attention, dynamic=False)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()

    for variant in (all_at_once, pad_inlined, packed_bits, mask_mod):
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

        build_peak = torch.cuda.max_memory_allocated() - resident

        # Attention with the block masks this variant produced.
        block_masks, _held = fn(**inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            for bm in block_masks.values():
                flex(*qkv, block_mask=bm, scale=1.0 / head_dim)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            for bm in block_masks.values():
                flex(*qkv, block_mask=bm, scale=1.0 / head_dim)
        torch.cuda.synchronize()
        attn_ms = (time.perf_counter() - t0) / repeats * 1000

        log(
            variant=variant.__name__,
            build_gib=f"{build_peak / 1024**3:.2f}",
            build_ms=f"{elapsed * 1000:.1f}",
            attn_ms=f"{attn_ms:.1f}",
        )
        del block_masks, _held
        torch.cuda.empty_cache()
