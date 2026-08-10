"""Process-wide environment that must be set before CUDA is touched.

Imported by the train and eval entry points, which call :func:`_set_alloc_conf`
at the top of their ``setup_dist`` -- the same place, and for the same reason,
as ``NCCL_NET_GDR_LEVEL``: it is the last point in ``main`` that is guaranteed
to run before the thing it configures initializes.
"""

import os

# The attention masks are the largest transient the net allocates: three
# (B, S, S) bool tensors, so `tokens_per_gpu * S` bytes each, plus (B, S, S, K)
# intermediates. With a `ctx_size_list` spanning 512..8192 that swings 16x from
# batch to batch, and the fixed-size segments of the default allocator end up
# holding the pool in blocks sized for the last context length seen -- a later
# batch at a longer context then cannot find a contiguous segment, and logs
# `memory allocation failed with OOM ... free: 30 MB` against an 80 GB card
# that is not actually full. Expandable segments are virtual-memory backed and
# grow in place, so mixed allocation sizes stop fragmenting the pool.
_ALLOC_CONF = "expandable_segments:True"


def _set_alloc_conf() -> None:
    """Configure the CUDA caching allocator. Must run before the first
    allocation; an explicit setting in the environment wins."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)
