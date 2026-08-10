"""Process-wide settings that must be applied before CUDA and NCCL come up.

Imported by the train and eval entry points, which call :func:`_setup_env` at
the top of their ``setup_dist`` -- the same place, and for the same reason, as
``NCCL_NET_GDR_LEVEL``: it is the last point in ``main`` that is guaranteed to
run before the things it configures initialize.

Every environment variable here is set with ``setdefault``, so anything the job
already exports wins.
"""

import os

import torch

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

# NCCL flight recorder: a ring buffer of recent collective metadata, dumped when
# a collective times out. This is the missing evidence for the failure mode the
# run actually hits -- a multi-node job that goes silent with every rank parked
# in a collective -- which is otherwise diagnosable only by inference. The
# buffer costs a few MB per rank and the dump fires only on failure.
_FR_BUFFER_SIZE = "20000"
_FR_DUMP_TEMP_FILE = "/tmp/nccl_trace_rank_"

# Inductor cache visibility: hit/miss/bypass lines say whether a restart is
# reusing the on-disk compile cache or paying full compile cost. Worth having
# permanently -- time_to_first_step has ranged from 3m43s to 25m54s across
# restarts of the same run, and this is what distinguishes the causes.
_TORCH_LOGS = [
    "+torch._inductor.codecache",
    "+torch._functorch._aot_autograd.autograd_cache",
]

# Dynamo compiles one graph per distinct input shape, and `ctx_size_list` has
# five entries before eval's shapes are counted. The default limit of 8 is thin
# enough that a sixth shape would silently drop the model back to eager for the
# rest of the run.
_DYNAMO_CACHE_SIZE_LIMIT = 16


def _setup_env() -> None:
    """Allocator, NCCL diagnostics, compile limits and thread counts. Must run
    before the first CUDA allocation and before ``init_process_group``."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)

    os.environ.setdefault("TORCH_FR_BUFFER_SIZE", _FR_BUFFER_SIZE)
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ.setdefault("TORCH_FR_DUMP_TEMP_FILE", _FR_DUMP_TEMP_FILE)

    os.environ["TORCH_LOGS"] = ",".join(
        filter(None, [os.environ.get("TORCH_LOGS", ""), *_TORCH_LOGS])
    )

    torch._dynamo.config.cache_size_limit = _DYNAMO_CACHE_SIZE_LIMIT

    # One rank per GPU, `num_workers` loader processes behind each, and only
    # `cpus_per_task` cores to share: intra-op thread pools at their default
    # width oversubscribe the node several times over. One thread per process
    # is the right shape when the parallelism is already in the processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
