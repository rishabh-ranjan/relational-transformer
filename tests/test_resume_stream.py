"""A resumed run must draw the batches an uninterrupted run would have drawn.

The stream is a pure function of two counters -- the ctx-size selector and the
rustler sampler's batch counter -- so "resume exactly" means starting both where
the interrupted run left them. This brute-forces the loop those counters drive
and checks the closed-form entry point agrees with it, which is the only thing
standing between a correct resume and a silently skewed one.
"""

from __future__ import annotations

import random

import pytest

from rt.data.datasets import resume_positions

CFG = dict(
    seed=12345,
    train_tokens_per_gpu=2**17,
    total_bs=256,
    world_size=1,
)


def _grad_accum(ctx: int) -> int:
    bs = max(1, CFG["train_tokens_per_gpu"] // ctx)
    if CFG["total_bs"] < CFG["world_size"] * bs:
        return 1
    return CFG["total_bs"] // (CFG["world_size"] * bs)


def _ctx_at(ctx_list: list[int], t: int) -> int:
    if len(ctx_list) == 1:
        return ctx_list[0]
    return random.Random(CFG["seed"] + t).choice(ctx_list)


def _next_index(target: int, worker_id: int, stride: int) -> int:
    """The first yield index >= target that belongs to `worker_id`."""
    i = worker_id
    while i < target:
        i += stride
    return i


def _walk(ctx_list, start_step, worker_id, stride):
    """The counters an uninterrupted run holds after `start_step` steps.

    Written from the DataLoader's semantics rather than from the implementation:
    yields are handed out round-robin, index `i` going to worker `i % stride`.

    Multi-ctx: one yield is one optimizer step, and that yield makes
    `grad_accum` sampler calls. Single-ctx: one yield is one microbatch, so
    `start_step` optimizer steps consume `start_step * grad_accum` of them and
    the sampler advances once per yield.
    """
    if len(ctx_list) == 1:
        consumed = start_step * _grad_accum(ctx_list[0])
        nxt = _next_index(consumed, worker_id, stride)
        return nxt, nxt
    ctx_step = _next_index(start_step, worker_id, stride)
    mine = [t for t in range(worker_id, start_step, stride)]
    sampler_step = worker_id + stride * sum(
        _grad_accum(_ctx_at(ctx_list, t)) for t in mine
    )
    return ctx_step, sampler_step


@pytest.mark.parametrize("ctx_list", [[128, 256, 512, 1024], [1024]])
@pytest.mark.parametrize("stride", [1, 2, 8])
@pytest.mark.parametrize("start_step", [0, 1, 7, 100, 2300, 12_345])
def test__resume_positions__match_an_uninterrupted_walk(
    ctx_list: list[int], stride: int, start_step: int
) -> None:
    for worker_id in range(stride):
        assert resume_positions(
            train_ctx_size_list=ctx_list,
            start_step=start_step,
            worker_id=worker_id,
            stride=stride,
            **CFG,
        ) == _walk(ctx_list, start_step, worker_id, stride)


def test__resume_at_zero__is_a_fresh_run() -> None:
    for worker_id in (0, 1, 5):
        assert resume_positions(
            train_ctx_size_list=[128, 256, 512, 1024],
            start_step=0,
            worker_id=worker_id,
            stride=8,
            **CFG,
        ) == (worker_id, worker_id)


def test__resume_is_a_continuation__not_a_replay_and_not_a_jump() -> None:
    """Resuming at N then walking M lands where walking N+M once lands."""
    ctx_list = [128, 256, 512, 1024]
    for stride in (1, 4):
        for worker_id in range(stride):
            a = resume_positions(
                train_ctx_size_list=ctx_list, start_step=500,
                worker_id=worker_id, stride=stride, **CFG,
            )
            whole = resume_positions(
                train_ctx_size_list=ctx_list, start_step=1500,
                worker_id=worker_id, stride=stride, **CFG,
            )
            # continue the walk from `a` for another 1000 optimizer steps
            ctx_step, sampler_step = a
            while ctx_step < 1500:
                ctx = random.Random(CFG["seed"] + ctx_step).choice(ctx_list)
                bs = max(1, CFG["train_tokens_per_gpu"] // ctx)
                ga = (
                    1 if CFG["total_bs"] < CFG["world_size"] * bs
                    else CFG["total_bs"] // (CFG["world_size"] * bs)
                )
                sampler_step += stride * ga
                ctx_step += stride
            assert (ctx_step, sampler_step) == whole
