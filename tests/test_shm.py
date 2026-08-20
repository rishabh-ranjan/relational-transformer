"""Worker tensors must not leave named segments behind in /dev/shm.

A killed worker cannot clean up after itself. Under the `file_system` sharing
strategy its segments outlive it and nothing reclaims them, so a node that runs
many jobs -- or one job that rebuilds an evaluator many times -- fills /dev/shm
and every process on it wedges on "No space left on device". Under
`file_descriptor` the pages are anonymous and the kernel frees them with the
last holder, whatever killed it.
"""

from __future__ import annotations

import glob

import pytest

torch = pytest.importorskip("torch")


def test__sharing_strategy__is_leak_proof() -> None:
    import rt.data.resolve  # noqa: F401  -- sets it at import

    assert torch.multiprocessing.get_sharing_strategy() == "file_descriptor", (
        "file_system leaks a named /dev/shm segment per shared tensor whenever "
        "a worker is killed rather than exited"
    )


def test__nofile_limit__is_raised_to_the_hard_limit() -> None:
    """`file_descriptor` costs one open fd per shared tensor."""
    import resource

    import rt.data.resolve  # noqa: F401

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft == hard


@pytest.mark.slow
def test__dataloader_rebuilds__leave_no_named_segments() -> None:
    from torch.utils.data import DataLoader, Dataset

    import rt.data.resolve  # noqa: F401

    class Tiny(Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, i: int) -> "torch.Tensor":
            return torch.zeros(128, 128)

    before = set(glob.glob("/dev/shm/torch_*"))
    for _ in range(3):
        loader = DataLoader(Tiny(), batch_size=8, num_workers=2)
        assert sum(int(b.shape[0]) for b in loader) == 64
        del loader
    assert not set(glob.glob("/dev/shm/torch_*")) - before


def test__evaluator_entry__members_are_the_third_element() -> None:
    """The in-loop eval's grid entry is `(tag, ctx_sizes, members)`.

    The memory guard reaches into `evaluators[0]` for a member to run one
    eval-shaped batch through. Indexing the wrong element hands it an `int` --
    a ctx size -- and every training run dies with `'int' object has no
    attribute 'mem_guard'` the moment it reaches its first step, after the whole
    preprocess has been paid for.
    """
    import inspect

    import rt.train._train as t

    src = inspect.getsource(t.main)
    assert "evaluators[0][2][0].mem_guard" in src, (
        "the guard must take a member (element 2), not a ctx size (element 1)"
    )
    unpack = [l for l in src.splitlines() if "in evaluators:" in l and "for" in l]
    assert any("tag, ctxs, members" in l for l in unpack), (
        "the entry shape changed; the guard's index has to change with it"
    )
