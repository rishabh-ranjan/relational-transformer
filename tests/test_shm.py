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
