"""Copy preprocessed data onto node-local storage before it is mmapped.

Training reads its data by mmap and random access, which needs the working set
resident in the node's page cache. A shared filesystem whose client evicts
cached pages (Lustre's lock LRU, a client cache cap) turns every item into
network faults. Staging copies the directories onto a local disk first; a
copy of a few hundred GB over a fast interconnect is a couple of minutes, a
run that faults over it does not finish.
"""

import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rt.progress import log

MARKER = ".staged"


def stage_paths(
    stage_dir: str,
    paths: list[str],
    *,
    local_rank: int,
    barrier: Callable[[], None],
) -> list[str]:
    """Return ``paths`` relocated under ``stage_dir``, copying them there first.

    ``stage_dir`` has environment variables and ``~`` expanded, so a submit
    script can name the job's node-local scratch without knowing the job id
    (``"$TMPDIR/hf"``). Each path is copied to ``<stage_dir>/<its basename>``
    by local rank 0 of every node, one top-level entry per thread; every rank
    then waits on ``barrier`` before reading. A ``.staged`` marker written
    last makes the copy idempotent: a requeue onto the same node, or a second
    run in a held allocation, finds it and skips the copy.

    Every path must be an existing local directory; two with the same basename
    would collide and are rejected.
    """
    root = Path(os.path.expandvars(stage_dir)).expanduser()
    srcs = [Path(p).expanduser() for p in paths]
    for s in srcs:
        assert s.is_dir(), f"nothing to stage at {s}"
    names = [s.name for s in srcs]
    assert len(set(names)) == len(names), f"staged paths share a basename: {names}"
    if local_rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        for src in srcs:
            dst = root / src.name
            assert os.stat(src).st_dev != os.stat(root).st_dev, (
                f"stage_dir {root} is on the same filesystem as {src}; "
                "staging is for a different (local) disk"
            )
            if (dst / MARKER).exists():
                log(staged=str(dst), reused=True)
                continue
            tic = time.time()
            if dst.exists():
                shutil.rmtree(dst)
            _copy(src, dst)
            (dst / MARKER).touch()
            size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
            log(
                staged=str(dst),
                gb=f"{size / 1e9:.1f}",
                secs=f"{time.time() - tic:.0f}",
            )
    barrier()
    return [str(root / s.name) for s in srcs]


def _copy(src: Path, dst: Path) -> None:
    """``cp -r`` with the top-level entries copied in parallel: a preprocessed
    directory is hundreds of db directories, and one stream does not fill an
    interconnect."""
    dst.mkdir(parents=True)
    entries = sorted(src.iterdir())

    def one(e: Path) -> None:
        if e.is_dir():
            shutil.copytree(e, dst / e.name, symlinks=False)
        else:
            shutil.copy2(e, dst / e.name)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, entries))
