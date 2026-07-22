"""mlock: pin the preprocessed mixture in RAM across training restarts."""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm

from rt.data.tasks import resolve_db_task_list

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mlock.restype = ctypes.c_int

_PROT_READ = 0x1
_MAP_SHARED = 0x01
_MAP_FAILED = ctypes.c_void_p(-1).value


def mlock_file(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            raise RuntimeError(f"empty file: {path}")
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr == _MAP_FAILED:
            err = ctypes.get_errno()
            raise OSError(err, f"mmap failed for {path}: {os.strerror(err)}")
    finally:
        os.close(fd)
    if _libc.mlock(addr, size) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mlock failed for {path}: {os.strerror(err)}")
    return size


@dataclass
class MlockConfig:
    db_task_list: list[tuple[str, str]] | str
    """(db, task) pairs, a local JSON file, or a Hub path like
    stanford-star/the-join/db-task-lists/rt-j.json; the referenced dbs are
    locked."""

    pre_dir: str

    embedding_model_ref: str

    workers: int
    """parallel mlock workers; /dfs scales with concurrency (measured
    ~244MB/s single-stream vs ~1.2GB/s at 8+ parallel), so more workers
    saturate it faster."""


def mlock_main(cfg: MlockConfig) -> None:
    db_names = sorted({db for db, _ in resolve_db_task_list(cfg.db_task_list)})
    print(f"mlock: {len(db_names)} unique dbs", flush=True)

    def db_paths(db: str) -> list[str]:
        base = os.path.join(cfg.pre_dir, db)
        return [
            os.path.join(base, "nodes.rkyv"),
            os.path.join(base, f"text_emb_{cfg.embedding_model_ref}.bin"),
            os.path.join(base, "p2f_adj.rkyv"),
        ]

    def fmt_size(n: int) -> str:
        return f"{n / 2**30:.2f} GiB"

    page_size = os.sysconf("SC_PAGESIZE")

    def allocated_size(p: str) -> int:
        return os.stat(p).st_blocks * 512

    def footprint_size(p: str) -> int:
        size = os.stat(p).st_size
        return ((size + page_size - 1) // page_size) * page_size

    db_sizes: dict[str, int] = {}
    db_footprints: dict[str, int] = {}
    size_errors: dict[str, str] = {}
    for db in db_names:
        try:
            paths = db_paths(db)
            db_sizes[db] = sum(allocated_size(p) for p in paths)
            db_footprints[db] = sum(footprint_size(p) for p in paths)
        except Exception as e:
            size_errors[db] = f"{type(e).__name__}: {e}"

    total_size = sum(db_sizes.values())
    width = max((len(fmt_size(s)) for s in db_sizes.values()), default=0)

    locked_files = 0
    total = 0
    skipped = 0

    for db in db_names:
        if db in size_errors:
            print(
                f"\x1b[31m[{'ERROR':>{width}}] {db}  {size_errors[db]}\x1b[0m",
                flush=True,
            )
            skipped += 1

    def lock_db(db: str) -> tuple[str, int, Exception | None]:
        n = 0
        try:
            for p in db_paths(db):
                mlock_file(p)
                n += 1
        except Exception as e:
            return db, n, e
        return db, n, None

    pending = [db for db in db_names if db not in size_errors]
    total_footprint = sum(db_footprints[db] for db in pending)
    import time

    t0 = time.time()
    pbar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024)
    pbar.set_postfix_str(f"footprint={fmt_size(total_footprint)}")
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = [ex.submit(lock_db, db) for db in pending]
        for fut in as_completed(futures):
            db, n, err = fut.result()
            db_size = db_sizes[db]
            locked_files += n
            if err is not None:
                tqdm.write(
                    f"\x1b[31m[{fmt_size(db_size):>{width}}] {db}  "
                    f"ERROR: {type(err).__name__}: {err}\x1b[0m"
                )
                skipped += 1
                continue
            tqdm.write(f"[{fmt_size(db_size):>{width}}] {db}")
            total += db_size
            pbar.update(db_size)
    pbar.close()
    elapsed = time.time() - t0

    print(
        f"locked {locked_files} files, {fmt_size(total)} on disk, "
        f"{fmt_size(total_footprint)} memory footprint, "
        f"{skipped} dbs skipped, in {elapsed:.0f}s "
        f"({total / 2**30 / max(elapsed, 1e-9):.2f} GiB/s). "
        f"pid={os.getpid()}. sleeping until signaled.",
        flush=True,
    )

    def _fast_exit(signum: int, frame: object) -> None:
        # Proactively release all locked pages before exiting. Without this the
        # kernel reclaims ~1TB of mlocked memory lazily on process teardown,
        # which can exceed slurm's UnkillableStepTimeout on scancel and DRAIN the
        # node ("Kill task failed"). munlockall() makes teardown prompt.
        try:
            _libc.munlockall()
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGINT, _fast_exit)
    signal.signal(signal.SIGTERM, _fast_exit)
    signal.pause()
