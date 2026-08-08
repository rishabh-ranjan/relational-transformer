"""Text embeddings for preprocessed datasets (sentence-transformers)."""

import _thread
import contextlib
import os
import threading

import numpy as np
import orjson
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer


# Rows encoded before the result is moved off the device. Bounds GPU memory at
# roughly CHUNK x d_text x 4 bytes, independent of how much text a database has.
CHUNK = 1_000_000

# How often the watchdog below looks at the encoding workers.
WATCH_SECONDS = 10.0
# How long it then gives the main thread to die of the exception it raised there
# before killing the process outright.
WATCH_GRACE_SECONDS = 60.0


class WorkerDied(RuntimeError):
    """An encoding worker exited while the pool was still being fed."""


def _watch_workers(pool, stop: threading.Event) -> None:
    """Turn a dead encoding worker into a dead job.

    sentence-transformers' multi-process pool hands each chunk to a worker and
    then blocks reading results off a queue. When a worker dies -- on this
    cluster, `uncorrectable ECC error` from a bad card -- its chunks never come
    back, so the parent waits on a queue nothing will ever write to. The job
    stays R, holding every GPU it asked for, until its walltime runs out, and
    the traceback that explains it is in the middle of a log nobody is reading.
    One such hang cost 1h44m and ten cards on 2026-08-07.

    Failing is strictly better than that: the stage is idempotent, so slurm
    requeues it or `submit.py` resubmits it, and it lands somewhere else. What
    is not recoverable is not noticing.
    """
    while not stop.wait(WATCH_SECONDS):
        dead = [p for p in pool["processes"] if p.exitcode is not None]
        if not dead:
            continue
        which = ", ".join(f"{p.name} (exit {p.exitcode})" for p in dead)
        print(
            f"!! {len(dead)} of {len(pool['processes'])} encoding workers died "
            f"[{which}] -- their chunks will never come back, so failing here "
            f"rather than waiting on them",
            flush=True,
        )
        # The main thread is blocked in queue.get(), which no timeout of ours
        # reaches; an exception raised into it is what unblocks it. If that does
        # not take -- the interrupt lands somewhere that swallows it -- exit
        # anyway, because the whole point is to stop holding the GPUs.
        _thread.interrupt_main()
        # `stop`, not a plain sleep: the pool's context manager sets it on the
        # way out, so the interrupt landing where it was meant to lands this
        # thread here and ends it. Sleeping through that would exit a process
        # that had already handled the failure, and turn a caller that caught
        # WorkerDied into one that never returns from the `with`.
        if not stop.wait(WATCH_GRACE_SECONDS):
            print("!! still alive after the interrupt; exiting hard", flush=True)
            os._exit(1)
        return


@contextlib.contextmanager
def _watched_pool(model: SentenceTransformer, devices: list[str]):
    """A multi-process pool that outlives one `encode` call, and is watched.

    Owned here rather than left to `encode(device=[...])`, which starts and
    stops a pool per call: this loop calls it once per chunk, and a pool costs a
    model load on every device. Owning it is also what makes the workers
    visible, which is what the watchdog needs.
    """
    pool = model.start_multi_process_pool(devices)
    stop = threading.Event()
    watchdog = threading.Thread(
        target=_watch_workers, args=(pool, stop), name="worker-watchdog", daemon=True
    )
    watchdog.start()
    try:
        yield pool
    except KeyboardInterrupt as e:
        # The watchdog's interrupt, or a real one. Either way the pool is not
        # trustworthy; say which so a log makes the difference obvious.
        if any(p.exitcode is not None for p in pool["processes"]):
            raise WorkerDied(
                "an encoding worker died mid-pool (see the traceback it printed "
                "above -- on this cluster it is usually an uncorrectable ECC "
                "error from a bad GPU)"
            ) from e
        raise
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            model.stop_multi_process_pool(pool)


class TextEmbedder:
    def __init__(self, batch_size, embedder, device, chunk=CHUNK):
        device_type = torch.device(device).type
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedder}",
            device=device,
            model_kwargs={
                "dtype": torch.bfloat16 if device_type == "cuda" else torch.float32,
            },
        )
        self.batch_size = batch_size
        self.chunk = chunk

    def __call__(self, text_list, device):
        """Encode in chunks, moving each off the device before the next.

        Both paths chunk, for the same reason: whether the result accumulates on
        one GPU (`convert_to_tensor`) or in host memory (the multi-process
        path), holding the whole output at once means peak memory scales with
        the database. join-overture-maps has 8 GiB of text; unchunked that is
        ~21 GiB on a card, or ~120 GiB of fp32 in RAM, and both of those have
        killed a job here with an hour of rustler work already behind them.
        Chunked, the ceiling is CHUNK rows whatever the database.
        """
        if isinstance(device, list):
            # Multi-process path: a worker per device, fp32 numpy back
            # regardless of flags. ~4x on six GPUs, measured.
            with _watched_pool(self.model, device) as pool:
                out = [
                    self.model.encode_multi_process(
                        text_list[i : i + self.chunk],
                        pool,
                        batch_size=self.batch_size,
                        show_progress_bar=True,
                    ).astype(bfloat16)
                    for i in range(0, len(text_list), self.chunk)
                ]
            return np.concatenate(out) if out else np.empty((0, 0), dtype=bfloat16)

        out = []
        for i in range(0, len(text_list), self.chunk):
            piece = text_list[i : i + self.chunk]
            emb = self.model.encode(
                piece,
                batch_size=self.batch_size,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=True,
                device=device,
            )
            # bf16 -> int16 bitcast so torch .numpy() accepts it, then relabel
            # as bf16. On CPU the SBERT model loaded with fp32, so cast first --
            # the bitcast on raw fp32 silently misinterprets 4-byte floats as
            # 2xbf16 garbage and writes a .bin full of NaN/inf bit patterns.
            out.append(emb.to(torch.bfloat16).cpu().view(torch.int16).numpy())
            del emb
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        if not out:
            return np.empty((0, 0), dtype=bfloat16)
        stacked = np.concatenate(out)
        return stacked if stacked.dtype == bfloat16 else stacked.view(bfloat16)


def embed_texts(
    dataset_name,
    pre_dir: str,
    device,
    batch_size,
    embedder,
):
    if device is None:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            # Pass a string for 1 GPU. A list of len 1 routes SBERT into its
            # multi-process path, which skips length-sorted batching.
            device = [f"cuda:{i}" for i in range(n)] if n > 1 else "cuda:0"
            print(f"Using device(s): {device}")
        else:
            device = "cpu"

    init_device = device[0] if isinstance(device, list) else device

    text_path = f"{pre_dir}/{dataset_name}/text.json"
    # Read as bytes and drop them the moment they are parsed. Holding `raw`
    # alongside the parsed list doubles the peak for no reason, and these files
    # are not small -- join-overture-maps' text.json is 8 GiB, which is enough
    # for the difference to be an out-of-memory kill rather than a slow moment.
    with open(text_path, "rb") as f:
        raw = f.read()
    text_list = orjson.loads(raw)
    del raw
    print(f"Loaded {len(text_list)} texts from {text_path}")

    text_embedder = TextEmbedder(batch_size, embedder, init_device)
    emb = text_embedder(text_list, device=device)

    emb_path = f"{pre_dir}/{dataset_name}/text_emb_{embedder}.bin"
    emb.tofile(emb_path)
    print(f"Wrote {emb.shape} {emb.dtype} to {emb_path}")
