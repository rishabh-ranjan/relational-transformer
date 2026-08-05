"""Hold a preprocessed mixture resident in RAM, so repeated runs skip the reload.

By default every run populates its data into the page cache at startup. When
you are iterating on training code, that reload is wasted work on every restart.
Lock the data once with this (Ctrl-C releases it), and pass
``mmap_populate=False`` in your training script so reads hit the locked cache.

    pixi run python examples/mlock.py     # terminal 1, on the node you train on

Needs a high RLIMIT_MEMLOCK (`ulimit -l unlimited`, or slurm's
`--propagate=MEMLOCK`, which rt.slurm sets) to lock the whole mixture.
"""

from __future__ import annotations

from rt.data import mlock_main

PRE_DIR = "data/the-join-preprocessed"

if __name__ == "__main__":
    mlock_main(
        db_task_list=f"{PRE_DIR}/db-task-lists/rt-j.json",
        pre_dir=PRE_DIR,
        embedder_ref="all-MiniLM-L12-v2",
        # Networked filesystems populate faster with more concurrency.
        workers=32,
    )
