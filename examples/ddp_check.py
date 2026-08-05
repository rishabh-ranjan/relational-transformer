"""Cluster sanity check: does DDP come up under srun, and do signals reach ranks?

Run it through roach.slurm (see the bottom of this file). Two things are being
checked, both of which the training jobs depend on:

* every rank joins the process group and an all-reduce returns the expected sum,
  i.e. srun's one-task-per-GPU layout is enough to bring DDP up with no launcher;
* a SIGTERM aimed at the job reaches *every rank*, which is what lets a preempted
  run save a checkpoint before it dies.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path


def main(run_id: str, seconds: int, out_dir: str) -> None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(
        f"rank {rank}/{world_size} local_rank {local_rank} device {device} "
        f"host {os.uname().nodename}",
        flush=True,
    )

    if world_size > 1:
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        t = torch.full((1,), float(rank), device=device)
        dist.all_reduce(t)
        expected = world_size * (world_size - 1) / 2
        print(
            f"rank {rank}: all_reduce -> {t.item()} (expected {expected})", flush=True
        )
        assert t.item() == expected

    caught = {"sig": None}

    def on_signal(signum, _frame):
        caught["sig"] = signum
        print(f"rank {rank}: caught signal {signum}", flush=True)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)

    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    for i in range(seconds):
        time.sleep(1)
        if caught["sig"] is not None:
            # Stand-in for save_resume: written only if the signal arrived.
            (out / f"rank{rank}.signalled").write_text(f"{caught['sig']} at step {i}\n")
            print(f"rank {rank}: saved and exiting at step {i}", flush=True)
            break
    else:
        print(f"rank {rank}: finished {seconds}s without a signal", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
