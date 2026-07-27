"""Time RT-J inference: (1) rustler context construction vs (2) forward pass.

Both steps run sequentially in the main process (no DataLoader workers), so
the CPU context build is never overlapped with GPU compute. The forward pass
is bracketed by ``torch.cuda.synchronize()``; its time includes the H2D copy
(``net.predict`` moves the batch to the device internally).

Reports per-batch and per-item mean/std over ``timing_steps`` measured steps,
after ``warmup_steps`` discarded warmup steps.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
import tyro

from rt.data import get_tasks
from rt.data.datasets import RustlerDataset
from rt.model import load_rt_model


@dataclass
class TimingConfig:
    # checkpoint (local dir/file or Hub repo)
    ckpt: str = "stanford-star/rt-j/classification"
    # local directory of preprocessed datasets
    pre_dir: str = "/dfs/user/ranjanr/pre/relbench-preprocessed"
    db: str = "rel-f1"
    task: str = "driver-dnf"
    split: str = "test"
    # context (token) size fed to the model
    ctx_size: int = 8192
    # rows per batch
    batch_size: int = 1
    # rustler context-construction knobs (eval defaults)
    local_ctx_size: int = 256
    bfs_width: int = 32
    prefer_latest: bool = True
    num_walks: int = 10_000
    walk_length: int = 20
    shuffle_seed: int = 0
    context_seed: int = 0
    mmap_populate: bool = True
    vector_db_path: str | None = None
    # timing protocol
    warmup_steps: int = 3
    timing_steps: int = 10
    device: str = "cuda"


def report(name: str, times: list[float], batch_size: int) -> None:
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"{name:>24s}: {mean * 1e3:9.2f} ± {std * 1e3:7.2f} ms/batch"
        f"  ({mean / batch_size * 1e3:8.3f} ± {std / batch_size * 1e3:7.3f} ms/item)"
    )


def main(cfg: TimingConfig) -> None:
    torch.manual_seed(0)

    net, config = load_rt_model(cfg.ckpt, device=cfg.device, compile=False)
    net = net.to(torch.bfloat16)
    net.eval()
    print(f"loaded {config.get('name', cfg.ckpt)} on {cfg.device}")

    tasks = get_tasks(cfg.pre_dir, [(cfg.db, cfg.task)], (cfg.split,))
    assert len(tasks) == 1, f"expected 1 task, got {tasks}"
    task = tasks[0]

    ds = RustlerDataset(
        tasks=[task],
        pre_dir=cfg.pre_dir,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_size_list=[cfg.local_ctx_size],
        bfs_width_list=[cfg.bfs_width],
        num_walks=cfg.num_walks,
        walk_length=cfg.walk_length,
        prefer_latest_list=[cfg.prefer_latest],
        mask_prob_max=0.0,
        embedder=config["embedder"],
        d_text=config["d_text"],
        shuffle_seed=cfg.shuffle_seed,
        context_seed=cfg.context_seed,
        items_per_task=10_000_000,
        quiet=True,
        ignore_data_errors=False,
        mmap_populate=cfg.mmap_populate,
        timeout_per_item=3600.0,
        vector_db_path=cfg.vector_db_path,
        train_only_fallback=False,
    )
    num_batches = ds.num_items // cfg.batch_size  # full batches only
    total_steps = cfg.warmup_steps + cfg.timing_steps
    assert num_batches >= total_steps, (
        f"task has only {num_batches} full batches of {cfg.batch_size} "
        f"({ds.num_items} items) but warmup+timing needs {total_steps}; "
        f"lower batch_size or the step counts"
    )
    print(
        f"{task.db_name}/{task.table_name}/{task.split}: {ds.num_items} items, "
        f"bs={cfg.batch_size}, ctx={cfg.ctx_size}; "
        f"{cfg.warmup_steps} warmup + {cfg.timing_steps} timed steps"
    )

    ctx_times, fwd_times = [], []
    with torch.inference_mode():
        for step in range(total_steps):
            # step 1: rustler context construction (CPU, main process)
            tic = time.perf_counter()
            tup = ds.sampler.batch_py(step, cfg.batch_size, cfg.ctx_size)
            batch = ds._process_batch(tup)
            ctx_time = time.perf_counter() - tic

            batch_mask = batch.pop("batch_mask")
            assert batch_mask.all(), "phantom rows in a timed batch"

            # step 2: forward pass (H2D copy + GPU compute, fully synced)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            net.predict(batch, [cfg.ctx_size], cfg.device, task)
            torch.cuda.synchronize()
            fwd_time = time.perf_counter() - tic

            if step >= cfg.warmup_steps:
                ctx_times.append(ctx_time)
                fwd_times.append(fwd_time)

    report("context construction", ctx_times, cfg.batch_size)
    report("forward pass", fwd_times, cfg.batch_size)
    report("total", [c + f for c, f in zip(ctx_times, fwd_times)], cfg.batch_size)


if __name__ == "__main__":
    main(tyro.cli(TimingConfig, description=__doc__))
