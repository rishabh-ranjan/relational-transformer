"""Time the training sampler on one task, in-process. See [README.md](README.md).

Everything a `rt.train` job does before its first optimizer step, minus the
model and minus the loader's worker processes: `get_tasks`, `TrainDataset`, then
microbatches pulled one at a time in this process. A stall shows up here as a
batch that does not arrive, at a line number, rather than as a wandb run that
logs nothing.

The dataset arguments are `expts/fine_tune/submit.py`'s verbatim -- the point is
to reproduce that job's sampling, so a difference between the two would make the
timing meaningless.
"""

import time

from rt.data import TrainDataset, get_tasks


def main(
    db: str,
    task: str,
    pre_dir: str,
    num_batches: int,
    num_workers: int,
) -> None:
    """Writes nothing: the timings are the output, and they go to the job log."""
    tic = time.time()
    tasks = get_tasks(pre_dir, [(db, task)], ("train",))
    print(f"get_tasks: {len(tasks)} tasks in {time.time() - tic:.1f}s", flush=True)

    tic = time.time()
    ds = TrainDataset(
        tasks=tasks,
        pre_dir=pre_dir,
        train_ctx_size_list=[1024],
        train_tokens_per_gpu=2**17,
        total_bs=128,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_size_list=[1024],
        bfs_width_list=[32],
        num_walks=0,
        walk_length=20,
        prefer_latest_list=[True],
        mask_prob_max=0.0,
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        seed=0,
        items_per_task=1000_000_000,
        mask_prob_max_shared=None,
        mmap_populate=True,
        timeout_per_item=10.0,
        vector_db_path=None,
        train_only_fallback=False,
    )
    print(
        f"TrainDataset: {ds.num_items:_} items in {time.time() - tic:.1f}s", flush=True
    )

    # num_workers=0 iterates the dataset in this process, which is the whole
    # point: a sampler that blocks blocks here, where the traceback of a
    # SIGQUIT (or py-spy) is this file, instead of inside a worker whose stall
    # the parent only sees as silence.
    if num_workers:
        from torch.utils.data import DataLoader

        it = iter(DataLoader(ds, batch_size=None, num_workers=num_workers))
    else:
        it = iter(ds)

    tic = t0 = time.time()
    for i in range(num_batches):
        next(it)
        now = time.time()
        print(f"batch {i}: {now - tic:.1f}s (total {now - t0:.1f}s)", flush=True)
        tic = now
