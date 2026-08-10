# rel-avito sampling

Why a fine-tuning run on `rel-avito/user-clicks` logs nothing after its data
loads. The run (`expts/fine_tune`, 1k steps, one B200 on `il-lo`) prints
`train_tasks_loaded` and `eval_tasks_loaded` and then sits there: no step 1, no
eval, no wandb point. Everything before the first optimizer step has finished
except the first microbatch, which points at the sampler (`rustler`, `fly.rs`)
rather than at the model.

## Running it

```
pixi run python expts/avito/submit.py
```

One CPU-only job on blackwell1 -- the node the stalled run is on, so the same
data path and the same page cache. It prints how long `get_tasks` and
`TrainDataset` take, then the wall time of each of the first five microbatches;
logs land in `/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/avito`.

`probe.py` uses `num_workers=0` on purpose: the fine-tuning job's 16 workers
turn a blocked sampler into silence from the parent, while in-process a stall is
a stack this job's own SIGQUIT (or `py-spy dump`) can show.

The dataset arguments are copied from `expts/fine_tune/submit.py`. They have to
stay copied -- a probe that samples differently from the job it is explaining
measures nothing -- so change both or neither.

## What could be slow

In rough order of what would explain minutes rather than seconds:

- **Context assembly per item.** `bfs_width=32` at `ctx_size=1024` over
  rel-avito's `SearchStream`/`VisitStream` fanout is a far bigger neighborhood
  than rel-f1's; a hub row makes the BFS frontier explode.
- **`timeout_per_item=10.0`.** An item that times out costs ten seconds and is
  retried, so a task where most items time out crawls without ever erroring.
- **Cold `/dfs`.** `mmap_populate=True` faults the whole task in, and rel-avito
  is the largest database in the set on the slowest filesystem here.
- **The first batch only.** `tokens_per_gpu=2**17` at `ctx=1024` is 128 items
  per microbatch, all sampled before anything is yielded.

The timings the probe prints separate these: a slow `TrainDataset` line is the
mmap, a slow *first* batch that later batches do not repeat is warm-up, and
uniformly slow batches are the sampler itself.
