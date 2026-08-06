# Pretrain

Self-supervised pretraining of a Relational Transformer over the tasks in
`db_task_list` (the Join). Includes Muon+AdamW
optimization, stochastic weight averaging (SWA), periodic validation against
RelBench, checkpointing, and automatic selection of the best clf / reg checkpoint
by mean validation metric.

Checkpoints land in the per-run directory
`<out-root>/<entity>/<project>/<id>/` as `steps=<N>.safetensors` (live) and
`swa_steps=<N>.safetensors` (SWA); at the end the run copies the best classifier
and regressor to `best_clf.safetensors` / `best_reg.safetensors`. With
`keep_all_ckpts=False` only the checkpoints the best-so-far still points at are
kept -- the rest are deleted as they are superseded, and the latest weights stay
available in `resume.pt` (rewritten at every eval and once more at the end). Multi-GPU is
automatic under `torchrun`, and a run relaunched with the same `run_id`
resumes automatically from `resume.pt` in that same directory
(preemption-safe).

## Prerequisite: preprocessed data

Pretraining takes a `pre_dir` of preprocessed pretraining data (the
Join) and an `eval_pre_dir` of preprocessed RelBench for validation. Both are
**local directories** — either produced by [preprocess.md](preprocess.md) or
downloaded up front; nothing is fetched on demand (see
[downloads.md](downloads.md) for why, and for how to fetch a subset):

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed
```

Those two paths are what `examples/train.py` passes (`pre_dir="data/the-join-preprocessed"`,
`eval_pre_dir="data/relbench-preprocessed"`). The full preprocessed Join is
~1.5 TiB, so on a cluster fetch it **once** to shared storage and point every
run at that path.

The task mixture is given by `db_task_list` — `(db, task)` pairs as a
JSON file. Every name must be a task the db actually ships (recorded in its
`meta.json`). The curated lists ship with the data, under
`<pre_dir>/db-task-lists/`: `forecast.json` (every forecast task in the Join),
`autocomplete.json` (every `kind: autocomplete` task — predict a column of a db
table, train-split only), `all.json` (both), and `rt-j.json` (the curated RT-J
mixture, forecast + autocomplete).

## Running a training script

There is no CLI. `rt.train._train` is a function that takes every knob as a
required argument; a run is a script that calls it. Copy
[`examples/train.py`](../examples/train.py) — it passes the released RT-J
values — and edit what you want. Build the sampler once first
(`pixi run build-sampler`).

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python examples/train.py    # one GPU
```

## Multi-GPU single-node training

One process per GPU, each told who it is; the model is replicated per rank (full
model + optimizer on every rank, no sharding). Under slurm that is one line:

```bash
srun --ntasks-per-node=8 --gres=gpu:8 pixi run python examples/train.py
```

`roach.slurm.run` translates slurm's `SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS`
into torch's `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, so nothing else is needed — and
because each rank is a slurm task, a preemption signal reaches all of them.
Outside slurm, `torchrun --standalone --nproc-per-node=auto examples/train.py`
works the same way.

Give the process as much of the node's RAM as you can: by default each run
populates the preprocessed mixture into the page cache at startup
(`mmap_populate=True`) so the GPUs are fed instead of cold-faulting
the (large) data from shared storage per item.

## On a cluster

Submission lives in [`roach.slurm`](../src/roach/slurm/README.md),
a pinned dependency; that README is the reference for how a job is built and how
it survives preemption. In short: it refuses a dirty or unpushed tree, records
the commit, checks your arguments against the target's signature, and hands
slurm a script that clones that commit, builds the environment on the node, and
starts one rank per GPU.

```python
from roach.slurm import AMPERE, submit

submit("examples.train:train",
       args={"pre_dir": ..., "eval_pre_dir": ..., "out_root": ...},
       resources=AMPERE,   # or Resources(...) for a shape roach does not ship
       name="rt-j", setup=("pixi run build-sampler",),
       repo_root=..., log_root=..., clone_root=..., secrets_dir=...)
```

See [`expts/README.md`](../expts/README.md) for how experiments in this repo are
laid out, and [`expts/fine_tune/submit.py`](../expts/fine_tune/submit.py) for a
worked one -- it passes `rt.train:main` straight to `submit`, which is the least
boilerplate a run on this cluster can be.
Hard-won notes if you write your own launcher instead:

- **Name a run you may want to resume.** `run_id` names the output
  directory `<out_root>/<entity>/<project>/<run_id>/`; pass the same value again
  to pick the run's `resume.pt` back up (`roach.slurm.submit` mints one and reuses
  it across requeues). Resuming *requires* an explicit id:
  unset, it defaults to a per-rank timestamp, which names a fresh directory
  with nothing to resume from. Rank 0 is the only rank that writes there.
- **Static rendezvous.** Pass a fixed `--master-addr`/`--master-port` (derive a
  unique per-job port) rather than torchrun's dynamic c10d rendezvous — the
  dynamic store has wedged large jobs under load.
- **Full-node CPUs.** Give the training step every core on the node. Data
  loading (the rustler sampler's parallel mmap-populate and per-item context
  building) runs on rayon; a small cgroup CPU slice (e.g. Slurm's default
  `--cpus-per-task`) starves it and bottlenecks the GPUs.
- **Full-node RAM.** The preprocessed mixture is populated into the page cache;
  request the whole node's memory (`--exclusive`, `--mem-per-gpu`, or
  equivalent).
- **Preemption is safe.** SIGTERM saves `$OUT_DIR/resume.pt` and exits;
  relaunching with the same `OUT_DIR` resumes (Slurm: `--requeue` on a
  preemptible queue).
- **Shared storage for the clone.** Run from a repo checkout all nodes can
  read; the pixi env itself builds node-locally.
- **One copy of the data.** `pre_dir` is a plain path, so fetch the preprocessed
  mixture once to storage every node can read rather than per node. Startup
  populates it into the page cache, after which reads are RAM-speed.
- **Flaky InfiniBand?** `NCCL_IB_DISABLE=1` forces NCCL over TCP — slower but
  robust.

**Resume** is automatic from `$OUT_DIR/resume.pt` and **GPU-count flexible**: a
run preempted on 4×8 GPUs can resume on a single 4-GPU node with the same
`OUT_DIR` — the data stream is re-seeded by the resumed step, so nothing is
replayed and determinism holds across the world-size change. A time-based dump
every `resume_save_mins` minutes (20 in the examples) bounds lost progress.

## Avoiding data loading during debug iterations

By default each run re-populates the preprocessed data into RAM at startup. When
iterating on training code, that reload is wasted work on every restart. Lock the
data into the page cache **once** with a long-lived holder
([`examples/mlock.py`](../examples/mlock.py)), then train with
`mmap_populate=False` so reads hit the
locked cache:

```bash
# terminal 1: hold the data resident (Ctrl-C to release)
pixi run python examples/mlock.py
# terminal 2 (same node): train without re-populating
# terminal 2 (same node): train with mmap_populate=False in your script
pixi run python examples/train.py
```

This is purely a convenience for repeated local runs; it is **not required**.
(`mlock` needs a high `RLIMIT_MEMLOCK` — e.g. `ulimit -l unlimited` or
slurm `--propagate=MEMLOCK` — to lock the full mixture.)

## Loading checkpoints

A trained run's `best_clf.safetensors` / `best_reg.safetensors` (+ the run's
`config.json`) load directly:

```python
from rt.model import load_rt_model
model, config = load_rt_model("~/ckpts/run1/best_clf.safetensors", device="cuda")
```

The same call loads a released Hub checkpoint
(`load_rt_model("stanford-star/rt-j/classification")`). Use the resulting checkpoints for
[inference](inference.md).
