# Pretrain

Self-supervised pretraining of a Relational Transformer over every task in the
preprocessed datasets at `--pre-dir` (the Join). Includes Muon+AdamW
optimization, stochastic weight averaging (SWA), periodic validation against
RelBench, checkpointing, and automatic selection of the best clf / reg checkpoint
by mean validation metric.

Checkpoints land in `<out-dir>/` as `steps=<N>.safetensors` (live) and
`swa_steps=<N>.safetensors` (SWA); at the end the run copies the best classifier
and regressor to `best_clf.safetensors` / `best_reg.safetensors`. Multi-GPU is
automatic under `torchrun`, and the run resumes automatically from
`<out-dir>/resume.pt` (preemption-safe).

## Prerequisite: preprocessed data

Pretraining takes a `--pre-dir` of preprocessed pretraining data (the Join) and a
`--val-pre-dir` of preprocessed RelBench for validation — each a local path (see
[preprocess.md](preprocess.md)) or a Hub repo, downloaded and cached on demand.
The released RT-J data:

- `--pre-dir stanford-star/the-join-preprocessed`
- `--val-pre-dir stanford-star/relbench-preprocessed`

To reproduce the curated RT-J mixture exactly, add `--include-dbs-file docs/rt_j_dbs.txt` (otherwise every preprocessed db under `--pre-dir` is used).

## Single-GPU training

The `pretrain` task launches `torchrun --standalone --nproc-per-node=auto`, which
uses every visible GPU. Pin it to one GPU for a single-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 pixi run train \
  --pre-dir stanford-star/the-join-preprocessed \
  --val-pre-dir stanford-star/relbench-preprocessed \
  --out-dir ~/ckpts/run1
```

## Multi-GPU single-node training

Same command without the device pin — `--nproc-per-node=auto` picks up all GPUs
on the node, and the model is replicated per GPU via DDP (full model + optimizer
on every rank, no sharding):

```bash
pixi run train \
  --pre-dir stanford-star/the-join-preprocessed \
  --val-pre-dir stanford-star/relbench-preprocessed \
  --out-dir ~/ckpts/run1
```

Give the process as much of the node's RAM as you can: by default each run
populates the preprocessed mixture into the page cache at startup
(`--mmap-populate`, on by default) so the GPUs are fed instead of cold-faulting
the (large) data from shared storage per item.

## Multi-node training

Multi-node runs are plain `torchrun` — one launcher per node, each spawning one
worker per GPU:

```bash
# on every node (rank 0 on the head node):
torchrun --nnodes=<N> --nproc-per-node=<GPUS> \
  --node-rank=<i> --master-addr=<head-node> --master-port=<port> \
  -m rt.cli.train --pre-dir ... --val-pre-dir ... --out-dir ...
```

Wrap this in your cluster's launcher (Slurm, k8s, ...). Hard-won notes for
writing that wrapper:

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
- **Flaky InfiniBand?** `NCCL_IB_DISABLE=1` forces NCCL over TCP — slower but
  robust.

**Resume** is automatic from `$OUT_DIR/resume.pt` and **GPU-count flexible**: a
run preempted on 4×8 GPUs can resume on a single 4-GPU node with the same
`OUT_DIR` — the data stream is re-seeded by the resumed step, so nothing is
replayed and determinism holds across the world-size change. A time-based dump
every `--resume-save-mins` minutes (default 20) bounds lost progress.

## Avoiding data loading during debug iterations

By default each run re-populates the preprocessed data into RAM at startup. When
iterating on training code, that reload is wasted work on every restart. Lock the
data into the page cache **once** with a long-lived holder
(`rt.cli.mlock`), then train with `--no-mmap-populate` so reads hit the
locked cache:

```bash
# terminal 1: hold the data resident (Ctrl-C to release)
pixi run python -m rt.cli.mlock --pre-dir <PRE_DIR> --workers 32
# terminal 2 (same node): train without re-populating
pixi run train --pre-dir <PRE_DIR> --out-dir ~/ckpts/run1 --no-mmap-populate
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
