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

To reproduce the curated RT-J mixture exactly, add `--include-dbs-file
rt/recipe_rt_j.txt` (otherwise every preprocessed db under `--pre-dir` is used).

## Single-GPU training

The `pretrain` task launches `torchrun --standalone --nproc-per-node=auto`, which
uses every visible GPU. Pin it to one GPU for a single-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 pixi run pretrain \
  --pre-dir stanford-star/the-join-preprocessed \
  --val-pre-dir stanford-star/relbench-preprocessed \
  --out-dir ~/ckpts/run1
```

## Multi-GPU single-node training

Same command without the device pin — `--nproc-per-node=auto` picks up all GPUs
on the node, and the model is replicated per GPU via DDP (full model + optimizer
on every rank, no sharding):

```bash
pixi run pretrain \
  --pre-dir stanford-star/the-join-preprocessed \
  --val-pre-dir stanford-star/relbench-preprocessed \
  --out-dir ~/ckpts/run1
```

Give the process as much of the node's RAM as you can: by default each run
populates the preprocessed mixture into the page cache at startup
(`--mmap-populate`, on by default) so the GPUs are fed instead of cold-faulting
the (large) data from shared storage per item.

## Multi-node training

Multi-node runs under Slurm via `scripts/slurm_pretrain.sh` (one `torchrun` per
node). It is infrastructure-agnostic — pass your `--account` / `--partition` /
`--qos` and the data/out paths as env vars; nothing cluster-specific is
hardcoded. Use a **preemptible** queue (the launcher is preemption-safe via
`--requeue`) and run from a clone on **shared** storage so every node sees the
same repo + pixi manifest:

```bash
# 4 nodes x 8 GPUs = 32, preemptible, logging to wandb
PRE_DIR=stanford-star/the-join-preprocessed \
VAL_PRE_DIR=stanford-star/relbench-preprocessed \
OUT_DIR=$HOME/ckpts/rtj GPUS_PER_NODE=8 \
WANDB_PROJECT=rt-pretrain WANDB_NAME=rtj \
sbatch --nodes=4 --gres=gpu:a100:8 --exclusive \
       --account=<acct> --partition=<preemptible> --qos=<preemptible-qos> \
       scripts/slurm_pretrain.sh
```

Single node is the same with `--nodes=1` (no preemptible queue needed). Knobs
(env vars): `GPUS_PER_NODE`, `WANDB_PROJECT` / `WANDB_NAME` (omit for offline),
`NCCL_IB_DISABLE=1` (force NCCL over TCP if your InfiniBand is unreliable),
`EXTRA_ARGS` (forwarded to `pretrain.py`).

`--exclusive` is the portable way to give the job the node's full RAM (the
preprocessed mixture is populated into the page cache); some schedulers instead
require `--mem-per-gpu=<X>G` (and reject `--mem=0`). The launcher also pins the
step to all node cores so the parallel data loading isn't CPU-starved, and uses a
static rendezvous (fixed master addr/port) for robustness under load.

**Resume** is automatic from `$OUT_DIR/resume.pt` and **GPU-count flexible**: a
run preempted on 4×8 GPUs can resume on a single 4-GPU node with the same
`OUT_DIR` — the data stream is re-seeded by the resumed step, so nothing is
replayed and determinism holds across the world-size change. A time-based dump
every `--resume-save-mins` minutes (default 20) bounds lost progress.

## Avoiding data loading during debug iterations

By default each run re-populates the preprocessed data into RAM at startup. When
iterating on training code, that reload is wasted work on every restart. Lock the
data into the page cache **once** with a long-lived holder
(`scripts/mlock_recipe.py`), then train with `--no-mmap-populate` so reads hit the
locked cache:

```bash
# terminal 1: hold the data resident (Ctrl-C to release)
pixi run python scripts/mlock_recipe.py --pre-dir <PRE_DIR> --workers 32
# terminal 2 (same node): train without re-populating
pixi run pretrain --pre-dir <PRE_DIR> --out-dir ~/ckpts/run1 --no-mmap-populate
```

This is purely a convenience for repeated local runs; it is **not required**.
(`mlock_recipe.py` needs a high `RLIMIT_MEMLOCK` — e.g. `ulimit -l unlimited` or
slurm `--propagate=MEMLOCK` — to lock the full mixture.)

## Loading checkpoints

A trained run's `best_clf.safetensors` / `best_reg.safetensors` (+ the run's
`config.json`) load directly:

```python
from rt.checkpoints import load_rt_model
model, config = load_rt_model("~/ckpts/run1/best_clf.safetensors", device="cuda")
```

The same call loads a released Hub checkpoint
(`load_rt_model("stanford-star/rt-j/classification")`). Use the resulting checkpoints for
[inference](inference.md).
