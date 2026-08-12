# Pretrain

Self-supervised pretraining of a Relational Transformer over the tasks in
`db_task_list` (the Join). Includes Muon+AdamW
optimization, stochastic weight averaging (SWA), periodic validation against
RelBench, checkpointing, and automatic selection of the best clf / reg checkpoint
by mean validation metric.

Checkpoints land in the per-run directory
`<out-root>/<entity>/<project>/<id>/` as `steps=<N>.safetensors` (live) and
`swa_steps=<N>.safetensors` (SWA, when `swa_momentum` is set), with
`latest.safetensors` (and `latest_swa.safetensors`) pointing at the most recent
of each -- a stable path for a reader that wants the current weights of a run
still training, or of one with no val split and so no `best_*`. At the end the
run copies each net's best
classifier and regressor to `best_live_clf.safetensors` /
`best_swa_clf.safetensors` (and the `reg` pair), plus `best_clf.safetensors` /
`best_reg.safetensors` for whichever of the two nets won. Selection is by mean
**validation** metric only -- AUROC for clf, NMAE for reg -- so a test split
evaluated alongside never picks a checkpoint. With
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
JSON file. Names resolve against the tasks the db ships (recorded in its
`meta.json`); one the build cannot predict — a recommendation task, or an
entry left over from an older build — is reported and ignored, not fatal. The curated lists ship with the data, under
`<pre_dir>/db-task-lists/`: `forecast.json` (every forecast task in the Join),
`autocomplete.json` (every `kind: autocomplete` task — predict a column of a db
table, train-split only), `all.json` (both), and `rt-j.json` (the curated RT-J
mixture, forecast + autocomplete).

`train_splits` picks which splits of those tasks the training stream draws
from. `["train"]` is the usual choice; `["train", "val"]` fine-tunes on the
validation labels too, which means `eval_splits` must drop `"val"` — a split
that is trained on cannot select the checkpoint, and with no val metric the
final step is what the run keeps (and `early_stop_after_steps` must be `None`).

`swa_momentum` is the momentum of the running weight average kept beside the
live net, evaluated, checkpointed and selected on in parallel with it. `None`
turns SWA off: no second net is built, so there are no `swa_steps=` or
`best_swa_*` files and nothing is selected on a `swa/` metric. It cannot be
changed across a resume — the average is part of `resume.pt`.

`wd` decays every weight matrix — the hidden matrices Muon holds and the
encoders'/decoders' alike — and never a gain or a bias. That choice is made on
its own, not by which optimizer a parameter went to: Muon takes hidden matrices
only, but decay is about shape, so the two splits are not the same one.

`delta_finetune` trains a zero-initialized additive delta on frozen pretrained
weights rather than the weights themselves. The gradient of the delta is the
gradient of the weight, and Muon's lr scaling depends on shape alone, so the
update is unchanged — what moves is the point weight decay pulls to: the
pretrained weights instead of zero. It needs a `load_ckpt_path`, and with
`wd=0` it is exactly ordinary fine-tuning.

`db_cutoff` trims the database the contexts are built from to a split's
timestamp: `"test"` (relbench's `get_db(upto_test_timestamp=True)`), `"val"`
one split earlier, or `None` for no trim. Evaluate a split under its own
cutoff — a val metric scored against a test-cutoff database sees the rows the
val labels postdate, and stops predicting test.

`eval_ensemble_size` averages the in-loop eval over that many context seeds
before scoring — the **predictions** are averaged per item, as
`rt.eval.run_ensemble` does, not the per-seed scores. `1` is the ordinary
single-context eval and uses `eval_context_seed` directly; above that the seeds
are the same mixed family `rt.eval` sweeps, so member *i* matches. Each member
is one more evaluator, built once and reused at every eval point: one more pass
over the eval split per eval, and one more set of loader workers resident for
the run.

`optimizer` selects `"muon"` — hidden weight matrices to Muon, the per-sem-type
encoders/decoders and every 0/1-D parameter to AdamW, which is what the released
checkpoints were trained with — or `"adamw"`, one AdamW over all of them with the
same weight-decay split. It cannot be changed across a resume: the optimizer
state in `resume.pt` is per optimizer.

`lr_warmup_steps` and `lr_decay_steps` shape the learning rate: linear warmup
from 0 over the first, linear decay to 0 over the last that many steps of
`total_steps`, and `0` disables either end. The decay is measured back from
`total_steps`, so a run that ends before it — early stopping, or a
`total_steps` set far beyond where the run is meant to stop — never reaches the
decay at all.

`early_stop_after_steps` ends the run early once neither the live nor the SWA
val metric has improved on — or matched again — its best for that many steps,
checked at each eval. `None` runs the full `total_steps`. It needs `"val"` in
`eval_splits`; with nothing selected there is nothing to stop on.

## Running a training script

There is no CLI. `rt.train._train` is a function that takes every knob as a
required argument; a run is a script that calls it. Copy
[`examples/train.py`](../examples/train.py) — it passes the released RT-J
values — and edit what you want. `pixi install` builds the rustler sampler as
part of the environment; nothing else to build.

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
       name="rt-j",
       repo_root=..., log_root=..., clone_root=..., secrets_dir=...)
```

See [`expts/README.md`](../expts/README.md) for how experiments in this repo are
laid out, and [`expts/fine_tune/submit.py`](../expts/fine_tune/submit.py) for a
worked one -- it passes `rt.train:main` straight to `submit`, which is the least
boilerplate a run on this cluster can be.

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
