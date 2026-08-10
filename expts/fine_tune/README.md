# Fine-tuning

What a Relational Transformer gets from training on one task, and how that
compares with the task-specific baselines.

`submit.py` is the default: stochastic context without token masking,
warm-started from the published RT-P checkpoint. One arm it was chosen over
keeps a submit script of its own -- `submit_stoc.py`, random init instead of
RT-P; `scan_rtp.py` is gone, so compare by hand or bring it back from git.

Tasks are submitted smallest train set first, so the smallest (the leftmost
column of `results.md`) answers first.

## Running it

```
pixi run python expts/fine_tune/submit.py
```

One job per task, one GPU each. `plan()` hands out the best slots this cluster
will give a one-GPU job, best first; its docstring is the reasoning for the
order.

Logs and `args.json` land in `/dfs/user/ranjanr/slurm-logs/fine-tune`,
checkpoints and `params.json` under `/dfs/user/ranjanr/ckpts/rtv2/fine-tune/<run_id>`.

Neither preemption nor the wall clock needs you: both requeue and resume from
the run's own checkpoint (see [`roach.slurm`](../../src/roach/slurm/README.md)),
which matters most on `il-interactive`'s 12 hours.

## Context tuning on the fine-tuned checkpoints

Tuning and ensembling in one submission, one job per task:

```
pixi run python expts/fine_tune/submit_hpo_ens.py
```

Or as two, in order, which keeps the tuning off test entirely and lets each
phase have its own subset size and its own resource plan:

```
pixi run python expts/fine_tune/submit_hpo_only.py
pixi run python expts/fine_tune/submit_ens.py
```

`submit_hpo_ens.py` tunes on validation, then ensembles the winner over 4
context seeds on test. It chooses between 36 context configurations --
`ctx_size` in {512, 1024, 2048} x `local_ctx_size` in {512, 1024, 2048} x
`bfs_width` in {64, 128, 256} x `prefer_latest` in {True, False}, minus the
combinations with `local_ctx_size > ctx_size` -- but pays only the 18 passes
of `lcs_bw_pl_grid`: the three ctx sizes ride along on each pass as prefixes
of the contexts it already built. It logs the test curve to wandb like `submit_ens_only.py`, and takes
its task list, its checkpoints and its `items_per_task` from that script, so
the tuned and untuned numbers are the same weights on the same rows.

`submit_hpo_only.py` scores the `lcs_bw_pl_grid` on **validation** only and
writes `tuning.json` (every config's score, and the winner) beside each run's
`eval_out`. Nothing reads test. `submit_ens.py` then evaluates each task's
winner on test, averaged over context seeds, and writes the RelBench
submission dir; it reads the winner out of `tuning.json`, so the tuning runs
have to have finished.

`submit_ens_only.py` is ensembling with the tuning taken out: it waits on
nothing, fixes the context at the `(2048, 128, True)` the fine-tuning runs
evaluated with, and sweeps 16 context seeds, largest train set first. Every
ensembled run -- tuned or not -- scores the running average after each seed, so
its log carries the test metric at every ensemble size, not just the last.

That run is the one that logs to wandb: the test metric against `ens_size`,
with the task's published target drawn beside it, in the same keys and units
`rt.train` uses. Its workspace is the same script as the training project's,
pointed at the other axis:

```
pixi run python expts/fine_tune/workspace.py \
    --project 2026-08-10-fine_tune_ens_only --x ens_size
```

and the curve reads as a table -- one row per task, one column per ensemble
size, the published baseline first -- with

```
pixi run python expts/fine_tune/ens_table.py
```

which shows whatever has been scored so far, so it is worth running while the
jobs are still going.

All four load the fine-tuned weights `submit.py` produced: `ckpt_for` takes the
best-on-val checkpoint of the most recent run of that task, `best_clf`/
`best_reg` (the better of the live and the SWA net; `best_live_*` and
`best_swa_*` sit beside it).

## What is fixed and what is not

`submit.py` passes `rt.train:main` directly, with pretraining's hyperparameters
verbatim (the released RT-J recipe, `examples/train.py`), so an arm differs from
pretraining only in what it is trained on:

- `db_task_list` is one `(db, task)` pair instead of a mixture;
- `pre_dir` is the *benchmark* data, not the Join -- fine-tuning trains where it
  is evaluated, and train/eval differ only in split;
- `load_ckpt_path` is RT-P by default (`None` is the random-init arm). RT-P is
  mirrored at
  `/dfs/user/ranjanr/share/stanford-star/rt-p` (compute nodes have no Hub
  access), one subdirectory per task type; refresh it with
  `huggingface_hub.snapshot_download("stanford-star/rt-p", local_dir=...)`.

`total_steps` is the one number this experiment sets on its own: pretraining's
100k steps is a mixture's worth of data, not one task's.

## Baselines

`results.csv` is the task-specific baseline sweep -- every model in RelBench's
comparison, default config and after ~30 trials of random search -- and
`results.md` is the table built from it:

```
pixi run python expts/fine_tune/make_results.py
```

That script rewrites `results.md` wholesale, so edit the script, not the
markdown. It reads train-set sizes and regression stds from the
`stanford-star/relbench` dataset repo at run time; `results.csv` itself is the
committed artifact of a sweep that is expensive to reproduce.
