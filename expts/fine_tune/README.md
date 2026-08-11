# Fine-tuning

What a Relational Transformer gets from training on one task, and how that
compares with the task-specific baselines.

`submit.py` is the default: stochastic context without token masking,
warm-started from the published RT-P checkpoint.

Tasks are submitted smallest test set first, so the fastest answers land first.

## Running it

```
pixi run python expts/fine_tune/submit.py
```

One job per task, one GPU each. `RESOURCES` assigns a slot per task by hand;
work it out again against the live cluster every submission, and write the
reasoning into the comment above it.

Logs and `args.json` land under
`/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune`,
checkpoints and `params.json` under `/dfs/user/ranjanr/ckpts/rtv2/fine-tune/<run_id>`.

Neither preemption nor the wall clock needs you: both requeue and resume from
the run's own checkpoint (see [`roach.slurm`](../../src/roach/slurm/README.md)),
which matters most on `il-interactive`'s 12 hours.

## Workspaces

`workspace.py` writes the view the project's bare URL opens on, so there is
nothing to pick from the view menu. Rerun it whenever a task starts logging a
key the view has no panel for; it rewrites the layout wholesale, so edit
`workspace.py`, never the UI.

```
pixi run python expts/fine_tune/workspace.py --project 2026-08-11-fine_tune
```

It prints the URL it wrote: <https://wandb.ai/rtv2/2026-08-11-fine_tune>.

## What is fixed and what is not

`submit.py` passes `rt.train:main` directly, with pretraining's hyperparameters
verbatim (the released RT-J recipe, `examples/train.py`), so an arm differs from
pretraining only in what it is trained on:

- `db_task_list` is one `(db, task)` pair instead of a mixture;
- `pre_dir` is the *benchmark* data, not the Join -- fine-tuning trains where it
  is evaluated, and train/eval differ only in split;
- `load_ckpt_path` is RT-P. It is mirrored at
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
