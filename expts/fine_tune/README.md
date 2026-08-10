# Fine-tuning

What a Relational Transformer gets from training on one task, and how that
compares with the task-specific baselines.

`submit.py` is the default: stochastic context, warm-started from the published
RT-P checkpoint. The arms it was chosen over each keep a submit script and a
scan that reads the pair off wandb:

- `submit_stoc.py` -- random init instead of RT-P; `scan_rtp.py` is gone, so
  compare by hand or bring it back from git;
- `submit_nonstoc.py` -- one fixed context instead of a sampled one, read by
  `scan_stoc.py`.

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
