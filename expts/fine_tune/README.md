# Fine-tuning

What a Relational Transformer gets from training on one task, and how that
compares with the task-specific baselines.

The current arm is **every forecast task from random init, with the full
attention each block runs after its three relational attentions turned on**
(`skip_full_attn=False`). Tasks are submitted smallest train set first, so the
smallest (rel-f1 `driver-top3`, 1.4k rows, the leftmost column of `results.md`)
answers first.

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
- `load_ckpt_path` is the arm. `None` is random init; a checkpoint path is the
  pretrained arm;
- `skip_full_attn` is the other arm. `True` is the three relational attentions
  per block; `False` adds a dense attention (pad mask only) after them.

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
