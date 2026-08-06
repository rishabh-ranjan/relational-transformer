# Fine-tuning

What a Relational Transformer gets from training on one task, and how that
compares with the task-specific baselines.

The first arm is the control: **rel-f1 `driver-top3` from random init**. It is
the smallest training set in the benchmark (1.4k rows, the leftmost column of
`results.md`), so it is where a from-scratch model has the least to work with and
where pretraining should be worth the most -- which makes it the number the
fine-tuned arm has to beat.

## Running it

```
pixi run python expts/fine_tune/submit.py                # batch, 4xB200
pixi run python expts/fine_tune/submit.py --interactive  # in a held allocation
```

Both submit `expts.fine_tune.train:train` with the same arguments; the flag only
changes whose allocation it runs in. Use `--interactive` while the recipe is
still moving -- it starts in seconds instead of queuing, and a crash leaves the
allocation standing -- and the plain form for the run whose number you report.
Take the allocation first:

```python
from roach.slurm import BLACKWELL_INTERACTIVE, interactive
interactive.hold(BLACKWELL_INTERACTIVE, log_root="/dfs/user/ranjanr/slurm-logs/fine-tune")
```

The allocation is 2xB200 for at most 12 hours (the `il-interactive` QOS caps
both), while each fine-tuning run takes **one** GPU
(`BLACKWELL_INTERACTIVE_1GPU`) -- so two arms run side by side, and a run's world
size does not change with how much of the allocation happens to be free. It is
not requeued, so nothing that has to survive the night belongs in it. See
[`src/roach/slurm/README.md`](../../src/roach/slurm/README.md).

Logs and `args.json` land in `/dfs/user/ranjanr/slurm-logs/fine-tune`,
checkpoints and `params.json` under `/dfs/user/ranjanr/ckpts/rtv2/fine-tune/<run_id>`.

**When it stops.** A batch run is requeued and resumes from its own `resume.pt`;
nothing to do. An interactive one is not -- resubmit with the same `run_id`
(`submit(..., run_id=...)`) and it picks up the same checkpoint.

## What is fixed and what is not

`train.py` carries pretraining's hyperparameters verbatim (they come from
`expts/data_scaling/train.py`, which is RT-J's recipe), so an arm differs from
pretraining only in what it is trained on:

- `db_task_list` is one `(db, task)` pair instead of a mixture;
- `pre_dir` is the *benchmark* data, not the Join -- fine-tuning trains where it
  is evaluated, and train/eval differ only in split;
- `load_ckpt_path` is the arm. `None` is random init; a checkpoint path is the
  pretrained arm.

`total_steps` is the one number this experiment sets on its own (pretraining's
100k steps is a mixture's worth of data, not one task's), and it is a `submit.py`
constant so a change to it is a diff.

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
