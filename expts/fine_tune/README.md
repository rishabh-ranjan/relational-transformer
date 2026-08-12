# Fine-tuning

What a Relational Transformer gets from training on one task, and how that
compares with the task-specific baselines.

`submit.py`'s docstring says what the current arm is -- the values it lists
change every submission, so they are described where they live, not here.

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

**Progress is in wandb, not in the log.** After `time_to_first_step` a run's
stdout only prints `resume_saved_at_step` every `resume_save_mins`, so a quiet
log is the normal state and says nothing either way. Read the step count per run
from the API instead:

```python
import wandb
for r in wandb.Api().runs("rtv2/2026-08-12-fine_tune"):
    print(r.name, r.state, r.summary.get("step"))
```

A run is alive when that step is higher than the one you read last round. Keep
the log for what wandb does not carry: a traceback, `resumed_from`, a preemption.

Neither preemption nor the wall clock needs you: both requeue and resume from
the run's own checkpoint (see [`roach.slurm`](../../src/roach/slurm/README.md)),
which matters most on `il-interactive`'s 12 hours.

## Ensembling the fine-tuned checkpoints

```
pixi run python expts/fine_tune/submit_ens.py
```

One job per task, each waiting `afterok` on that task's fine-tuning job and
loading the `latest.safetensors` it leaves behind -- `submit.py` trains on val,
so nothing selects a checkpoint and the last step is the run. A task whose
training has already finished is submitted with no dependency.

It sweeps 8 context seeds at the context `submit.py` trains under, scoring the
running average over the **whole** test split after every seed: one job is the
metric at every ensemble size up to 8, and the last point is a RelBench-valid
number rather than the subsample the training curve carries. Preemption costs
one seed -- `rt.eval` writes `ensemble_resume.pt` beside `eval_out` and a
requeued attempt picks the sums back up.

## Workspaces

`workspace.py` writes the view the project's bare URL opens on, so there is
nothing to pick from the view menu. Rerun it whenever a task starts logging a
key the view has no panel for; it rewrites the layout wholesale, so edit
`workspace.py`, never the UI.

```
pixi run python expts/fine_tune/workspace.py --project 2026-08-12-fine_tune
```

It prints the URL it wrote: <https://wandb.ai/rtv2/2026-08-12-fine_tune>.

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
