# Pretraining

One pretraining run of the Relational Transformer on the Join, with the
benchmark's forecast tasks as in-loop validation. The checkpoint it produces is
what `expts/fine_tune` loads as its pretrained arm.

## Before the first submission

Anyone on the cluster can run this from their own checkout; nothing is tied to
one account. What the submitting user has to have:

- A slurm association on account `infolab` with the `il` and `il-lo` QOS
  (`sacctmgr show assoc user=$USER format=account,qos`).
- `/dfs/user/$USER/.secrets/{wandb,huggingface,github}`, each holding one
  token (`chmod 700` the directory). The job reads them on the node; the wandb
  key must belong to a member of the `rtv2` team, the entity the run logs to.
- `pixi` on the login host and `pixi install` run once in the checkout.
- Push access to `origin`: `submit` refuses a commit that is not on
  `origin/<branch>`, because the job clones that commit. A fork works the same.

Logs, checkpoints and per-node clones all land under the submitting user's own
`/dfs/user/$USER` and `/lfs/local/0/$USER`; the first job on a node sets that
node up (roach's ILC cluster env). Inputs are shared and read-only under
`/dfs/user/ranjanr/share`.

## Running it

```
pixi run python -m expts.pretrain.submit --gpus a100:8 --qos il            # new run; prints its run_id
pixi run python -m expts.pretrain.submit <run_id> --gpus a100:8 --qos il   # resume that run
pixi run python -m expts.pretrain.submit --gpus b200:2 --qos il --nodelist blackwell1
```

`--gpus b200:2` on `il` is 2 cards of blackwell1 (the `il` b200 sub-cap),
non-preemptible, 7d at a time; `il-interactive` gives the same 2 cards at a
higher priority but for 12h at a time. The run requeues and resumes across the
wall clock either way (each restart costs a cold start, see
[MONITOR.md](MONITOR.md)). `autoscale.py`
only knows the ampere shape.

With `--gpus a100:8`: 8xA100 per node, `--exclusive`. A single node goes on `il` -- not preemptible,
7d wall, but capped at 10 a100 per user, so it fits one node and nothing wider;
everything wider goes on the preemptible `il-lo` (21d wall). Prefer `il`
whenever the cap allows, *including for a job that will sit in the queue*: a
pending `il-lo` job is preemptible the moment it starts, and queueing on `il`
costs no more. How many nodes, and which queue, is not a constant -- it is
whatever the cluster will hand over right now:

```
pixi run python -m expts.pretrain.autoscale <run_id>   # take the widest free shape
```

Run that on a timer for the life of the run. It takes 4 whole nodes when 4 are
free, else 2, else 1, only ever moving to a shape slurm starts immediately, and
puts a single-node run on the non-preemptible `il` queue when that QOS's
10-a100 cap allows. **[MONITOR.md](MONITOR.md) is the operating manual** -- the
policy, what to watch, what is routine, and the failure modes that have
actually happened. Read it before babysitting this run.

Before trusting a multi-node shape -- after touching distributed setup, or on
nodes a run has just hung on -- [`smoke.py`](smoke.py) runs the same launch
path on rel-f1 for 20 steps, which takes about a minute:

```
pixi run python -m expts.pretrain.smoke --nodelist ampere3,ampere9
```

Neither preemption nor the wall clock needs you: both requeue and resume from
the run's own checkpoint (see [`roach.slurm`](https://github.com/rishabh-ranjan/roach)),
and resume is GPU-count flexible, which is what lets the shape change under a
running experiment without costing work.

Logs and `args.json` land in
`/dfs/user/$USER/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain`,
checkpoints and `params.json` under `/dfs/user/$USER/ckpts/rtv2/<project>/<run_id>`.

## Inputs

Both directories are produced by [`expts/preprocess`](../preprocess/README.md)
and have to exist before submitting:

- `pre_dir` -- the Join, preprocessed. `db_task_list` is its `rt-j.json`, the
  mixture the run trains on.
- `eval_pre_dir` -- RelBench, preprocessed. The validation tasks are
  [`eval-tasks.json`](eval-tasks.json) beside this file, passed inline and
  evaluated on `val` every `eval_freq` steps, which is what makes transfer
  visible during the run rather than after it. They are listed here rather than
  read from the published `forecast.json` by path because this repo lives on
  the submitting host's local disk, which a compute node cannot see.
