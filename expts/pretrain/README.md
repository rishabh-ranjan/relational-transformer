# Pretraining

One pretraining run of the Relational Transformer on the Join, with the
benchmark's forecast tasks as in-loop validation. The checkpoint it produces is
what `expts/fine_tune` loads as its pretrained arm.

## Running it

```
pixi run python expts/pretrain/submit.py            # new run; prints its run_id
pixi run python expts/pretrain/submit.py <run_id>   # resume that run
```

8xA100 per node, `--exclusive`, on the preemptible `il-lo` queue (21d wall).
How many nodes, and which queue, is not a constant -- it is whatever the
cluster will hand over right now:

```
pixi run python expts/pretrain/autoscale.py <run_id>   # take the widest free shape
```

Run that on a timer for the life of the run. It takes 4 whole nodes when 4 are
free, else 2, else 1, only ever moving to a shape slurm starts immediately, and
puts a single-node run on the non-preemptible `il` queue when that QOS's
10-a100 cap allows. **[MONITOR.md](MONITOR.md) is the operating manual** -- the
policy, what to watch, what is routine, and the failure modes that have
actually happened. Read it before babysitting this run.

Neither preemption nor the wall clock needs you: both requeue and resume from
the run's own checkpoint (see [`roach.slurm`](../../src/roach/slurm/README.md)),
and resume is GPU-count flexible, which is what lets the shape change under a
running experiment without costing work.

Logs and `args.json` land in
`/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain`,
checkpoints and `params.json` under `/dfs/user/ranjanr/ckpts/rtv2/<project>/<run_id>`.

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
