# Pretraining

One pretraining run of the Relational Transformer on the Join, with the
benchmark's forecast tasks as in-loop validation. The checkpoint it produces is
what `expts/fine_tune` loads as its pretrained arm.

## Running it

```
pixi run python expts/pretrain/submit.py
```

One job, 4xB200 on the preemptible `il-lo` queue (21d wall).

Logs and `args.json` land in
`/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain`,
checkpoints and `params.json` under `/dfs/user/ranjanr/ckpts/rtv2/<project>/<run_id>`.

Neither preemption nor the wall clock needs you: both requeue and resume from
the run's own checkpoint (see [`roach.slurm`](../../src/roach/slurm/README.md)).

## Inputs

Both directories are produced by [`expts/preprocess`](../preprocess/README.md)
and have to exist before submitting:

- `pre_dir` -- the Join, preprocessed. `db_task_list` is its `rt-j.json`, the
  mixture the run trains on.
- `eval_pre_dir` -- RelBench, preprocessed. `eval_db_task_list` is its
  `forecast.json`, evaluated on `val` every `eval_freq` steps, which is what
  makes transfer visible during the run rather than after it.
