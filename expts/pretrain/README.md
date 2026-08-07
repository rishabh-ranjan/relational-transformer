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

**Wall clock is the one stop that needs you.** Preemption requeues and resumes
itself; a `TIMEOUT` job does not, and slurm has no setting that changes that
(`JobRequeue`/`PreemptMode=REQUEUE` cover preemption and node failure only).
Resubmit with the same `run_id` and it picks up the same checkpoint. At 21 days
this run should not reach the limit; a 12h `il-interactive` one would.

## Inputs

Both directories are produced by [`expts/preprocess`](../preprocess/README.md)
and have to exist before submitting:

- `pre_dir` -- the Join, preprocessed. `db_task_list` is its `rt-j.json`, the
  mixture the run trains on.
- `eval_pre_dir` -- RelBench, preprocessed. `eval_db_task_list` is its
  `forecast.json`, evaluated on `val` every `eval_freq` steps, which is what
  makes transfer visible during the run rather than after it.
