# Pretraining

One pretraining run of the Relational Transformer on the Join, with the
benchmark's forecast tasks as in-loop validation on `val`. The checkpoint it
produces is what `expts/fine_tune` loads as its pretrained arm.

## Before the first submission

- A slurm association on account `infolab` with the `il` and `il-lo` QOS
  (`sacctmgr show assoc user=$USER format=account,qos`).
- `~/scratch/.secrets/{wandb,huggingface,github}`, one token each, directory
  `chmod 700`. The wandb key must belong to a member of the `rtv2` team.
- `pixi install` run once in the checkout, and push access to `origin`.
- Inputs, produced by [`expts/preprocess`](../preprocess/README.md):
  `pre_dir` (the Join) and `eval_pre_dir` (RelBench). The task lists,
  [`all_5gb_cutoff.json`](all_5gb_cutoff.json) and
  [`eval-tasks.json`](eval-tasks.json), sit beside this file and are read from
  the job's clone by repo-relative path, so commit them before submitting. If
  startup reports `tasks_skipped` or `ignored`, regenerate the lists with
  `expts/preprocess/finalize.py task-lists`.

Logs, checkpoints and per-node clones land under the submitting user's own
`~/scratch` and `~/`; inputs are read-only under `~/scratch/hf`.

## Running it

Write `run_id` and `resources` into [`submit.py`](submit.py), commit, push,
then

```
pixi run python -m expts.pretrain.submit    # prints the run_id
```

- `run_id=None` starts a new run; a run_id resumes that run from its
  `resume.pt`. Resume works at any GPU count.
- Ampere: whole nodes, `--exclusive`. One node fits under `il` (not
  preemptible, 7d wall, 10 a100 per user); wider goes on `il-lo` (preemptible,
  21d wall). Name exactly the idle nodes (`sinfo -p il -o "%n %t %C"`, 0
  allocated cpus) in `nodelist` to start immediately instead of queueing. A
  node the run used in the last few hours reaches the first step in minutes
  rather than tens of minutes.
- Blackwell: cards of blackwell1, shared. `il` gives 2 cards for 7d,
  `il-interactive` 2 for 12h at top priority, `il-lo` up to 4.

Preemption and the wall clock both requeue and resume from the run's own
checkpoint; no action is needed.

Logs and `args.json`: `~/scratch/relational-transformer/pretrain/slurm-logs/<run_id>_<jobid>.out`
(a new file per requeue, same run_id). Checkpoints and `params.json`:
`~/scratch/relational-transformer/pretrain/rtv2/<project>/<run_id>`.

## Watching it

1. `squeue -j <id> -h -o "%T %R %N"` every 60s, emit on change. Preemption
   reappears as PENDING under the same job id. When the job leaves the queue:
   `sacct -j <id> -X -n -o State,ExitCode`.
2. `tail -F` the `.out`, grepping
   `Traceback|Error|FAILED|Killed|OOM|out of memory|PREEMPT|requeu|restarts=|CANCELLED`
   plus a progress line. Filter out `task_skipped` bulk lines.

To find a run already in flight:

```
squeue -u $USER -h -n pretrain -o "%i|%T|%D|%N|%q"
ls -t ~/scratch/relational-transformer/pretrain/slurm-logs/*.out | head -1
```

The log name gives the run_id; `grep -c resume_saved_at_step` and `tail` give
the step. The run is finished when the log says so at `total_steps`
(100_001), not when the queue is empty.

On wandb, each attempt is its own run, id `<run_id>-<jobid>.<restart>`,
grouped under the run_id: group by `Group` and the attempts draw one curve on
the `step` axis.

Routine:

- `preempted_at_step: N` followed by a requeue; a `restarts=N` bump followed
  by `resumed_from`. The log says "wall clock is near" on preemption too.
- `time_to_first_step` of 25-45m on a node the run has not used recently, ~4m
  on one it has (page-cache population; compile is ~1m of it).
- Allocator OOM *warnings*.
- `ReqNodeNotAvail` for a few minutes after cancelling a hung job: the node is
  draining, wait.

Escalate:

- A crash that repeats across restarts, or an OOM error.
- The job leaving the queue with a non-zero exit.
- Time-to-first-step longer than the interval between preemptions: move to a
  non-preemptible queue or a smaller shape.
- A multi-node job with no log line well past the expected first step and
  every rank holding ~2 GB at 100% util is hung in a collective. Cancel it and
  run [`smoke.py`](smoke.py) on those nodes before putting the run back there.

## Smoke test

Write the shape into [`smoke.py`](smoke.py), commit, push, then

```
pixi run python -m expts.pretrain.smoke
```

Run it after touching distributed setup and on nodes a run has just hung on.
Pass/fail is in its docstring. Its logs go under `.../expts/pretrain/smoke`.
