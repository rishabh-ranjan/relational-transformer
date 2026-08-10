# Running and babysitting the pretraining run

Everything a session needs to start this run, keep it on the best hardware the
cluster will give it, and not lose work when it is interrupted. Written for
whoever picks it up next, with no context from the session that wrote it.

The run is long (100k steps, days), preemptible, and resumes from its own
checkpoint. So babysitting is about keeping a job *scheduled and as wide as
possible* -- never about redoing work.

## Picking up a run already in flight

Nothing needs to be told to you. Find the run, then start the loop:

```
squeue -u $USER -h -n pretrain -o "%i|%T|%D|%N|%q"   # job id | state | nodes | hosts | qos
ls -t /dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain/*.out | head -1
```

The log file is `<run_id>_<jobid>.out`, so its name gives you the run_id -- the
one thing everything else is keyed on. `grep -c resume_saved_at_step` on it and
`tail` tell you where the run has got to; `total_steps` in `args.json` beside it
is where it is going (100_001).

If `squeue` shows no `pretrain` job, the run is stopped, not finished: check
the last log's tail and `sacct -j <jobid> -X -n -o State,ExitCode` for why, then
resubmit it with its run_id (below). A run is finished when its log says so and
the step count has reached `total_steps` -- not because the queue is empty.

Then, for the life of the run, every ~10 minutes:

```
pixi run python expts/pretrain/autoscale.py <run_id>
```

and keep the two watches of [Watching it](#watching-it) armed. Everything else
in this file is what to do when one of them fires.

## The one thing to know first

**A run is its `run_id`.** It names the checkpoint directory under
`/dfs/user/ranjanr/ckpts/rtv2/<project>/<run_id>`, and passing it back to
`submit.py` resumes from `resume.pt` -- model, optimizers, schedulers, SWA,
step, best-so-far -- instead of starting at step 0.

```
pixi run python expts/pretrain/submit.py                    # new run, prints its run_id
pixi run python expts/pretrain/submit.py <run_id>           # resume that run
```

Resume is **GPU-count flexible**: a run stopped on 32 GPUs comes back on 8, or
the other way round, and the data stream is re-seeded by the resumed step so
nothing is replayed. That is what makes the scaling policy below free to take.

Submitting *without* a run_id when you meant to resume silently starts a second
run from scratch. Read the run_id out of the last log or `ls -t` the ckpt dir
before you submit.

## Keeping it on the best shape: `autoscale.py`

```
pixi run python expts/pretrain/autoscale.py <run_id> --dry-run   # decide, print
pixi run python expts/pretrain/autoscale.py <run_id>             # decide, act
```

One pass reads the cluster, picks the shape, and gets there. Idempotent: a pass
with nothing to do prints a line and exits. **Run it every ~10 minutes while
the run is alive**, and after every preemption.

The policy it implements:

- **Widest whole-node shape wins** -- 4 nodes, else 2, else 1, at 8 a100 each.
- **Only whole nodes count.** The job is `--exclusive` (the mixture is
  populated into each node's page cache and wants the node's memory), so a node
  carrying anyone else's job is no use, even if its GPUs are idle. The test is
  allocated CPUs == 0, not free GPUs.
- **Never queue for a bigger shape.** An upgrade is worth having only if slurm
  starts it *now*, so the job names exactly the nodes already idle. A queued
  4-node job that waits is strictly worse than a 2-node job that runs.
- **Upgrade only on a strictly higher node count.** Cancelling a running job
  costs ~45 minutes of page-cache population; a lateral move is pure loss. The
  job's own nodes count as available when weighing an upgrade -- they return to
  the pool when it is cancelled, so 2 -> 4 needs two *more* free nodes.
- **A pending job is not progress.** If the run is queued and anything whole is
  free, replace it with a job that starts.
- **One node goes to `il` when it fits.** `il` is not preemptible but caps a100
  at 10 per user, so it holds one node's eight and nothing wider. If this
  user's other `il` jobs already hold more than 2, it falls back to `il-lo`.
  This applies to a **queued** single node as much as a running one: `il-lo` is
  preemptible, so a job waiting there is strictly worse than the same job
  waiting on `il`, and waiting costs nothing either way. Whenever the run is
  `PENDING`, check its qos with `squeue -j <id> -o "%q"` -- if it says `il-lo`
  while `il` has room (`squeue -u $USER -o "%q|%b"` shows what is held), cancel
  and resubmit with `--qos il`. `autoscale.py` gets this right now; it did not
  always.
- **Everything wider is `il-lo`**: preemptible, effectively uncapped, 21d wall.

## Watching it

Two cheap watches, both worth having:

1. **Queue state** -- poll `squeue -j <id> -h -o "%T %R %N"` every 60s, emit on
   change. Catches PENDING -> RUNNING, the nodes it landed on, and preemption
   (which reappears as PENDING under the *same* job id, since slurm requeues).
   When it leaves the queue entirely, `sacct -j <id> -X -n -o State,ExitCode`.
2. **The log** -- `tail -F` the `.out` and grep for
   `Traceback|Error|FAILED|Killed|OOM|out of memory|PREEMPT|requeu|restarts=|CANCELLED`
   plus a progress signal. Grep the failure signatures, not only the happy
   path: silence from a success-only filter is indistinguishable from a
   crashloop. Filter out `task_skipped`-style bulk lines or the watch drowns.

Logs and `args.json` are under
`/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain`,
named `<run_id>_<jobid>.out`. A requeued job appends to a new file for the new
job id, same run_id.

`restarts=N` in the log banner is the requeue counter. A bump, followed by the
run resuming from its checkpoint, is normal and needs no action.

## What is normal and what is not

Routine, do not escalate:

- **Preemption on `il-lo`.** The ranks catch the signal, save at the next step
  boundary (`preempted_at_step: N`), and slurm requeues. Note that slurm sends
  the job's requested `USR1` at the start of the *preemption* grace window too,
  so the log says "wall clock is near" for a preemption -- the action taken is
  right either way.
- **`time_to_first_step: ~45m`.** Page-cache population (`mmap_populate=True`)
  plus ~1m compile. It is why short slots make no progress at all, and why
  upgrades are only worth a strictly wider shape.
- **`tasks_skipped` / `ignored` lines at startup.** Tasks the build cannot
  predict. Zero is expected now; a non-zero count means the published task
  lists and the build disagree (see "Inputs the run needs").

Escalate to a human:

- A crash that repeats across restarts, or an OOM (as opposed to the allocator
  OOM *warning*, which is recoverable).
- The job leaving the queue with a non-zero exit.
- Time-to-first-step consistently exceeding the interval between preemptions --
  the run is livelocked and needs a non-preemptible queue or a smaller shape.

## Gotchas that have actually bitten

- **`/lfs/local/0` is a per-node symlink** to `/lfs/<host>/0`. Anything that
  resolves a path on one node and uses it on another breaks; multi-node jobs
  pass `--chdir=$REPO_DIR` (the unresolved path) for exactly this reason.
- **Every node of a multi-node job needs setting up**, not just the batch node:
  node-local HOME, pixi and the clone. `roach.slurm.bootstrap` does this with
  one srun task per node. A node that has never been used pays ~7 minutes the
  first time.
- **A silent multi-node run is hung, not slow.** No log line for well over the
  ~45m first step, GPUs at 100% util holding only ~2 GB on every rank: the
  ranks are stuck in a collective and the job will sit there until its NCCL
  watchdog times out. [`smoke.py`](smoke.py) says in a minute whether a pair of
  nodes can train at all -- run it before putting a real run back on them.
- **Cancelling a hung job drains its nodes for a few minutes.** The ranks do
  not die on SIGKILL either, so slurm marks the node `Kill task failed` and a
  resubmit sits at `ReqNodeNotAvail`. It clears on its own -- wait rather than
  resubmitting somewhere worse.
- **Nodes go bad and come back.** ampere9 once failed every job in its first
  second (`mkdir`: Input/output error) and was excluded for a while; it is fine
  now. Treat an exclusion as a hypothesis with a date on it -- check with
  `sinfo` and a short job before re-including or continuing to avoid a node.
- **A concurrent `git add -A` in this clone** can sweep your edits into someone
  else's commit. Check `git log -1 --stat` after committing.
- **`submit.py` refuses a dirty or unpushed tree**, and this clone is shared, so
  a pass can fail on a working tree that is clean again a minute later.
  `autoscale.py` cancels before it submits, so a failed submit leaves the run
  with *no job*; it now retries once and then exits non-zero with the command to
  rerun. A non-zero autoscale pass means the run is down -- act on it.

## Inputs the run needs

Both must exist before submitting, and are produced by
[`expts/preprocess`](../preprocess/README.md):

- `pre_dir` -- the Join, preprocessed. `db_task_list` is its `rt-j.json`.
- `eval_pre_dir` -- RelBench, preprocessed. The in-loop validation tasks are
  listed in [`eval-tasks.json`](eval-tasks.json) beside this file and passed
  inline, because this repo lives on the submitting host's local disk and a
  path into it does not resolve on a compute node.

If startup reports skipped or ignored tasks, the lists and the build have
diverged: regenerate the lists with
`expts/preprocess/finalize.py task-lists`, which drops tasks whose target
column preprocessing dropped and tasks whose type is not modelled.
