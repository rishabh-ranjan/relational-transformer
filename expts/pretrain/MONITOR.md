# Babysitting a pretraining run

The run is long, preemptible, and resumes from its own checkpoint, so
monitoring is about keeping a job *scheduled*, not about restarting work. These
are the rules a future run (or agent) should follow directly.

## Submit

`submit.py` pins the shape. Today that is 8xA100, `--exclusive`, qos `il-lo`
(21d wall, preemptible), with an explicit `nodelist`.

- `--exclusive` is deliberate: the mixture is populated into the page cache, so
  the job wants the node's whole memory. With it, `cpus_per_task` is 128/8=16
  (the 14-per-gpu site limit only applies to shared jobs).
- The node set is expressed as a `nodelist` of the healthy nodes, not as an
  exclusion, because `Resources` has no exclude field.
- `submit` clones the commit you submitted from, so **commit and push before
  submitting** or the job runs stale code.

## Watch

Two watches, both cheap:

1. **Queue state** -- poll `squeue -j <id> -h -o "%T %R %N"` every 60s and emit
   only on change. This catches PENDING->RUNNING, the node it landed on, and a
   preemption (which shows up as the job going back to PENDING under the *same*
   job id, since slurm requeues it). When the job leaves the queue entirely,
   read the terminal state from `sacct -j <id> -X -n -o State,ExitCode`.
2. **The log** -- `tail -F` the job's `.out` and grep for
   `Traceback|Error|FAILED|assert|Killed|OOM|out of memory|PREEMPT|requeu|restarts=|CANCELLED`
   plus a progress signal. Grep the failure signatures, not only the happy path:
   silence from a success-only filter looks identical to a crashloop.

`restarts=N` in the log banner is the requeue counter; a bump with the run
resuming from its checkpoint is normal and needs no action.

## Bad nodes

Nodes go bad and come back. Treat the node set as something to revise, not as a
constant:

- Drop a node from the `nodelist` when jobs on it fail in ways healthy nodes do
  not. (ampere9 was the case that motivated this.)
- A requeued job keeps the nodelist it was submitted with. If it is stuck
  PENDING because its nodes are all busy, widen it in place rather than
  resubmitting -- resubmitting loses queue priority:

  ```
  scontrol update JobId=<id> NodeList=ampere1,...,ampere8
  ```

- Before re-including a node that was dropped, check it is actually healthy
  (`sinfo -p il -o "%n %G %t %C %m"`, plus a short job on it) -- an exclusion is
  a hypothesis with a date on it, not a permanent fact.
- Prefer whichever ampere node is fully idle: an `--exclusive` job cannot start
  on a node carrying any other job.

## When to interrupt a human

Preemption, requeue, and pending are routine -- do not escalate them. Escalate
a crash that repeats across restarts, an OOM, or a job that has left the queue
with a non-zero exit.
