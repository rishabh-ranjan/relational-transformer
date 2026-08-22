# Experiments

One directory per experiment, laid out however that experiment wants. It has to
submit jobs and it has to be committed: jobs clone the commit you submit from.

## Layout

[repaper/README.md](repaper/README.md) is the order of operations across the
experiments under `repaper/` that regenerate the RT-J paper's results.

`fine_tune/submit.py` is the shape to copy: submit the entry point directly,
every argument spelled out at the call, a sweep as a loop around that call.
Keep a derived input (a curated task list, a subset) beside the file that uses
it.

## Editing submit.py

A submit script is edited, not configured — no arguments, no flags, no
`--dry-run`. Expect to change the file every submission, and **commit every
submission before submitting**: the job clones that commit. Rewrite the file
freely, git holds the variants.

Nothing in it is settled, and the resource plan least of all. **Work out which
gpus and which qos this submission should ask for at the moment you submit** —
the `roach` skill's cluster page says how — and write that answer into the
file. The cluster itself is the human's call alone: a submission goes to ILC
unless the instruction names Marlowe, whose allocation is metered and shared,
and then with exactly the partition and node count the instruction gave. What the plan said last time is a record of a different cluster and a
different instruction, not a default to inherit: read it as one more variant
git is holding for you. The same goes for the task list, the hyperparameters,
and every other value in the call.

## Submitting and watching

Jobs go through [`roach.slurm`](https://github.com/rishabh-ranjan/roach), a
pinned dependency. **The cluster workflow — submitting, the ILC budget,
watching every job, rebalancing, preemption — is the `roach` skill**
(`~/.claude/skills/roach/`), not this file; read it before submitting.

```python
from roach.slurm import submit
from roach.slurm.clusters.ilc import BLACKWELL, ILC

submit("expts.<name>.<module>:<function>", args={...}, resources=BLACKWELL,
       cluster=ILC, name=..., job_env="expts/job_env.sh", repo_root=...,
       log_root="~/scratch/relational-transformer/<expt>/slurm-logs",
       clone_root="~/roach_clones", secrets_dir="~/scratch/.secrets")
```

Those three paths are the same strings on every cluster (`~` is the cluster's
home; the dotfiles make `~/scratch` its shared store), so a Marlowe submission
changes only `cluster=MARLOWE` and `resources=` from
`roach.slurm.clusters.marlowe`.

What is this repo's, on top of that:

- **`job_env="expts/job_env.sh"` at every call.** It is this project's per-job
  shell (cargo cache, `/dev/shm` sweep, `RLIMIT_MEMLOCK`) and roach sources it
  on every node before the clone is built.
- **A probe is an experiment**: give it a directory under `expts/`, commit it,
  submit it as its own target. An uncommitted target cannot be run at all.
- **Take an output root as an argument and put everything under it.** The clone
  is read-only (roach's README says why).
- **Move a checkpointing run by its `run_id`.** Read the id off the
  submission's output, check `resume.pt`'s mtime is recent, cancel, submit
  with `run_id=`; the new log must say `resumed_from: ... step: <the step it
  was at>`. A job that reports `time_to_first_step` with no `resumed_from` has
  restarted from zero.
- **What a progress line costs here:** a training step is seconds; an eval
  pass over a whole test split is minutes to hours; the first minutes of any
  job go on populating `pre_dir` into the page cache, longer with
  `compile=True`.

## What every experiment owes

A README that someone who was not there can follow, in order, to run the work
again.

- **The README opens with the commands, in order** — what to run, what it costs,
  what to do when it stops. Numbers measured, and said to be measurements.
- **Every input is fetched by code in the directory**, not by a command someone
  typed once.
- **Anything derived, expensive and depended on is committed beside the code.**
  If regenerating it needs a resource that may be gone, keep the artifact, not
  the recipe alone.
`preprocess/` is the worked example.

## Commit what a re-run needs, delete the rest

- **Commit what a future re-run would want**: the submit script, a list that
  cannot be recomputed, a measurement that cost a sweep.
- **Delete everything else, wherever it landed** — scratch clones, probe logs,
  throwaway scripts, half-finished output. Nothing sweeps the shared
  filesystem, the node-local disks or `/tmp` for you.
- **Never delete checkpoints.** Not a cancelled run's, not a superseded run's,
  not a "broken" run's: `resume.pt` is the only way to continue a run, and
  the released weights of one run are the warm start of the next. No
  instruction in this repo authorises `rm` of anything under an `out_root`;
  only the user does, explicitly, naming the run.
- **The test is "would I read this next time, or write it again?"** Keep it only
  when re-deriving it is the expensive part.
- **Clean up when the question is answered**, not later.

## Teardown

"Tear down X" is one instruction, and it means all of this, without asking
which parts:

- **Kill every slurm job of X's** -- running, pending and requeued alike --
  and nothing else's.
- **Delete every file that exists only for X**: its submit script, its scan,
  any derived input only it reads. Deleted, not commented out; git holds it.
- **Take X out of everything that mentions it**: the experiment's README, a
  shared workspace or results script, a sweep list that still names it.
- **Delete X's scratch**: clones and logs under `~`, `/tmp` and
  `~/scratch` alike. Its checkpoints stay (never delete checkpoints);
  say in the README where they are.
- **Keep the finding, not the machinery** -- a decision X settled belongs in
  the experiment's README, in a sentence.
- **Commit and push it as one change.** A half-torn-down experiment reads as a
  live one.

Whatever the answer was, the runs are over: an arm that keeps training after
its question is answered is spending someone else's cards.

## What is specific to this repo

- **Nothing to build past `pixi install`.** The rustler sampler is a compiled
  extension, but the project is an editable dependency of its own environment,
  so building the clone's environment builds it. Pass `setup=` only for work
  that is not part of the environment (fetching a model, say).
- **Data is a local directory.** Nothing is fetched from the Hub at run time
  (see [docs/downloads.md](../docs/downloads.md)); point `pre_dir` at a path
  every node can read, and expect a job's first minutes to go on populating it
  into the page cache.
- **Entry points take every argument explicitly.** `rt.train._train` has no
  defaults; [`examples/`](../examples/) has the released values to start from.
