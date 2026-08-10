# Experiments

One directory per experiment, laid out however that experiment wants. It has to
submit jobs and it has to be committed: jobs clone the commit you submit from.

## Layout

- **One file where one file will do**, `fine_tune/submit.py` being the shape to
  copy: submit the entry point directly, every argument spelled out at the call.
  A sweep is a loop around that call. A second file has to earn itself.
- **No module-level `CONSTANTS` for a value one call site consumes.** Write the
  value in the argument that takes it. Bulky or wanted twice: a function that
  returns it, beside its use (`targets_for(db, task)`). `TASKS`, which the sweep
  loops over, is the exception.
- **Keep a derived input beside the file that uses it** — a curated task list, a
  subset.

## Editing submit.py

- **Edit it, do not configure it.** No arguments, no flags, no `--dry-run`, no
  `if variant == ...`. Expect to change the file every submission.
- **Comment out to switch.** Leave the shape you are not using sitting there
  commented; coming back to it is uncommenting.
- **Commit every submission**, before submitting: the job clones that commit.
  Rewrite the file freely, git holds the variants.

## Watch every job you submit

A job is not finished when `sbatch` returns. Nothing alerts you, and a broken
run holds its GPU for the full wall clock while filling the shared filesystem.
Shortly after a submission, and again as it runs:

- read the log and confirm the run reaches steps and the loss moves;
- `ls -laS` the log directory and `df -h` the output filesystem;
- cancel what is broken, delete what it wrote, fix the cause, resubmit.

## Submitting

Jobs go through [`roach.slurm`](../src/roach/slurm/README.md) — read that first
for `submit()`, resource presets, and how a run survives preemption.

```python
from roach.slurm import BLACKWELL, submit

submit("expts.<name>.<module>:<function>", args={...}, resources=BLACKWELL,
       name=..., setup=("pixi run build-sampler",),
       repo_root=..., log_root=..., clone_root=..., secrets_dir=...)
```

- **Submit it, do not `srun` it**, a two-minute probe included. The repo lives
  on the submitting host's `/lfs`, which the compute node does not have; an
  `srun` from inside an allocation is a step of that job and inherits its
  limits; `Resources` carries the account, partition and constraint.
- **A probe is an experiment**: give it a directory under `expts/`, commit it,
  submit it as its own target. An uncommitted target cannot be run at all.
- **Write nowhere but the paths you pass the entry point.** The clone is shared
  by every job at that commit on a node and is read-only while jobs run; take an
  output root as an argument and put everything under it. See the
  [read-only section](../src/roach/slurm/README.md#the-clone-is-read-only).
- **`/tmp` is for a job's own scratch on its node, named after the run** — never
  for code, never for anything to be read afterwards (it is node-local, and a
  `#SBATCH -o /tmp/...` log vanishes).

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
- **What the code can prevent, it prevents** — a failure hit once is made
  impossible, not documented. Prose is for what code cannot fix: bad hardware, a
  preemptible queue, a rule a future change could break.

`preprocess/` is the worked example.

## Commit what a re-run needs, delete the rest

- **Commit what a future re-run would want**: the submit script, a list that
  cannot be recomputed, a measurement that cost a sweep.
- **Delete everything else, wherever it landed** — scratch clones, probe logs
  and checkpoints, throwaway scripts, half-finished output. Nothing sweeps the
  shared filesystem, the node-local disks or `/tmp` for you.
- **The test is "would I read this next time, or write it again?"** Keep it only
  when re-deriving it is the expensive part.
- **Clean up when the question is answered**, not later.

## Say it once

- **Every fact has one home, and the others point at it.** These conventions
  live here; an experiment's README covers that experiment; a module docstring
  covers that module. No docstring re-explains that submit scripts are edited
  rather than configured, that a job clones the commit you submit from, or what
  `clone_root` and `setup=` are for.
- **Operational instructions only.** No history, no incident stories, no fixed
  bugs, no rationale for a settled choice. Write what to run and what to check;
  git holds how it came about.
- **Only what the reader cannot get from the code.** Prefer a link to a summary.

## What is specific to this repo

- **`setup=("pixi run build-sampler",)`** — the rustler sampler is a compiled
  extension, so a job builds it inside its clone.
- **Data is a local directory.** Nothing is fetched from the Hub at run time
  (see [docs/downloads.md](../docs/downloads.md)); point `pre_dir` at a path
  every node can read, and expect a job's first minutes to go on populating it
  into the page cache.
- **Entry points take every argument explicitly.** `rt.train._train` has no
  defaults; [`examples/`](../examples/) has the released values to start from.
