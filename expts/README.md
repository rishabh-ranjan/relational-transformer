# Experiments

One directory per experiment, laid out however that experiment wants. It has to
submit jobs and it has to be committed — jobs clone the commit you submit from,
so the directory is the record of what was run.

**One file where one file will do, and arguments written where they are passed.**
`fine_tune/submit.py` is the shape to copy: it submits `rt.train:main` directly,
with every argument spelled out at the call, so there is no wrapper function to
keep in step with the entry point and no constant defined three screens away from
its only use. The call *is* the recipe, and a change to the experiment is a diff
in one place. A sweep is a loop around that call; an experiment that needs a
derived input (a curated task list, a subset) keeps it beside the file that uses
it. Nothing here enforces a shape, but a second file has to earn itself.

## What every experiment owes

**Enough to run the work again, and a README that says how.** Someone who was
not there should be able to reproduce it by reading that file, in order, without
reconstructing anything from a conversation or a shell history.

- **Every input is fetched by code in the directory**, not by a command someone
  typed once. Data downloaded by hand is a step nobody can repeat.
- **Anything derived, expensive and depended on is committed beside the code** —
  measured sizes, a curated list that cannot be recomputed. If regenerating it
  needs a resource that may be gone, keep the artifact, not the recipe alone.
- **What the code can prevent, it prevents.** A failure that was hit once should
  be impossible the next time, not documented. Prose is for what code cannot
  fix: a machine with bad hardware, a preemptible queue, a rule a future change
  could break.
- **The README opens with the commands, in order** — what to run, what it costs,
  what to do when it stops. Numbers that were measured, not guessed, and said to
  be measurements.

Not a diary. It is the difference between an experiment that can be re-run and
one that merely happened; `preprocess/` is the worked example.

## Submitting

Jobs go through [`roach.slurm`](../src/roach/slurm/README.md) — read that
first; it covers `submit()`, resource presets, and how a run survives
preemption. It lives in this repo, so the commit a job clones pins the
submission machinery along with everything else.

```python
from roach.slurm import BLACKWELL, submit

submit("expts.<name>.<module>:<function>", args={...}, resources=BLACKWELL,
       name=..., setup=("pixi run build-sampler",),
       repo_root=..., log_root=..., clone_root=..., secrets_dir=...,
       clone_ttl_days=...)
```

A sweep is a python loop around that call — conditional resources, staggered
submissions, resumed run ids, whatever the experiment needs.

**Design the entry point to write nowhere but the paths you pass it.** A clone
is shared by every job at that commit on a node, so the checkout is read-only
while jobs run — a relative output path, a checkpoint saved next to the code, or
a scratch file named after the dataset rather than the run is now two processes
writing one file. Take an output root as an argument and put everything under
it. Roach cannot enforce this, and breaking it shows up as corrupt output rather
than an error; the [read-only section](../src/roach/slurm/README.md#the-clone-is-read-only)
has the details.

## What is specific to this repo

- **`setup=("pixi run build-sampler",)`** — the rustler sampler is a compiled
  extension, so a job has to build it inside its clone.
- **Data is a local directory.** Nothing is fetched from the Hub at run time
  (see [docs/downloads.md](../docs/downloads.md)); point `pre_dir` at a path
  every node can read, and expect a job's first minutes to go on populating it
  into the page cache.
- **Entry points take every argument explicitly.** `rt.train._train` has no
  defaults; [`examples/`](../examples/) has the released values to start from.
