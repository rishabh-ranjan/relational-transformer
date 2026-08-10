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

**No module-level `CONSTANTS` for values one call site consumes.** A name in
caps at the top of the file is the "constant three screens away" this layout
exists to avoid: write the value in the argument that takes it. When a value is
bulky or wanted in two places, a function that returns it (`targets_for(db,
task)` in `fine_tune/submit.py`) keeps it next to its use and lets the second
caller ask for exactly what it needs — a shared `TARGETS` dict does neither.
`TASKS` is the exception the rule allows: the sweep loops over it, and it is
edited every submission.

**`submit.py` is edited, not configured.** It is scratch that happens to be
committed: expect to change it every time you submit, because what you submit
next depends on what slurm has free right now — a different card, a different
QOS, half the sweep, one task you want to see again. So it takes no arguments
and carries no flags, no `--dry-run`, no `if variant == ...`. A knob that exists
to avoid editing the file is a code path that has to keep working; editing the
file is the interface.

Commenting out is the right way to switch. Leave the shape you are not using
sitting there commented, so coming back to it is uncommenting rather than
rewriting from memory — a commented-out resource tier or task list is a record
of what was tried, and costs nothing.

Git is what makes this safe: every submission is committed (the job clones that
commit), so the file's history is the log of every variant that ran, and any of
them can be recovered exactly. Rewrite it freely.

**Scripts live here, not in `/tmp`.** A one-off probe -- "why is this run
stuck?", "how slow is this loader?" -- is still an experiment: give it a
directory under `expts/` and submit it the same way. Writing it to `/tmp`
instead breaks two things at once. Roach jobs clone the commit you submit from,
so a target that is not committed cannot be run at all; and `/tmp` is
node-local, so a job's output written there lands on whichever node ran it and
is unreadable from anywhere else (a `#SBATCH -o /tmp/...` log simply vanishes).
`/tmp` is for a job's *own* scratch, on the node, named after the run --
never for the code, and never for anything you intend to read afterwards.
Delete the directory when the question is answered; see "Commit what a re-run
needs" below.

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

## Commit what a re-run needs, delete the rest

Two halves of one rule, and the second is the one that gets skipped.

**Commit anything a future re-run would want**: the submit script, a curated
list that cannot be recomputed, a measurement that cost a sweep to produce. If
regenerating it needs a resource that may be gone, keep the artifact and not
only the recipe.

**Delete everything else, wherever it landed** — the scratch clone, the probe
job's logs and checkpoints, the throwaway script that answered one question, the
half-finished output of a run nobody will read. Not only under `expts/`: the
shared filesystem, the node-local disks and `/tmp` are where this collects, and
nothing sweeps them for you. A file kept "just in case" is a file the next
person has to work out the status of.

The test is not "was this useful?" but **"would I read this next time, or write
it again?"** A one-off script that took ten minutes and depended on a state of
the world that has since moved on is faster to rewrite than to trust, so it is
residue however clever it was. Keep it only when re-deriving it is the expensive
part.

Clean up when the question is answered, not later: the person who knows which
files were the experiment and which were the scaffolding is you, now.

## Say it once

**Every fact has one home, and the others point at it.** These conventions live
here; an experiment's README covers what is specific to that experiment; a module
docstring covers what is specific to that module. A paragraph that would be true
of any experiment in this directory belongs here and nowhere else — restating it
in a submit script's docstring means two copies to keep true, and the copy nobody
edits is the one that goes stale.

So: no docstring re-explains that submit scripts are edited rather than
configured, that the job clones the commit you submit from, or what `clone_root`
and `setup=` are for — [`roach.slurm`](../src/roach/slurm/README.md) and this
file already do. Same downward: an experiment README does not re-derive its own
module docstrings.

Concise, and only what the reader cannot get from the code. Prefer a link to a
summary.

## Submitting

Jobs go through [`roach.slurm`](../src/roach/slurm/README.md) — read that
first; it covers `submit()`, resource presets, and how a run survives
preemption. It lives in this repo, so the commit a job clones pins the
submission machinery along with everything else.

```python
from roach.slurm import BLACKWELL, submit

submit("expts.<name>.<module>:<function>", args={...}, resources=BLACKWELL,
       name=..., setup=("pixi run build-sampler",),
       repo_root=..., log_root=..., clone_root=..., secrets_dir=...)
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
