# Working under expts/

Private internal research code. The repo-wide rules in
[`../CLAUDE.md`](../CLAUDE.md) still hold; these override them here.
[`README.md`](README.md) has the workflow (submitting, watching jobs, cleanup).

## Code style

- **One file where one file will do.** `fine_tune/submit.py` is the shape to
  copy: submit the entry point directly, every argument spelled out at the call;
  a sweep is a loop around that call. A second file has to earn itself.
- **No module-level `CONSTANTS` for a value one call site consumes.** Write the
  value in the argument that takes it. Bulky or wanted twice: a function that
  returns it, beside its use (`targets_for(db, task)`). `TASKS`, which a sweep
  loops over, is the exception.
- **Keep a derived input beside the file that uses it** — a curated task list,
  a subset.
- **Edit a submit script, do not configure it.** No arguments, no flags, no
  `--dry-run`, no `if variant == ...`. Expect to change the file every
  submission, and commit before submitting: the job clones that commit.
- **Comment out to switch.** Leave the shape you are not using sitting there
  commented; coming back to it is uncommenting.
- **What the code can prevent, it prevents** — a failure hit once is made
  impossible, not documented. Prose is for what code cannot fix: bad hardware, a
  preemptible queue, a rule a future change could break.
- **Research code, not a public API.** No back-compat shims, no deprecation
  paths, no defensive generality for a caller that does not exist. Delete a
  shape you replaced rather than keeping both.

## Docstrings

No docstring re-explains that submit scripts are edited rather than configured,
that a job clones the commit you submit from, or what `clone_root` and `setup=`
are for — [`README.md`](README.md) owns those.
