# Working under expts/

Private internal research code: never exported to the broader public, unlike the
rest of the repo (the GitHub repo itself is private either way). The repo-wide
rules in
[`../CLAUDE.md`](../CLAUDE.md) still hold; these override them here.
[`README.md`](README.md) has the workflow (submitting, watching jobs, cleanup).

## Code style

- **One file where one file will do.** `pretrain/submit_ilc.py` is the shape
  to copy: submit the entry point directly, every argument spelled out at the
  call; a sweep is a loop around that call (`fine_tune/submit.py`). A second
  file has to earn itself (`fine_tune/run.py` does: one task is four entry
  points in sequence, with values derived between them).
- **Comment out to switch.** Commented-out code is not commentary: leave the
  shape you are not using sitting there commented; coming back to it is
  uncommenting.
- **No module-level `CONSTANTS` for a value one call site consumes.** Write the
  value in the argument that takes it. Bulky or wanted twice: a function that
  returns it, beside its use (`targets_for(db, task)`). `TASKS`, which a sweep
  loops over, is the exception.
- **Keep a derived input beside the file that uses it** — a curated task list,
  a subset.
- **Edit a submit script, do not configure it.** No arguments, no flags, no
  `--dry-run`. Expect to change the file every submission, and commit before
  submitting: the job clones that commit.
- **No helper wrapping the submit call.** The `submit(...)` call sits in the
  loop body, spelled out; a parameter threaded through a `submit_one(...)` puts
  the value that changes in a different place from the value that does not, and
  every knob has to be in one place to be edited in one place.
- **What the code can prevent, it prevents** — a failure hit once is made
  impossible, not documented.
- **Research code, not a public API.** No back-compat shims, no deprecation
  paths, no defensive generality for a caller that does not exist. Delete a
  shape you replaced rather than keeping both.
