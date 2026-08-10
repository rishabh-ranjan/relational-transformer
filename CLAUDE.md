# Working on this repo

Environment and commands: see [Development](README.md#development).

## Comments and docs are operational

A comment says what the code does, or what someone changing it has to know.
Never how it came to be: no history, no "previously we ...", no incident
stories, no bug a past edit fixed, no note that a value was tuned or a line
reordered. If a comment only makes sense to someone who saw the previous
version, delete it — git holds that.

Say it once:

- Every fact has one home, and the others point at it. These rules live here;
  [`expts/README.md`](expts/README.md) covers experiment workflow; an
  experiment's README covers that experiment; a module docstring covers that
  module. No docstring re-explains that submit scripts are edited rather than
  configured, that a job clones the commit you submit from, or what
  `clone_root` and `setup=` are for.
- Operational instructions only. No history, no incident stories, no fixed bugs,
  no rationale for a settled choice. Write what to run and what to check.
- Only what the reader cannot get from the code. Prefer a link to a summary.

## Code style

- **One file where one file will do.** `expts/fine_tune/submit.py` is the shape
  to copy: submit the entry point directly, every argument spelled out at the
  call; a sweep is a loop around that call. A second file has to earn itself.
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
  impossible, not documented. Prose is for what code cannot fix.

## Structure

- No CLI. RT is a library; a run is a script that calls it. Start from
  [`examples/`](examples/) and edit.
- Entry points take every argument explicitly (`rt.train._train` has no
  defaults); `examples/` holds the released values.
- Data is a local directory. Nothing is fetched from the Hub at run time; point
  `pre_dir` at a path every node can read.
- `README.md`, `docs/` and `expts/README.md` are written for humans and the
  broader public: keep opinionated development rules out of them, they belong
  here. `expts/README.md` holds experiment *workflow* (submitting, watching
  jobs, what an experiment owes, cleanup).
