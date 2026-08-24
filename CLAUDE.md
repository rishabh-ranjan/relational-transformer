# Working on this repo

The whole repository is private on GitHub. "Public" here never means git
visibility: it means the code is meant to be *exported* to the broader public
one day — released with the package, read by someone who arrived from the paper.
`src/`, `examples/`, `docs/`, `byod/`, `README.md` are public in that sense.
`expts/` is private internal research and stays that way.

## No comments, no docstrings

Code says what it does; git says what changed. Functional pragmas (`# noqa`,
`# type:`, `#!`, `#SBATCH`) are tool instructions, not comments, and stay.
Prose for humans lives in `docs/` and the READMEs, nowhere else.

## Code style

Outside `expts/`, write for eventual release.

- **Explicit over defaulted.** Public entry points take every argument at the
  call site; do not add a default that hides a choice from the caller.
- **Fail loudly, do not handle corner cases.** Support the path that is meant to
  work; for everything else assert. No `try`/`except` that swallows, no fallback
  branch, no defensive default that lets a wrong config run. Every unsupported
  case must crash loudly at the point it is first knowable — a silent bug is far
  worse than a crash.
- **Match the surrounding module** in naming and idiom rather than importing a
  new style.
- **Keep examples runnable and minimal.** An example shows one path end to end.
- Public API changes ripple into `docs/` and released checkpoints — update the
  docs in the same change.

Working under `expts/` is private research code and follows different rules; see
[`expts/CLAUDE.md`](expts/CLAUDE.md).

## Structure

- No CLI. RT is a library; a run is a script that calls it. Start from
  [`examples/`](examples/) and edit.
- Entry points take every argument explicitly (`rt.train._train` has no
  defaults); `examples/` holds the released values.
- Data is a local directory. Nothing is fetched from the Hub at run time; point
  `pre_dir` at a path every node can read.
- `README.md`, `docs/` and `expts/README.md` are written for humans and the
  broader public: keep opinionated development rules out of them, they belong in
  a `CLAUDE.md`. `expts/README.md` holds experiment *workflow* (submitting,
  watching jobs, what an experiment owes, cleanup).
