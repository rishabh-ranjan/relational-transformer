# Working on this repo

## Environment

[pixi](https://pixi.sh) manages one self-contained environment (Python, PyTorch +
CUDA, Rust, all dependencies), built on first use. There is nothing to build past
`pixi install`: the rustler extension is compiled as part of the editable install.

```bash
pixi run pytest                        # the test suite
pixi run python examples/train.py      # or eval.py, preprocess.py, ...
```

## Comments and docs are operational

A comment says what the code does, or what someone changing it has to know.
Never how it came to be: no history, no "previously we ...", no incident
stories, no bug a past edit fixed, no note that a value was tuned or a line
reordered. If a comment only makes sense to someone who saw the previous
version, delete it — git holds that.

Say it once:

- Every fact has one home, and the others point at it. Conventions live in
  [`expts/README.md`](expts/README.md); an experiment's README covers that
  experiment; a module docstring covers that module. Do not re-explain elsewhere.
- Only what the reader cannot get from the code. Prefer a link to a summary.
- No rationale for a settled choice.

## Structure

- No CLI. RT is a library; a run is a script that calls it. Start from
  [`examples/`](examples/) and edit.
- Entry points take every argument explicitly (`rt.train._train` has no
  defaults); `examples/` holds the released values.
- Data is a local directory. Nothing is fetched from the Hub at run time; point
  `pre_dir` at a path every node can read.
- `README.md` and `docs/` are written for humans and the broader public: keep
  development rules out of them, they belong here. `expts/README.md` is the
  exception and holds the experiment conventions in full.
