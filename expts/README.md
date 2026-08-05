# Experiments

One directory per experiment. Each holds the recipe, the sweep that submits it,
and whatever data lists it needs — nothing else, and no slurm boilerplate.

```
expts/data_scaling/
  train.py     # the recipe: one function, calling rt.train.main with fixed arguments
  submit.py    # the sweep: cluster paths, and a loop that varies one thing
  *.json       # the task lists this experiment defines
```

## Submitting

Jobs go through [`roach.slurm`](https://github.com/rishabh-ranjan/roach/blob/main/roach/slurm/README.md)
— read that first; it explains `submit()`, the resource presets, and how a run
survives preemption. It is a pinned dependency (see `pyproject.toml`), so a run
cannot change because roach moved.

```bash
pixi run python expts/data_scaling/submit.py
```

Two things that are specific to this repo:

- **`setup=("pixi run build-sampler",)`** — the rustler sampler is a compiled
  extension, so every job builds it inside its clone. Experiments here must pass
  this.
- **Data is a local directory.** Nothing is fetched from the Hub at run time;
  see [docs/downloads.md](../docs/downloads.md). Point `pre_dir` at a path every
  node can read (`/dfs/...` here), and expect the first minutes of a job to be
  spent populating it into the page cache.

## Starting a new experiment

1. `mkdir expts/<name>`, add `train.py` with one function that calls the library
   (`rt.train.main` takes every knob as a required argument — see
   [`examples/train.py`](../examples/train.py) for the released values to start from).
2. Add `submit.py`: the cluster paths at the top, the sweep as a plain loop.
3. Run it from a clean, pushed checkout. The job clones that commit, so the
   sweep file is the record of what was run.

Check the wiring without touching the scheduler:

```python
from roach.slurm import check_args
check_args("expts.<name>.train:train", {**BASE, "run_id": "x"})
```
