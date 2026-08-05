# Experiments

One directory per experiment, laid out however that experiment wants. The only
convention is that something in it submits jobs, and that it is committed —
jobs clone the commit you submit from, so the directory is the record of what
was run.

`data_scaling/` happens to split into a recipe (`train.py`), a sweep
(`submit.py`) and its task lists; an experiment that runs one job, or evaluates
checkpoints, or sweeps a grid, will look different. Nothing here enforces a
shape.

## Submitting

Jobs go through [`roach.slurm`](https://github.com/rishabh-ranjan/roach/blob/main/roach/slurm/README.md)
— read that first; it covers `submit()`, resource presets, and how a run
survives preemption. The job clones the roach that submitted it, so upgrading
roach cannot change a job already queued.

```python
from roach.slurm import BLACKWELL, submit

submit("expts.<name>.<module>:<function>", args={...}, resources=BLACKWELL,
       name=..., setup=("pixi run build-sampler",),
       repo_root=..., log_root=..., clone_root=..., secrets_dir=...)
```

A sweep is a python loop around that call — conditional resources, staggered
submissions, resumed run ids, whatever the experiment needs.

## What is specific to this repo

- **`setup=("pixi run build-sampler",)`** — the rustler sampler is a compiled
  extension, so a job has to build it inside its clone.
- **Data is a local directory.** Nothing is fetched from the Hub at run time
  (see [docs/downloads.md](../docs/downloads.md)); point `pre_dir` at a path
  every node can read, and expect a job's first minutes to go on populating it
  into the page cache.
- **Entry points take every argument explicitly.** `rt.train._train` has no
  defaults; [`examples/`](../examples/) has the released values to start from.
