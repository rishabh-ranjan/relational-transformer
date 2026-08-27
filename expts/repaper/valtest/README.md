# Default vs tuned context (RT-J paper rerun)

The appendix table comparing the shared default context (8192, 256, 32, pl=1)
against each task's tuned configuration on the 8192-row test subsample at a
single context seed, with the selected (ctx, lcs, bw, pl) columns.

Nothing is submitted here: both columns are the single-seed points
(`curve["1"]`, context seed `member_context_seed(0, 0)`) of the two
ensemble curves in [`../enscurve`](../enscurve) -- `default` and `tuned`
run the same 8192-row subsample (shuffle_seed=0, db_cutoff=None) at exactly
these two configurations -- so the table is the n_ens=1 point of the
ensembling figure, read off the same runs.

Waits on [`../tune/tuned_configs.json`](../tune) and both enscurve variants.

```bash
pixi run python -m expts.repaper.valtest.collect   # -> results.json (commit) + wandb run
```

`collect.py` asserts each curve's configuration (default, or the task's
`best_cfg`) and protocol before reading it, and logs the run `valtest` to
`rtv2/<RUN_TAG>-repaper-valtest` with the flat keys the paper's
`gen/appendix/tables.py` reads (`default/<db>/<table>`, `tuned/...`,
`cfg/...`).
