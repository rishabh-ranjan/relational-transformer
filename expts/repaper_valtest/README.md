# Default vs tuned context (RT-J paper rerun)

The appendix table comparing the shared default context (8192, 256, 32, pl=1)
against each task's tuned configuration on the 8192-row test subsample at a
single context seed, with the selected (ctx, lcs, bw, pl) columns.

Waits on [`../repaper_tune/tuned_configs.json`](../repaper_tune). The default
column is read from ``../repaper_scaling``'s `subsampled/rt` arm (identical
protocol at ctx 8192), so only the 21 tuned runs are submitted here, through
the same runner.

```bash
pixi run python -m expts.repaper_valtest.submit
pixi run python -m expts.repaper_valtest.collect   # -> results.json (commit) + wandb run
```
