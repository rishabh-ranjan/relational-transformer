# Context-ensembling curves (RT-J paper rerun)

The test-time-compute ensembling figure: metric vs number of ensembled context
seeds, for the shared default context (8192, 256, 32, pl=1) and the per-task
tuned context (from [`../tune`](../tune)); plus its per-task
appendix figures.

## Running it

```bash
pixi run python -m expts.repaper.enscurve.submit    # 21 jobs per variant, 1 GPU each
pixi run python -m expts.repaper.enscurve.reduce    # -> wandb runs `default`, `tuned`
```

The tuned variant waits on `../tune/tuned_configs.json`; the default
variant runs any time.

## Protocol

Per (variant, task): 16 independent context seeds (the `rt.eval` ensemble seed
family off base seed 0) at the variant's fixed configuration, on the fixed
8192-row test subsample (shuffle_seed=0), db_cutoff=None. Raw per-row
predictions are averaged over the first k seeds and scored on the normalized
scale at every k=1..16 -- the same quantity `rt.eval`'s ensembling scores, not
a mean of per-seed scores. Jobs resume per seed
(`<db>__<table>.state.npz`), so preemption costs one seed.

Curves land under
`/dfs/user/ranjanr/ckpts/rtv2/repaper-enscurve/<variant>/<db>__<table>.json`;
`reduce.py` aggregates them into `rtv2/2026-08-19-repaper-enscurve` runs
`default` / `tuned` (keys `ens_size`, `test/avg_mae`, `test/avg_auc`,
`per_task/relbench/<db>/<table>/test/{mae,auc}`).
