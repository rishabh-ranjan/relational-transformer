# Context-hyperparameter tuning grid (RT-J paper rerun)

Per-task grid search over the context sampler's (ctx, lcs, bw, pl), on
validation, feeding three downstream results: the tuned arm of the
ensemble-curve figure, the default-vs-tuned appendix table
(`../valtest`), and the top-4-configs leaderboard ensemble
(`../submit`).

## Running it

```bash
pixi run python -m expts.repaper.tune.submit     # 21 tuning jobs, 1 GPU each
pixi run python -m expts.repaper.tune.collect    # -> tuned_configs.json; commit it
```

The 2026-08-19 round did not submit: its grids are the rt-j grids of
[`../../icl`](../../icl/README.md) (`tune-rt-j-<db>-<table>/tuning.json`
under `~/scratch/relational-transformer/icl/rtv2/2026-08-25-icl/`, run
2026-08-26 on the same checkpoint pair, data and protocol), which is the path
`collect.py`'s `grid()` reads; the repaper path is the commented alternative.
The one argument that differs between the two submit scripts is
`num_workers` (8 here, the job's cpu count there), which is DataLoader
parallelism: the context of a target row is drawn from
`(context_seed, item index, node index)` alone (`rustler/src/fly.rs`).

## Protocol

Grid: ctx {512, 1024, 2048, 4096, 8192} x lcs {256, 512, 1024, 2048, 4096,
8192 | lcs <= ctx} x bw {8, 32, 128} x pl {True, False} = **120 configurations
per task** (2520 total). Each configuration is scored on the task's validation
split -- 4096 rows (shuffle_seed=0), the prediction averaged over 4 context
seeds (`val_ensemble_size=4`), `db_cutoff=None`. AUROC ranks clf configurations,
normalized MAE ranks reg.

A job is `rt.eval:main` in tune-only mode (`splits=["val"]`), one per task,
resumable per grid entry (`ensemble_resume.pt`); a preemption costs at most
one (grid entry, seed) pass. `tuning.json` lands under
`~/scratch/ckpts/rtv2/<RUN_TAG>-repaper-tune/tune--<db>--<table>/`.

`tuned_configs.json` (committed once the grid finishes) holds, per task: the
best configuration and its val score, the top-4 configurations by val score,
the full score table, and the path of the grid it was read from.

## Measured runtimes

The icl rt-j grids, 2026-08-26 (`sacct`): a task with 4096 validation rows
is ~10.5 h on an a100 (rel-amazon/item-ltv 10h29, item-churn 10h27, user-ltv
10h54) and ~4-6 h on a b200; the smaller validation splits scale down with
their rows (rel-event/user-repeat 1h10, rel-f1 tasks ~15 min,
rel-trial/study-outcome 1h40).
