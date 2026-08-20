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

## Protocol

Grid: ctx {512, 1024, 2048, 4096, 8192} x lcs {256, 512, 1024, 2048, 4096,
8192 | lcs <= ctx} x bw {16, 64, 256} x pl {True, False} = **120 configurations
per task** (2520 total). Each configuration is scored on the task's validation
split -- 4096 rows (shuffle_seed=0), the prediction averaged over 4 context
seeds (`val_ensemble_size=4`), `db_cutoff=None`. AUROC ranks clf configurations,
normalized MAE ranks reg.

A job is `rt.eval:main` in tune-only mode (`splits=["val"]`), one per task,
resumable per grid entry (`ensemble_resume.pt`); a preemption costs at most
one (grid entry, seed) pass. `tuning.json` lands under
`/dfs/user/ranjanr/ckpts/rtv2/2026-08-19-repaper-tune/tune--<db>--<table>/`.

`tuned_configs.json` (committed once the grid finishes) holds, per task: the
best configuration and its val score, the top-4 configurations by val score,
and the full score table.

## Measured runtimes

(filled in from the first finished jobs)
