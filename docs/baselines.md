# Baselines

`rel2tab` tabular baselines run through the **same eval path** as RT
([inference.md](inference.md)) and produce the same RelBench submissions, so they
are directly comparable. A baseline is a `(featurizer, predictor)` pair: each
task's in-context training labels (and optional features) are fed to a tabular
predictor, and the result is scored with RelBench's own leaderboard evaluator.

```bash
pixi run baseline --featurizer entity --predictor ridge \
  --eval.pre-dir stanford-star/relbench-preprocessed --eval.out-dir baseline_out
```

- **Featurizers** (`--featurizer`): `global`, `entity`, `rt` (RT embeddings —
  pass a checkpoint with `--rt-ckpt`).
- **Predictors** (`--predictor`): `mean`, `linear`, `ridge`, `xgboost`.

The `global`/`entity` featurizers with the `mean`/`linear`/`ridge` predictors
need no GPU (only the `rt` featurizer runs a model). `--eval.out-dir` is a valid
RelBench submission directory, scored and re-validatable exactly like RT's eval
output. The `--eval.*` flags (context sizes, grid, task selection, …) are the
same `EvalConfig` as `eval`'s — see
[context engineering](inference.md#context-engineering).
