# Fine-tuning

Per-task fine-tuning of a Relational Transformer on the 21 RelBench v1 entity
tasks, and the leaderboard submissions it produces: the RelArena-α RT-PluRel
recipe ([arXiv:2608.16319](https://arxiv.org/abs/2608.16319), Appendix F),
expressed with this repo's own entry points instead of the RelArena harness.

## Running it

```bash
# 1. one job per (model, task); edit submit.py first, commit, push
pixi run python -m expts.fine_tune.submit

# 2. watch (the roach skill says how); progress is in wandb and the logs
ls -t ~/scratch/relational-transformer/fine_tune/slurm-logs/*.out | head

# 3. gather the prediction tables, validate them, write the leaderboard zips
pixi run python -m expts.fine_tune.collect
```

`submit.py` is edited every submission: `MODELS` and `TASKS` say what runs,
`RESOURCES` says where, worked out against the live cluster
([`../README.md`](../README.md)). It refuses a task whose job is already
queued -- two jobs on one task would write the same directories.

One job is one task end to end (`run.py`), on one GPU, and is idempotent per
stage: a requeued or resubmitted job resumes the stage it was in (`rt.train`
from `resume.pt`, the context search per grid entry, the test ensemble per
seed) and skips the stages whose output exists. So to move a job, cancel it and
resubmit it; nothing is lost but the minutes since the last checkpoint.

What it costs (RelArena-α's run of the same recipe, one GPU per task, wall
clock over all four stages): 2.5 h on the rel-f1 tasks, 3-6 h on most, 10-20 h
on rel-hm/item-sales, rel-stack/user-engagement and rel-amazon/user-churn.
Numbers measured here go in the table below as they come in.

Everything lands under `~/scratch/relational-transformer/fine_tune/`:
`slurm-logs/`, one directory per stage at
`<entity>/<project>/<model>-<db>-<task>-<stage>/`, and the submission packages
under `leaderboard/<project>/`.

## The recipe

Four stages, all existing entry points; `run.py` only derives the values
between them. `db_cutoff=None` throughout: per-row temporal masking is the
only trim of the database a context is built from, as in the RT-J paper's
runs (RelArena bounded contexts at the split timestamp instead, because its
harness hands the model a censored database).

1. **Selection arm** -- `rt.train.main` on `train`, delta-fine-tuned from the
   warm start (or trained from scratch for `rt`): Muon, constant lr 5e-4, wd
   0.1, batch 256, an EMA of the weights (`swa_momentum=0.9999`), up to 50k
   steps. Every batch draws its context shape from the cross product of ctx
   {128, 256, 512, 1024} x local ctx {128, 256, 512, 1024} x bfs width
   {16, 64, 256} x prefer-latest {F, T}. Every 100 steps the EMA net is scored
   on 1024 val rows under the two endpoint shapes, `(1024, 1024, 256, F)` and
   `(128, 128, 16, T)`, each a 4-seed context ensemble; the best step by the
   task's metric (AUROC / nMAE) is kept as `best_swa_*.safetensors`, and the
   arm stops after 10k steps without an improvement in either shape.
2. **Context search** -- `rt.eval.main` on `val` alone, with that checkpoint
   frozen: the 60 shapes of the grid above (local ctx <= ctx), each a 4-seed
   ensemble over 4096 val rows (`shuffle_seed=1`, so not the rows the step was
   chosen on). The winner is in `tuning.json`.
3. **Reporting arm** -- `rt.train.main` on `train + val`, same recipe, no
   evaluation, for the selected step scaled by the row ratio
   `(train + val) / train`; the EMA net at the last step is the model.
4. **Test ensemble** -- `rt.eval.main` on `test` under the chosen shape, eight
   context seeds averaged before the sigmoid / denormalization, scored through
   `relbench.submit.evaluate_task` and written as the leaderboard prediction
   table `<db>__<task>.csv`.

`selection.json` beside the selection arm records the step, the row counts,
the refit budget and the chosen context.

`MODELS` names the warm start: `rt-plurel` is
[`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel)
(mirrored at `~/scratch/hf/stanford-star/rt-plurel`, one head per task type),
`rt` is a random initialization under the same protocol, `rt-j` the RT-J
release. The data is `~/scratch/hf/stanford-star/relbench-preprocessed`, built
by [`../preprocess`](../preprocess/README.md) from `stanford-star/relbench`
(RelBench v3's `relbench-v1`).

## Submitting to the leaderboard

`collect.py` copies every finished task's prediction table into
`~/scratch/relational-transformer/fine_tune/leaderboard/<project>/<model>/`,
scores them with RelBench's own validator, prints ours beside the paper's
RT-PluRel numbers, and -- once a board is complete -- runs
`python -m relbench.submit` to write `<model>-classification.zip` and
`<model>-regression.zip`. Attach those to a
[submission issue](https://github.com/stanford-star/relbench/issues/new?template=submit.yml);
"In-context?" is **No** for every model here (each trains on the target
database).

## Reference numbers

`relarena_paper.csv` is Tables 4 and 5 of the RelArena-α paper (test AUROC,
test MAE in native units) for every method it ran; `submit.py` turns the
`rt-plurel` column and the best other column into the wandb `target/` lines.
`results.csv` is RelArena-α's `baseline_results/results.csv` (every evaluated
config of every method, val and test), and `results.md` is the table
`make_results.py` builds from it:

```bash
pixi run python expts/fine_tune/make_results.py
```

## Workspaces

`workspace.py` writes the wandb project view; rerun it whenever a run starts
logging a key the view has no panel for.

```bash
pixi run python expts/fine_tune/workspace.py --project 2026-08-24-fine_tune
```
