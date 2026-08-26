# In-context learning

RT-PluRel and RT-J on the 21 RelBench v1 entity tasks with no gradient step
on the target database, and the RelBench v3 leaderboard submissions they
produce (In-context: **Yes**). Two stages, both `rt.eval` passes with the released
checkpoint pair frozen: a per-task context search on validation, then the
top-4 contexts each run with 4 context seeds on the full test split and
averaged. [`../fine_tune`](../fine_tune/README.md) is the same submission
machinery around a fine-tuned model; [`../repaper/tune`](../repaper/tune/README.md)
and [`../repaper/submit`](../repaper/submit/README.md) are this protocol as
the RT-J paper ran it.

## Running it

```bash
# 1. 21 context-search jobs, one per task, one GPU each; edit submit.py, commit, push
pixi run python -m expts.icl.submit

# 2. watch (the roach skill says how); a job logs a line per (config, task)
ls -t ~/scratch/relational-transformer/icl/slurm-logs/*.out | head

# 3. rank the grids -> tuned_configs.json; commit it
pixi run python -m expts.icl.collect

# 4. 84 ensemble units (21 tasks x top-4 configs); rewrite ENS in submit.py
#    against the live cluster first, commit, push
pixi run python -m expts.icl.submit

# 5. average the 16 predictions per task, write and validate the prediction
#    tables, package the leaderboard zips
pixi run python -m expts.icl.collect
```

`submit.py` submits whatever is not queued or done: a context search for every
task without a `tuning.json`, an ensemble unit for every `(task, config)` in
`tuned_configs.json` without a `result.json`. So steps 1 and 4 are the same
command, and re-running it fills gaps. `TUNE` and `ENS` are the resource
plans, worked out against the live cluster at each submission
([`../README.md`](../README.md)).

Both stages resume: the context search per grid entry (`ensemble_resume.pt`),
an ensemble unit per context seed (`state.npz`). A preemption costs at most
one (grid entry, seed) pass on validation or one full test pass.

Everything lands under `~/scratch/relational-transformer/icl/`: `slurm-logs/`,
one directory per job at `rtv2/2026-08-25-icl/tune-<model>-<db>-<task>/` and
`rtv2/2026-08-25-icl/ens-<model>-<db>-<task>-cfg<k>/`, and the submission
package under `leaderboard/2026-08-25-icl/<model>/`. (rt-plurel's tuning ran
before the model was in the id; its `tune-rel-*` / `ens-rel-*` directories
are reached through `tune-rt-plurel-rel-*` symlinks.)

## The recipe

Checkpoints (`MODELS` in `submit.py`):
[`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel)
and [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j),
mirrored at `~/scratch/hf/stanford-star/{rt-plurel,rt-j}` (the rt-j mirror's
`model.safetensors` match the Hub's LFS sha256 at revision `1819386c`,
checked 2026-08-26), `classification/` for the 12 classification tasks and
`regression/` for the 9 regression tasks. Every run id carries the model:
`tune-<model>-<db>-<task>`, `ens-<model>-<db>-<task>-cfg<k>`;
`tuned_configs.json` is keyed by model, then task.
Data: `~/scratch/hf/stanford-star/relbench-preprocessed`, built by
[`../preprocess`](../preprocess/README.md). `db_cutoff=None` throughout:
per-row temporal masking is the only trim of the database a context is built
from. Sampler: `num_walks=10_000`, `walk_length=20`, the released inference
values. `tokens_per_gpu=2**18` (32 rows per batch at ctx 8192).

1. **Context search** -- `rt.eval.main` on `val` alone. Grid: ctx
   {512, 1024, 2048, 4096, 8192} x lcs {256, 512, 1024, 2048, 4096, 8192 |
   lcs <= ctx} x bfs width {8, 32, 128} x prefer-latest {T, F} = **120
   configurations per task** (the released default (8192, 256, 32, T) is one
   of them), 36 sampler passes each scored at every ctx off
   a prefix of the largest. Each configuration is scored on 4096 validation
   rows (`shuffle_seed=0`; the whole split where it is smaller), the
   prediction averaged over 4 context seeds (`val_ensemble_size=4`). AUROC
   ranks classification, normalized MAE regression. The scores are in
   `tuning.json`; `collect.py` ranks them into `tuned_configs.json` (best
   config, top-4, full table per task), committed here.
2. **Test ensemble** -- `run.py`, one unit per (task, config): the config's
   context sampled with 4 seeds (`member_context_seed(0, k)`, k = 0..3) on
   the full test split, the raw per-row predictions (logits / normalized
   values) summed into `state.npz`. `collect.py` averages the 16 predictions
   of a task's 4 units, then sigmoids (clf) / denormalizes (reg) through
   `rt.eval.relbench._emit_and_score`, which also proves the node-index join
   against relbench's ground truth (`cls=1.000` / `|dy|`), scores the table
   with `relbench.submit.evaluate_task`, and writes the leaderboard prediction
   table `<db>__<task>.csv`.

`results.json` beside the prediction tables records, per task, the official
metric, the alignment guard, the top-4 configs with their validation scores,
and each unit's own 1..4-seed test curve (so the ensemble can be read against
its best single config).

## Watching it on wandb

Everything logs to `rtv2/2026-08-25-icl`; `workspace.py` writes the project
view (rerun it whenever a run starts logging a key the view has no panel for):

```bash
pixi run python expts/icl/workspace.py
```

- A context-search job is the run `<model>/<db>/<task>/tune` (rt-plurel's
  first tuning runs predate the model in the name): the validation metric of
  every configuration against `tune/idx` (`tune/<metric>/val/<db>/<task>`,
  the running best folded in as `tune/best/...`), and the configuration
  itself on the same axis (`tune/ctx_size`, `tune/local_ctx_size`,
  `tune/bfs_width`, `tune/prefer_latest`).
- An ensemble unit is the run `<model>/<db>/<task>/cfg<k>`: the test metric of that
  configuration against `ens_size` 1..4 (`<metric>/test/<db>/<task>`), with
  RT-J's in-context number and RT-PluRel's fine-tuned number as `target/`
  lines.
- `collect.py` adds two summary runs per model once a stage is complete:
  `<model>/tune`, the same tune curves for all 21 tasks plus
  `tune/<metric>/val/mean` and `tune/best/<metric>/val/mean`; and
  `<model>/top4x4`, the top-k-configs x 4-seeds
  test metric against `ens_size` 4, 8, 12, 16 per task and averaged
  (`<metric>/test/mean`), with the official RelBench numbers as
  `relbench/<metric>/test/...`.

Values are on the leaderboard scale, in percent (AUROC %, nMAE %). Runs are
grouped by `run_name`, so a requeued job's attempts read as one curve.

## What it costs

Measured on this round where it says so; the rest is projected from the RT-J
rerun of the same grid on this cluster (`sacct`, 2026-08-20/21).

- Context search, measured 2026-08-25 on a100 (14 workers): one sampler pass
  over 4096 val rows scored at every ctx up to 8192 is ~400 s, all of it
  predict (load 3-8 s), the same on rel-amazon and rel-avito; the first grid
  entry (lcs 256, five ctx sizes, 4 seeds) took 27 min, so a task with 4096
  val rows is ~12-14 h on an a100 (entries with a larger lcs score fewer ctx
  sizes), 6h15 on a b200 in the RT-J rerun; the smaller validation splits
  scale down with their rows (rel-event/user-repeat 1h15, rel-f1/driver-dnf
  2h, rel-trial/study-outcome 3h30 in the RT-J rerun).
- Test ensemble: ~30 test rows/s on an a100 at ctx 8192 including sampling
  (the 16-seed 8192-row curves ran ~1h05 on rel-amazon); a unit is four
  passes, so ~13h for each rel-amazon user task (352k rows), ~6h30 for the
  item tasks, ~9h30 rel-stack/user-badge, ~6h post-votes, ~4h rel-hm/item-sales,
  ~3h user-engagement, under 2h for everything else; ~280 a100-hours in all,
  less wherever the tuned ctx is below 8192.

| task | val rows | test rows |
|---|---:|---:|
| rel-amazon/user-churn | 409,792 | 351,885 |
| rel-amazon/user-ltv | 409,792 | 351,885 |
| rel-amazon/item-ltv | 166,978 | 178,334 |
| rel-amazon/item-churn | 177,689 | 166,842 |
| rel-stack/user-badge | 247,398 | 255,360 |
| rel-stack/post-votes | 156,216 | 160,903 |
| rel-hm/item-sales | 105,542 | 105,542 |
| rel-stack/user-engagement | 85,838 | 88,137 |
| rel-hm/user-churn | 76,556 | 74,575 |
| rel-avito/user-clicks | 21,183 | 47,996 |
| rel-avito/user-visits | 29,979 | 36,129 |
| rel-trial/site-success | 19,740 | 22,617 |
| rel-trial/study-adverse | 3,596 | 3,098 |
| rel-event/user-attendance | 2,013 | 1,958 |
| rel-event/user-ignore | 2,013 | 1,958 |
| rel-avito/ad-ctr | 1,766 | 1,816 |
| rel-trial/study-outcome | 960 | 825 |
| rel-f1/driver-position | 499 | 760 |
| rel-f1/driver-top3 | 588 | 726 |
| rel-f1/driver-dnf | 566 | 702 |
| rel-event/user-repeat | 268 | 246 |

`TASKS` in `submit.py` is in this order (test rows, the ensemble stage's
cost), so a resource plan reads down the list.

## Submitting to the leaderboard

`collect.py` writes each model's prediction tables to
`~/scratch/relational-transformer/icl/leaderboard/2026-08-25-icl/<model>/preds/`,
scores them with RelBench's own validator, prints ours beside RT-J's paper
in-context numbers and RT-PluRel's fine-tuned ones, and -- once both boards
are complete -- runs `python -m relbench.submit` to write
`<model>-icl-classification.zip` and `<model>-icl-regression.zip` beside
them. Attach those to a
[submission issue](https://github.com/stanford-star/relbench/issues/new?template=submit.yml)
per model: Name `RT-PluRel (in-context)` / `RT-J (in-context)`,
**In-context? Yes** (both checkpoint pairs were pretrained on the Join, which
holds no RelBench database; the only per-task choice is the context
configuration, made on validation), URL
`https://huggingface.co/stanford-star/rt-plurel` /
`https://huggingface.co/stanford-star/rt-j`, Note the recipe in one line
(120-config context grid tuned on 4096 val rows x 4 seeds; test = top-4
configs x 4 context seeds, averaged).

## Reference numbers

`reference.csv`, in leaderboard units (AUROC %, nMAE %): `rt-j-icl` is RT-J's
top-4 x 4-seeds full-test ensemble from the RT-J paper's appendix table
(`overleaf-rtj` `figures/appendix/rt_vs_rdblearn_body.tex`, generated from
the `rtj-top4x4` run), the in-context number to beat; `rt-plurel-ft` is the
RelArena-α paper's fine-tuned RT-PluRel
([`../fine_tune/relarena_paper.csv`](../fine_tune/relarena_paper.csv), MAE
divided by the train-split std from `regression_stds.json`), an upper
reference.
