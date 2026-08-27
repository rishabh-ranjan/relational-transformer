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

What it costs, measured on the 2026-08-24 sweep (one GPU per task, all four
stages, every attempt of every job summed, restarts included): 480 GPU-hours
for the 63 jobs (181 rt-plurel, 163 rt, 135 rt-j); 2.7-3.7 h on the rel-f1
tasks, 3-7 h on most, 10-21 h on the tasks whose selection arm ran to its
50k-step ceiling (rel-amazon/user-churn, rel-hm/user-churn, rel-hm/item-sales,
rel-stack/user-badge, and for rt-j rel-amazon/user-ltv and item-ltv). A b200 is
~2.5x an a100 per step. The rt-plurel and rt jobs took 21.5 h of wall clock
with 12 high-priority slots plus the `il-lo` pool; the 21 rt-j jobs, submitted
the next night, took 17.9 h sharing those slots with another sweep. The
per-task table is at the end.

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
`rt` is a random initialization under the same protocol, `rt-j` is
[`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) (mirrored at
`~/scratch/hf/stanford-star/rt-j`, checked file by file against the Hub). The
data is `~/scratch/hf/stanford-star/relbench-preprocessed`, built
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

## Results (2026-08-24 sweep)

8-seed test ensembles scored by `relbench.submit`, beside the RelArena-α
paper's RT-PluRel and its best other method (AUROC higher is better, MAE in
native units lower is better). The 18 non-rel-f1 tasks are comparable to the
paper: their val and test splits have a single timestamp, so the per-row bound
`db_cutoff=None` applies is the split timestamp RelArena bounded at. The three
rel-f1 tasks are not: 26-33 horizons over 3-6 years past the cutoff, so a
context reaches later races and earlier test horizons, and the numbers are far
higher than the paper's.

| task | metric | rt-plurel | rt-j | rt (scratch) | paper RT-PluRel | best paper baseline |
|---|---|---|---|---|---|---|
| rel-hm/item-sales | mae | 0.03977 | 0.04611 | 0.03839 | 0.0403 | 0.0532 |
| rel-stack/user-engagement | auroc | 0.9094 | 0.9082 | 0.9074 | 0.8968 | 0.9067 |
| rel-amazon/user-churn | auroc | 0.7130 | 0.7129 | 0.7130 | 0.7135 | 0.7086 |
| rel-trial/study-adverse | mae | 39.81 | 39.95 | 40.9 | 32.7 | 39.8 |
| rel-hm/user-churn | auroc | 0.7036 | 0.7021 | 0.7024 | 0.7044 | 0.7057 |
| rel-amazon/item-churn | auroc | 0.8318 | 0.8299 | 0.8296 | 0.8327 | 0.8305 |
| rel-event/user-attendance | mae | 0.2435 | 0.2461 | 0.2488 | 0.241 | 0.239 |
| rel-amazon/user-ltv | mae | 14.23 | 15 | 13.89 | 13.9 | 14.4 |
| rel-avito/user-visits | auroc | 0.6684 | 0.6670 | 0.6621 | 0.6709 | 0.6688 |
| rel-stack/user-badge | auroc | 0.8928 | 0.8936 | 0.8933 | 0.8916 | 0.8887 |
| rel-event/user-ignore | auroc | 0.8124 | 0.8315 | 0.8141 | 0.8476 | 0.8787 |
| rel-amazon/item-ltv | mae | 42.61 | 42.96 | 44.32 | 43 | 46.8 |
| rel-trial/site-success | mae | 0.3835 | 0.3067 | 0.3729 | 0.41 | 0.325 |
| rel-stack/post-votes | mae | 0.0684 | 0.07804 | 0.07029 | 0.0635 | 0.0649 |
| rel-avito/user-clicks | auroc | 0.6901 | 0.6758 | 0.6859 | 0.5834 | 0.6788 |
| rel-avito/ad-ctr | mae | 0.03519 | 0.0355 | 0.03686 | 0.0348 | 0.0311 |
| rel-trial/study-outcome | auroc | 0.7319 | 0.6964 | 0.6955 | 0.7235 | 0.7647 |
| rel-event/user-repeat | auroc | 0.8113 | 0.7821 | 0.7897 | 0.7914 | 0.7846 |
| rel-f1/driver-dnf | auroc | 0.8031 | 0.7251 | 0.7993 | 0.7315 | 0.7322 |
| rel-f1/driver-top3 | auroc | 0.9069 | 0.8794 | 0.8384 | 0.7589 | 0.8108 |
| rel-f1/driver-position | mae | 2.736 | 2.825 | 3.425 | 3.818 | 3.762 |

`rt` is the same protocol from a random initialization -- the no-pretraining
control. Means over the 12 classification tasks (AUROC %) and the 9 regression
tasks (nMAE %, as `relbench.submit` reports): rt-plurel 78.96 / 28.14, rt-j
77.53 / 27.07, rt 77.76 / 29.26. The submission packages are
`~/scratch/relational-transformer/fine_tune/leaderboard/2026-08-24-fine_tune/{rt-plurel,rt-j,rt}-{classification,regression}.zip`,
validated by `python -m relbench.submit` (12/12 and 9/9 tasks each).

## What bit

- **ampere7's a100 comes up with a foreign process holding 22 GB**, so a job
  placed there dies in CUDA OOM at its first forward. Six attempts died that
  way (three at submission, three requeued onto it); `a100()` excludes it,
  as does ampere4 (local disk 99% full). A requeue keeps the original
  submission's flags, so `scontrol update ExcNodeList=` on every pending job
  is what stops a preempted job landing there.
- **A node without the `~/scratch` symlink fails every job at startup**:
  slurm creates the log path as a real directory before `ilc.env.sh` can make
  the link, and the env script then refuses the real directory. ampere1, 2, 4,
  7 and 9 were like that; a one-line job per node (`rm -rf` the stray tree,
  `ln -s /dfs/user/$USER ~/scratch`) fixed them.
- **A job preempted in the seconds after it wrote its prediction table is
  requeued anyway**; the requeued attempt finds every stage done and exits in a
  minute, so it only costs a card briefly.
- **Moving a job is nearly free**: `scancel` delivers SIGTERM, `rt.train`
  saves `resume.pt` at the next step, and the resubmission resumes at exactly
  that step; the only cost is the restart (clone check, page-cache populate,
  compile: 1-10 min). Every one of the eleven moves resumed at its cancel step.
- **`il`'s ten count across sessions**, and its b200 sub-cap of two too: a
  second session's job on `il` blocks a promotion with `QOSMaxGRESPerUser`.
- **A 21-day `il-lo` job never backfills while free a100s are `PLANNED`** for
  a higher-priority whole-node job, however many sit idle; a 6-hour wall clock
  fits under the plan and starts at once. The rt-j jobs asked for 6 h on
  `il-lo`: roach requeues at the limit and the job resumes, so a long task
  simply ran in 6-hour slices (with a wait of up to an hour between them) until
  a high-tier slot took it.

## Per-task record

Selected step, refit steps, chosen context `(ctx, local_ctx, bfs_width,
prefer_latest)` and wall clock (all attempts; the cards it ran on).

| task | rt-plurel: step / refit / context / wall clock | rt-j: step / refit / context / wall clock | rt: step / refit / context / wall clock |
|---|---|---|---|
| rel-hm/item-sales | 46400 / 47293 / (1024, 128, 16, True) / 9.6 h (b200) | 22800 / 23239 / (1024, 128, 16, True) / 6.8 h (a100+b200) | 44800 / 45662 / (1024, 128, 16, True) / 9.2 h (b200) |
| rel-stack/user-engagement | 16000 / 17010 / (1024, 1024, 64, True) / 4.7 h (b200) | 7600 / 8080 / (1024, 1024, 64, True) / 3.2 h (b200) | 9800 / 10419 / (128, 128, 64, False) / 3.4 h (b200) |
| rel-amazon/user-churn | 49000 / 53265 / (512, 128, 16, True) / 19.4 h (a100+b200) | 48700 / 52939 / (256, 128, 16, True) / 10.2 h (b200) | 50000 / 54352 / (512, 128, 16, True) / 21.4 h (a100+b200) |
| rel-trial/study-adverse | 8400 / 9098 / (1024, 512, 256, True) / 7.8 h (a100) | 7400 / 8015 / (1024, 512, 64, True) / 7.3 h (a100) | 17100 / 18519 / (1024, 1024, 256, True) / 12.1 h (a100) |
| rel-hm/user-churn | 27400 / 27948 / (1024, 1024, 256, False) / 18.2 h (a100) | 20800 / 21216 / (512, 512, 64, True) / 6.5 h (b200) | 40300 / 41105 / (1024, 1024, 256, False) / 19.9 h (a100+b200) |
| rel-amazon/item-churn | 20800 / 22258 / (512, 512, 64, True) / 17.1 h (a100) | 12600 / 13483 / (512, 512, 256, True) / 5.9 h (a100+b200) | 16400 / 17550 / (1024, 512, 64, True) / 14.8 h (a100) |
| rel-event/user-attendance | 13900 / 15355 / (256, 128, 16, False) / 11.0 h (a100) | 3100 / 3425 / (1024, 256, 256, False) / 7.9 h (a100+b200) | 14300 / 15797 / (512, 512, 16, False) / 5.1 h (a100+b200) |
| rel-amazon/user-ltv | 31900 / 34677 / (1024, 1024, 16, True) / 13.3 h (a100+b200) | 48700 / 52939 / (1024, 1024, 16, True) / 14.9 h (a100+b200) | 50000 / 54352 / (1024, 1024, 256, False) / 11.6 h (a100+b200) |
| rel-avito/user-visits | 3900 / 5250 / (1024, 1024, 256, True) / 6.5 h (a100) | 1500 / 2020 / (512, 512, 256, True) / 4.8 h (a100) | 3900 / 5250 / (512, 512, 256, True) / 6.2 h (a100) |
| rel-stack/user-badge | 33600 / 36055 / (128, 128, 256, False) / 16.2 h (a100+b200) | 48300 / 51829 / (1024, 128, 256, False) / 12.9 h (a100+b200) | 40900 / 43889 / (512, 128, 256, True) / 11.9 h (a100+b200) |
| rel-event/user-ignore | 2600 / 2873 / (512, 512, 64, False) / 5.0 h (a100) | 700 / 774 / (1024, 256, 16, True) / 4.0 h (a100) | 4300 / 4750 / (1024, 128, 64, True) / 5.5 h (a100) |
| rel-amazon/item-ltv | 21500 / 22826 / (1024, 512, 16, False) / 16.9 h (a100) | 45200 / 47988 / (1024, 512, 256, False) / 14.9 h (a100+b200) | 12900 / 13696 / (1024, 256, 16, False) / 7.3 h (a100+b200) |
| rel-trial/site-success | 1500 / 1696 / (1024, 1024, 64, False) / 4.7 h (a100) | 300 / 340 / (1024, 1024, 256, True) / 4.0 h (a100) | 2200 / 2487 / (1024, 1024, 256, True) / 4.9 h (a100) |
| rel-stack/post-votes | 2400 / 2553 / (128, 128, 16, False) / 5.1 h (a100) | 6200 / 6595 / (1024, 512, 16, True) / 8.3 h (a100) | 6000 / 6382 / (128, 128, 16, True) / 6.7 h (a100) |
| rel-avito/user-clicks | 1400 / 1899 / (1024, 1024, 256, False) / 5.6 h (a100) | 200 / 272 / (1024, 1024, 256, False) / 4.2 h (a100) | 1000 / 1357 / (1024, 1024, 256, False) / 4.9 h (a100) |
| rel-avito/ad-ctr | 600 / 808 / (1024, 1024, 256, False) / 4.4 h (a100) | 100 / 135 / (1024, 1024, 256, False) / 3.5 h (a100) | 500 / 674 / (1024, 1024, 256, False) / 3.7 h (a100) |
| rel-trial/study-outcome | 200 / 217 / (1024, 1024, 64, True) / 3.2 h (a100) | 1500 / 1621 / (1024, 256, 64, False) / 3.7 h (a100) | 500 / 541 / (1024, 1024, 64, True) / 3.2 h (a100) |
| rel-event/user-repeat | 300 / 321 / (1024, 1024, 256, True) / 3.2 h (a100) | 100 / 107 / (1024, 512, 64, True) / 3.2 h (a100) | 400 / 428 / (512, 256, 64, True) / 3.4 h (a100) |
| rel-f1/driver-dnf | 1500 / 1575 / (1024, 1024, 64, True) / 3.7 h (a100) | 2100 / 2205 / (1024, 512, 64, True) / 3.6 h (a100) | 100 / 105 / (512, 512, 256, True) / 2.7 h (a100) |
| rel-f1/driver-top3 | 100 / 144 / (512, 256, 64, False) / 2.9 h (a100) | 100 / 144 / (1024, 128, 256, True) / 2.7 h (a100) | 100 / 144 / (128, 128, 256, True) / 2.7 h (a100) |
| rel-f1/driver-position | 1000 / 1067 / (1024, 512, 256, True) / 3.1 h (a100) | 300 / 321 / (1024, 256, 16, False) / 2.7 h (a100) | 400 / 427 / (1024, 1024, 16, True) / 2.8 h (a100) |

The probe that preceded the sweep (rel-f1/driver-dnf at reduced budgets,
15 min) left its checkpoints under
`~/scratch/relational-transformer/fine_tune/rtv2/2026-08-24-fine_tune-probe/`.

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
