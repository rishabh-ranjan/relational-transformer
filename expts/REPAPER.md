# Regenerating every result of the RT-J paper

The `repaper_*` experiments are the paper's results sections, end to end:
from the released pretraining checkpoints to the wandb runs the paper's
figure scripts read, the leaderboard submission package, and the two
generated appendix tables. This file is the order of operations across them;
each experiment's README owns its own commands and protocol. Read
[README.md](README.md) first for how submitting and watching jobs works here.

## 0. Point the round at its checkpoint

Edit [`repaper_config.py`](repaper_config.py): `RUN_TAG` (a new tag puts a
new round's wandb runs in new projects) and `CKPT_CLF` / `CKPT_REG` (local
mirrors of the checkpoint pair under evaluation; compute nodes have no Hub
access). Set the same `RUN_TAG` in the paper repo's `gen/__init__.py`
(rishabh-ranjan/overleaf-rtj, branch `rr-main`). Commit both.

Then clear what the previous round produced that depends on the checkpoint --
every job no-ops on an output that already exists, so stale results would
otherwise be kept as finished:

```bash
R=/dfs/user/ranjanr/ckpts/rtv2
rm -rf $R/repaper-scaling/fulltest/rt $R/repaper-scaling/subsampled/rt \
       $R/repaper-scaling/abl $R/repaper-enscurve $R/repaper-valtest \
       $R/repaper-submit $R/<RUN_TAG>-repaper-tune
S=/dfs/user/ranjanr/share/relational-transformer/repaper
rm -rf $S/features/*/rt_features $S/vector_db/rt $S/leaderboard
```

What stays valid across checkpoints (same preprocessed data, same default
context): the SQL and RDBLearn feature blobs and their FAISS indices, the
classic-relbench cache, the TabICL checkpoints, the semantics-ablated data
copy, and the eight baseline arms under `repaper-scaling/{fulltest,subsampled}/
{sql,rdblearn}_{tabicl,lgbm}` -- their contexts depend only on the sampler.
Delete those too if the preprocessed data or the default context changed
(`rm -rf $R/repaper-scaling $S/features $S/vector_db
$S/relbench-preprocessed-nosem`).

Artifacts that are not checkpoint-dependent and are already in place are
listed in each README; every one of them is produced by a script in its
directory, so a clean machine rebuilds them with the same commands.

## 1. One-time fetches (login node; compute nodes have no internet)

[`repaper_baselines/README.md`](repaper_baselines/README.md), steps 0-2:
TabICL checkpoints, the classic-relbench cache, `pixi run install-rdblearn`,
and `check_alignment`.

## 2. Submit, in dependency order

Everything below is `pixi run python -m expts.<experiment>.submit` after
editing that script's `__main__` block to the stage it names. Every job is
idempotent per output file, so a script is resubmitted to fill gaps.

| stage | experiment | waits for |
|---|---|---|
| features | `repaper_baselines` (sql, rdblearn, rt) | fetches |
| FAISS indices | `repaper_baselines` (`submit_vecdb`) | a feature set's 21 blobs |
| RT arms + ablations | `repaper_scaling` (rt, rand, bfs*, nosem) | checkpoint only |
| baseline arms | `repaper_scaling` (sql/rdblearn x tabicl/lgbm) | feature blobs |
| vdb arms | `repaper_scaling` (vdb_rdblearn, vdb_rt) | FAISS indices |
| tuning grid | `repaper_tune` | checkpoint only -- the critical path |
| ensemble curves, default | `repaper_enscurve` | checkpoint only |
| `tuned_configs.json` | `repaper_tune` (`collect`, commit) | all 21 grids |
| ensemble curves, tuned | `repaper_enscurve` | `tuned_configs.json` |
| default-vs-tuned table | `repaper_valtest` | `tuned_configs.json`, `subsampled/rt` |
| leaderboard ensemble | `repaper_submit` | `tuned_configs.json` |
| pretraining ablations | `repaper_pretrain_abl` | nothing; lowest priority |

Submit the checkpoint-only stages together; they are the bulk of the GPU
work. Measured on this cluster (one a100): a full-test RT pass over rel-amazon
is 2-3 h, a tuning grid 4-7 h, the default ensemble curve 1-3 h per task;
the LightGBM arms are cpu-bound (a full-test rel-stack task is ~5 h at 32
cpus); the TabICL arms run on an a100 in a few hours per big task and also on
an rtx8000 at roughly a third of the speed. The whole eval family fits in
about a day of wall clock when the ampere pool is shared as usual.

## 3. Reduce and regenerate

When an arm is complete (21 task files), its reduce logs the wandb run the
paper reads: `repaper_scaling/reduce.py` (all 18 runs of the fulltest,
subsampled and ablation projects), `repaper_enscurve/reduce.py`,
`repaper_valtest/collect.py`, `repaper_submit/reduce.py` (which also writes
the submission CSVs; then `pixi run python -m relbench.submit <csv dir>`
validates and packages them -- the zips stay under
`$S/leaderboard/`, the upload is the issue form on rishabh-ranjan/relbench).
Commit `repaper_tune/tuned_configs.json`, `repaper_valtest/results.json`.

In the paper repo (`pixi run <task>`; the tasks are in its `pixi.toml`):
`results baselines baselines_extended per_task_baselines per_task_ensemble
per_task_retriever per_task_semantics ensemble retriever semantics mask_prob
task_mix tables compile`. Each figure script reads wandb and writes its PDF;
`tables` writes the two appendix table bodies. Inspect every PDF and update
the observation paragraphs -- the numbers in the prose are not generated.

## 4. What to know before it bites

- `db_cutoff=None` everywhere: per-row temporal masking is the only trim. The
  released pretraining run predates the knob, so the ablations match it with
  `None` too (`"test"` would resolve every Join source through the Hub and
  crash on databases without a test timestamp).
- The cpu-only partition (`il-cpu`) admits one QOS whose whole per-user
  budget a standing dev-node allocation holds; cpu work runs as zero-gres
  `il-lo` jobs on the `il` partition instead.
- A 1-gpu ampere job is capped at 252154M of memory by the submit plugin.
- Never pin a sweep to `hyperturing2`: its future slots are back-fill-claimed
  by higher-priority `il-lo` work a day out and pinned jobs park on
  `ReqNodeNotAvail`. `hyperturing1`'s cards throw ECC errors.
- blackwell1's eight b200s are normally held by preemptible `il-lo` jobs; an
  `il` (2 b200) or `il-interactive` (2) job preempts onto them within a few
  grace rounds -- the fastest cards on the cluster for the longest poles.
- Jobs that resume mid-run: tuning (per grid entry), ensemble and leaderboard
  units (per seed), pretraining (every 20 min). Scaling and featurize jobs
  restart their task on preemption; the biggest full-test tasks are hours.
- Committing a change under `rustler/` runs cargo hooks; they run through
  `pixi run` so they see the project's python.
