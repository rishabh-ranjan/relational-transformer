# Context-scaling evals (RT-J paper rerun)

The eval family behind the paper's context-scaling figures: the intro figure,
the main baselines comparison (fig:baselines) and its per-task appendix, the
extended-context baselines appendix, and the retriever and schema-semantics
ablations. One arm = one curve = one wandb run; every arm is a method plus a
context-sampler configuration over the same 21 RelBench tasks, at the paper's
shared default context (lcs=256, bw=32, prefer_latest=1) unless the arm
ablates exactly that.

Prerequisites: the feature blobs, FAISS indices, and TabICL checkpoints from
[`../baselines`](../baselines) (baseline and vdb arms only;
the RT arms need nothing but the released checkpoints mirrored under
`~/scratch/hf/stanford-star/rt-j`).

## Running it

```bash
# 1. the semantics-ablated data (once; needs only cpu + disk): uncomment the
#    make_nosem_data call in submit.py, comment out the arm loop, run

# 2. submit arms: write the arm list and the resources into submit.py, commit,
#    run. Jobs are idempotent per (arm, task): resubmitting fills in missing
#    task JSONs only.
pixi run python -m expts.repaper.scaling.submit

# 3. when an arm is complete, aggregate it into its wandb run
pixi run python -m expts.repaper.scaling.reduce
```

Per-task results land under
`~/scratch/ckpts/rtv2/repaper-scaling/<arm>/<db>__<table>.json`
(metric on the normalized scale + mean in-context labels per ctx size); logs
under
`~/scratch/slurm-logs/rishabh-ranjan/relational-transformer/expts/repaper/scaling`.

## Protocol

- **fulltest arms** (5 methods x ctx 256..8192): the full official test
  splits; what fig 2 / fig:baselines / the per-task appendix read.
- **subsampled arms** (5 methods): fixed 8192-row test subsample
  (shuffle_seed=0), baselines extended to 131072-cell contexts; what the
  extended-baselines appendix reads. `subsampled/rt` doubles as the
  random-walk arm of the retriever ablation and the semantics-on arm.
- **abl arms** (RT only, 8192-row subsample): `rand` (num_walks=0 and
  prefer_latest=False -- walk-free, uniformly random same-table fallback),
  `bfs32`/`bfs256` (one BFS around the target: local_ctx_size=8192 at the
  stated width), `vdb_rdblearn`/`vdb_rt` (FAISS-similarity seed selection over
  the corresponding feature space; the job's setup step rebuilds rustler with
  the `vecdb` feature), and `nosem` (the derived
  `repaper/relbench-preprocessed-nosem` data whose column-name embeddings are
  deranged; `make_nosem_data.py`).
- Everything is evaluated with `db_cutoff=None` (per-row temporal masking
  is the only trim), context_seed=0, a single context seed, and no tuning or
  ensembling.

Metrics are computed on rustler's normalized target scale, which for
regression equals RelBench's NMAE (rustler normalizes by the same train std)
and for classification is AUROC (sigmoid-invariant) -- so nothing here writes
submission CSVs; the leaderboard run lives in `../submit`.

`reduce.py` refuses to aggregate an incomplete arm, logs one history row per
ctx size under the `ctx_scaling/steps=0/test/*` keys (aggregates) and
`per_task/ctx_scaling/steps=0/relbench/<db>/<table>/test/*` (per task), into:

| wandb project | arms |
|---|---|
| `rtv2/2026-08-19-repaper-fulltest` | the five fulltest arms |
| `rtv2/2026-08-19-repaper-subsampled` | the five subsampled arms |
| `rtv2/2026-08-19-repaper-abl` | the six ablation arms + `abl/rw`,`abl/sem` re-logged from `subsampled/rt` |

## Placement

One a100 per RT or TabICL job through `il-lo` (TabICL also runs on an rtx8000
at about a third of the speed); 32-cpu zero-gres slots for LightGBM, whose
per-row fits `predict_batch` fans across the job's cpus. The rel-amazon
full-test passes are the longest poles and where the high tiers go;
[`../README.md`](../README.md) has the cluster rules.

## Measured runtimes

One a100, startup (clone build + compile) included, full-test RT arm:
rel-f1 tasks 1:27-2:58, rel-avito/ad-ctr 4:06, rel-event tasks 3:45-5:34
(700-2000 test rows each). The four rel-amazon tasks (167k-352k rows) are the
long poles, projected 5-12 h each at the measured ~8-45 rows/s. Subsampled
arms are bounded by 8192 rows/task.
