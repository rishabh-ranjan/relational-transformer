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
`~/scratch/relational-transformer/repaper/scaling/slurm-logs`.

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

`reduce.py` skips an incomplete arm and an arm it has logged (a `wandb-*.json`
marker beside the task files names the run), logs one history row per
ctx size under the `ctx_scaling/steps=0/test/*` keys (aggregates) and
`per_task/ctx_scaling/steps=0/relbench/<db>/<table>/test/*` (per task), into:

| wandb project | arms |
|---|---|
| `rtv2/<RUN_TAG>-repaper-fulltest` | the five fulltest arms |
| `rtv2/<RUN_TAG>-repaper-subsampled` | the five subsampled arms |
| `rtv2/<RUN_TAG>-repaper-abl` | the six ablation arms + `abl/rw`,`abl/sem` re-logged from `subsampled/rt` |

## Placement

One a100 per RT or TabICL job through `il-lo` (TabICL also runs on an rtx8000
at about a third of the speed); 24-cpu zero-gres slots on the cpu-only
partition (`il-cpu`: rambo and furiosa; until 2026-08-27 15:00 they ran under
`il` on hyperturing1, saturating the interactive node) for LightGBM, whose
per-row fits `predict_batch` fans across the job's cpus. The rel-amazon
full-test passes are the longest poles and where the high tiers go;
[`../README.md`](../README.md) has the cluster rules.

## Measured runtimes

2026-08-27 (`sacct`), startup included. Full-test RT arm: rel-amazon
user-churn / user-ltv 3h21 each on a b200, item-ltv 1h42 on a b200,
item-churn 2h53 on an a100; rel-stack user-badge 4h24, post-votes 2h48,
user-engagement 1h32, rel-hm item-sales 1h53, user-churn 1h25, rel-avito
user-clicks 0h53, user-visits 0h40, rel-trial site-success 0h24, everything
under 2000 test rows 2-6 min (a100). Full-test TabICL arms (a100): rel-amazon
user tasks ~1h50, item tasks ~1h, rel-stack user-badge 1h22 on a b200
(sql) and 5h29 on an a100 (rdblearn), post-votes 1h00-1h19, rel-hm item-sales
1h10-1h31; full-test LightGBM at 16 cpus: rel-amazon user tasks ~1h30.
Subsampled RT and every RT ablation pass: 5-15 min. Subsampled TabICL at
131k cells: rel-stack user-badge 7h07 (rdblearn) / 5h30 (sql) on an a100,
rel-hm item-sales and rel-stack post-votes 1-3 h, the rest under an hour;
subsampled LightGBM at 16 cpus under 1h15. The whole scaling stage ran
00:16-09:01 on up to 12 high-tier slots plus il-lo. The full-test extension
pieces (2026-08-28/29, one context size per job): on a b200 the 131k pieces
took 8h (rel-amazon user-ltv, rdblearn), 9h13 (rel-stack post-votes, sql),
10h (rel-hm item-sales, rdblearn) and 11h53 (rel-stack post-votes,
rdblearn), 5-9x under their a100 projections; the 65k rdblearn pieces took
10h30 on an a100; LightGBM fits at 24 cpus on rambo/furiosa ran 3x faster
than at 16 on hyperturing1 (all 42 in under two hours).
