# 4DBInfer context scaling

RT-J against a tabular baseline on the 4DBInfer tasks, over the context sweep
256 → 8192. Two tables: mean AUROC over the classification tasks, mean NMAE over
the regression tasks.

| | |
|---|---|
| methods | `rt` (RT-J) and `rdblearn_tabicl` = `precomputed_rdblearn` + `tabicl_batched` |
| clf tasks (5) | `dbinfer-amazon/churn`, `dbinfer-diginetica/ctr`, `dbinfer-retailrocket/cvr`, `dbinfer-stackexchange/churn`, `dbinfer-stackexchange/upvote` |
| reg tasks (1) | `dbinfer-amazon/rating` |
| context points | 256, 512, 1024, 2048, 4096, 8192 |
| test subsample | 1024 rows per task |

Data is [`stanford-star/dbinfer`](https://huggingface.co/datasets/stanford-star/dbinfer),
the port of the 4DBInfer benchmark rebuilt from the upstream archives.

## The four stages

```bash
./expts/dbinfer/slurm_preprocess.sh              # 1. rustler format + text embeddings
./expts/dbinfer/slurm_featurize.sh               # 2. depth-2 DFS feature matrices
RT_METHODS=rt ./expts/dbinfer/slurm_eval.sh      # 3. the eval (rt-j first)
pixi run python expts/dbinfer/reduce.py \
    --out-dir /dfs/user/$USER/dbinfer-scaling --per-task    # 4. the tables
```

Stage 2 is only needed for `rdblearn_tabicl`; `RT_METHODS=rt` skips straight past
it. Every stage is idempotent and skips completed work, so a requeue after
preemption or a rerun to mop up failures is safe and cheap.

Stage 3 submits one job per **(method, task)** -- 12 points, not 72. `eval.py`
builds each target's context once at ctx=8192 and reads all six context points off
as prefixes of it, so one job produces a task's entire row. That is what removes
the per-ctx output directories and the column-merge step the RelBench campaign's
launcher needed.

## Why `lbl` is in the tables

`mean_labels` -- the mean number of labeled rows visible in a target's context --
is a pure function of the sampler: context config, seeds, `items_per_task`. It has
nothing to do with which model consumed the context. So both methods must report
the same value at every context point, and `reduce.py` exits non-zero if they do
not. When they disagree, the methods were not measured on the same x-axis and the
metric columns are not comparable, whatever they say.

The 1024-row test subsample is part of the same guarantee: the sampler picks it
from `(task, items_per_task, shuffle_seed)` alone, so both methods score the same
1024 rows.

## Choices worth knowing about

**Untuned context config.** The RelBench campaign selected
`(local_ctx_size, bfs_width, prefer_latest)` per task by argmax over a validation
grid. No such grid exists for 4DBInfer, so a single uniform `(256, 32, True)` is
used for every task and both methods, with `local_ctx_size` clamped to the context
point. The numbers are therefore untuned, equally so for both methods.

**Tasks that are not here.** 4DBInfer has 12 tasks; this runs 6.

* `dbinfer-avs/repeater` -- dropped. Its `Transaction` table is 349,655,789 rows,
  ~5x the largest table in the-join, and it is expensive in both stage 1 and
  stage 2.
* `dbinfer-outbrain-small/ctr` -- dropped. The upstream subsample has almost no
  referential integrity: 58 of 69,543 distinct train entities exist in `Event`. The
  number it yields would describe the defect, not a method.
* `dbinfer-seznam/{charge,prepay}` -- multiclass, and RT has no multiclass head.
* `dbinfer-amazon/purchase`, `dbinfer-diginetica/purchase` -- link prediction under
  4DBInfer's MRR-over-supplied-candidates protocol, which is not what RelBench's
  link metric computes.

That leaves one regression task, so Table 2's "mean NMAE" is a mean of one. It is
reported with `n_reg=1` stated rather than dropped.

**Label leakage, both sides.** Two tasks take their label from a column that is
also in the database (`retailrocket/cvr` ← `View.added_to_cart`, `amazon/rating` ←
`Review.rating`). RT's sampler drops such a column on rows sharing the target's
timestamp and keeps strictly-past ones. fastdfs aggregates with a strict `<`
against the task cutoff (`include_cutoff_time=False`, pinned in
`rel2tab/featurizers/rdblearn_featurizer.py` so an upstream default flip cannot
silently change it), which is the same rule. Neither method sees its own label.

**Both methods see the same schema.** `fastdfs`'s `DBInferAdapter` filters each
table to the columns `metadata.yaml` declares, which is exactly the column set the
`stanford-star/dbinfer` port publishes -- so, for instance, neither sees
`Posts.Score`, the column `stackexchange/upvote`'s label is derived from.

**Row order is the contract.** `PrecomputedFeaturizer` maps a node to a feature row
by `node_idx - min_offset`, so the archive's task rows must line up one-for-one with
the preprocessed label tables. The featurizer asserts equal row counts per split;
`featurize.py --verify-rows <dbinfer-build-dir>` additionally compares the label
sequences against the published parquets, which is the check that would catch a
reordering. Run it at least once per database.

## Layout

| path | |
|---|---|
| `tasks.json` | the `(db, task)` list |
| `eval.py` | the eval driver -- a copy of `src/rt/cli/eval.py` extended with a context sweep, its own metrics, and per-task JSON output |
| `featurize.py` | stage 2: DFS matrices for `PrecomputedFeaturizer` |
| `reduce.py` | stage 4: JSONs → tables, plus the `mean_labels` check |
| `rel2tab/` | restored from `b999183^:src/rt/rel2tab/`, pruned to this one baseline |
| `slurm_*.sh` | one per stage |
