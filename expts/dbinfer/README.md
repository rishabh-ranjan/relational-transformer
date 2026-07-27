# 4DBInfer context scaling

RT-J against a tabular baseline on the 4DBInfer tasks, over the context sweep
256 → 8192. One table: mean AUROC over the classification tasks. There is no
regression table -- see "Tasks that are not here".

| | |
|---|---|
| methods | `rt` (RT-J) and `rdblearn_tabicl` = `precomputed_rdblearn` + `tabicl_batched` |
| clf tasks (4) | `dbinfer-diginetica/ctr`, `dbinfer-retailrocket/cvr`, `dbinfer-stackexchange/churn`, `dbinfer-stackexchange/upvote` |
| reg tasks | none -- see below |
| context points | 256, 512, 1024, 2048, 4096, 8192 |
| test subsample | 1024 rows per task |

Data is [`stanford-star/dbinfer`](https://huggingface.co/datasets/stanford-star/dbinfer),
the port of the 4DBInfer benchmark rebuilt from the upstream archives.

## Running it

```bash
./expts/dbinfer/launch.sh                              # everything, chained
RT_SKIP_FEATURIZE=1 ./expts/dbinfer/launch.sh           # the rt-j path only
pixi run python expts/dbinfer/reduce.py \
    --out-dir /dfs/user/$USER/dbinfer-scaling --per-task
```

`launch.sh` queues all three compute stages immediately and lets slurm sequence
them, rather than waiting for one to finish before submitting the next:

| stage | shape | placement |
|---|---|---|
| 1 preprocess | array of 3, one per db | `il-lo`, 48G GPU node, off ampere |
| 2 featurize | array of 3, chained per db | `il-lo`, CPU-only, off ampere |
| 3 eval `rt` | after all of stage 1 | **`il`**, 8x a100, not ampere4 |
| 3 eval `rdblearn_tabicl` | after all of stage 2 | `il-lo`, 8x a100, not ampere4 |

Stage 2 is chained per *database* (`afterok:<arrayjob>_<idx>`), so a db starts its
DFS as soon as its own preprocess finishes rather than waiting for the slowest of
the four.

The QOS split is forced by the cluster, not preference: the `il` QOS caps a user at
`gres/gpu:a100=10`, which is exactly one 8-GPU node. RT-J is the priority, so it
takes `il` and the baseline takes `il-lo`. Everything that is not the eval stays off
the ampere nodes so it cannot compete for them. The `il-cpu` partition is unusable
here -- its only QOS caps a user at `cpu=8/mem=60G` in total, which a dev-node
allocation already occupies.

Stage 3 is one job per **method**, not per (method, task): `eval.py` loops the task
list internally and builds each target's context once at ctx=8192, reading all six
context points off as prefixes. So one job produces a whole table row -- which is
what removes the per-ctx output directories and the column-merge step the RelBench
campaign's launcher needed one job per (task, ctx) to work around.

Every stage skips work whose output already exists, so a requeue after preemption
or a rerun to mop up a failure is safe and cheap, and re-running `launch.sh` is the
intended way to top up.

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
* `dbinfer-amazon/{churn,rating}` -- dropped on cost. Preprocessing amazon means
  embedding 21.2M texts, and its review bodies are long: measured at ~808 texts/s
  on a 2080ti, that is ~7 h for the MiniLM pass alone, against ~12 min for
  diginetica and retailrocket. It was the only regression task in the collection,
  so **the regression table goes with it** -- 4DBInfer's other non-clf tasks are
  2 link-prediction and 2 multiclass, none of which RT has a head for.
* `dbinfer-outbrain-small/ctr` -- dropped. The upstream subsample has almost no
  referential integrity: 58 of 69,543 distinct train entities exist in `Event`. The
  number it yields would describe the defect, not a method.
* `dbinfer-seznam/{charge,prepay}` -- multiclass, and RT has no multiclass head.
* `dbinfer-amazon/purchase`, `dbinfer-diginetica/purchase` -- link prediction under
  4DBInfer's MRR-over-supplied-candidates protocol, which is not what RelBench's
  link metric computes.

So this measures 4 of 4DBInfer's 12 tasks, all classification. `reduce.py` still
emits the regression table; with no reg tasks present it prints "(no reg tasks)".

**Label leakage, both sides.** One task takes its label from a column that is also
in the database (`retailrocket/cvr` ← `View.added_to_cart`). RT's sampler drops such a column on rows sharing the target's
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
| `launch.sh` | submits all stages at once, chained by slurm dependencies |
| `slurm_*.sh` | one per stage |
