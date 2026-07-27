# Inference

Run a trained RT checkpoint on preprocessed data: load the checkpoint, sample an
in-context context for each test row, run the model's forward pass, and (for
RelBench) score the predictions with RelBench's own evaluator. There is no
fine-tuning — RT predicts zero-shot from the context it is given.

A checkpoint is a local path or a Hub model repo such as
`stanford-star/rt-j/classification`. All tasks in the configured task list are
evaluated regardless of the checkpoint; if the checkpoint's `config.json` says
it was selected for one task type (the released `classification` / `regression`
checkpoints), eval prints a note and still runs both.

## Prerequisite: preprocessed data

Inference takes an `--eval.pre-dir` of preprocessed data: a **local directory**,
either produced by you (see [preprocess.md](preprocess.md)) or downloaded up
front (see [downloads.md](downloads.md)). Data is never fetched on demand;
checkpoints still are. To reproduce the RelBench numbers:

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed

# the checkpoint still comes from the Hub, on demand
pixi run eval --model.load-ckpt-path stanford-star/rt-j/classification \
  --eval.pre-dir data/relbench-preprocessed
```

## Inference with default context

The command above runs **simple** inference: one default context config
(`--eval.lcs-bw-pl-grid 256 32 True`, total `--eval.ctx-size-list 8192`) on the
test split of every task in the default task list
(`data/relbench-preprocessed/db-task-lists/forecast.json`, the 21-task RelBench
benchmark). For each test row the sampler builds a context (a sampled
neighborhood of the relational graph), the model does a single forward pass,
and predictions are keyed back to each row by its seed node index. Because that
key is the seed node index and not a row position, per-row predictions stay
aligned regardless of eval row order.

Eval runs single-process on one GPU by default. For multi-GPU eval, launch it
under torchrun:

```bash
torchrun --nproc-per-node=8 -m rt.cli.eval --model.load-ckpt-path ...
```

Rows are sharded across ranks and gathered back on rank 0, which scores them
and writes the submission CSVs; the results are identical to a single-GPU run.

**Your own database (not RelBench).** `rt.eval` is wired to the RelBench
benchmark, but the pieces compose directly for any database — prediction is just
the model's forward over a sampled context. The [fully worked Colab
notebook](../byod/colab.ipynb) walks through it end-to-end (DuckDB
database → tasks defined in SQL → preprocess → forward pass → map outputs →
score) on the released RT-J checkpoints.

## Inference on a subset of tasks

The task set is `--eval.db-task-list`: `(db, task)` pairs given inline or as a
JSON file of pairs. To run one task:

```bash
pixi run eval --model.load-ckpt-path stanford-star/rt-j/classification \
  --eval.pre-dir data/relbench-preprocessed \
  --eval.db-task-list rel-f1 driver-top3
```

That reads just that task's data out of `--eval.pre-dir`, so it's the quickest
way to try the model end-to-end — and you can fetch only that database:
`hf download stanford-star/relbench-preprocessed --repo-type dataset --local-dir
data/relbench-preprocessed --include "rel-f1/*" "db-task-lists/*"`. Curated
lists ship with the data:
`<pre_dir>/db-task-lists/{forecast,autocomplete,all}.json`.

## Evaluate with the RelBench evaluator

Eval writes a valid RelBench **submission directory** to
`<logger.out-root>/<entity>/<project>/<id>/eval_out`: one
`<dataset>__<task>.csv` prediction table per task, scored through **RelBench's own
leaderboard evaluator** (`relbench.leaderboard`). Eval denormalizes regression
predictions to the original target scale (`y = pred*std + mean`, train-split
stats), maps classification logits to probabilities (sigmoid), and keys each
prediction to its relbench `(entity_col, time_col)`. It prints per-task and mean
test metrics — **AUROC** for clf, **NMAE** for reg.

Re-validate / re-score a submission dir any time, and submit it via the [RelBench
leaderboard procedure](https://relbench.stanford.edu):

```bash
pixi run python -m relbench.leaderboard eval_out
```

## Context engineering

Because RT predicts from context alone, the **context sampled for each row** is
the main quality knob. All are CLI flags on `eval`:

| flag | meaning | default |
|---|---|---|
| `--eval.ctx-size-list` | total context size (cells) the model attends over (one value for standalone eval) | 8192 |
| `--eval.lcs-bw-pl-grid` | `(local_ctx_size, bfs_width, prefer_latest)` context configs; one entry = use it directly, several = tune per task on validation | `256 32 True` |
| `--eval.num-walks` | random walks used to rank same-table neighbors | 10000 |
| `--eval.walk-length` | max length of each random walk | 20 |

Within a grid entry, larger `local_ctx_size` (max cells per BFS expansion
around the seed) and `bfs_width` (max DB nodes kept per BFS level) pull more
relational neighborhood into each row's context (more signal, more tokens);
`--eval.ctx-size-list` caps the total. `prefer_latest` controls *which* same-table
neighbors win that budget — the most recent rows (`True`, default) or the most
frequent (`False`). The best setting is task-dependent — which motivates tuning
and ensembling below.

`--eval.shuffle-seed` (default 0) seeds the per-task subset selection and item
shuffle. Fixing it while capping rows with `--eval.items-per-task N` evaluates
the *same* N validation rows across every config — the basis for a
like-for-like context grid search.

## Context tuning

Rather than fix one config, **tune** the context per task: pass several
`--eval.lcs-bw-pl-grid` entries and eval evaluates each on the **validation**
split, keeping the best per task before scoring test (here with a single test
seed, so no averaging yet):

```bash
pixi run eval \
  --model.load-ckpt-path stanford-star/rt-j/regression \
  --eval.pre-dir data/relbench-preprocessed \
  --eval.lcs-bw-pl-grid 256 32 True 512 64 True \
  --eval.ensemble-size 1
```

## Context ensembling

Context sampling is stochastic, so averaging predictions over several context
**seeds** reduces variance. Set `--eval.ensemble-size N` (> 1): the per-task
tuned config runs with N independent context seeds on test and the per-row
predictions are averaged before scoring:

```bash
pixi run eval \
  --model.load-ckpt-path stanford-star/rt-j/regression \
  --eval.pre-dir data/relbench-preprocessed \
  --eval.lcs-bw-pl-grid 256 32 True 512 64 True \
  --eval.ensemble-size 4
```

Tuning (on validation) and ensembling (on test) engage automatically whenever
the grid has more than one entry or `--eval.ensemble-size` exceeds 1: pick the
best context config per task, then average that config over the seeds.

## Optional: FAISS vector-DB sampler

The default sampler is FAISS-free. The opt-in FAISS vector-db sampler (for
nearest-neighbor context retrieval) is built manually and additionally needs
cmake + a BLAS:

```bash
maturin develop --release --features vecdb
```

## Legacy checkpoints (RT-v1, RT-PluRel)

The released checkpoints of the earlier papers use their original
architectures, kept verbatim in `rt.model.legacy` (state-dict compatible with
the published `.pt` files). Dedicated eval CLIs reproduce the published
context configuration (ctx 1024, one BFS neighborhood around the seed,
bfs_width 256, no random-walk tier) and write RelBench leaderboard submission
dirs:

```bash
# RT-v1 (ICLR 2026): task-wise pretrain_<db>_<task>.pt from stanford-star/rt-v1
pixi run python -m rt.cli.legacy.eval_v1 --out-dir v1_sub

# RT-PluRel (ICML 2026), stanford-star/rt-plurel:
pixi run python -m rt.cli.legacy.eval_plurel --mode synth      --out-dir plurel_synth_sub
pixi run python -m rt.cli.legacy.eval_plurel --mode synth-real --out-dir plurel_sr_sub
```

`--mode synth` uses the best synthetic-only pretraining checkpoint (same for
all tasks); `--mode synth-real` uses the task-wise continued-pretraining
checkpoints. All three are in-context: no checkpoint ever trained on the
target task's database (v1, synth) or task (synth-real).

`data/relbench-preprocessed/legacy` (from
`stanford-star/relbench-preprocessed`, subdir `legacy/`) holds RelBench re-preprocessed
with `rt.cli.legacy.preprocess`, which applies the RT-v1-era boolean-typing
rules (binary targets and a few db columns become a real Boolean semantic
type instead of z-scored numbers) before the regular pipeline. The legacy
nets read classification targets from their BCE-trained boolean head, so this
is the data they need; both CLIs default to it. `eval_plurel` additionally
defaults to the paper's bfs_width 128. Metrics reproduce the papers within noise except
RT-v1 on rel-avito, which degrades for sampler-level reasons outside these
configs.
