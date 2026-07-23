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

Inference takes an `--eval.pre-dir` of preprocessed data — either a local path
you produced (see [preprocess.md](preprocess.md)) or a Hub repo such as
`stanford-star/relbench-preprocessed`, downloaded and cached on demand. So you
can reproduce the RelBench numbers with nothing downloaded up front:

```bash
# checkpoint and data both come from the Hub
pixi run eval --model.load-ckpt-path stanford-star/rt-j/classification \
  --eval.pre-dir stanford-star/relbench-preprocessed --eval.out-dir eval_out
```

## Inference with default context

The command above runs **simple** inference: one default context config
(`--eval.lcs-bw-pl-grid 256 32 True`, total `--eval.ctx-sizes 8192`) on the
test split of every task in the default task list
(`stanford-star/relbench/db-task-lists/forecast.json`, the 21-task RelBench
benchmark). For each test row the sampler builds a context (a sampled
neighborhood of the relational graph), the model does a single forward pass,
and predictions are keyed back to each row by its seed node index. Eval is
single-process (one GPU) so per-row predictions stay aligned regardless of eval
row order.

**Your own database (not RelBench).** `rt.eval` is wired to the RelBench
benchmark, but the pieces compose directly for any database — prediction is just
the model's forward over a sampled context. The [fully worked Colab
notebook](../byod/colab.ipynb) walks through it end-to-end (DuckDB
database → tasks defined in SQL → preprocess → forward pass → map outputs →
score) on the released RT-J checkpoints.

## Inference on a subset of tasks

The task set is `--eval.db-task-list`: `(db, task)` pairs given inline, as a
local JSON file of pairs, or as a Hub path to such a file. To run one task:

```bash
pixi run eval --model.load-ckpt-path stanford-star/rt-j/classification \
  --eval.pre-dir stanford-star/relbench-preprocessed \
  --eval.db-task-list rel-f1 driver-top3 --eval.out-dir eval_out
```

This downloads only that task's data (the Hub `--eval.pre-dir` is fetched on
demand), so it's the quickest way to try the model end-to-end. Curated lists
ship on the Hub: `stanford-star/relbench/db-task-lists/{forecast,autocomplete,all}.json`.

## Evaluate with the RelBench evaluator

`--eval.out-dir` is a valid RelBench **submission directory**: one
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
| `--eval.ctx-sizes` | total context size (cells) the model attends over (one value for standalone eval) | 8192 |
| `--eval.lcs-bw-pl-grid` | `(local_ctx_size, bfs_width, prefer_latest)` context configs; one entry = use it directly, several = tune per task on validation | `256 32 True` |
| `--eval.num-walks` | random walks used to rank same-table neighbors | 10000 |
| `--eval.walk-length` | max length of each random walk | 20 |

Within a grid entry, larger `local_ctx_size` (max cells per BFS expansion
around the seed) and `bfs_width` (max DB nodes kept per BFS level) pull more
relational neighborhood into each row's context (more signal, more tokens);
`--eval.ctx-sizes` caps the total. `prefer_latest` controls *which* same-table
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
  --eval.pre-dir stanford-star/relbench-preprocessed \
  --eval.lcs-bw-pl-grid 256 32 True 512 64 True \
  --eval.ensemble-size 1 --eval.out-dir eval_out
```

## Context ensembling

Context sampling is stochastic, so averaging predictions over several context
**seeds** reduces variance. Set `--eval.ensemble-size N` (> 1): the per-task
tuned config runs with N independent context seeds on test and the per-row
predictions are averaged before scoring:

```bash
pixi run eval \
  --model.load-ckpt-path stanford-star/rt-j/regression \
  --eval.pre-dir stanford-star/relbench-preprocessed \
  --eval.lcs-bw-pl-grid 256 32 True 512 64 True \
  --eval.ensemble-size 4 --eval.out-dir eval_out
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

By default these CLIs read `stanford-star/relbench-preprocessed/legacy`:
RelBench re-preprocessed with `rt.cli.legacy.preprocess`, which applies the
RT-v1-era boolean-typing rules (binary targets and a few db columns become a
real Boolean semantic type instead of z-scored numbers) before the regular
pipeline. With it, `bool_as_num=False` reads classification targets from the
BCE-trained boolean head, matching the legacy models' training. Pass
`--pre-dir stanford-star/relbench-preprocessed --bool-as-num` for the
modern-typed data instead. Metrics reproduce the papers within noise except
RT-v1 on rel-avito, which degrades for sampler-level reasons outside these
configs.
