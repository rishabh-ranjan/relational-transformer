# Baseline featurization (RT-J paper rerun)

Everything the paper's tabular baselines and retriever ablations need before
any eval can run: the `rel2tab/` library (label-matched featurizer+predictor
models the eval drives like an RT net), precomputed per-row features for every
RelBench task table, and the FAISS indices for the vector-similarity retriever
arms. The evals themselves live in [`../scaling`](../scaling).

## Running it, in order

```bash
# 0. one-time fetches, on the login node (compute nodes have no internet)
pixi run python -m expts.repaper.baselines.fetch_tabicl
RELBENCH_CACHE_DIR=~/scratch/hf/relational-transformer/repaper/relbench-cache \
    pixi run -e featurize python -m expts.repaper.baselines.populate_relbench_cache

# 1. one-time install into the featurize env (jobs repeat it via setup=)
pixi run install-rdblearn

# 2. prove classic-relbench row order matches the preprocessed data
RELBENCH_CACHE_DIR=~/scratch/hf/relational-transformer/repaper/relbench-cache \
    pixi run -e featurize python -m expts.repaper.baselines.check_alignment

# 3. featurize (comment out the stages not wanted in submit.py), then the
#    indices (the commented-out vector-db loop); a finished table is skipped at
#    submit time, so resubmitting fills in the missing blobs only
pixi run python -m expts.repaper.baselines.submit
```

Feature blobs land under
`~/scratch/hf/relational-transformer/repaper/features/<db>/{sql,rdblearn,rt}_features/`,
FAISS indices under `.../repaper/vector_db/{rdblearn,rt}/`, one
`<table>_vectors.bin` + `<table>_meta.json` (and `.index`) per task table.
Logs land under
`~/scratch/relational-transformer/repaper/baselines/slurm-logs`.

## What each featurizer is

- **`featurize_sql.py`** -- the "LLM Data Scientist" features: the committed
  DuckDB queries in [`sql_queries/`](sql_queries/) (written once by a Claude
  Code agent against each database), run over the full database
  (`upto_test_timestamp=False`) and z-scored globally.
- **`featurize_rdblearn.py`** -- RDBLearn's fastdfs depth-2 DFS features
  (agg primitives max/min/mean/count/mode/std, `dfs2sql` engine, per-row
  temporal cutoffs from the task's time column), transformed by the fitted
  RDBLearn preprocessor.
- **`featurize_rt.py`** -- RT-J row embeddings (the masked target cell's
  final-layer state over a walk-free 256-cell local context), one file per
  table, for the RT-similarity retriever arm.

Feature row `r` of a table is node `min_offset + r` in the preprocessed data;
`check_alignment.py` is what makes that mapping trustworthy for the two
classic-relbench featurizers, and every featurize job asserts its row counts
against `table_info.json`.

## Environments

The two classic-relbench featurizers run in the `featurize` pixi environment
(classic relbench 2.x API + fastdfs + duckdb; rdblearn arrives via the
`install-rdblearn` pixi task -- its pins conflict with autogluon's, so it is
installed `--no-deps` at a pinned commit). Everything else runs in the default
environment. Jobs pass `pixi_env="featurize"` and build the env in `setup=`.

## The vecdb sampler build

The FAISS indices are read by rustler's opt-in `vecdb` cargo feature, which
the default build does not include. Jobs that pass `vector_db_path` (the
`abl/vdb_*` arms in `../scaling`) rebuild the extension in their
clone's setup step:

```python
setup=("pixi run maturin develop --uv --release --features vecdb",)
```

cmake / g++ / openblas for that build ship in the default environment.
