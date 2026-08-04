# Preprocess

RT trains and predicts on a custom on-disk format produced by the `rustler`
preprocessor from **any dataset in relbench format**. A dataset is a local path, or a HuggingFace Hub spec `org/repo[/subdir]`
(e.g. `stanford-star/relbench/rel-f1`).

Preprocessing runs `download/resolve → rustler → text embeddings` and writes a
self-contained `<out-dir>/<name>/` directory. Text embeddings use all visible
GPUs automatically (Sentence-Transformers multi-process); rustler itself is
multithreaded (rayon).

## Preprocess one database in RelBench format

```bash
`examples/preprocess.py`
```

This writes `~/scratch/pre/rel-f1/` containing rustler artifacts,
which are used by RT dataloaders.

Any dataset in relbench format works by swapping the `--dataset` argument — the
manifest is the sole source of relational metadata; the parquet files carry only
native dtypes. Useful flags: `--skip-tasks` (ingest db tables only), `--no-embed`,
`--embedder`, `--batch-size`, and `--upload-repo <hub repo>` (preprocess
and push in one step).

## Preprocess many databases efficiently

To preprocess a whole Hub collection (e.g. the 650-database [the Join](https://huggingface.co/datasets/stanford-star/the-join)):

```bash
pixi run python -c "from rt.preprocess import ls; ls(repo='stanford-star/the-join', revision=None)"
`examples/preprocess.py`
  --repo stanford-star/the-join --out-dir ~/scratch/the-join-pre \
  --shard 0 --num-shards 1 --skip-existing
```

`--skip-existing` makes the pass resumable (datasets whose embeddings are already
written are skipped). `--shard i --num-shards N` splits the collection across a
job array (e.g. a preemptible Slurm array with `--array=0-63` mapping the task id to `--shard`).

## Using preprocessed data

Everywhere a `pre_dir` is taken (see [inference](inference.md) and
[pretrain](train.md)) it is a **local directory**: what this preprocessor wrote,
or a published collection you downloaded with `hf download --local-dir` (see
[downloads.md](downloads.md)). Nothing is fetched on demand, so you never have
to upload anything to use your own data, and a run's data is a path you can
inspect. The layout is the same either way — one subdirectory per database — so
your own output and a downloaded collection are interchangeable.
