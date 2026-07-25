# Downloads

Our HuggingFace org [`stanford-star`](https://huggingface.co/stanford-star)
provides raw data, preprocessed data, and model checkpoints.

**Data is downloaded up front, by you; only checkpoints are fetched on demand.**
A `pre_dir` is always a local directory. Preprocessed datasets run to hundreds
of GiB and every rank and dataloader worker of a run reads them, so on-demand
fetching meant thousands of Hub requests per run (HTTP 429 rate limits, even
when the bytes were already cached, because each call still revalidates over the
network) and a separate copy per machine. One explicit `hf download` into a path
you choose is faster, and the data a run used is a directory you can inspect.

Download the preprocessed data you need with the `hf` CLI:

```bash
# Preprocessed "the Join" (rustler artifacts, ready for RT) -- the pretraining data
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed

# Preprocessed RelBench (rustler artifacts, ready for RT) -- validation/eval data
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed
```

Those are the paths the scripts default to (`--train.pre-dir data/the-join-preprocessed`,
`--eval.pre-dir data/relbench-preprocessed`); pass your own to put them elsewhere.
Each repo also ships its curated task lists under `db-task-lists/`, so they
arrive with the data they refer to.

The full preprocessed Join is **~1.5 TiB**. To fetch only what a run needs, keep
the core rustler artifacts plus the one text embedder you train with, and skip
`text.json`:

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed --max-workers 16 \
  --include "db-task-lists/*" "*/meta.json" "*/table_info.json" "*/column_index.json" \
            "*/nodes.rkyv" "*/offsets.rkyv" "*/p2f_adj.rkyv" \
            "*/text_emb_all-MiniLM-L12-v2.bin"
```

Narrow it further with `--include "<db>/*"` per database (a `db-task-lists/*.json`
entry names the dbs a mixture needs). Sizes: `nodes.rkyv` ~1055 GiB,
`p2f_adj.rkyv` ~193 GiB, `text_emb_all-MiniLM-L12-v2.bin` ~150 GiB,
`offsets.rkyv` ~42 GiB, `text.json` ~27 GiB.

Raw data (only needed to re-run preprocessing yourself, see
[preprocess.md](preprocess.md)) and checkpoints:

```bash
# Raw "the Join" (650+ databases in RelBench format)
pixi run hf download stanford-star/the-join --repo-type dataset

# Raw RelBench databases (RelBench format)
pixi run hf download stanford-star/relbench --repo-type dataset

# RT-J checkpoints (classifier under classification/, regressor under regression/)
pixi run hf download stanford-star/rt-j --repo-type model
```

Checkpoints are the one thing still resolved from the Hub on demand: a single
small file fetched once by one process, so `load_rt_model("stanford-star/rt-j")`
and `--model.load-ckpt-path stanford-star/rt-j` keep working without a manual
download. The preprocessor also reads its *raw* inputs straight from the Hub.

Without `--local-dir` these land in the shared HuggingFace cache
(`~/.cache/huggingface/hub`, or `$HF_HOME`), which is what you want for
checkpoints and raw preprocessing inputs. For a `pre_dir`, use `--local-dir` so
the path you pass to the scripts is a plain directory. Useful flags:
`--include`/`--exclude` (glob patterns to grab a subset), `--max-workers`
(parallel downloads), and `--revision` (pin a branch, tag, or commit).
