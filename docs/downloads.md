# Downloads

Our HuggingFace org [`stanford-star`](https://huggingface.co/stanford-star)
provides a lot of resources — raw data, preprocessed data, and model
checkpoints — usable directly. The scripts here take care of HuggingFace
downloads automatically (everywhere a dataset, `pre_dir`, or checkpoint is
taken, you can pass a Hub spec and it is fetched and cached on demand), so you
rarely need to download anything by hand.

Automatic downloads can get rate-limited on large/bulk pulls, though. To
warm the cache up front, use the `hf` CLI to pre-fetch the whole repo:

```bash
# Raw "the Join" (650+ databases in RelBench format)
pixi run hf download stanford-star/the-join --repo-type dataset

# Preprocessed "the Join" (rustler artifacts, ready for RT)
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset

# Raw RelBench databases (RelBench format)
pixi run hf download stanford-star/relbench --repo-type dataset

# Preprocessed RelBench (rustler artifacts, ready for RT)
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset

# RT-J checkpoints (classifier under classification/, regressor under regression/)
pixi run hf download stanford-star/rt-j --repo-type model
```

These download into the shared HuggingFace cache (`~/.cache/huggingface/hub`,
or `$HF_HOME`). Afterwards just keep passing the **same Hub specs** to the
scripts — `hf` sees the files are already present and serves them from the
cache instead of re-downloading. No custom paths to manage; the cache location
is entirely HuggingFace's concern. Useful flags: `--include`/`--exclude` (glob
patterns to grab a subset, e.g. one database from `the-join`), `--max-workers`
(parallel downloads), and `--revision` (pin a branch, tag, or commit).
