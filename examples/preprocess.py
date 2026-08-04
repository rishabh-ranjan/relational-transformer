"""Preprocess a relbench-format dataset into the tensors RT trains on.

No CLI: uncomment the call you want, edit the arguments, run it.

    pixi run python examples/preprocess.py

``one`` handles a single dataset, ``many`` a whole Hub collection (optionally
sharded across a slurm array), ``ls`` prints what is in a collection, and
``upload`` publishes the result. Raw inputs may be a local path or a Hub spec.
"""

from __future__ import annotations

from rt.preprocess import ls, many, one, upload  # noqa: F401


def preprocess_one_dataset() -> None:
    one(
        dataset="stanford-star/relbench/rel-f1",
        out_dir="data/relbench-preprocessed",
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


def preprocess_a_collection() -> None:
    """`shard`/`num_shards` split the work across a job array; each job takes
    every num_shards-th dataset."""
    many(
        repo="stanford-star/the-join",
        out_dir="data/the-join-preprocessed",
        shard=0,
        num_shards=1,
        skip_existing=True,
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


if __name__ == "__main__":
    preprocess_one_dataset()
