"""Pin the preprocessed pretraining mixture in RAM across training restarts.

mmap+mlocks every database's rustler input files (nodes, text embeddings,
adjacency) into the page cache and sleeps holding the locks until
SIGINT/SIGTERM. Run pretraining with --no-train.mmap-populate alongside it and
restarts skip re-loading the ~1TB mixture from shared storage. Purely an
optional convenience for fast debug iterations; needs a high RLIMIT_MEMLOCK
(ulimit -l unlimited).
"""

import tyro

from rt.mlock_recipe import MlockConfig, main


def default_config() -> MlockConfig:
    return MlockConfig(
        pre_dir="stanford-star/the-join-preprocessed",
        include_dbs_file=None,
        embedding_model_ref="all-MiniLM-L12-v2",
        workers=32,
    )


if __name__ == "__main__":
    main(tyro.cli(MlockConfig, default=default_config(), description=__doc__))
