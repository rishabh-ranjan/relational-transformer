"""Preprocess RelBench datasets with legacy (RT-v1) boolean typing.

Applies the RT-v1 per-database boolean casting rules to the source parquets,
then runs the regular rustler `pre` + embedding pipeline. Results upload
under ``legacy/<db>/`` of --upload-repo; consume with
``pre_dir=<repo>/legacy``.

    python -m rt.cli.legacy.preprocess --dataset stanford-star/relbench/rel-f1 \
        --out-dir ~/scratch/pre-legacy --upload-repo stanford-star/relbench-preprocessed
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from rt.preprocess.legacy import preprocess_one_legacy


@dataclass
class Config:
    # Local path or org/repo[/subdir] of the source RelBench dataset.
    dataset: str
    # Preprocessed-data output root.
    out_dir: str
    # Hub repo to upload the result to (under legacy/<db>/); None = no upload.
    upload_repo: str | None = None
    embedding_model: str = "all-MiniLM-L12-v2"
    batch_size: int = 8192
    public: bool = True
    revision: str | None = None


def main(cfg: Config) -> None:
    preprocess_one_legacy(
        cfg.dataset,
        Path(cfg.out_dir).expanduser(),
        embedding_model=cfg.embedding_model,
        batch_size=cfg.batch_size,
        upload_repo=cfg.upload_repo,
        private=not cfg.public,
        revision=cfg.revision,
    )


if __name__ == "__main__":
    main(tyro.cli(Config, description=__doc__))
