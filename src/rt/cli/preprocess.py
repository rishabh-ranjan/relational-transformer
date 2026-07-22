"""CLI for rt.preprocess. All defaults live here; see rt.preprocess for logic."""

from typing import Union

import tyro
from typing_extensions import Annotated

from rt.preprocess import (
    ListConfig,
    ManyConfig,
    OneConfig,
    UploadConfig,
    main,
)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 1024


def default_one() -> OneConfig:
    return OneConfig(
        dataset=tyro.MISSING,
        out_dir=tyro.MISSING,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        batch_size=DEFAULT_BATCH_SIZE,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


def default_many() -> ManyConfig:
    return ManyConfig(
        repo=tyro.MISSING,
        out_dir=tyro.MISSING,
        shard=0,
        num_shards=1,
        skip_existing=False,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        batch_size=DEFAULT_BATCH_SIZE,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


def default_list() -> ListConfig:
    return ListConfig(repo=tyro.MISSING, revision=None)


def default_upload() -> UploadConfig:
    return UploadConfig(
        pre_dir=tyro.MISSING,
        repo=tyro.MISSING,
        public=False,
        bulk=False,
    )


Config = Union[
    Annotated[OneConfig, tyro.conf.subcommand("one", default=default_one())],
    Annotated[ManyConfig, tyro.conf.subcommand("many", default=default_many())],
    Annotated[UploadConfig, tyro.conf.subcommand("upload", default=default_upload())],
    Annotated[ListConfig, tyro.conf.subcommand("list", default=default_list())],
]

if __name__ == "__main__":
    main(tyro.cli(Config))
