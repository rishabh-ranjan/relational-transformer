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

def default_one() -> OneConfig:
    return OneConfig(
        dataset="stanford-star/relbench/rel-f1",
        out_dir="~/scratch/pre",
        embedding_model="all-MiniLM-L12-v2",
        batch_size=1024,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


def default_many() -> ManyConfig:
    return ManyConfig(
        repo="stanford-star/the-join",
        out_dir="~/scratch/the-join-pre",
        shard=0,
        num_shards=1,
        skip_existing=False,
        embedding_model="all-MiniLM-L12-v2",
        batch_size=1024,
        skip_tasks=False,
        embed=True,
        upload_repo=None,
        public=False,
        revision=None,
    )


def default_list() -> ListConfig:
    return ListConfig(repo="stanford-star/the-join", revision=None)


def default_upload() -> UploadConfig:
    return UploadConfig(
        pre_dir="~/scratch/pre",
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
