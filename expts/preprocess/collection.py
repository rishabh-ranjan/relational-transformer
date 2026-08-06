"""Which collection a command acts on, and where its pieces live.

One pipeline, two collections: the Join (639 databases, ~1.4 TiB out) and
RelBench (7 databases, ~230 GiB). They differ only in paths and in two details
that are properties of the published repo rather than of the work -- a curated
task list that cannot be recomputed, and directories a mirror upload must not
delete -- so they are data here rather than a second copy of the experiment.

Every command takes the collection as its first argument:

    pixi run python expts/preprocess/submit.py the-join
    pixi run python expts/preprocess/download.py relbench
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
SHARE = "/dfs/user/ranjanr/share/stanford-star"


@dataclass(frozen=True)
class Collection:
    name: str
    source_repo: str
    """Hub repo of the raw data, and what `meta.json` records as its origin."""
    target_repo: str
    curated: str | None
    """A task list that cannot be derived from the build, kept here because the
    upload that replaces the repo would otherwise be the thing that loses it.
    None when every list the collection publishes is derivable."""
    keep: tuple[str, ...]
    """Directories the published repo carries that are not a database of this
    build, and that a mirror upload must therefore not delete."""
    legacy: bool
    """Whether this collection also publishes a `legacy/` tree: the same
    databases under RT-v1's boolean typing, which the released RT-v1
    checkpoints need (see rt.preprocess.legacy). Built and swapped separately,
    because a half-written one on the Hub breaks those checkpoints."""

    @property
    def raw_dir(self) -> str:
        return f"{SHARE}/{self.name}"

    @property
    def out_dir(self) -> str:
        return f"{SHARE}/{self.name}-preprocessed"

    @property
    def legacy_dir(self) -> str:
        """Built beside the collection, not inside it: the main upload must be
        able to go out without carrying a half-finished legacy tree with it."""
        return f"{SHARE}/{self.name}-preprocessed-legacy"

    @property
    def sizes(self) -> Path:
        """Expected output bytes per database, from the previous build."""
        return HERE / f"sizes-{self.name}.json"

    @property
    def curated_path(self) -> Path | None:
        return HERE / self.curated if self.curated else None


COLLECTIONS = {
    c.name: c
    for c in (
        Collection(
            name="the-join",
            source_repo="stanford-star/the-join",
            target_repo="stanford-star/the-join-preprocessed",
            # rt-j is a whitelist of 475 databases, 126 excluded wholesale and
            # none partially: nothing in the build says which.
            curated="rt-j-dbs.json",
            keep=("db-task-lists",),
            legacy=False,
        ),
        Collection(
            name="relbench",
            source_repo="stanford-star/relbench",
            target_repo="stanford-star/relbench-preprocessed",
            # all three of its lists are derivable from the metas
            curated=None,
            keep=("db-task-lists", "legacy"),
            legacy=True,
        ),
    )
}


def pick(argv: list[str]) -> Collection:
    """The collection named on the command line."""
    names = [a for a in argv[1:] if not a.startswith("-")]
    if len(names) != 1 or names[0] not in COLLECTIONS:
        raise SystemExit(
            f"name one collection: {' | '.join(sorted(COLLECTIONS))}\n"
            f"  e.g. {Path(argv[0]).name} the-join"
        )
    return COLLECTIONS[names[0]]
