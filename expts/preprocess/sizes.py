"""How big each database's preprocessed output comes out.

Written by the previous published build, read here for two things a rebuild
cannot know in advance: how much cpu and memory to ask slurm for per database,
and how far along the sweep really is. Both matter because the collection is
extremely lopsided -- the median database preprocesses to ~43 MiB and the
largest to 76 GiB, and the top 20 are half the bytes -- so counting datasets
would report a sweep as nearly finished while most of the work remained.

Regenerate with ``python expts/preprocess/sizes.py <collection>`` (needs the
Hub; it reads file sizes only, downloading nothing).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.collection import Collection, pick  # noqa: E402


def load(collection: Collection) -> dict[str, int]:
    """dataset name -> expected output bytes."""
    return json.loads(collection.sizes.read_text())


def write(collection: Collection) -> None:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(
        collection.target_repo, repo_type="dataset", files_metadata=True
    )
    per_db: dict[str, int] = defaultdict(int)
    for f in info.siblings:
        db, _, rest = f.rfilename.partition("/")
        # only the collection's own databases: not db-task-lists, not legacy/,
        # not anything else the published repo carries alongside them
        if rest and db not in collection.keep and "/" not in rest:
            per_db[db] += f.size or 0
    collection.sizes.write_text(
        json.dumps(dict(sorted(per_db.items())), indent=1) + "\n"
    )
    total = sum(per_db.values())
    print(f"{len(per_db)} databases, {total / 2**30:.1f} GiB -> {collection.sizes}")


if __name__ == "__main__":
    write(pick(sys.argv))
