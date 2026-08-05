"""How big each database's preprocessed output comes out.

Written by the previous published build, read here for two things a rebuild
cannot know in advance: how much cpu and memory to ask slurm for per database,
and how far along the sweep really is. Both matter because the collection is
extremely lopsided -- the median database preprocesses to ~43 MiB and the
largest to 76 GiB, and the top 20 are half the bytes -- so counting datasets
would report a sweep as nearly finished while most of the work remained.

Regenerate with ``python expts/preprocess/sizes.py`` (needs the Hub; it reads
file sizes only, downloading nothing).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

SIZES = Path(__file__).with_name("sizes.json")
REPO = "stanford-star/the-join-preprocessed"


def load() -> dict[str, int]:
    """dataset name -> expected output bytes."""
    return json.loads(SIZES.read_text())


def write(repo: str = REPO) -> None:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo, repo_type="dataset", files_metadata=True)
    per_db: dict[str, int] = defaultdict(int)
    for f in info.siblings:
        if "/" in f.rfilename and f.rfilename.startswith("join-"):
            per_db[f.rfilename.split("/", 1)[0]] += f.size or 0
    SIZES.write_text(json.dumps(dict(sorted(per_db.items())), indent=1) + "\n")
    total = sum(per_db.values())
    print(f"{len(per_db)} databases, {total / 2**40:.3f} TiB -> {SIZES}")


if __name__ == "__main__":
    write()
