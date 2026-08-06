"""How big each database's preprocessed output comes out.

Written by the previous published build, read here for two things a rebuild
cannot know in advance: how much cpu and memory to ask slurm for per database,
and how far along the sweep really is. Both matter because these collections are
extremely lopsided -- the Join's median database preprocesses to ~43 MiB and its
largest to 76 GiB -- so counting datasets would report a sweep as nearly
finished while most of the work remained.

Two numbers per database, because the two stages scale on different things:

* ``out`` -- total output bytes. rustler tracks this closely (~21 s/GiB).
* ``text`` -- bytes of ``text.json``. The embedding stage tracks *text*, and
  output size predicts it badly: rel-amazon is 48% of RelBench's output and 78%
  of its strings, and join-bird-codebase-comments took 5983 s to embed 1.1 GiB
  of text while join-overture-maps took 9398 s for 8.0 GiB.

Text bytes are themselves a proxy -- the cost is per string, not per byte -- so
anything with the build in front of it should prefer ``num_text_strings`` from
``meta.json``. This is for what has not been built yet.

Regenerate with ``python expts/preprocess/sizes.py`` for whichever collection
``submit.py`` currently names (needs the Hub; it reads file sizes only,
downloading nothing).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.submit import KEEP, SIZES, TARGET_REPO  # noqa: E402


def out_bytes(sizes: dict[str, dict[str, int]], name: str, default: int) -> int:
    return sizes.get(name, {}).get("out", default)


def text_bytes(sizes: dict[str, dict[str, int]], name: str, default: int) -> int:
    return sizes.get(name, {}).get("text", default)


def write() -> None:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(TARGET_REPO, repo_type="dataset", files_metadata=True)
    per_db: dict[str, dict[str, int]] = defaultdict(lambda: {"out": 0, "text": 0})
    for f in info.siblings:
        db, _, rest = f.rfilename.partition("/")
        # only the collection's own databases: not db-task-lists, not legacy/,
        # not anything else the published repo carries alongside them
        if rest and db not in KEEP and "/" not in rest:
            per_db[db]["out"] += f.size or 0
            if rest == "text.json":
                per_db[db]["text"] = f.size or 0
    SIZES.write_text(
        json.dumps({k: per_db[k] for k in sorted(per_db)}, indent=1) + "\n"
    )
    out = sum(v["out"] for v in per_db.values())
    text = sum(v["text"] for v in per_db.values())
    print(
        f"{len(per_db)} databases, {out / 2**30:.1f} GiB out, "
        f"{text / 2**30:.1f} GiB text -> {SIZES}"
    )


if __name__ == "__main__":
    write()
