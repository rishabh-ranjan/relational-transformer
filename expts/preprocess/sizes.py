"""How big each database's preprocessed output comes out.

Written by the previous published build, read here to size each job and to
weight progress. Two numbers per database, because the two stages scale on
different things (see [README.md](README.md)):

* ``out`` -- total output bytes, which rustler tracks.
* ``text`` -- bytes of ``text.json``, which the embedding stage tracks.

Text bytes are themselves a proxy -- the cost is per string, not per byte -- so
anything with the build in front of it should prefer ``num_text_strings`` from
``meta.json``. This is for what has not been built yet.

Regenerate for whichever collection ``submit.py`` currently names.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.submit import KEEP, SIZES, TARGET_REPO  # noqa: E402
from huggingface_hub import HfApi


def write() -> None:
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
