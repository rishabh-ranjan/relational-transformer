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
