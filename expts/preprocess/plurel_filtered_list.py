import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq
import yaml

from expts.preprocess.submit import OUT_DIR, RAW_DIR  # noqa: E402

SCORES_DIR = Path("/dfs/user/vigneshk/scratch/raw/plurel-05-10-2026-v8-2000")
NUM_TRAIN_DBS = 1900
MAX_FKEYS = 5
MAX_MAJORITY_FRAC = 0.99


def keep(cs: dict | None, task_type: str) -> bool:
    if cs is None:
        return True
    if cs.get("is_source_node", False):
        return False
    if cs.get("n_unique", 0) < 2:
        return False
    if task_type == "clf" and (
        cs.get("n_classes", 0) < 2 or cs.get("majority_frac", 0.0) > MAX_MAJORITY_FRAC
    ):
        return False
    if task_type == "reg" and cs.get("std", 0.0) < 1e-4:
        return False
    return True


def main() -> None:
    raw = Path(RAW_DIR).expanduser()
    pairs, kept_dbs = [], 0
    for seed in range(3000, 5000):
        name = f"plurel-{seed}"
        manifest = yaml.safe_load((raw / name / "manifest.yaml").read_text())
        if any(
            len(t.get("fkeys") or {}) > MAX_FKEYS for t in manifest["tables"].values()
        ):
            continue
        scores = json.loads((SCORES_DIR / name / "scores.json").read_text())
        for table in sorted(manifest["tables"]):
            for f in pq.read_schema(raw / name / "db" / f"{table}.parquet"):
                if "feature" not in f.name or str(f.type) not in (
                    "bool",
                    "int64",
                    "double",
                ):
                    continue
                task_type = "clf" if str(f.type) == "bool" else "reg"
                if keep(scores.get(table, {}).get(f.name), task_type):
                    pairs.append([name, f"{table}-{f.name}"])
        kept_dbs += 1
        if kept_dbs == NUM_TRAIN_DBS:
            break
    assert kept_dbs == NUM_TRAIN_DBS
    out = Path(OUT_DIR).expanduser() / "db-task-lists" / "rt-plurel-train.json"
    out.write_text(json.dumps(sorted(pairs), indent=1) + "\n")
    print(f"{len(pairs)} tasks over {len({p[0] for p in pairs})} dbs -> {out}")


if __name__ == "__main__":
    main()
