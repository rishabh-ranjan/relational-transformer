import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq
import yaml

from expts.preprocess.preprocess import is_done, is_rustler_done  # noqa: E402
from expts.preprocess.submit import EMBEDDER, OUT_DIR, RAW_DIR, SOURCE_REPO  # noqa: E402
from rt.preprocess import run_rustler_pre  # noqa: E402

MAX_FKEYS = 5

TASK_TYPE = {
    "bool": "binary_classification",
    "int64": "regression",
    "double": "regression",
}


def synth_one(raw_db: Path) -> int:
    manifest = yaml.safe_load((raw_db / "manifest.yaml").read_text())
    tasks_dir = raw_db / "tasks"
    if tasks_dir.exists():
        shutil.rmtree(tasks_dir)
    if any(len(t.get("fkeys") or {}) > MAX_FKEYS for t in manifest["tables"].values()):
        return 0
    n = 0
    for table_name in sorted(manifest["tables"]):
        schema = pq.read_schema(raw_db / "db" / f"{table_name}.parquet")
        for field in schema:
            if "feature" not in field.name:
                continue
            task_type = TASK_TYPE.get(str(field.type))
            if task_type is None:
                continue
            d = tasks_dir / f"{table_name}-{field.name}"
            d.mkdir(parents=True)
            (d / "manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "kind": "autocomplete",
                        "entity_table": table_name,
                        "target_col": field.name,
                        "task_type": task_type,
                        "time_col": None,
                    }
                )
            )
            n += 1
    return n


def rebuild_one(name: str) -> str:
    raw_root = Path(RAW_DIR).expanduser()
    out_root = Path(OUT_DIR).expanduser()
    pre = out_root / name
    assert is_done(pre, EMBEDDER), f"{name}: embedding not done"
    meta = json.loads((pre / "meta.json").read_text())
    emb = meta["text_embeddings"]
    old_text = (pre / "text.json").read_bytes()
    n = synth_one(raw_root / name)
    for f in ("meta.json", "table_info.json", "column_index.json", "text.json"):
        (pre / f).unlink()
    run_rustler_pre(
        raw_root / name, out_root, source=f"{SOURCE_REPO}/{name}", skip_tasks=False
    )
    assert (pre / "text.json").read_bytes() == old_text, f"{name}: text.json changed"
    meta = json.loads((pre / "meta.json").read_text())
    assert len(meta["tasks"]) == n, f"{name}: {len(meta['tasks'])} tasks, wrote {n}"
    meta["text_embeddings"] = emb
    (pre / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    assert is_done(pre, EMBEDDER)
    return f"{name}: {n} tasks"


def main() -> None:
    out_root = Path(OUT_DIR).expanduser()
    names = sorted(
        p.parent.name for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
    )
    todo = [n for n in names if is_rustler_done(out_root / n)]
    assert len(todo) == len(names), f"{len(names) - len(todo)} dbs not rustled yet"
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, line in enumerate(ex.map(rebuild_one, todo)):
            if i % 200 == 0:
                print(f"[{i}/{len(todo)}] {line}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
