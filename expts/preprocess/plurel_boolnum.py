import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from expts.preprocess.plurel_tasks import synth_one
from expts.preprocess.preprocess import is_done
from expts.preprocess.submit import EMBEDDER, OUT_DIR, RAW_DIR
from rt.preprocess import run_rustler_pre

CAST_RAW = "~/scratch/hf/stanford-star/plurel-boolnum"
CAST_OUT = "~/scratch/hf/stanford-star/plurel-boolnum-preprocessed"


def one(name: str) -> str:
    raw = Path(RAW_DIR).expanduser() / name
    pre = Path(OUT_DIR).expanduser() / name
    craw = Path(CAST_RAW).expanduser() / name
    cout = Path(CAST_OUT).expanduser()
    cpre = cout / name
    if is_done(cpre, EMBEDDER):
        return f"{name}: done"
    if craw.exists():
        shutil.rmtree(craw)
    craw.mkdir(parents=True)
    shutil.copy(raw / "manifest.yaml", craw / "manifest.yaml")
    (craw / "db").mkdir()
    for f in sorted((raw / "db").glob("*.parquet")):
        df = pd.read_parquet(f)
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype("float64")
        df.to_parquet(craw / "db" / f.name, index=False)
    n = synth_one(craw)
    run_rustler_pre(craw, cout, source=f"boolnum/{name}", skip_tasks=False)
    meta = json.loads((cpre / "meta.json").read_text())
    assert len(meta["tasks"]) == n
    assert (cpre / "text.json").read_bytes() == (pre / "text.json").read_bytes(), name
    src = json.loads((pre / "meta.json").read_text())["text_embeddings"]
    emb_file = src[EMBEDDER]["file"]
    shutil.copy(pre / emb_file, cpre / emb_file)
    meta["text_embeddings"] = src
    (cpre / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    assert is_done(cpre, EMBEDDER)
    return f"{name}: {n} tasks"


def main() -> None:
    names = sorted(
        p.parent.name for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
    )
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, line in enumerate(ex.map(one, names)):
            if i % 200 == 0:
                print(f"[{i}/{len(names)}] {line}", flush=True)
    lists = Path(OUT_DIR).expanduser() / "db-task-lists" / "rt-plurel-train.json"
    d = Path(CAST_OUT).expanduser() / "db-task-lists"
    d.mkdir(exist_ok=True)
    shutil.copy(lists, d / "rt-plurel-train.json")
    print("done", flush=True)


if __name__ == "__main__":
    main()
