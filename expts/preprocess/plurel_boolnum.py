import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import orjson
import pandas as pd

from expts.preprocess.plurel_tasks import synth_one
from expts.preprocess.submit import EMBEDDER, OUT_DIR, RAW_DIR
from rt.preprocess import run_rustler_pre
from rt.preprocess.embed import TextEmbedder

CAST_RAW = "~/scratch/hf/stanford-star/plurel-boolnum"
CAST_OUT = "~/scratch/hf/stanford-star/plurel-boolnum-preprocessed"


def one(name: str) -> str:
    raw = Path(RAW_DIR).expanduser() / name
    craw = Path(CAST_RAW).expanduser() / name
    cout = Path(CAST_OUT).expanduser()
    if (cout / name / "text.json").exists():
        return f"{name}: rustled"
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
    meta = json.loads((cout / name / "meta.json").read_text())
    assert len(meta["tasks"]) == n
    return f"{name}: {n} tasks"


def main() -> None:
    names = sorted(
        p.parent.name for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
    )
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, line in enumerate(ex.map(one, names)):
            if i % 200 == 0:
                print(f"[rustle {i}/{len(names)}] {line}", flush=True)

    cout = Path(CAST_OUT).expanduser()
    embedder = TextEmbedder(1024, EMBEDDER, "cpu")
    for i, name in enumerate(names):
        d = cout / name
        meta = json.loads((d / "meta.json").read_text())
        if EMBEDDER in meta.get("text_embeddings", {}):
            continue
        texts = orjson.loads((d / "text.json").read_bytes())
        emb = embedder(texts, device="cpu")
        emb.tofile(d / f"text_emb_{EMBEDDER}.bin")
        meta["text_embeddings"] = {
            EMBEDDER: {"file": f"text_emb_{EMBEDDER}.bin", "d_text": int(emb.shape[1])}
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        if i % 200 == 0:
            print(f"[embed {i}/{len(names)}] {name}: {emb.shape}", flush=True)

    lists = Path(OUT_DIR).expanduser() / "db-task-lists" / "rt-plurel-train.json"
    d = cout / "db-task-lists"
    d.mkdir(exist_ok=True)
    shutil.copy(lists, d / "rt-plurel-train.json")
    print("done", flush=True)


if __name__ == "__main__":
    main()
