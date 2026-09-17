import functools
import json
import pickle
from pathlib import Path

import numpy as np

CAVEAT = """\
PROVENANCE NOTE -- the gnn featurizer is not bit-exactly regenerable.

pyg-lib samples neighbours in parallel with one RandintEngine per thread. Every
engine is seeded deterministically (vslNewStream(.., MT19937, 1)), so nothing
draws from entropy, but which thread draws for which seed node depends on thread
scheduling, and torch_geometric.seed.seed_everything reaches none of it. The
featurize jobs ran with sampler_threads > 1 -- a deliberate trade, because the
21 tasks are ~33 M rows of 2-hop sampling -- so re-running the featurizer
produces different features and slightly different metrics.

The gnn blobs are therefore the artifact of record, not the code that made
them: keep them, and do not regenerate one that a result was computed from. Each
carries its own sha256 and `reproducible: false` in its `<table>_meta.json`,
written when it was created; that metadata is reported below but not verified
against the file, because re-reading ~150 GiB to check is not worth it.

rdblearn, rt-j and rt-plurel are unaffected -- those featurizers are
deterministic and their blobs can be rebuilt from code.\
"""


# Metadata only, never the bytes. Verifying a blob by re-hashing it means
# reading ~150 GiB across the four feature roots, and it would only tell us
# something for the gnn ones: the other three featurizers are deterministic, so
# a changed blob there is simply rebuilt. The hash reported here is the one
# featurize_gnn wrote when it created the file.
@functools.cache
def blob_provenance(features_root: str, subdir: str, db: str, table: str) -> dict:
    feat_dir = Path(features_root).expanduser() / db / subdir
    meta_path = feat_dir / f"{table}_meta.json"
    vectors = feat_dir / f"{table}_vectors.bin"
    if not meta_path.is_file() or not vectors.is_file():
        return {"present": False}
    meta = json.loads(meta_path.read_text())
    return {
        "present": True,
        "n_features": meta["n_features"],
        "total_nodes": meta["total_nodes"],
        "bytes": vectors.stat().st_size,
        # Written at creation time; absent for a blob made before provenance was
        # recorded (rel-f1 and rel-event, from the width grid).
        "sha256": meta.get("sha256"),
        "reproducible": meta.get("reproducible"),
        "config": meta.get("config"),
    }


def collate(round_dir: str, out_dir: str) -> None:
    root = Path(round_dir).expanduser()
    rows = []
    for pkl in sorted(root.rglob("*.pkl")):
        with open(pkl, "rb") as f:
            result = pickle.load(f)
        cfg = result["config"]
        ctx = sorted(result["per_ctx"])
        for c in ctx:
            entry = result["per_ctx"][c]
            preds = result["preds"][c]
            labels = result["labels"]
            rows.append(
                {
                    "arm": pkl.parent.name,
                    "method": result["method"],
                    "db": result["db"],
                    "table": result["table"],
                    "task": result["task"],
                    "task_type": result["task_type"],
                    "ctx": int(c),
                    "metric_name": entry["metric_name"],
                    "metric_value": entry["metric_value"],
                    "n": entry["n"],
                    "features_root": cfg["features_root"],
                    "retriever": cfg["retriever"],
                    "context_sampler": cfg["context_sampler"],
                    "context_seed": cfg["context_seed"],
                    "n_distinct_preds": len(np.unique(preds)),
                    "pred_mean": float(np.mean(preds)),
                    "pred_std": float(np.std(preds)),
                    "label_mean": float(np.mean(labels)),
                    "blob": blob_provenance(
                        cfg["features_root"],
                        {
                            "rdblearn": "rdblearn_features",
                            "rt": "rt_features",
                            "gnn": "gnn_features",
                            "sql": "sql_features",
                        }[result["method"].rsplit("_", 1)[0]],
                        result["db"],
                        result["table"],
                    ),
                }
            )

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "collated.pkl", "wb") as f:
        pickle.dump(
            {"caveat": CAVEAT, "rows": rows}, f, protocol=pickle.HIGHEST_PROTOCOL
        )
    (out / "collated.json").write_text(
        json.dumps({"caveat": CAVEAT, "rows": rows}, indent=2, sort_keys=True)
    )

    print(CAVEAT)
    print()
    print(f"{len(rows)} results from {root}")
    print()
    by_task: dict[tuple[str, str], dict[str, float]] = {}
    arms: set[str] = set()
    for r in rows:
        by_task.setdefault((r["task"], r["metric_name"]), {})[r["arm"]] = r[
            "metric_value"
        ]
        arms.add(r["arm"])
    order = sorted(arms)
    width = max((len(t) for t, _ in by_task), default=10)
    print(f"{'task':{width}} {'metric':8} " + " ".join(f"{a:>18}" for a in order))
    for (task, metric), vals in sorted(by_task.items()):
        cells = " ".join(
            f"{vals[a]:>18.4f}" if a in vals else f"{'-':>18}" for a in order
        )
        print(f"{task:{width}} {metric:8} {cells}")

    absent = [f"{r['arm']}/{r['task']}" for r in rows if not r["blob"]["present"]]
    if absent:
        print(f"\nBLOB MISSING -- result can no longer be traced: {absent}")
    print(f"\nwrote {out / 'collated.pkl'} and {out / 'collated.json'}")
