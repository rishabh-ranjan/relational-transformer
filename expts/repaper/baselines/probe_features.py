import json
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import norm

from expts.repaper.config import SHARE

TASKS = [
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "user-ltv"),
    ("rel-stack", "post-votes"),
]


def clip4(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -4.0, 4.0)


def rankgauss(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    n = len(x)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        vals, inv = np.unique(x[:, j], return_inverse=True)
        mean_rank = np.bincount(inv, weights=ranks) / np.bincount(inv)
        u = (mean_rank[inv] + 0.5) / n
        out[:, j] = norm.ppf(u)
    return out


def dropzero(x: np.ndarray) -> np.ndarray:
    keep = (x == 0).mean(axis=0) <= 0.5
    print(f"  dropzero keeps {keep.sum()}/{len(keep)} columns", flush=True)
    return x[:, keep]


TRANSFORMS = {"clip4": clip4, "rankgauss": rankgauss, "dropzero": dropzero}

if __name__ == "__main__":
    root = Path(SHARE).expanduser()
    for db, table in TASKS:
        src = root / "features" / db / "rdblearn_features"
        meta = json.loads((src / f"{table}_meta.json").read_text())
        x = np.fromfile(src / f"{table}_vectors.bin", dtype=np.float32).reshape(
            meta["total_nodes"], meta["n_features"]
        )
        for name, fn in TRANSFORMS.items():
            print(f"{db}/{table} {name}", flush=True)
            dst = root / "features_probe" / name / db / "rdblearn_features"
            dst.mkdir(parents=True, exist_ok=True)
            y = fn(x.astype(np.float64)).astype(np.float32)
            assert np.isfinite(y).all()
            y.tofile(dst / f"{table}_vectors.bin")
            m = dict(meta, n_features=y.shape[1])
            (dst / f"{table}_meta.json").write_text(json.dumps(m))
            for extra in src.glob(f"{table}_*"):
                if extra.name not in (f"{table}_vectors.bin", f"{table}_meta.json"):
                    shutil.copy(extra, dst / extra.name)
