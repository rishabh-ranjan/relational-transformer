"""Derive the schema-semantics-ablated copy of the preprocessed eval data.

The ablation removes schema semantics by deranging the column-name embeddings:
for each database, a Sattolo permutation (a uniform random cycle, so no column
keeps its own name) over the column indices in ``column_index.json``, applied
to the rows of the text-embedding table those indices address. Cells that
shared a column name still share one, cells that differed still differ, and
nothing else about the data changes -- every other file is a symlink to the
original, so the derived tree costs only the embedding files it rewrites.

The permutation is deterministic per (db, seed). Table-name embeddings are
not touched (their rows are not in ``column_index.json``).
"""

import hashlib
import json
from pathlib import Path


def derange(indices: list[int], seed_material: str) -> dict[int, int]:
    """Sattolo's algorithm over ``indices``: a uniform random cyclic
    permutation, so every index maps to a different one."""
    import numpy as np

    assert len(indices) >= 2, f"need >= 2 columns to derange ({seed_material})"
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
    )
    shuffled = list(indices)
    for i in range(len(shuffled) - 1, 0, -1):
        j = int(rng.integers(0, i))  # strictly below i: no fixed points
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return dict(zip(indices, shuffled))


def main(*, pre_dir: str, out_dir: str, embedder: str, d_text: int, seed: int) -> None:
    import numpy as np

    pre = Path(pre_dir).expanduser()
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # The task lists ship with the data; the derived tree points at the same one.
    lists = out / "db-task-lists"
    if not lists.exists():
        lists.symlink_to(pre / "db-task-lists")

    pairs = json.loads((pre / "db-task-lists" / "forecast.json").read_text())
    for db in sorted({db for db, _ in pairs}):
        src = pre / db
        dst = out / db
        dst.mkdir(exist_ok=True)
        emb_name = f"text_emb_{embedder}.bin"
        for f in src.iterdir():
            link = dst / f.name
            if f.name == emb_name or link.exists():
                continue
            link.symlink_to(f)

        col_index = json.loads((src / "column_index.json").read_text())
        perm = derange(sorted(col_index.values()), f"{db}:{seed}")

        emb = np.fromfile(src / emb_name, dtype=np.uint16).reshape(-1, d_text)
        assert max(col_index.values()) < emb.shape[0]
        deranged = emb.copy()
        for orig, repl in perm.items():
            deranged[orig] = emb[repl]
        deranged.tofile(dst / emb_name)
        print(f"{db}: deranged {len(perm)} column-name embeddings", flush=True)
