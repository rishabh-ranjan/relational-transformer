from pathlib import Path

import numpy as np

from rt.augment.plan import TRANSFORM_BY_NAME
from rt.augment.stats import load


def main(
    *,
    stats_path: str,
    out_path: str,
    embedder: str,
    batch_size: int,
    device: str,
) -> None:
    from sentence_transformers import SentenceTransformer

    stats = load(stats_path)
    keys: list[str] = []
    names: list[str] = []
    for key in sorted(stats.derived):
        entry = stats.derived[key]
        transform = TRANSFORM_BY_NAME[entry.transform]
        keys.append(key)
        # The model reads a column through the embedding of its NAME, so a
        # derived column is given a name that says what it is. rt-j has never
        # seen these strings, but "empirical cdf" and "signed log1p" land
        # somewhere meaningful in the same sentence-embedding space the real
        # column names live in.
        names.append(transform.display_name(*entry.sources))

    model = SentenceTransformer(f"sentence-transformers/{embedder}", device=device)
    vectors = model.encode(
        names,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    assert vectors.shape[0] == len(keys), (
        f"{vectors.shape[0]} embeddings for {len(keys)} derived columns"
    )

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **dict(zip(keys, vectors)))
    print(
        f"{stats.db}: embedded {len(keys)} derived column names "
        f"({vectors.shape[1]}-d) -> {out}",
        flush=True,
    )
    for key, name in list(zip(keys, names))[:3]:
        print(f"  {key}  ->  {name!r}", flush=True)
