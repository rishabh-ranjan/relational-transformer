from pathlib import Path

import numpy as np

from rt.augment.plan import (
    TRANSFORM_BY_NAME,
    ablation_names,
    build_plan,
    load_column_index,
    load_name_embeddings,
)
from rt.augment.stats import load


def ablation_keys_and_names(
    *, stats, pre_dir: str, real_names_path: str, include: list[str],
    max_modal_share: float, d_text: int,
) -> tuple[list[str], list[str]]:
    real = build_plan(
        stats=stats,
        column_index=load_column_index(pre_dir, stats.db),
        name_embeddings=load_name_embeddings(real_names_path),
        include=include,
        max_modal_share=max_modal_share,
        d_text=d_text,
    )
    names = ablation_names(real)
    return names, names


def main(
    *,
    stats_path: str,
    out_path: str,
    embedder: str,
    batch_size: int,
    device: str,
    ablation_of: dict | None,
) -> None:
    from sentence_transformers import SentenceTransformer

    stats = load(stats_path)
    keys: list[str] = []
    names: list[str] = []
    if ablation_of is not None:
        # The token-count ablation names its copies per source column, so they
        # are keyed by the name itself rather than by a derived key -- several
        # copies of one column share a transform and would otherwise collide.
        keys, names = ablation_keys_and_names(
            stats=stats,
            pre_dir=ablation_of["pre_dir"],
            real_names_path=ablation_of["real_names_path"],
            include=ablation_of["include"],
            max_modal_share=ablation_of["max_modal_share"],
            d_text=ablation_of["d_text"],
        )
        return _embed_and_save(
            SentenceTransformer, embedder, device, batch_size, keys, names,
            out_path, stats.db,
        )
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

    return _embed_and_save(
        SentenceTransformer, embedder, device, batch_size, keys, names, out_path,
        stats.db,
    )


def _embed_and_save(
    SentenceTransformer, embedder, device, batch_size, keys, names, out_path, db
) -> None:
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
    assert len(set(keys)) == len(keys), "duplicate embedding keys"

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **dict(zip(keys, vectors)))
    print(
        f"{db}: embedded {len(keys)} column names ({vectors.shape[1]}-d) -> {out}",
        flush=True,
    )
    for key, name in list(zip(keys, names))[:3]:
        print(f"  {key}  ->  {name!r}", flush=True)
