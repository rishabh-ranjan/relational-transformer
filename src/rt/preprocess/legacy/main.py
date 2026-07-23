"""Legacy (RT-v1-era) RelBench preprocessing.

The RT-v1 ``pre.rs`` hard-coded per-database rules that cast binary columns
(classification task targets and a few db columns) to polars ``Boolean``
before featurization, which made them a real Boolean semantic type
(``sem_types == 3``, values in ``boolean_values``, BCE-trained decoder head).
The current manifest-driven ``pre.rs`` types columns purely by parquet dtype,
and the RelBench source parquets store those columns as ints — so they become
z-scored numbers, a mismatch for the released RT-v1 checkpoints.

This module reproduces the RT-v1 rules as a *dataset transform*: it copies a
RelBench dataset directory, casts the legacy-boolean columns to parquet
``Boolean`` dtype, then runs the unchanged rustler ``pre`` + embedding
pipeline on the transformed copy. The output is bit-compatible with the
regular preprocessed layout and is published under a ``legacy/`` folder of
the ``*-preprocessed`` Hub repo (use ``pre_dir=<repo>/legacy``).

Faithfulness notes (vs ``rt-v1:rustler/src/pre.rs``):
- All ``cast_col_to_bool`` task-target rules are ported, except
  rel-event/user-attendance: RT-v1 also cast that *regression* target to
  bool (its paper skipped rel-event); the leaderboard needs numeric
  attendance predictions, so it stays numeric here.
- All ``make_column_boolean`` (equality-to-first-non-null) db-column rules
  are ported.
- rel-amazon ``product.category`` keeps RT-v1's first-list-element text
  (the current pipeline stringifies the whole list).
- RT-v1's rel-event row-dropping (``drop_nulls`` on event_attendees /
  user_friends) is not ported: it changed node indexing for robustness of
  the old loader, not typing semantics, and the current pipeline handles
  nulls.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# (db, table) -> columns cast to Boolean via `col != 0` (RT-v1 cast_col_to_bool;
# polars int -> bool casts nonzero to true). Task-table rules apply to every
# split parquet of that task.
CAST_TO_BOOL: dict[tuple[str, str], list[str]] = {
    ("rel-amazon", "user-churn"): ["churn"],
    ("rel-amazon", "item-churn"): ["churn"],
    ("rel-stack", "user-engagement"): ["contribution"],
    ("rel-stack", "user-badge"): ["WillGetBadge"],
    ("rel-trial", "study-outcome"): ["outcome"],
    ("rel-f1", "driver-dnf"): ["did_not_finish"],
    ("rel-f1", "driver-top3"): ["qualifying"],
    ("rel-hm", "user-churn"): ["churn"],
    ("rel-event", "user-repeat"): ["target"],
    ("rel-event", "user-ignore"): ["target"],
    ("rel-avito", "user-visits"): ["num_click"],
    ("rel-avito", "user-clicks"): ["num_click"],
}

# (db, table) -> columns binarized as `col == first non-null value` (RT-v1
# make_column_boolean).
BINARIZE_FIRST: dict[tuple[str, str], list[str]] = {
    ("rel-stack", "postLinks"): ["LinkTypeId"],
    ("rel-trial", "studies"): ["has_dmc"],
    ("rel-trial", "eligibilities"): ["adult", "child", "older_adult"],
}


def _transform_df(df, db_name: str, table_name: str):
    import polars as pl

    for col in CAST_TO_BOOL.get((db_name, table_name), []):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Boolean).alias(col))

    for col in BINARIZE_FIRST.get((db_name, table_name), []):
        if col in df.columns:
            s = df[col].cast(pl.String)
            first = s.drop_nulls().first()
            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(None)
                .otherwise(s == first)
                .alias(col)
            )

    if db_name == "rel-amazon" and table_name == "product" and "category" in df.columns:
        # RT-v1 kept only the first list element as the category text.
        df = df.with_columns(
            pl.col("category").list.first().cast(pl.String).alias("category")
        )

    return df


def transform_dataset(dataset_dir: Path, out_dataset_dir: Path, db_name: str) -> Path:
    """Copy ``dataset_dir`` to ``out_dataset_dir`` with the RT-v1 boolean rules
    applied to the relevant parquets. Everything else is copied verbatim."""
    import polars as pl

    out_dataset_dir = Path(out_dataset_dir)
    if out_dataset_dir.exists():
        shutil.rmtree(out_dataset_dir)
    out_dataset_dir.mkdir(parents=True)

    for src in sorted(Path(dataset_dir).rglob("*")):
        rel = src.relative_to(dataset_dir)
        dst = out_dataset_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix == ".parquet":
            # db/<table>.parquet or tasks/<task>/<split>.parquet
            parts = rel.parts
            table_name = Path(parts[-1]).stem if parts[0] == "db" else parts[1]
            df = pl.read_parquet(src)
            out = _transform_df(df, db_name, table_name)
            out.write_parquet(dst)
            changed = [
                c for c in out.columns
                if c in df.columns and out.schema[c] != df.schema[c]
            ]
            if changed:
                print(f"  {rel}: {', '.join(f'{c}->{out.schema[c]}' for c in changed)}",
                      flush=True)
        else:
            shutil.copy2(src, dst)
    return out_dataset_dir


def preprocess_one_legacy(
    spec: str,
    out_dir: Path,
    *,
    embedding_model: str,
    batch_size: int,
    upload_repo: str | None,
    private: bool,
    revision: str | None,
) -> Path:
    """Legacy variant of :func:`rt.preprocess.main.preprocess_one`: resolve the
    dataset, apply the RT-v1 boolean transform, run the regular rustler `pre` +
    embedding pipeline, and (optionally) upload under ``legacy/<name>`` of
    ``upload_repo``."""
    from rt.preprocess.main import (
        dataset_name,
        embed_dataset,
        resolve_dataset_dir,
        run_rustler_pre,
        update_meta_with_embeddings,
    )

    out_dir = Path(out_dir).expanduser()
    dataset_dir = resolve_dataset_dir(spec, revision=revision)
    name = dataset_name(dataset_dir)

    tf_dir = out_dir / "_transformed" / name
    print(f"=== legacy-transforming {name} ({spec}) -> {tf_dir} ===", flush=True)
    transform_dataset(dataset_dir, tf_dir, name)

    pre_dataset_dir = out_dir / name
    print(f"=== preprocessing {name} -> {pre_dataset_dir} ===", flush=True)
    run_rustler_pre(tf_dir, out_dir, source=spec, skip_tasks=False)
    d_text = embed_dataset(pre_dataset_dir, embedding_model, batch_size)
    update_meta_with_embeddings(pre_dataset_dir, embedding_model, d_text)

    if upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(upload_repo, repo_type="dataset", private=private, exist_ok=True)
        print(f"uploading {pre_dataset_dir} -> {upload_repo}/legacy/{name}", flush=True)
        api.upload_folder(
            folder_path=str(pre_dataset_dir),
            path_in_repo=f"legacy/{name}",
            repo_id=upload_repo,
            repo_type="dataset",
            commit_message=f"add legacy (RT-v1 boolean typing) preprocessed {name}",
        )
        print(f"uploaded {upload_repo}/legacy/{name}", flush=True)
    return pre_dataset_dir
