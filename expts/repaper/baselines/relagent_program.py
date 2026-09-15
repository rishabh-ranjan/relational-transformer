import json
from pathlib import Path


def load_program(path: str) -> list[dict]:
    program = json.loads(Path(path).expanduser().read_text())

    combiner = program.get("combiner_code") or ""
    assert not combiner.strip(), (
        f"{path} carries combiner_code; it is agent-authored python and is not "
        f"executed here. Fold the combination into the feature SQL instead."
    )

    queries = program["feature_queries"]
    assert queries, f"{path} has no feature_queries"
    names = [q["name"] for q in queries]
    assert len(names) == len(set(names)), f"{path} has duplicate query names: {names}"
    for q in queries:
        assert q.get("sql", "").strip(), f"{path}: query {q['name']!r} has empty sql"
    return queries


def build_feature_frame(
    con,
    queries: list[dict],
    task_rows,
    entity_col: str,
    time_col: str,
    sql_timeout_seconds: float,
):
    import pandas as pd

    # RelAgent's own naming: its queries select FROM eval_table.
    con.register("eval_table", task_rows)

    join_cols = [entity_col, time_col]
    # Deliberately not RelAgent's drop_duplicates() on the base: the blob has to
    # stay one row per task row in node-idx order. (entity, time) is unique per
    # task row -- check_alignment.py asserts it -- so deduping would be a no-op
    # and reordering would silently misalign the features.
    base = task_rows[join_cols].copy().reset_index(drop=True)
    n_rows = len(base)
    diagnostics = {}

    for q in queries:
        name, sql = q["name"], q["sql"]
        con.execute(f"SET statement_timeout = {int(sql_timeout_seconds * 1000)}")
        fdf = con.execute(sql).df()

        assert entity_col in fdf.columns, (
            f"query {name!r} does not select the entity column {entity_col!r}; "
            f"it returned {list(fdf.columns)}"
        )
        feature_join_cols = [c for c in join_cols if c in fdf.columns]
        fdf = fdf.drop_duplicates(subset=feature_join_cols, keep="first")
        feat_cols = [c for c in fdf.columns if c not in feature_join_cols]
        assert feat_cols, f"query {name!r} produced no feature columns"

        renamed = fdf[feature_join_cols + feat_cols].rename(
            columns={c: f"{name}__{c}" for c in feat_cols}
        )
        base = base.merge(renamed, on=feature_join_cols, how="left")
        assert len(base) == n_rows, (
            f"query {name!r} changed the row count ({len(base)} vs {n_rows}); "
            f"its {feature_join_cols} are not unique after dedup"
        )
        block = [f"{name}__{c}" for c in feat_cols]
        diagnostics[name] = {
            "n_features": len(block),
            "missing_rate": float(base[block].isna().any(axis=1).mean()),
            "joined_on": feature_join_cols,
        }

    feat_columns = [c for c in base.columns if c not in join_cols]
    assert feat_columns, "no feature columns after merging every block"
    # RelAgent's _coerce_ml_feature_column: duckdb hands back nullable Int64 and
    # booleans, and a column can mix ints and fractions across rows.
    frame = pd.DataFrame(
        {
            c: pd.to_numeric(base[c], errors="coerce").astype("float64")
            for c in feat_columns
        }
    )
    return frame, diagnostics
