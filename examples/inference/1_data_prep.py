"""Step 1 — Convert your SQL database to RT's dataset format.

Reads the tables named in ``config.TABLES`` from ``config.SQL_URI`` (DuckDB file,
Postgres, or MySQL), and writes a RelBench-format dataset directory:

    <DATA_DIR>/<DB_NAME>/
      manifest.yaml     # the schema: primary keys, foreign keys, time columns
      db/<table>.parquet

    pixi run python examples/inference/1_data_prep.py
"""

from pathlib import Path

import config
import yaml


def read_tables(uri: str, table_names) -> dict:
    """Read the named tables into DataFrames.

    A ``.duckdb`` / ``.db`` file path is read natively; anything else is treated
    as a SQLAlchemy URI (Postgres: install ``psycopg2-binary``, MySQL: ``pymysql``).
    """
    if uri.endswith((".duckdb", ".db")):
        import duckdb

        if not Path(uri).exists():
            raise FileNotFoundError(f"DuckDB file not found: {uri}")
        con = duckdb.connect(uri, read_only=True)
        try:
            return {t: con.execute(f'SELECT * FROM "{t}"').df() for t in table_names}
        finally:
            con.close()

    import pandas as pd
    from sqlalchemy import create_engine

    engine = create_engine(uri)
    try:
        return {t: pd.read_sql(f'SELECT * FROM "{t}"', engine) for t in table_names}
    finally:
        engine.dispose()


def main():
    print(f"[step 1] reading tables from {config.SQL_URI}")
    raw = read_tables(config.SQL_URI, list(config.TABLES))

    import pandas as pd

    out = Path(config.DATA_DIR) / config.DB_NAME
    (out / "db").mkdir(parents=True, exist_ok=True)
    for name, cfg in config.TABLES.items():
        df = raw[name]
        if cfg.get("time_col"):
            df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]])
        df.to_parquet(out / "db" / f"{name}.parquet", index=False)

    yaml.safe_dump(
        {"name": config.DB_NAME, "tables": config.TABLES},
        open(out / "manifest.yaml", "w"),
        sort_keys=False,
    )
    print(f"[step 1] wrote RelBench database '{config.DB_NAME}' -> {out}")
    for name, cfg in config.TABLES.items():
        print(
            f"   - {name}: {len(raw[name])} rows | pkey={cfg.get('pkey')} "
            f"| fkeys={cfg.get('fkeys') or {}} | time={cfg.get('time_col')}"
        )


if __name__ == "__main__":
    main()
