"""Step 0 (demo only) — derive labeled rows from the demo shop database.

Builds the churn label files that the pre-filled ``config.py`` points at: for each
(customer, monthly cutoff) with purchase history before the cutoff, did they buy
nothing in the next 28 days? The last cutoff becomes the test split, the one before
it the val split, everything earlier the train split.

When you point config.py at your own database, skip this step and supply your own
labeled parquet/csv files instead.

    pixi run python examples/inference/0_make_demo_labels.py
"""

from pathlib import Path

import config
import duckdb


def main():
    out = Path(config.DATA_DIR) / "labels"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config.SQL_URI, read_only=True)
    con.execute("""
    CREATE TEMP TABLE cutoffs AS                       -- monthly reference dates
      SELECT TIMESTAMP '2020-01-01' + (d * INTERVAL 1 DAY) AS cutoff FROM range(360, 692, 28) t(d);
    CREATE TEMP TABLE eligible AS                      -- customers with history before the cutoff
      SELECT DISTINCT t.customer_id, k.cutoff FROM transactions t JOIN cutoffs k ON t.timestamp < k.cutoff;
    CREATE TEMP TABLE labels AS                        -- churn: no purchase in the next 28 days?
      SELECT e.customer_id, e.cutoff AS timestamp, count(t.transaction_id) = 0 AS churn FROM eligible e
      LEFT JOIN transactions t ON t.customer_id = e.customer_id
           AND t.timestamp > e.cutoff AND t.timestamp <= e.cutoff + INTERVAL 28 DAY
      GROUP BY e.customer_id, e.cutoff;
    """)
    cuts = [r[0] for r in con.execute("SELECT cutoff FROM cutoffs ORDER BY cutoff").fetchall()]
    for split, cond, param in [
        ("train", "timestamp < ?", cuts[-2]),
        ("val", "timestamp = ?", cuts[-2]),
        ("test", "timestamp = ?", cuts[-1]),
    ]:
        path = out / f"churn_{split}.parquet"
        con.execute(f"COPY (FROM labels WHERE {cond}) TO '{path}' (FORMAT PARQUET)", [param])
        n = con.execute(f"SELECT count(*) FROM labels WHERE {cond}", [param]).fetchone()[0]
        print(f"[step 0] wrote {n:4d} labeled rows -> {path}")
    con.close()


if __name__ == "__main__":
    main()
