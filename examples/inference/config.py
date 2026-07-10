"""Describe YOUR database, prediction task, and checkpoint here — the only file you edit.

It ships pre-filled with the demo shop database at ``examples/byod/mini-shop.duckdb``
(labels are derived by ``0_make_demo_labels.py``), so the walkthrough runs out of the
box. Point it at your own database and change the schema + task to run on your data.
Then run, in order:

    1_data_prep.py   # convert your database to RT's dataset format
    2_task_prep.py   # define the prediction task
    3_predict.py     # download the checkpoint, preprocess, predict, score
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Everything the walkthrough writes (dataset dir, preprocessed tensors) goes here.
DATA_DIR = _HERE / "out"

# A short name for this dataset (becomes a directory under DATA_DIR).
DB_NAME = "mini-shop"

# --- how to reach your database ----------------------------------------------
# DuckDB   : a .duckdb file path (the demo default)
# Postgres : "postgresql+psycopg2://user:password@host:5432/dbname"
# MySQL    : "mysql+pymysql://user:password@host:3306/dbname"
# (install psycopg2-binary / pymysql for Postgres / MySQL.)
SQL_URI = str(_HERE.parent / "byod" / "mini-shop.duckdb")

# --- your relational schema ---------------------------------------------------
# One entry per table you want to include. For each:
#   pkey     : the primary-key column
#   time_col : the row-timestamp column (omit if the table has no time)
#   fkeys    : {foreign_key_column: table_it_points_to}
TABLES = {
    "customers": {"pkey": "customer_id"},
    "products": {"pkey": "product_id"},
    "transactions": {
        "pkey": "transaction_id",
        "time_col": "timestamp",
        "fkeys": {"customer_id": "customers", "product_id": "products"},
    },
}

# --- your prediction task ------------------------------------------------------
# You predict `target_col` for an `entity_table` entity as of a given time.
# The split files are the LABELED rows: each has the entity column, the time
# column, and the target column (parquet or csv). `test` is required; `train`
# is also used for regression target de-normalization.
TASK = {
    "name": "customer-churn",
    "entity_table": "customers",
    "entity_col": "customer_id",
    "time_col": "timestamp",
    "target_col": "churn",
    "task_type": "binary_classification",  # or "regression"
    "splits": {
        "train": str(DATA_DIR / "labels" / "churn_train.parquet"),
        "val": str(DATA_DIR / "labels" / "churn_val.parquet"),
        "test": str(DATA_DIR / "labels" / "churn_test.parquet"),
    },
}

# --- pretrained checkpoint ------------------------------------------------------
# Any RT checkpoint on the Hugging Face Hub (or a local path) works here —
# `rt.checkpoints.load_rt_model` resolves all of these:
#
#   "stanford-star/rt-j/classification"                        # RT-J classifier
#   "stanford-star/rt-j/regression"                            # RT-J regressor
#   "stanford-star/rt-plurel/synthetic-pretrain_rdb_1024_size_4b.pt"   # PluRel
#   "stanford-star/rt-plurel/cntd-pretrain_rel-f1_driver-top3.pt"      # PluRel, cntd-pretrained
#   "~/ckpts/my-run/best_clf.safetensors"                      # your own training run
#
# Browse https://huggingface.co/stanford-star for all released checkpoints.
CHECKPOINT = "stanford-star/rt-j/classification"
