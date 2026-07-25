"""The compiled Rust engine: symbols + preprocess end-to-end."""

from __future__ import annotations


def test_extension_symbols():
    import rt.rustler as r

    assert hasattr(r, "Sampler")
    assert hasattr(r, "preprocess")  # present only when built with --features pre


def test_preprocess_end_to_end(synthetic_dataset, tmp_path):
    from rt.rustler import preprocess

    out = tmp_path / "out"
    preprocess(str(synthetic_dataset), str(out), skip_tasks=True)

    # preprocess writes to <out>/<dataset name>/
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    files = {p.name for p in produced.rglob("*") if p.is_file()}
    # the on-disk format the Sampler consumes
    assert {"nodes.rkyv", "table_info.json", "column_index.json"} <= files


def test_preprocess_zero_variance_columns_emit_no_nan(tmp_path):
    """Constant columns must not normalize by a zero std.

    A boolean (or datetime) column with a single distinct value has zero
    variance; dividing the centered value by it yields 0.0/0.0 = NaN, which then
    reaches the model as a garbage cell value. Cell values written by preprocess
    are always finite.
    """
    from datetime import datetime

    import polars as pl
    import yaml

    from rt.rustler import preprocess

    ds = tmp_path / "constants"
    (ds / "db").mkdir(parents=True)
    pl.DataFrame(
        {
            "id": range(4),
            "flag": [True] * 4,  # zero-variance boolean
            "when": [datetime(2024, 1, 1)] * 4,  # zero-variance datetime
            "amount": [2.5] * 4,  # zero-variance numeric (already guarded)
        }
    ).write_parquet(ds / "db" / "t.parquet")
    (ds / "manifest.yaml").write_text(
        yaml.safe_dump({"name": "constants", "tables": {"t": {"pkey": "id"}}})
    )

    out = tmp_path / "out"
    preprocess(str(ds), str(out), skip_tasks=True)

    # rkyv node records are packed back to back, so field offsets are not
    # 4-aligned; look for the bit patterns a f32 NaN is written as instead of
    # scanning at a stride. 0.0/0.0 gives -NaN on x86, but accept either sign.
    blob = (out / "constants" / "nodes.rkyv").read_bytes()
    for nan_bytes in (bytes.fromhex("0000c0ff"), bytes.fromhex("0000c07f")):
        assert nan_bytes not in blob, "NaN cell value in preprocessed nodes"
