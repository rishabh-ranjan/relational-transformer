"""Shared fixtures for the relational-transformer test suite.

These tests exercise the *installed wheel* (the public `rt` API + the compiled
`rt._rustler` engine), so run them against a built + installed package -- e.g.
`local/test.sh`, or the CI `test` job. torch/polars-dependent tests skip
cleanly when those aren't present.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

TINY_DIMS = dict(num_blocks=2, d_model=16, d_text=8, num_heads=2, d_ff=32)


@pytest.fixture(scope="session")
def tiny_dims() -> dict:
    return dict(TINY_DIMS)


@pytest.fixture()
def tiny_checkpoint(tmp_path, tiny_dims):
    """A real checkpoint dir (config.json + model.safetensors) for a tiny model.

    Returns ``(checkpoint_dir, source_model)``.
    """
    pytest.importorskip("torch")
    from rt import RelationalTransformer
    from rt.checkpoints import CONFIG_FILE, MODEL_FILE, save_model

    src = RelationalTransformer(
        **tiny_dims, compile=False, materialize_attn_masks=True
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_model(src.state_dict(), ckpt / MODEL_FILE)
    (ckpt / CONFIG_FILE).write_text(
        json.dumps({"model": tiny_dims, "embedding_model": "test-embed"})
    )
    return ckpt, src


@pytest.fixture()
def synthetic_dataset(tmp_path):
    """A tiny hand-rolled dataset in relbench-3.0.0 layout (``manifest.yaml``
    next to ``db/<table>.parquet``), for the preprocess round-trip.

    Two tables in the shape rustler cares about: an entity table with a pkey and
    no time column, and an activity table with a pkey, a time column, and a fkey
    into the entity table. Column dtypes cover the branches ``normalize_df``
    dispatches on -- string, int, float, bool, and datetime.
    """
    pl = pytest.importorskip("polars")
    import yaml

    n_users, n_events = 10, 18
    users = pl.DataFrame(
        {
            "user_id": range(n_users),
            "name": [f"user {i}" for i in range(n_users)],
            "plan": ["free" if i % 3 else "pro" for i in range(n_users)],
            "credit": [round(1.5 * i, 2) for i in range(n_users)],
            "active": [i % 4 != 0 for i in range(n_users)],
        }
    )
    events = pl.DataFrame(
        {
            "event_id": range(n_events),
            "user_id": [i % n_users for i in range(n_events)],
            "kind": ["click" if i % 2 else "view" for i in range(n_events)],
            "amount": [float(i % 7) for i in range(n_events)],
            "timestamp": [
                datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_events)
            ],
        }
    )

    ds = tmp_path / "synth"
    (ds / "db").mkdir(parents=True)
    users.write_parquet(ds / "db" / "users.parquet")
    events.write_parquet(ds / "db" / "events.parquet")

    manifest = {
        "name": "synth",
        "tables": {
            "users": {"pkey": "user_id"},
            "events": {
                "pkey": "event_id",
                "time_col": "timestamp",
                "fkeys": {"user_id": "users"},
            },
        },
    }
    (ds / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return ds
