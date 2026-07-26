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


def test_remove_columns_reaches_the_sampler_for_a_non_autocomplete_task(
    synthetic_dataset_with_external_task, tmp_path
):
    """`remove_columns` is honored for every task kind, not just autocomplete.

    Walks the whole path a leakage column takes: task `manifest.yaml` ->
    `meta.json` (rustler's preprocess) -> `Task.leakage_columns` -> the per-`(table,
    column)` indices handed to the sampler as `columns_to_drop`. fly.rs then skips those
    cell indices wherever they appear, with no dependence on the task's kind.
    """
    import json

    from rt.data.resolve import get_column_index
    from rt.data.tasks import get_tasks
    from rt.rustler import preprocess

    out = tmp_path / "out"
    preprocess(str(synthetic_dataset_with_external_task), str(out))
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    pre_dir, db_name = str(out), produced.name

    # the preprocessor carries remove_columns through for a kind: external task
    meta = json.loads((produced / "meta.json").read_text())
    (task_meta,) = [t for t in meta["tasks"] if t["name"] == "spend"]
    assert task_meta["kind"] == "external"
    assert task_meta["remove_columns"] == [["events", "amount"]]

    # ... and task resolution turns it into leakage_columns
    (task,) = get_tasks(pre_dir, [[db_name, "spend"]], ["train"])
    assert task.leakage_columns == (("events", "amount"),)

    # ... which resolve to the (table, column) index the sampler drops. The index is
    # per (table, column), so dropping it cannot touch a same-named column elsewhere.
    drop_idx = get_column_index("amount", "events", db_name, pre_dir)
    target_idx = get_column_index(task.target_column, task.table_name, db_name, pre_dir)
    assert drop_idx != target_idx
    kind_idx = get_column_index("kind", "events", db_name, pre_dir)
    assert drop_idx != kind_idx  # a sibling column of the same table survives


def test_sampler_drops_remove_columns_for_a_non_autocomplete_task(
    synthetic_dataset_with_external_task, tmp_path
):
    """The dropped column really is absent from sampled contexts.

    Samples every item twice -- once with `columns_to_drop` empty, once with the task's
    `leakage_columns` -- and compares the column indices that actually land in a context.
    The `kind: external` shape is the one that used to slip through: the leaking column
    (`events.amount`) is in a different table from the label rows, so a rule keyed on the
    target's own row or its exact timestamp could not see it.
    """
    import json

    import ml_dtypes
    import numpy as np

    from rt.data.resolve import get_column_index
    from rt.data.tasks import get_tasks
    from rt.rustler import Sampler, preprocess

    out = tmp_path / "out"
    preprocess(str(synthetic_dataset_with_external_task), str(out))
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    pre_dir, db_name, d_text, embedder = str(out), produced.name, 8, "test-embed"

    # The sampler mmaps text embeddings; zeros are enough here -- the assertions are about
    # which columns reach a context, not their values.
    n_text = len(json.loads((produced / "text.json").read_text()))
    (produced / f"text_emb_{embedder}.bin").write_bytes(
        np.zeros((n_text, d_text), dtype=ml_dtypes.bfloat16).tobytes()
    )

    (task,) = get_tasks(pre_dir, [[db_name, "spend"]], ["train"])
    info = json.loads((produced / "table_info.json").read_text())
    span = info[f"{task.table_name}:Train"]
    drop_idx = get_column_index("amount", "events", db_name, pre_dir)
    sibling_idx = get_column_index("kind", "events", db_name, pre_dir)
    target_idx = get_column_index(task.target_column, task.table_name, db_name, pre_dir)

    def context_columns(columns_to_drop):
        sampler = Sampler(
            [(db_name, task.table_name, span["node_idx_offset"], span["num_nodes"])],
            0,
            0,
            1,  # global_rank, local_rank, world_size
            [64],  # local_ctx_size_list
            [8],  # bfs_width_list
            0,
            10,  # num_walks, walk_length
            [False],  # prefer_latest_list
            0.0,  # mask_prob_max
            embedder,
            pre_dir,
            d_text,
            0,
            0,  # shuffle_seed, context_seed
            [target_idx],
            [columns_to_drop],
            -1,  # items_per_task
            True,
            False,
            0,  # quiet, ignore_data_errors, num_prev_skipped
            True,  # mmap_populate
            10.0,  # timeout_per_item
            None,
            False,  # vector_db_path, train_only_fallback
        )
        seen: set[int] = set()
        for batch_idx in range(sampler.num_items):
            batch = dict(sampler.batch_py(batch_idx, 2, 64))
            keep = ~batch["is_padding"].astype(bool)
            seen |= {int(c) for c, k in zip(batch["col_name_idxs"], keep) if k}
        return seen

    without = context_columns([])
    assert drop_idx in without, "fixture is not exercising the column at all"

    # resolved the way RustlerDataset does, from the task's own leakage_columns
    drops = [
        get_column_index(col, table, db_name, pre_dir)
        for table, col in task.leakage_columns
    ]
    assert drops == [drop_idx]
    with_drop = context_columns(drops)
    assert drop_idx not in with_drop  # the leakage column is gone
    assert sibling_idx in with_drop  # a sibling column of the same table stays
    assert target_idx in with_drop  # the target itself stays
    assert with_drop == without - {drop_idx}  # and nothing else changed
