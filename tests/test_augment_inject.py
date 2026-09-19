import numpy as np
import pytest
import torch

from rt.augment.inject import augment_batch
from rt.augment.plan import (
    NUMBER_SEM_TYPE,
    SYNTHETIC_COL_BASE,
    AugmentPlan,
    PlannedColumn,
)
from rt.augment.stats import QUANTILE_PROBS, ColumnStats, DerivedStats
from rt.augment.transforms import DerivedColumn, Ecdf, PairProduct, SignedLog1p

D_TEXT = 4
TEXT_SEM_TYPE = 1


def _column_stats(name: str, values: np.ndarray) -> ColumnStats:
    return ColumnStats(
        column=name,
        table="t",
        table_type="Db",
        count=len(values),
        n_null=0,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        quantiles=tuple(float(v) for v in np.percentile(values, list(QUANTILE_PROBS))),
        pre_mean=0.0,
        pre_std=1.0,
        n_distinct=len(np.unique(values)),
        degenerate=False,
    )


def _planned(transform, sources, source_values, index, source_col_idxs):
    stats = tuple(_column_stats(s, v) for s, v in zip(sources, source_values))
    derived = DerivedStats(
        key="k",
        transform=transform.name,
        sources=sources,
        count=100,
        mean=0.0,
        std=1.0,
        modal_share=0.0,
        fingerprint=transform.fingerprint_for(sources),
    )
    return PlannedColumn(
        derived=DerivedColumn(
            transform=transform, sources=sources, source_stats=stats, derived=derived
        ),
        source_col_idxs=source_col_idxs,
        synthetic_col_idx=SYNTHETIC_COL_BASE + index,
        name_embedding_row=index,
    )


def _plan(columns) -> AugmentPlan:
    return AugmentPlan(
        db="rel-f1",
        columns=tuple(columns),
        name_embeddings=torch.arange(
            len(columns) * D_TEXT, dtype=torch.float32
        ).reshape(len(columns), D_TEXT).to(torch.bfloat16),
    )


def _batch(batch_size: int, seq_len: int) -> dict:
    return {
        "batch_mask": torch.ones(batch_size, dtype=torch.bool),
        "node_idxs": torch.zeros((batch_size, seq_len), dtype=torch.int32),
        "table_name_idxs": torch.zeros((batch_size, seq_len), dtype=torch.int32),
        "col_name_idxs": torch.zeros((batch_size, seq_len), dtype=torch.int32),
        "class_value_idxs": torch.full((batch_size, seq_len), -1, dtype=torch.int32),
        "sem_types": torch.zeros((batch_size, seq_len), dtype=torch.int32),
        "timestamps": torch.zeros((batch_size, seq_len), dtype=torch.int32),
        "seed_node_idxs": torch.zeros((batch_size, seq_len), dtype=torch.int32),
        "bfs_depths": torch.zeros((batch_size, seq_len), dtype=torch.int32),
        "f2p_nbr_idxs": torch.zeros((batch_size, seq_len, 5), dtype=torch.int32),
        "is_padding": torch.zeros((batch_size, seq_len), dtype=torch.bool),
        "is_targets": torch.zeros((batch_size, seq_len), dtype=torch.bool),
        "is_task_nodes": torch.zeros((batch_size, seq_len), dtype=torch.bool),
        "number_values": torch.zeros((batch_size, seq_len, 1), dtype=torch.bfloat16),
        "datetime_values": torch.zeros((batch_size, seq_len, 1), dtype=torch.bfloat16),
        "text_values": torch.zeros((batch_size, seq_len, D_TEXT), dtype=torch.bfloat16),
        "col_name_values": torch.zeros(
            (batch_size, seq_len, D_TEXT), dtype=torch.bfloat16
        ),
    }


def _one_numeric_cell_batch() -> dict:
    b = _batch(1, 3)
    b["node_idxs"][0] = torch.tensor([10, 10, 11], dtype=torch.int32)
    b["col_name_idxs"][0] = torch.tensor([7, 8, 7], dtype=torch.int32)
    b["sem_types"][0] = torch.tensor(
        [NUMBER_SEM_TYPE, TEXT_SEM_TYPE, NUMBER_SEM_TYPE], dtype=torch.int32
    )
    b["number_values"][0, :, 0] = torch.tensor([1.5, 0.0, -2.0])
    b["timestamps"][0] = torch.tensor([99, 99, 55], dtype=torch.int32)
    b["table_name_idxs"][0] = torch.tensor([3, 3, 3], dtype=torch.int32)
    return b


def test_no_columns_is_the_identity():
    b = _one_numeric_cell_batch()
    plan = AugmentPlan(db="d", columns=(), name_embeddings=torch.zeros((0, D_TEXT)))
    assert augment_batch(b, plan) is b


def test_emits_one_cell_per_matching_source_cell():
    b = _one_numeric_cell_batch()
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    assert out["node_idxs"].shape == (1, 5)
    new = out["col_name_idxs"][0, 3:]
    assert (new == SYNTHETIC_COL_BASE).all()
    assert (~out["is_padding"][0]).sum() == 5


def test_derived_cell_copies_its_parents_row_identity():
    b = _one_numeric_cell_batch()
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    assert out["node_idxs"][0, 3:].tolist() == [10, 11]
    assert out["timestamps"][0, 3:].tolist() == [99, 55]
    assert out["table_name_idxs"][0, 3:].tolist() == [3, 3]
    assert out["sem_types"][0, 3:].tolist() == [NUMBER_SEM_TYPE, NUMBER_SEM_TYPE]
    assert out["class_value_idxs"][0, 3:].tolist() == [-1, -1]
    assert not out["is_targets"][0, 3:].any()


def test_derived_cell_carries_its_own_name_embedding():
    b = _one_numeric_cell_batch()
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    got = out["col_name_values"][0, 3].float()
    assert torch.allclose(got, plan.name_embeddings[0].float())
    assert not torch.allclose(got, out["col_name_values"][0, 0].float())


def test_a_target_cell_produces_no_derived_cell():
    b = _one_numeric_cell_batch()
    b["is_targets"][0, 0] = True
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    emitted = out["col_name_idxs"][0] >= SYNTHETIC_COL_BASE
    assert emitted.sum() == 1
    assert out["node_idxs"][0][emitted].tolist() == [11]


def test_padding_cells_produce_nothing():
    b = _one_numeric_cell_batch()
    b["is_padding"][0, 2] = True
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    emitted = out["col_name_idxs"][0] >= SYNTHETIC_COL_BASE
    assert emitted.sum() == 1
    assert out["node_idxs"][0][emitted].tolist() == [10]


def test_text_cells_of_the_same_column_id_are_not_augmented():
    b = _batch(1, 2)
    b["col_name_idxs"][0] = torch.tensor([7, 7], dtype=torch.int32)
    b["sem_types"][0] = torch.tensor(
        [TEXT_SEM_TYPE, NUMBER_SEM_TYPE], dtype=torch.int32
    )
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    assert (out["col_name_idxs"][0] >= SYNTHETIC_COL_BASE).sum() == 1


def test_pair_product_only_fires_when_both_parents_share_a_node():
    b = _batch(1, 3)
    b["node_idxs"][0] = torch.tensor([10, 10, 11], dtype=torch.int32)
    b["col_name_idxs"][0] = torch.tensor([7, 8, 7], dtype=torch.int32)
    b["number_values"][0, :, 0] = torch.tensor([2.0, 3.0, 5.0])
    values = np.linspace(-3, 3, 100)
    plan = _plan(
        [_planned(PairProduct(), ("a of t", "b of t"), [values, values], 0, (7, 8))]
    )
    out = augment_batch(b, plan)
    emitted = out["col_name_idxs"][0] >= SYNTHETIC_COL_BASE
    assert emitted.sum() == 1
    assert out["node_idxs"][0][emitted].tolist() == [10]
    assert out["number_values"][0][emitted].squeeze(-1).float().item() == pytest.approx(
        6.0, abs=0.1
    )


def test_pair_product_does_not_cross_rows_of_the_batch():
    b = _batch(2, 2)
    b["node_idxs"][0] = torch.tensor([10, 10], dtype=torch.int32)
    b["node_idxs"][1] = torch.tensor([10, 10], dtype=torch.int32)
    b["col_name_idxs"][0] = torch.tensor([7, 8], dtype=torch.int32)
    b["col_name_idxs"][1] = torch.tensor([7, 7], dtype=torch.int32)
    b["number_values"][0, :, 0] = torch.tensor([2.0, 3.0])
    b["number_values"][1, :, 0] = torch.tensor([9.0, 9.0])
    values = np.linspace(-3, 3, 100)
    plan = _plan(
        [_planned(PairProduct(), ("a of t", "b of t"), [values, values], 0, (7, 8))]
    )
    out = augment_batch(b, plan)
    assert (out["col_name_idxs"][0] >= SYNTHETIC_COL_BASE).sum() == 1
    assert (out["col_name_idxs"][1] >= SYNTHETIC_COL_BASE).sum() == 0


def test_rows_with_different_counts_are_padded_to_one_width():
    b = _batch(2, 2)
    b["node_idxs"][0] = torch.tensor([10, 11], dtype=torch.int32)
    b["node_idxs"][1] = torch.tensor([12, 13], dtype=torch.int32)
    b["col_name_idxs"][0] = torch.tensor([7, 7], dtype=torch.int32)
    b["col_name_idxs"][1] = torch.tensor([7, 9], dtype=torch.int32)
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    assert out["node_idxs"].shape == (2, 4)
    assert (~out["is_padding"][0]).sum() == 4
    assert (~out["is_padding"][1]).sum() == 3
    assert out["is_padding"][1, 3]


def test_every_batch_key_is_rebuilt():
    b = _one_numeric_cell_batch()
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    assert set(out) == set(b)
    for key in b:
        if key == "batch_mask":
            continue
        assert out[key].dtype == b[key].dtype, key
        assert out[key].shape[0] == b[key].shape[0], key
        assert out[key].shape[1] == b[key].shape[1] + 2, key


def test_original_cells_are_untouched():
    b = _one_numeric_cell_batch()
    values = np.linspace(-3, 3, 100)
    plan = _plan([_planned(SignedLog1p(), ("a of t",), [values], 0, (7,))])
    out = augment_batch(b, plan)
    for key, tensor in b.items():
        if key == "batch_mask":
            continue
        assert torch.equal(out[key][:, : tensor.shape[1]], tensor), key


def test_two_transforms_on_one_source_get_distinct_column_ids():
    b = _one_numeric_cell_batch()
    values = np.linspace(-3, 3, 100)
    plan = _plan(
        [
            _planned(SignedLog1p(), ("a of t",), [values], 0, (7,)),
            _planned(Ecdf(), ("a of t",), [values], 1, (7,)),
        ]
    )
    out = augment_batch(b, plan)
    ids = out["col_name_idxs"][0][out["col_name_idxs"][0] >= SYNTHETIC_COL_BASE]
    assert set(ids.tolist()) == {SYNTHETIC_COL_BASE, SYNTHETIC_COL_BASE + 1}


def test_plan_rejects_duplicate_synthetic_ids():
    values = np.linspace(-3, 3, 100)
    a = _planned(SignedLog1p(), ("a of t",), [values], 0, (7,))
    with pytest.raises(ValueError, match="duplicate synthetic column ids"):
        AugmentPlan(
            db="d", columns=(a, a), name_embeddings=torch.zeros((2, D_TEXT))
        )


def test_plan_rejects_an_embedding_count_mismatch():
    values = np.linspace(-3, 3, 100)
    a = _planned(SignedLog1p(), ("a of t",), [values], 0, (7,))
    with pytest.raises(ValueError, match="name embeddings for"):
        AugmentPlan(db="d", columns=(a,), name_embeddings=torch.zeros((3, D_TEXT)))
