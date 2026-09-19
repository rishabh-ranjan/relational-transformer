import numpy as np
import pytest
import torch

from rt.augment import (
    ARTIFACT_VERSION,
    QUANTILE_PROBS,
    ColumnStats,
    DerivedStats,
    NumericStats,
    ecdf,
    ecdf_saturated,
    pair_product,
    sample_pairs,
    signed_log1p,
    standardize,
)
from rt.augment.stats import load, save
from rt.augment.transforms import DerivedColumn, Ecdf, PairProduct, SignedLog1p


def _probs() -> torch.Tensor:
    return torch.tensor([p / 100.0 for p in QUANTILE_PROBS], dtype=torch.float32)


def _column(values: np.ndarray, name: str = "x of t") -> ColumnStats:
    q = np.percentile(values, list(QUANTILE_PROBS))
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
        quantiles=tuple(float(v) for v in q),
        pre_mean=float(values.mean()),
        pre_std=float(values.std(ddof=1)),
        n_distinct=len(np.unique(values)),
        degenerate=False,
    )


def test_ecdf_recovers_uniform_ranks_on_a_uniform_column():
    values = np.linspace(0.0, 1.0, 10001)
    col = _column(values)
    knots = torch.tensor(col.quantiles, dtype=torch.float32).expand(len(values), -1)
    got = ecdf(torch.tensor(values, dtype=torch.float32), knots, _probs())
    assert torch.allclose(got, torch.tensor(values, dtype=torch.float32), atol=2e-3)


def test_ecdf_is_invariant_to_affine_rescaling():
    rng = np.random.default_rng(0)
    values = rng.lognormal(size=5000)
    col = _column(values)
    v = torch.tensor(values, dtype=torch.float32)

    raw_knots = torch.tensor(col.quantiles, dtype=torch.float32).expand(len(values), -1)
    raw = ecdf(v, raw_knots, _probs())

    z = (v - col.pre_mean) / col.pre_std
    z_knots = torch.tensor(col.standardized_quantiles(), dtype=torch.float32).expand(
        len(values), -1
    )
    on_z = ecdf(z, z_knots, _probs())

    assert torch.allclose(raw, on_z, atol=1e-5)


def test_ecdf_clamps_outside_the_fitted_range():
    values = np.linspace(0.0, 1.0, 1001)
    col = _column(values)
    probe = torch.tensor([-100.0, 0.0, 0.5, 1.0, 100.0], dtype=torch.float32)
    knots = torch.tensor(col.quantiles, dtype=torch.float32).expand(len(probe), -1)
    got = ecdf(probe, knots, _probs())
    assert got[0].item() == pytest.approx(0.0)
    assert got[-1].item() == pytest.approx(1.0)
    assert got.min() >= 0.0 and got.max() <= 1.0


def test_ecdf_on_a_constant_column_is_finite_and_does_not_divide_by_zero():
    values = np.full(100, 3.0)
    knots = torch.full((5, len(QUANTILE_PROBS)), 3.0)
    probe = torch.tensor([2.0, 3.0, 3.0, 4.0, 3.0], dtype=torch.float32)
    got = ecdf(probe, knots, _probs())
    assert torch.isfinite(got).all()
    del values


def test_ecdf_saturation_flags_values_outside_the_knots():
    knots = torch.tensor([[0.0, 1.0, 2.0]]).expand(4, -1)
    probe = torch.tensor([-1.0, 0.5, 1.5, 9.0])
    got = ecdf_saturated(probe, knots)
    assert got.tolist() == [True, False, False, True]


def test_ecdf_saturation_does_not_flag_the_fitted_endpoints():
    knots = torch.tensor([[0.0, 1.0, 2.0]]).expand(2, -1)
    probe = torch.tensor([0.0, 2.0])
    assert ecdf_saturated(probe, knots).tolist() == [False, False]


def test_ecdf_saturation_is_zero_on_the_fitting_population():
    rng = np.random.default_rng(12)
    values = rng.lognormal(size=5000)
    col = _column(values)
    knots = torch.tensor(col.quantiles, dtype=torch.float32).expand(len(values), -1)
    got = ecdf_saturated(torch.tensor(values, dtype=torch.float32), knots)
    assert not got.any()


def test_signed_log1p_is_odd_and_monotone():
    v = torch.tensor([-100.0, -1.0, -0.5, 0.0, 0.5, 1.0, 100.0])
    got = signed_log1p(v)
    assert torch.allclose(got, -signed_log1p(-v), atol=1e-6)
    assert (got.diff() > 0).all()
    assert got[3].item() == pytest.approx(0.0)


def test_signed_log1p_compresses_tails():
    v = torch.tensor([1.0, 10.0, 100.0])
    got = signed_log1p(v)
    assert (v[2] / v[0]).item() == pytest.approx(100.0)
    assert (got[2] / got[0]).item() < 10.0
    step = signed_log1p(torch.tensor([1.0, 2.0, 3.0]))
    assert (step[2] - step[1]) < (step[1] - step[0])


def test_pair_product_mean_is_the_correlation_of_standardized_columns():
    rng = np.random.default_rng(1)
    n = 200000
    a = rng.normal(size=n)
    b = 0.6 * a + np.sqrt(1 - 0.36) * rng.normal(size=n)
    za = (a - a.mean()) / a.std(ddof=1)
    zb = (b - b.mean()) / b.std(ddof=1)
    prod = pair_product(torch.tensor(za), torch.tensor(zb))
    assert prod.mean().item() == pytest.approx(np.corrcoef(a, b)[0, 1], abs=5e-3)


def test_sample_pairs_is_deterministic_and_within_range():
    cols = [f"c{i} of t" for i in range(8)]
    first = sample_pairs(cols, n_pairs=6, seed=0)
    assert first == sample_pairs(cols, n_pairs=6, seed=0)
    assert len(first) == 6
    assert len(set(first)) == 6
    for left, right in first:
        assert left in cols and right in cols and left < right


def test_sample_pairs_degenerate_cases():
    assert sample_pairs(["a of t"], n_pairs=4, seed=0) == []
    assert sample_pairs(["a of t", "b of t"], n_pairs=0, seed=0) == []
    assert len(sample_pairs([f"c{i} of t" for i in range(3)], n_pairs=99, seed=0)) == 3


def test_standardize_gives_zero_mean_unit_variance():
    rng = np.random.default_rng(2)
    v = torch.tensor(rng.lognormal(size=10000), dtype=torch.float32)
    out = standardize(v, v.mean(), v.std(unbiased=True))
    assert out.mean().item() == pytest.approx(0.0, abs=1e-4)
    assert out.std(unbiased=True).item() == pytest.approx(1.0, abs=1e-4)


def test_fingerprint_changes_with_params_and_sources():
    a = Ecdf().fingerprint_for(("x of t",))
    b = Ecdf().fingerprint_for(("y of t",))
    c = SignedLog1p().fingerprint_for(("x of t",))
    assert a != b and a != c


def test_derived_stats_rejects_a_mismatched_fingerprint():
    stats = NumericStats(
        db="rel-f1",
        version=1,
        columns={},
        derived={
            "ecdf(x of t)": DerivedStats(
                key="ecdf(x of t)",
                transform="ecdf",
                sources=("x of t",),
                count=10,
                mean=0.5,
                std=0.3,
                modal_share=0.0,
                fingerprint="deadbeefdeadbeef",
            )
        },
    )
    with pytest.raises(ValueError, match="do not describe this transform"):
        stats.derived_for("ecdf(x of t)", "0000000000000000")
    assert stats.derived_for("ecdf(x of t)", "deadbeefdeadbeef").mean == 0.5


def test_derived_stats_rejects_a_zero_std():
    with pytest.raises(ValueError, match="std must be positive"):
        DerivedStats(
            key="k",
            transform="ecdf",
            sources=("x of t",),
            count=1,
            mean=0.0,
            std=0.0,
            modal_share=0.0,
            fingerprint="f",
        )


def test_column_stats_rejects_a_wrong_quantile_count():
    with pytest.raises(ValueError, match="quantiles for"):
        ColumnStats(
            column="x of t",
            table="t",
            table_type="Db",
            count=1,
            n_null=0,
            mean=0.0,
            std=1.0,
            minimum=0.0,
            maximum=1.0,
            quantiles=(0.0, 1.0),
            pre_mean=0.0,
            pre_std=1.0,
            n_distinct=2,
            degenerate=False,
        )


def test_display_names_are_readable_and_carry_the_table():
    assert Ecdf().display_name("points of standings") == (
        "points (empirical cdf) of standings"
    )
    assert SignedLog1p().display_name("points of standings") == (
        "points (signed log1p) of standings"
    )
    assert PairProduct().display_name("a of t", "b of t") == (
        "a times b (product) of t"
    )


def test_pair_product_rejects_a_cross_table_pair():
    with pytest.raises(ValueError, match="one table"):
        PairProduct().display_name("a of t", "b of u")


def test_artifact_round_trips(tmp_path):
    rng = np.random.default_rng(3)
    col = _column(rng.normal(size=1000), name="points of standings")
    stats = NumericStats(
        db="rel-f1",
        version=ARTIFACT_VERSION,
        columns={col.column: col},
        derived={
            "ecdf(points of standings)": DerivedStats(
                key="ecdf(points of standings)",
                transform="ecdf",
                sources=("points of standings",),
                count=1000,
                mean=0.5,
                std=0.29,
                modal_share=0.0,
                fingerprint=Ecdf().fingerprint_for(("points of standings",)),
            )
        },
    )
    path = tmp_path / "numeric_stats.json"
    save(stats, path)
    back = load(path)
    assert back.db == "rel-f1"
    assert back.column("points of standings").quantiles == col.quantiles
    assert (
        back.derived_for(
            "ecdf(points of standings)",
            Ecdf().fingerprint_for(("points of standings",)),
        ).count
        == 1000
    )


def test_load_rejects_a_foreign_quantile_grid(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"db": "d", "version": 2, "quantile_probs": [0.0, 50.0, 100.0], '
        '"columns": {}, "derived": {}}'
    )
    with pytest.raises(ValueError, match="do not match this build"):
        load(path)


def _derived_column(values: np.ndarray, transform):
    col = _column(values, name="points of standings")
    z = torch.tensor((values - col.pre_mean) / col.pre_std, dtype=torch.float32)
    probe = DerivedColumn(
        transform=transform,
        sources=("points of standings",),
        source_stats=(col,),
        derived=DerivedStats(
            key="tmp",
            transform=transform.name,
            sources=("points of standings",),
            count=len(values),
            mean=0.0,
            std=1.0,
            modal_share=0.0,
            fingerprint=transform.fingerprint_for(("points of standings",)),
        ),
    )
    raw = probe.raw_values(z)
    fitted = DerivedStats(
        key=probe.key,
        transform=transform.name,
        sources=("points of standings",),
        count=len(values),
        mean=float(raw.mean()),
        std=float(raw.std(unbiased=True)),
        modal_share=0.0,
        fingerprint=transform.fingerprint_for(("points of standings",)),
    )
    return DerivedColumn(
        transform=transform,
        sources=("points of standings",),
        source_stats=(col,),
        derived=fitted,
    ), z


@pytest.mark.parametrize("transform", [Ecdf(), SignedLog1p()])
def test_derived_column_output_is_standardized_on_its_fitting_population(transform):
    rng = np.random.default_rng(7)
    values = rng.lognormal(sigma=1.5, size=20000)
    column, z = _derived_column(values, transform)
    out = column.values(z)
    assert out.mean().item() == pytest.approx(0.0, abs=1e-4)
    assert out.std(unbiased=True).item() == pytest.approx(1.0, abs=1e-4)


def test_derived_column_rejects_stats_from_a_different_transform():
    rng = np.random.default_rng(8)
    col = _column(rng.normal(size=100), name="points of standings")
    with pytest.raises(ValueError, match="standardize a different transform"):
        DerivedColumn(
            transform=Ecdf(),
            sources=("points of standings",),
            source_stats=(col,),
            derived=DerivedStats(
                key="k",
                transform="signed_log1p",
                sources=("points of standings",),
                count=1,
                mean=0.0,
                std=1.0,
                modal_share=0.0,
                fingerprint=SignedLog1p().fingerprint_for(("points of standings",)),
            ),
        )


def test_derived_column_rejects_mismatched_source_stats():
    rng = np.random.default_rng(9)
    col = _column(rng.normal(size=100), name="other of standings")
    with pytest.raises(ValueError, match="source stats are for"):
        DerivedColumn(
            transform=Ecdf(),
            sources=("points of standings",),
            source_stats=(col,),
            derived=DerivedStats(
                key="k",
                transform="ecdf",
                sources=("points of standings",),
                count=1,
                mean=0.0,
                std=1.0,
                modal_share=0.0,
                fingerprint=Ecdf().fingerprint_for(("points of standings",)),
            ),
        )


def test_derived_column_rejects_wrong_arity():
    rng = np.random.default_rng(10)
    col = _column(rng.normal(size=100), name="a of t")
    with pytest.raises(ValueError, match="arity 2 but was given 1"):
        DerivedColumn(
            transform=PairProduct(),
            sources=("a of t",),
            source_stats=(col,),
            derived=DerivedStats(
                key="k",
                transform="pair_product",
                sources=("a of t",),
                count=1,
                mean=0.0,
                std=1.0,
                modal_share=0.0,
                fingerprint=PairProduct().fingerprint_for(("a of t",)),
            ),
        )


def test_saturation_is_only_defined_for_the_empirical_cdf():
    rng = np.random.default_rng(11)
    values = rng.normal(size=500)
    column, z = _derived_column(values, SignedLog1p())
    with pytest.raises(TypeError, match="only defined for the empirical"):
        column.saturation(z)
