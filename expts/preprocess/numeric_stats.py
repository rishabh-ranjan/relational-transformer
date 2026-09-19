import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from rt.augment.stats import (
    ARTIFACT_VERSION,
    QUANTILE_PROBS,
    ColumnStats,
    DerivedStats,
    NumericStats,
    save,
)
from rt.augment.transforms import (
    DerivedColumn,
    Ecdf,
    PairProduct,
    SignedLog1p,
    sample_pairs,
)

TRANSFORMS = {"ecdf": Ecdf(), "signed_log1p": SignedLog1p()}


def is_numeric(t: pa.DataType) -> bool:
    return (
        pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_decimal(t)
        or pa.types.is_duration(t)
        or pa.types.is_boolean(t)
    )


def load_column(path: Path, column: str) -> tuple[np.ndarray, int]:
    table = pq.read_table(path, columns=[column])
    arr = table.column(0)
    if pa.types.is_duration(arr.type):
        arr = arr.cast(pa.int64())
    values = arr.to_numpy(zero_copy_only=False).astype(np.float64)
    n_total = len(values)
    values = values[np.isfinite(values)]
    return values, n_total - len(values)


def column_stats(
    *,
    key: str,
    table: str,
    values: np.ndarray,
    n_null: int,
    min_distinct: int,
) -> ColumnStats:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    quantiles = np.percentile(values, list(QUANTILE_PROBS))
    n_distinct = len(np.unique(values))
    return ColumnStats(
        column=key,
        table=table,
        table_type="Db",
        count=len(values),
        n_null=int(n_null),
        mean=mean,
        std=std,
        minimum=float(values.min()),
        maximum=float(values.max()),
        quantiles=tuple(float(q) for q in quantiles),
        # pre.rs forces a zero standard deviation to 1.0 before dividing; the
        # store was written that way, so the value that recovers a stored cell
        # is this one and not `std`.
        pre_mean=mean,
        pre_std=std if std != 0.0 else 1.0,
        n_distinct=n_distinct,
        degenerate=n_distinct < min_distinct or std == 0.0,
    )


def standardized(values: np.ndarray, stats: ColumnStats) -> torch.Tensor:
    return torch.from_numpy((values - stats.pre_mean) / stats.pre_std).to(torch.float32)


def fit_derived(
    *, transform, sources: tuple[str, ...], source_stats, raw: torch.Tensor
) -> DerivedStats | None:
    seed = DerivedStats(
        key="seed",
        transform=transform.name,
        sources=sources,
        count=int(raw.shape[0]) if transform.arity == 1 else 0,
        mean=0.0,
        std=1.0,
        modal_share=0.0,
        fingerprint=transform.fingerprint_for(sources),
    )
    probe = DerivedColumn(
        transform=transform,
        sources=sources,
        source_stats=source_stats,
        derived=seed,
    )
    out = probe.raw_values(*raw) if isinstance(raw, tuple) else probe.raw_values(raw)
    n = int(out.shape[0])
    if n < 2:
        return None
    std = float(out.std(unbiased=True))
    if not np.isfinite(std) or std == 0.0:
        return None
    mode = out.mode().values
    return DerivedStats(
        key=probe.key,
        transform=transform.name,
        sources=sources,
        count=n,
        mean=float(out.mean()),
        std=std,
        modal_share=float((out == mode).float().mean()),
        fingerprint=transform.fingerprint_for(sources),
    )


def main(
    *,
    db: str,
    raw_dir: str,
    pre_dir: str,
    out_path: str,
    transforms: list[str],
    n_pairs_per_table: int,
    pair_seed: int,
    min_distinct: int,
) -> None:
    raw = Path(raw_dir).expanduser() / db
    pre = Path(pre_dir).expanduser() / db
    unknown = sorted(set(transforms) - set(TRANSFORMS) - {"pair_product"})
    assert not unknown, f"unknown transforms {unknown}; have {sorted(TRANSFORMS)}"

    manifest = yaml.safe_load((raw / "manifest.yaml").read_text())
    # The store is the authority on which columns became cells: pre.rs drops
    # constant columns and applies an identifier policy, so a column present in
    # the parquet may have no cells at all. Keying off column_index.json means
    # this job cannot disagree with the store about that.
    column_index = json.loads((pre / "column_index.json").read_text())

    columns: dict[str, ColumnStats] = {}
    derived: dict[str, DerivedStats] = {}
    per_table: dict[str, list[str]] = {}

    for table, spec in sorted(manifest["tables"].items()):
        path = raw / "db" / f"{table}.parquet"
        if not path.is_file():
            continue
        skip = {spec.get("pkey"), spec.get("time_col")} | set(spec.get("fkeys") or {})
        for field in pq.ParquetFile(path).schema_arrow:
            if field.name in skip or not is_numeric(field.type):
                continue
            key = f"{field.name} of {table}"
            if key not in column_index:
                print(f"  skip {key}: not in the store", flush=True)
                continue
            values, n_null = load_column(path, field.name)
            if len(values) < 2:
                print(f"  skip {key}: {len(values)} usable values", flush=True)
                continue
            stats = column_stats(
                key=key,
                table=table,
                values=values,
                n_null=n_null,
                min_distinct=min_distinct,
            )
            columns[key] = stats
            if stats.degenerate:
                print(
                    f"  {key}: degenerate "
                    f"(n_distinct={stats.n_distinct}, std={stats.std:.6g})",
                    flush=True,
                )
                continue
            per_table.setdefault(table, []).append(key)
            z = standardized(values, stats)
            for name in transforms:
                if name == "pair_product":
                    continue
                got = fit_derived(
                    transform=TRANSFORMS[name],
                    sources=(key,),
                    source_stats=(stats,),
                    raw=z,
                )
                if got is None:
                    print(f"  {key}: {name} output is constant; skipped", flush=True)
                    continue
                derived[got.key] = got

    if "pair_product" in transforms:
        product = PairProduct()
        for table, keys in sorted(per_table.items()):
            pairs = sample_pairs(keys, n_pairs=n_pairs_per_table, seed=pair_seed)
            for left, right in pairs:
                left_col = left.partition(" of ")[0]
                right_col = right.partition(" of ")[0]
                path = raw / "db" / f"{table}.parquet"
                both = pq.read_table(path, columns=[left_col, right_col])
                a = both.column(0).to_numpy(zero_copy_only=False).astype(np.float64)
                b = both.column(1).to_numpy(zero_copy_only=False).astype(np.float64)
                # A product needs both parents, so the population is the rows
                # where both are present -- not the union of each column's own
                # non-null rows, which would not be row-aligned.
                keep = np.isfinite(a) & np.isfinite(b)
                if keep.sum() < 2:
                    print(
                        f"  skip product {left} x {right}: <2 shared rows", flush=True
                    )
                    continue
                za = standardized(a[keep], columns[left])
                zb = standardized(b[keep], columns[right])
                got = fit_derived(
                    transform=product,
                    sources=(left, right),
                    source_stats=(columns[left], columns[right]),
                    raw=(za, zb),
                )
                if got is None:
                    print(f"  skip product {left} x {right}: constant", flush=True)
                    continue
                derived[got.key] = got

    stats = NumericStats(
        db=db, version=ARTIFACT_VERSION, columns=columns, derived=derived
    )
    save(stats, out_path)
    n_degenerate = sum(1 for c in columns.values() if c.degenerate)
    print(
        f"{db}: {len(columns)} numeric columns ({n_degenerate} degenerate), "
        f"{len(derived)} derived -> {out_path}",
        flush=True,
    )
