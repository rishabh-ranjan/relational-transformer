from dataclasses import dataclass, field

import torch

from rt.augment.stats import (
    QUANTILE_PROBS,
    ColumnStats,
    DerivedStats,
    fingerprint,
)

_SPAN_EPS = 1e-12


def ecdf(
    values: torch.Tensor, knots: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    if knots.shape[0] != values.shape[0]:
        raise ValueError(
            f"ecdf: {knots.shape[0]} knot rows for {values.shape[0]} values"
        )
    if knots.shape[1] != probs.shape[0]:
        raise ValueError(
            f"ecdf: {knots.shape[1]} knots per row but {probs.shape[0]} probabilities"
        )
    n_knots = knots.shape[1]
    v = values.to(torch.float32)
    k = knots.to(torch.float32)
    idx = torch.searchsorted(k.contiguous(), v.unsqueeze(-1).contiguous())
    idx = idx.squeeze(-1).clamp(1, n_knots - 1)
    lo_v = k.gather(1, (idx - 1).unsqueeze(1)).squeeze(1)
    hi_v = k.gather(1, idx.unsqueeze(1)).squeeze(1)
    lo_p = probs[idx - 1]
    hi_p = probs[idx]
    span = hi_v - lo_v
    frac = torch.where(
        span > _SPAN_EPS,
        (v - lo_v) / span.clamp(min=_SPAN_EPS),
        torch.zeros_like(v),
    )
    return (lo_p + frac.clamp(0.0, 1.0) * (hi_p - lo_p)).clamp(0.0, 1.0)


def ecdf_saturated(values: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    v = values.to(torch.float32)
    k = knots.to(torch.float32)
    return (v < k[:, 0]) | (v > k[:, -1])


def signed_log1p(values: torch.Tensor) -> torch.Tensor:
    v = values.to(torch.float32)
    return torch.sign(v) * torch.log1p(v.abs())


def pair_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left.to(torch.float32) * right.to(torch.float32)


@dataclass(frozen=True)
class Ecdf:
    name: str = "ecdf"
    arity: int = 1
    needs_pass2: bool = True
    params: dict = field(default_factory=lambda: {"probs": list(QUANTILE_PROBS)})

    def fingerprint_for(self, sources: tuple[str, ...]) -> str:
        return fingerprint(self.name, self.params, sources)

    def derived_key(self, source: str) -> str:
        return f"ecdf({source})"

    def display_name(self, source: str) -> str:
        column, table = _split_key(source)
        return f"{column} (empirical cdf) of {table}"

    def apply_unary(self, values: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        probs = torch.tensor(
            [p / 100.0 for p in QUANTILE_PROBS],
            dtype=torch.float32,
            device=values.device,
        )
        return ecdf(values, knots, probs)

    def reference_values(self, column: ColumnStats, raw: torch.Tensor) -> torch.Tensor:
        knots = torch.tensor(
            column.standardized_quantiles(), dtype=torch.float32
        ).expand(raw.shape[0], -1)
        return self.apply_unary(raw, knots)


@dataclass(frozen=True)
class SignedLog1p:
    name: str = "signed_log1p"
    arity: int = 1
    needs_pass2: bool = True
    params: dict = field(default_factory=lambda: {"input_space": "standardized"})

    def fingerprint_for(self, sources: tuple[str, ...]) -> str:
        return fingerprint(self.name, self.params, sources)

    def derived_key(self, source: str) -> str:
        return f"signed_log1p({source})"

    def display_name(self, source: str) -> str:
        column, table = _split_key(source)
        return f"{column} (signed log1p) of {table}"

    def apply_unary(self, values: torch.Tensor) -> torch.Tensor:
        return signed_log1p(values)

    def reference_values(self, column: ColumnStats, raw: torch.Tensor) -> torch.Tensor:
        del column
        return self.apply_unary(raw)


@dataclass(frozen=True)
class PairProduct:
    name: str = "pair_product"
    arity: int = 2
    needs_pass2: bool = True
    params: dict = field(default_factory=lambda: {"input_space": "standardized"})

    def fingerprint_for(self, sources: tuple[str, ...]) -> str:
        return fingerprint(self.name, self.params, sources)

    def derived_key(self, left: str, right: str) -> str:
        return f"pair_product({left},{right})"

    def display_name(self, left: str, right: str) -> str:
        left_col, table = _split_key(left)
        right_col, right_table = _split_key(right)
        if table != right_table:
            raise ValueError(
                f"pair_product pairs must come from one table, got "
                f"{table!r} and {right_table!r}"
            )
        return f"{left_col} times {right_col} (product) of {table}"

    def apply_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return pair_product(left, right)


@dataclass(frozen=True)
class Identity:
    name: str = "identity"
    arity: int = 1
    needs_pass2: bool = True
    params: dict = field(default_factory=lambda: {"input_space": "standardized"})

    def fingerprint_for(self, sources: tuple[str, ...]) -> str:
        return fingerprint(self.name, self.params, sources)

    def derived_key(self, source: str) -> str:
        return f"identity({source})"

    def display_name(self, source: str) -> str:
        return source

    def apply_unary(self, values: torch.Tensor) -> torch.Tensor:
        return values.to(torch.float32)


@dataclass(frozen=True)
class PairLeft:
    name: str = "pair_left"
    arity: int = 2
    needs_pass2: bool = True
    params: dict = field(default_factory=lambda: {"input_space": "standardized"})

    def fingerprint_for(self, sources: tuple[str, ...]) -> str:
        return fingerprint(self.name, self.params, sources)

    def derived_key(self, left: str, right: str) -> str:
        return f"pair_left({left},{right})"

    def display_name(self, left: str, right: str) -> str:
        _, table = _split_key(left)
        _, right_table = _split_key(right)
        if table != right_table:
            raise ValueError(
                f"pair_left pairs must come from one table, got {table!r} and "
                f"{right_table!r}"
            )
        return left

    def apply_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del right
        return left.to(torch.float32)


def sample_pairs(columns: list[str], n_pairs: int, seed: int) -> list[tuple[str, str]]:
    if n_pairs < 0:
        raise ValueError(f"n_pairs must be non-negative, got {n_pairs}")
    ordered = sorted(columns)
    if len(ordered) < 2 or n_pairs == 0:
        return []
    all_pairs = [
        (ordered[i], ordered[j])
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    ]
    generator = torch.Generator().manual_seed(seed)
    if n_pairs >= len(all_pairs):
        return all_pairs
    picked = torch.randperm(len(all_pairs), generator=generator)[:n_pairs]
    return [all_pairs[i] for i in sorted(picked.tolist())]


def _split_key(key: str) -> tuple[str, str]:
    column, sep, table = key.partition(" of ")
    if not sep:
        raise ValueError(
            f"column key {key!r} is not of the form '<column> of <table>', "
            f"which is the string pre.rs interns"
        )
    return column, table


def standardize(
    values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (values.to(torch.float32) - mean) / std


@dataclass(frozen=True)
class DerivedColumn:
    transform: Ecdf | SignedLog1p | PairProduct | Identity | PairLeft
    sources: tuple[str, ...]
    source_stats: tuple[ColumnStats, ...]
    derived: DerivedStats

    def __post_init__(self) -> None:
        if len(self.sources) != self.transform.arity:
            raise ValueError(
                f"{self.transform.name} has arity {self.transform.arity} but "
                f"was given {len(self.sources)} sources {self.sources}"
            )
        if len(self.source_stats) != len(self.sources):
            raise ValueError(
                f"{self.key}: {len(self.source_stats)} source stats for "
                f"{len(self.sources)} sources"
            )
        for source, stats in zip(self.sources, self.source_stats):
            if stats.column != source:
                raise ValueError(
                    f"{self.key}: source stats are for {stats.column!r}, not {source!r}"
                )
        expected = self.transform.fingerprint_for(self.sources)
        if self.derived.fingerprint != expected:
            raise ValueError(
                f"{self.key}: derived stats carry fingerprint "
                f"{self.derived.fingerprint!r} but this transform is "
                f"{expected!r}; the stored mean and std standardize a "
                f"different transform. Recompute pass 2."
            )

    @property
    def key(self) -> str:
        if self.transform.arity == 1:
            return self.transform.derived_key(self.sources[0])
        return self.transform.derived_key(*self.sources)

    @property
    def display_name(self) -> str:
        return self.transform.display_name(*self.sources)

    def raw_values(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self.transform.arity:
            raise ValueError(
                f"{self.key}: {len(inputs)} inputs for arity {self.transform.arity}"
            )
        if isinstance(self.transform, Ecdf):
            knots = torch.tensor(
                self.source_stats[0].standardized_quantiles(),
                dtype=torch.float32,
                device=inputs[0].device,
            ).expand(inputs[0].shape[0], -1)
            return self.transform.apply_unary(inputs[0], knots)
        if isinstance(self.transform, (SignedLog1p, Identity)):
            return self.transform.apply_unary(inputs[0])
        if isinstance(self.transform, (PairProduct, PairLeft)):
            return self.transform.apply_pair(inputs[0], inputs[1])
        raise TypeError(f"unknown transform {self.transform!r}")

    def values(self, *inputs: torch.Tensor) -> torch.Tensor:
        return standardize(
            self.raw_values(*inputs), self.derived.mean, self.derived.std
        )

    def saturation(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.transform, Ecdf):
            raise TypeError(
                f"{self.key}: saturation is only defined for the empirical "
                f"cdf, whose knots bound the fitted range"
            )
        knots = torch.tensor(
            self.source_stats[0].standardized_quantiles(),
            dtype=torch.float32,
            device=values.device,
        ).expand(values.shape[0], -1)
        return ecdf_saturated(values, knots)
