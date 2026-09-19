import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

QUANTILE_PROBS = (
    0.0,
    0.1,
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    75.0,
    90.0,
    95.0,
    99.0,
    99.9,
    100.0,
)

ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class ColumnStats:
    column: str
    table: str
    table_type: str
    count: int
    n_null: int
    mean: float
    std: float
    minimum: float
    maximum: float
    quantiles: tuple[float, ...]
    pre_mean: float
    pre_std: float
    n_distinct: int
    degenerate: bool

    def __post_init__(self) -> None:
        if len(self.quantiles) != len(QUANTILE_PROBS):
            raise ValueError(
                f"{self.column}: {len(self.quantiles)} quantiles for "
                f"{len(QUANTILE_PROBS)} probabilities {QUANTILE_PROBS}"
            )
        if self.pre_std == 0.0:
            raise ValueError(
                f"{self.column}: pre_std is zero; pre.rs forces a zero standard "
                f"deviation to 1.0, so a zero here means the stats were not "
                f"computed the way the store was written"
            )

    def standardized_quantiles(self) -> tuple[float, ...]:
        return tuple((q - self.pre_mean) / self.pre_std for q in self.quantiles)


@dataclass(frozen=True)
class DerivedStats:
    key: str
    transform: str
    sources: tuple[str, ...]
    count: int
    mean: float
    std: float
    modal_share: float
    fingerprint: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.modal_share <= 1.0:
            raise ValueError(
                f"{self.key}: modal_share must be a fraction, got "
                f"{self.modal_share}"
            )
        if self.std <= 0.0:
            raise ValueError(
                f"{self.key}: std must be positive, got {self.std}; a transform "
                f"whose output is constant must be recorded with std 1.0 and "
                f"its source column marked degenerate"
            )


@dataclass(frozen=True)
class NumericStats:
    db: str
    version: int
    columns: dict[str, ColumnStats]
    derived: dict[str, DerivedStats]

    def column(self, key: str) -> ColumnStats:
        try:
            return self.columns[key]
        except KeyError:
            raise KeyError(
                f"no column stats for {key!r} in db {self.db!r}; the artifact "
                f"holds {len(self.columns)} columns. Column keys are spelled "
                f"'<column> of <table>', the same string pre.rs interns."
            ) from None

    def derived_for(self, key: str, fingerprint: str) -> DerivedStats:
        try:
            got = self.derived[key]
        except KeyError:
            raise KeyError(
                f"no derived stats for {key!r} in db {self.db!r}; run the "
                f"pass-2 stats job for this transform set before using it"
            ) from None
        if got.fingerprint != fingerprint:
            raise ValueError(
                f"{key}: derived stats were computed for transform "
                f"fingerprint {got.fingerprint!r} but the configured transform "
                f"is {fingerprint!r}. The stored mean and std do not describe "
                f"this transform; recompute pass 2."
            )
        return got


def fingerprint(transform_name: str, params: dict, source_keys: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"transform": transform_name, "params": params, "sources": list(source_keys)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save(stats: NumericStats, path: str | Path) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "db": stats.db,
        "version": stats.version,
        "quantile_probs": list(QUANTILE_PROBS),
        "columns": {k: asdict(v) for k, v in sorted(stats.columns.items())},
        "derived": {k: asdict(v) for k, v in sorted(stats.derived.items())},
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(p)


def load(path: str | Path) -> NumericStats:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text())
    if payload["version"] != ARTIFACT_VERSION:
        raise ValueError(
            f"{p}: artifact version {payload['version']}, expected "
            f"{ARTIFACT_VERSION}; regenerate it"
        )
    probs = tuple(payload["quantile_probs"])
    if probs != QUANTILE_PROBS:
        raise ValueError(
            f"{p}: quantile probabilities {probs} do not match this build's "
            f"{QUANTILE_PROBS}; the stored knots describe a different grid"
        )
    columns = {
        k: ColumnStats(**{**v, "quantiles": tuple(v["quantiles"])})
        for k, v in payload["columns"].items()
    }
    derived = {
        k: DerivedStats(**{**v, "sources": tuple(v["sources"])})
        for k, v in payload["derived"].items()
    }
    return NumericStats(
        db=payload["db"],
        version=payload["version"],
        columns=columns,
        derived=derived,
    )
