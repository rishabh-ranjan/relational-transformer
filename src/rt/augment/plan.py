import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rt.augment.stats import NumericStats
from rt.augment.transforms import (
    DerivedColumn,
    Ecdf,
    Identity,
    PairLeft,
    PairProduct,
    SignedLog1p,
)

SYNTHETIC_COL_BASE = 1 << 24
NUMBER_SEM_TYPE = 0

TRANSFORM_BY_NAME = {
    "ecdf": Ecdf(),
    "signed_log1p": SignedLog1p(),
    "pair_product": PairProduct(),
    # The token-count ablation: same number of extra cells in the same places,
    # carrying the untransformed z-scored value instead of a derived one.
    "identity": Identity(),
    "pair_left": PairLeft(),
}
ABLATION_OF = {"ecdf": "identity", "signed_log1p": "identity",
               "pair_product": "pair_left"}


@dataclass(frozen=True)
class PlannedColumn:
    derived: DerivedColumn
    source_col_idxs: tuple[int, ...]
    synthetic_col_idx: int
    name_embedding_row: int


@dataclass(frozen=True)
class AugmentPlan:
    db: str
    columns: tuple[PlannedColumn, ...]
    name_embeddings: torch.Tensor

    def __post_init__(self) -> None:
        if self.name_embeddings.shape[0] != len(self.columns):
            raise ValueError(
                f"{self.db}: {self.name_embeddings.shape[0]} name embeddings "
                f"for {len(self.columns)} planned columns"
            )
        synthetic = [c.synthetic_col_idx for c in self.columns]
        if len(set(synthetic)) != len(synthetic):
            raise ValueError(f"{self.db}: duplicate synthetic column ids")
        for c in self.columns:
            if c.synthetic_col_idx < SYNTHETIC_COL_BASE:
                raise ValueError(
                    f"{c.derived.key}: synthetic column id "
                    f"{c.synthetic_col_idx} is below the reserved base "
                    f"{SYNTHETIC_COL_BASE}, so it could collide with a real "
                    f"column of the store"
                )

    @property
    def unary(self) -> tuple[PlannedColumn, ...]:
        return tuple(c for c in self.columns if c.derived.transform.arity == 1)

    @property
    def pairs(self) -> tuple[PlannedColumn, ...]:
        return tuple(c for c in self.columns if c.derived.transform.arity == 2)

    def to(self, device) -> "AugmentPlan":
        return AugmentPlan(
            db=self.db,
            columns=self.columns,
            name_embeddings=self.name_embeddings.to(device),
        )


def build_plan(
    *,
    stats: NumericStats,
    column_index: dict[str, int],
    name_embeddings: dict[str, np.ndarray],
    include: list[str],
    max_modal_share: float,
    d_text: int,
) -> AugmentPlan:
    unknown = sorted(set(include) - set(TRANSFORM_BY_NAME))
    if unknown:
        raise ValueError(
            f"unknown transforms {unknown}; have {sorted(TRANSFORM_BY_NAME)}"
        )
    planned: list[PlannedColumn] = []
    vectors: list[np.ndarray] = []
    for key in sorted(stats.derived):
        derived_stats = stats.derived[key]
        if derived_stats.transform not in include:
            continue
        transform = TRANSFORM_BY_NAME[derived_stats.transform]
        sources = derived_stats.sources
        source_stats = tuple(stats.column(s) for s in sources)
        if any(s.degenerate for s in source_stats):
            continue
        # A transform whose output piles onto one value carries almost
        # nothing and still costs a token, so it is dropped rather than
        # emitted. Zero-inflated counts do this to the ecdf.
        if derived_stats.modal_share > max_modal_share:
            continue
        missing = [s for s in sources if s not in column_index]
        if missing:
            raise KeyError(
                f"{key}: source columns {missing} are not in the store's "
                f"column_index.json, so no cell of theirs can ever appear"
            )
        if key not in name_embeddings:
            raise KeyError(
                f"{key}: no column-name embedding; run the name embedding job "
                f"for this transform set"
            )
        vector = name_embeddings[key]
        if vector.shape != (d_text,):
            raise ValueError(
                f"{key}: name embedding has shape {vector.shape}, expected "
                f"({d_text},)"
            )
        column = DerivedColumn(
            transform=transform,
            sources=sources,
            source_stats=source_stats,
            derived=derived_stats,
        )
        planned.append(
            PlannedColumn(
                derived=column,
                source_col_idxs=tuple(column_index[s] for s in sources),
                synthetic_col_idx=SYNTHETIC_COL_BASE + len(planned),
                name_embedding_row=len(planned),
            )
        )
        vectors.append(vector)
    embeddings = (
        torch.from_numpy(np.stack(vectors)).to(torch.bfloat16)
        if vectors
        else torch.zeros((0, d_text), dtype=torch.bfloat16)
    )
    return AugmentPlan(db=stats.db, columns=tuple(planned), name_embeddings=embeddings)


def load_column_index(pre_dir: str, db: str) -> dict[str, int]:
    path = Path(pre_dir).expanduser() / db / "column_index.json"
    return json.loads(path.read_text())


def load_name_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def ablation_names(real: AugmentPlan) -> list[str]:
    """Display names for the ablation mirror of `real`, in plan order.

    Each copy is numbered per source column. A copy must NOT reuse its source's
    name: identical name and identical value would make it an exact duplicate
    token, which is a degenerate input the model never saw in pretraining, and a
    different intervention from "one more column". Numbering keeps the structure
    matched to the real augmentation -- new column, new name, new token -- with
    only the value left untransformed.
    """
    seen: dict[str, int] = {}
    names = []
    for column in real.columns:
        source = column.derived.sources[0]
        seen[source] = seen.get(source, 0) + 1
        col, _, table = source.partition(" of ")
        names.append(f"{col} (copy {seen[source]}) of {table}")
    return names


def build_ablation_plan(
    *,
    real: AugmentPlan,
    stats: NumericStats,
    name_embeddings: dict[str, np.ndarray],
    d_text: int,
) -> AugmentPlan:
    """A plan that mirrors `real` cell for cell with untransformed values.

    A unary transform becomes an identity copy of its one source; a pair becomes
    a copy of its LEFT source that stays gated on both parents, so it fires on
    exactly the same rows. The result adds the same number of tokens in the same
    positions, carrying values the model already has.

    Copies of one source share a DerivedStats -- the output distribution is the
    same for every copy -- but each gets its own synthetic column id and its own
    name embedding.
    """
    names = ablation_names(real)
    planned: list[PlannedColumn] = []
    vectors: list[np.ndarray] = []
    for column, name in zip(real.columns, names):
        transform = TRANSFORM_BY_NAME[ABLATION_OF[column.derived.transform.name]]
        sources = column.derived.sources
        key = transform.derived_key(*sources)
        derived_stats = stats.derived_for(key, transform.fingerprint_for(sources))
        if name not in name_embeddings:
            raise KeyError(
                f"{name!r}: no column-name embedding; run the ablation name "
                f"embedding job for this plan"
            )
        vector = name_embeddings[name]
        if vector.shape != (d_text,):
            raise ValueError(
                f"{name!r}: name embedding has shape {vector.shape}, expected "
                f"({d_text},)"
            )
        planned.append(
            PlannedColumn(
                derived=DerivedColumn(
                    transform=transform,
                    sources=sources,
                    source_stats=column.derived.source_stats,
                    derived=derived_stats,
                ),
                source_col_idxs=column.source_col_idxs,
                synthetic_col_idx=SYNTHETIC_COL_BASE + len(planned),
                name_embedding_row=len(planned),
            )
        )
        vectors.append(vector)
    embeddings = (
        torch.from_numpy(np.stack(vectors)).to(torch.bfloat16)
        if vectors
        else torch.zeros((0, d_text), dtype=torch.bfloat16)
    )
    return AugmentPlan(db=stats.db, columns=tuple(planned), name_embeddings=embeddings)
