"""Binary-classification targets for Stage 5b.

Three thresholds pinned by AskUserQuestion 2026-05-26 (PLAN_Stage5b.md §3,
§10). Single source of truth so the sweep script, classifier, tests, and
notebook all agree.

Binarisation happens at fit/eval time. No Stage 5 packaging change is
needed -- both source columns (`boulder_count`, `fractional_area`) are
already present on every `y_*.parquet` produced by `src.dataset.package_split`.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinaryTarget:
    """One binary classification target.

    `binarize(y_df)` extracts `source_col` from a Stage 5 y dataframe and applies
    the threshold via the named comparison, returning an int8 0/1 array suitable
    for LightGBM `objective="binary"`.
    """

    id: str
    source_col: str
    threshold: float
    comparison: str  # ">=" or ">"
    label: str       # human-readable for plots / tables

    def binarize(self, y_df: pd.DataFrame) -> np.ndarray:
        if self.source_col not in y_df.columns:
            raise KeyError(
                f"BinaryTarget {self.id!r} expects column {self.source_col!r}, "
                f"got {list(y_df.columns)}"
            )
        col = y_df[self.source_col].to_numpy()
        op = _COMPARISON_OPS[self.comparison]
        return op(col, self.threshold).astype(np.int8)


_COMPARISON_OPS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    ">=": operator.ge,
    ">": operator.gt,
}


BINARY_TARGETS: tuple[BinaryTarget, ...] = (
    BinaryTarget(
        id="bc_ge_1",
        source_col="boulder_count",
        threshold=1.0,
        comparison=">=",
        label="boulder_count ≥ 1",
    ),
    # PLAN_FM 1b count-target re-read (Brian, 2026-06-12): bc_ge_1 is presence
    # (saturated 0.93 positive at S=64), the wrong operationalization of a
    # count-defined rich/poor split. These thresholds are grounded in the S=64
    # per-tile boulder_count distribution (scripts/probes/_fm_count_dist.py):
    # bc_ge_50 -> pos_rate 0.483 (near-median split, real PR-AUC headroom);
    # bc_ge_100 -> pos_rate 0.352, base-rate-matched to fa_gt_1e-2 (0.354) for
    # an apples-to-apples "same selectivity, different definition" comparison.
    BinaryTarget(
        id="bc_ge_50",
        source_col="boulder_count",
        threshold=50.0,
        comparison=">=",
        label="boulder_count ≥ 50",
    ),
    BinaryTarget(
        id="bc_ge_100",
        source_col="boulder_count",
        threshold=100.0,
        comparison=">=",
        label="boulder_count ≥ 100",
    ),
    BinaryTarget(
        id="fa_gt_1e-3",
        source_col="fractional_area",
        threshold=1e-3,
        comparison=">",
        label="fractional_area > 1e-3",
    ),
    BinaryTarget(
        id="fa_gt_1e-2",
        source_col="fractional_area",
        threshold=1e-2,
        comparison=">",
        label="fractional_area > 1e-2",
    ),
)

BINARY_TARGETS_BY_ID: dict[str, BinaryTarget] = {t.id: t for t in BINARY_TARGETS}


def get_target(id: str) -> BinaryTarget:
    """Look up a registered binary target by id. Raises KeyError on miss."""
    if id not in BINARY_TARGETS_BY_ID:
        raise KeyError(
            f"unknown binary target {id!r}; registered: {list(BINARY_TARGETS_BY_ID)}"
        )
    return BINARY_TARGETS_BY_ID[id]
