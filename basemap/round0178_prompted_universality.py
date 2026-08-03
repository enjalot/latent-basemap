"""Frozen contract for the training-disjoint prompted OOD recovery."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from basemap import round0167_prompted_universality as base
from basemap.round0176_prompted_universality import (
    exact_training_overlap_report,
)


ROUND_ID = "0178"
CAPABILITY = "jina-prompted-universality-panel-v1"
PROMPTED_MAP_ORDER = (
    "r0115-prompted-2m-seed42",
    "r0117-prompted-2m-seed43",
    "r0171-prompted-8m-seed42",
)
CONTROL_ROWS = 60_000
EXPECTED_CONTROL_ROWS_SCANNED = 60_626
EXPECTED_CONTROL_TRAINING_TEXT_REJECTS = 626
EXPECTED_CONTROL_DUPLICATE_TEXT_REJECTS = 0
EXPECTED_CONTROL_SELECTION_SHA256 = (
    "e531f8ebaedaa5e3d1d1f3000245d27b751cf5cd1070fe3d5dc21aeb5b52096a"
)


class Round0178Error(base.Round0167Error):
    """The R0178 recovery contract changed."""


def _configure_base() -> None:
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    base.Round0167Error = Round0178Error


def twonn_correlations(
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _configure_base()
    return base.twonn_correlations(cells)


def retention_verdict(value: float) -> str:
    _configure_base()
    return base.retention_verdict(value)


__all__ = [
    "CAPABILITY",
    "CONTROL_ROWS",
    "EXPECTED_CONTROL_DUPLICATE_TEXT_REJECTS",
    "EXPECTED_CONTROL_ROWS_SCANNED",
    "EXPECTED_CONTROL_SELECTION_SHA256",
    "EXPECTED_CONTROL_TRAINING_TEXT_REJECTS",
    "PROMPTED_MAP_ORDER",
    "ROUND_ID",
    "Round0178Error",
    "exact_training_overlap_report",
    "retention_verdict",
    "twonn_correlations",
]
