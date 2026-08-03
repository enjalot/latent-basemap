"""Frozen contract for the R0171-backed prompted OOD universality panel."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from basemap import round0167_prompted_universality as base


ROUND_ID = "0172"
CAPABILITY = "jina-prompted-universality-panel-v1"
PROMPTED_MAP_ORDER = (
    "r0115-prompted-2m-seed42",
    "r0117-prompted-2m-seed43",
    "r0171-prompted-8m-seed42",
)


class Round0172Error(base.Round0167Error):
    """The corrected prompted-universality contract changed."""


def _configure_base() -> None:
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    base.Round0167Error = Round0172Error


def twonn_correlations(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _configure_base()
    return base.twonn_correlations(cells)


def retention_verdict(value: float) -> str:
    _configure_base()
    return base.retention_verdict(value)


__all__ = [
    "CAPABILITY",
    "PROMPTED_MAP_ORDER",
    "ROUND_ID",
    "Round0172Error",
    "retention_verdict",
    "twonn_correlations",
]
