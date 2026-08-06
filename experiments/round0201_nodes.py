"""Execute R0201's float64-reranked Track-D localization."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.round0201_pile_loss_localization import (
    CAPABILITY,
    ROUND_ID,
    Round0201Error,
    synthesize,
)
from experiments import round0194_nodes as base


EXPECTED_FP32_BOUNDARY_TIES = {
    "seed42_half": 0,
    "seed42_full": 0,
    "seed43_half": 0,
    "seed43_full": 0,
    "seed44_half": 0,
    "seed44_full": 1,
}
_CELL_ORDER = tuple(EXPECTED_FP32_BOUNDARY_TIES)
_SEARCH_RECEIPTS: list[dict[str, Any]] = []


def _rerank_candidates_float64(
    values: np.ndarray,
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rerank a bounded candidate set with exact float64 arithmetic."""
    points = np.asarray(values, dtype=np.float32)
    queries = np.asarray(query, dtype=np.float32)
    ids = np.asarray(candidates, dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or queries.ndim != 2
        or queries.shape[1] != 2
        or ids.ndim != 2
        or len(queries) != len(ids)
        or k <= 0
        or ids.shape[1] <= k
        or np.any(ids < 0)
        or np.any(ids >= len(points))
    ):
        raise Round0201Error("R0201 candidate rerank geometry changed")
    delta64 = points[ids].astype(np.float64) - queries[:, None, :].astype(
        np.float64
    )
    squared64 = np.sum(delta64 * delta64, axis=2, dtype=np.float64)
    # The last lexsort key is primary: squared distance, then canonical row ID.
    order = np.lexsort((ids, squared64), axis=1)
    ordered_ids = np.take_along_axis(ids, order, axis=1)
    ordered_squared64 = np.take_along_axis(squared64, order, axis=1)
    gap64 = ordered_squared64[:, k] - ordered_squared64[:, k - 1]
    if np.any(gap64 <= 0) or not np.isfinite(gap64).all():
        raise Round0201Error(
            "R0201 float64 k-boundary remains tied or nonfinite"
        )
    delta32 = points[ids] - queries[:, None, :]
    squared32 = np.sum(delta32 * delta32, axis=2, dtype=np.float32)
    order32 = np.argsort(squared32, axis=1, kind="stable")
    ordered_squared32 = np.take_along_axis(squared32, order32, axis=1)
    gap32 = ordered_squared32[:, k] - ordered_squared32[:, k - 1]
    if np.any(gap32 < 0) or not np.isfinite(gap32).all():
        raise Round0201Error("R0201 fp32 diagnostic boundary is invalid")
    return np.ascontiguousarray(ordered_ids[:, :k]), {
        "distance_rerank_dtype": "float64-from-exact-float32-coordinates",
        "boundary_secondary_order": "canonical-row-id",
        "minimum_boundary_gap_squared_l2_float64": float(gap64.min()),
        "zero_boundary_gaps_float64": int(np.count_nonzero(gap64 == 0)),
        "minimum_boundary_gap_squared_l2_float32_diagnostic": float(
            gap32.min()
        ),
        "zero_boundary_gaps_float32_diagnostic": int(
            np.count_nonzero(gap32 == 0)
        ),
    }


def _float64_exact_low_fraction(
    coordinates: np.ndarray, anchor_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(coordinates, dtype=np.float32)
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    if (
        values.shape != (base.PILE_ROWS, 2)
        or anchors.shape != (base.ANCHORS,)
        or not np.isfinite(values).all()
    ):
        raise Round0201Error("R0201 Pile coordinate geometry changed")
    tree = base.cKDTree(values, compact_nodes=True, balanced_tree=True)
    _distances, candidates = tree.query(
        values[anchors],
        k=base.K_FRAC + base.TREE_OVERSELECT + 1,
        eps=0.0,
        p=2,
        workers=base.TREE_WORKERS,
    )
    filtered = np.empty(
        (base.ANCHORS, base.K_FRAC + base.TREE_OVERSELECT), dtype=np.int64
    )
    for row, anchor in enumerate(anchors):
        selected = candidates[row][candidates[row] != anchor]
        if len(selected) < base.K_FRAC + base.TREE_OVERSELECT:
            raise Round0201Error("R0201 cKDTree self exclusion did not close")
        filtered[row] = selected[: base.K_FRAC + base.TREE_OVERSELECT]
    neighbors, rerank = _rerank_candidates_float64(
        values, values[anchors], filtered, k=base.K_FRAC
    )
    receipt = {
        "algorithm": "scipy-cKDTree-exact-query-plus-float64-candidate-rerank",
        "scipy_version": base.scipy.__version__,
        "workers": base.TREE_WORKERS,
        "overselect": base.TREE_OVERSELECT,
        "boundary_tie_policy": (
            "float64 squared-L2 over exact float32 coordinates; fail on any "
            "remaining k567/k568 tie; canonical row ID is secondary order"
        ),
        "deterministic_membership_proved": True,
        **rerank,
    }
    _SEARCH_RECEIPTS.append(receipt)
    return neighbors, receipt


def _checked_synthesize(
    scores: Mapping[int, Mapping[str, np.ndarray]],
    labels: Mapping[int, np.ndarray],
    predictors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if len(_SEARCH_RECEIPTS) != len(_CELL_ORDER):
        raise Round0201Error("R0201 did not evaluate the exact six-cell family")
    observed = {
        cell: int(receipt["zero_boundary_gaps_float32_diagnostic"])
        for cell, receipt in zip(_CELL_ORDER, _SEARCH_RECEIPTS, strict=True)
    }
    if observed != EXPECTED_FP32_BOUNDARY_TIES:
        raise Round0201Error(
            f"R0201 fp32 boundary-tie diagnosis changed: {observed}"
        )
    result = synthesize(scores, labels, predictors)
    result["boundary_rerank_validation"] = {
        "cell_order": list(_CELL_ORDER),
        "expected_fp32_boundary_ties": dict(EXPECTED_FP32_BOUNDARY_TIES),
        "observed_fp32_boundary_ties": observed,
        "float64_zero_boundary_ties_all_cells": True,
    }
    return result


def _configure() -> None:
    _SEARCH_RECEIPTS.clear()
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.Round0194Error = Round0201Error
    base.synthesize = _checked_synthesize
    base._exact_low_fraction = _float64_exact_low_fraction


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure()
    base.run_job(active, job)


__all__ = [
    "EXPECTED_FP32_BOUNDARY_TIES",
    "_checked_synthesize",
    "_float64_exact_low_fraction",
    "_rerank_candidates_float64",
    "run_job",
]
