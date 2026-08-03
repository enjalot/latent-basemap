"""Frozen design helpers for the R0175 approximate-UMAP OOS baseline."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .panel_v2 import ffr_from_neighbors, recall_at_k_from_neighbors


ROUND_ID = "0175"
CAPABILITY = "jina-aumap-oos-baseline-v1"
SCALES = ("200k", "500k", "2m")
ROWS = {"200k": 200_000, "500k": 500_000, "2m": 2_000_000}
HELD_HASHES = {
    "200k": "0e81ac067567",
    "500k": "7d94f88eb0bc",
    "2m": "cd1208a56d17",
}
N_QUERIES = 20_000
NEIGHBORS = 15
K_HIT = 10
FRAC = 0.001
SEED = 123
EPSILON = 1.0e-8


class Round0175Error(RuntimeError):
    """The registered aUMAP baseline contract was violated."""


def project_coordinates(
    teacher: np.ndarray,
    neighbor_ids: np.ndarray,
    cosine_distances: np.ndarray,
    *,
    weighted: bool,
) -> np.ndarray:
    """Apply the official approx-umap v0.2.0 inverse-distance rule."""
    teacher = np.asarray(teacher, dtype=np.float32)
    neighbor_ids = np.asarray(neighbor_ids)
    distances = np.asarray(cosine_distances, dtype=np.float32)
    if (
        teacher.ndim != 2
        or teacher.shape[1] != 2
        or neighbor_ids.ndim != 2
        or neighbor_ids.shape != distances.shape
        or neighbor_ids.shape[1] != NEIGHBORS
        or neighbor_ids.dtype.kind not in "iu"
        or neighbor_ids.size == 0
        or int(neighbor_ids.min()) < 0
        or int(neighbor_ids.max()) >= len(teacher)
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
    ):
        raise Round0175Error("aUMAP projection inputs are malformed")
    neighbor_coordinates = teacher[neighbor_ids]
    if weighted:
        similarity = 1.0 / (distances + EPSILON)
        denominator = similarity.sum(axis=1, keepdims=True)
        if not np.isfinite(similarity).all() or np.any(denominator <= 0):
            raise Round0175Error("aUMAP inverse-distance weights are invalid")
        projected = np.sum(
            (similarity / denominator)[:, :, None] * neighbor_coordinates,
            axis=1,
        )
    else:
        projected = neighbor_coordinates.mean(axis=1)
    projected = np.asarray(projected, dtype=np.float32)
    if projected.shape != (len(neighbor_ids), 2) or not np.isfinite(projected).all():
        raise Round0175Error("aUMAP projected coordinates are malformed")
    return projected


def projection_metrics(
    high_truth: np.ndarray,
    low_neighbors: np.ndarray,
) -> dict[str, float]:
    high_truth = np.asarray(high_truth)
    low_neighbors = np.asarray(low_neighbors)
    if (
        high_truth.ndim != 2
        or high_truth.shape[1] < K_HIT
        or low_neighbors.ndim != 2
        or low_neighbors.shape[0] != high_truth.shape[0]
        or low_neighbors.shape[1] < K_HIT
    ):
        raise Round0175Error("projection neighbor panels are malformed")
    high = high_truth[:, :K_HIT]
    return {
        "ffr": float(ffr_from_neighbors(high, low_neighbors, K_HIT)),
        "recall_at_10": float(
            recall_at_k_from_neighbors(high, low_neighbors, K_HIT)
        ),
    }


def build_synthesis(
    *,
    official_probe: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
    historical_context: Mapping[str, Any],
) -> dict[str, Any]:
    if set(cells) != set(SCALES):
        raise Round0175Error("R0175 scale cells are missing or unexpected")
    if (
        official_probe.get("package") != "approx-umap==0.2.0"
        or official_probe.get("max_abs_error", 1.0) > 1.0e-6
        or official_probe.get("passed") is not True
    ):
        raise Round0175Error("official aUMAP formula probe did not qualify")
    summaries: dict[str, Any] = {}
    for scale in SCALES:
        cell = cells[scale]
        if (
            cell.get("scale") != scale
            or cell.get("rows") != ROWS[scale]
            or cell.get("held_hash") != HELD_HASHES[scale]
            or cell.get("n_queries") != N_QUERIES
            or cell.get("neighbors") != NEIGHBORS
            or cell.get("guards_passed") is not True
        ):
            raise Round0175Error(f"R0175 {scale} cell contract changed")
        weighted = cell.get("aumap_inverse_distance")
        unweighted = cell.get("unweighted_knn15")
        high_performance = cell.get("high_search_performance")
        coordinate_performance = cell.get("coordinate_projection_performance")
        rank1_diagnostics = (
            high_performance.get("rank1_cosine_distance_diagnostics")
            if isinstance(high_performance, Mapping)
            else None
        )
        if (
            not isinstance(weighted, Mapping)
            or not isinstance(unweighted, Mapping)
            or not isinstance(high_performance, Mapping)
            or not isinstance(coordinate_performance, Mapping)
            or not isinstance(rank1_diagnostics, Mapping)
        ):
            raise Round0175Error(f"R0175 {scale} metrics are absent")
        summaries[scale] = {
            "aumap_inverse_distance": dict(weighted),
            "unweighted_knn15": dict(unweighted),
            "delta_weighted_minus_unweighted": {
                metric: float(weighted[metric]) - float(unweighted[metric])
                for metric in ("ffr", "recall_at_10")
            },
            "execution_performance": {
                "high_search": dict(high_performance),
                "coordinate_projection": dict(coordinate_performance),
                "low_search_seconds_both_methods": float(
                    cell["low_search_seconds_both_methods"]
                ),
                "total_wall_seconds": float(cell["total_wall_seconds"]),
                "peak_rss_gib": float(cell["peak_rss_gib"]),
            },
            "historical_parametric_context": historical_context.get(scale),
        }
    return {
        "schema": "round0175-aumap-oos-synthesis-v1",
        "round_id": ROUND_ID,
        "outcome": "aumap-oos-baseline-measured",
        "official_formula_probe": dict(official_probe),
        "scales": summaries,
        "selector": "execution-validity only; every quality contrast is diagnostic",
        "teacher_role": (
            "frozen standard-UMAP transductive coordinates; R0175 measures only "
            "the non-parametric out-of-sample transform"
        ),
        "search_adapter": (
            "exact fp32 cosine IndexFlatIP; official approx-umap v0.2.0 "
            "inverse-distance weighting applied to the returned k15 rows"
        ),
        "map_or_training_claim": False,
        "production_or_publishing": False,
        "capabilities": [CAPABILITY],
    }


__all__ = [
    "CAPABILITY",
    "EPSILON",
    "FRAC",
    "HELD_HASHES",
    "K_HIT",
    "NEIGHBORS",
    "N_QUERIES",
    "ROUND_ID",
    "ROWS",
    "SEED",
    "SCALES",
    "Round0175Error",
    "build_synthesis",
    "project_coordinates",
    "projection_metrics",
]
