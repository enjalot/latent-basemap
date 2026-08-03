"""Contract tests for the R0175 approximate-UMAP OOS baseline."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0175_aumap_baseline import (
    HELD_HASHES,
    NEIGHBORS,
    N_QUERIES,
    ROWS,
    SCALES,
    Round0175Error,
    build_synthesis,
    project_coordinates,
    projection_metrics,
)


def test_inverse_distance_projection_matches_registered_formula() -> None:
    teacher = np.arange(80, dtype=np.float32).reshape(40, 2)
    ids = np.tile(np.arange(NEIGHBORS, dtype=np.int32), (3, 1))
    distances = np.tile(
        np.linspace(0.05, 0.75, NEIGHBORS, dtype=np.float32), (3, 1)
    )
    observed = project_coordinates(teacher, ids, distances, weighted=True)
    weights = 1.0 / (distances + 1.0e-8)
    expected = np.sum(
        (weights / weights.sum(axis=1, keepdims=True))[:, :, None] * teacher[ids],
        axis=1,
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1.0e-6)
    unweighted = project_coordinates(teacher, ids, distances, weighted=False)
    np.testing.assert_allclose(unweighted, teacher[ids].mean(axis=1))


def test_projection_rejects_bad_neighbor_ids() -> None:
    teacher = np.zeros((20, 2), dtype=np.float32)
    ids = np.zeros((2, NEIGHBORS), dtype=np.int32)
    ids[0, 0] = 20
    with pytest.raises(Round0175Error):
        project_coordinates(
            teacher, ids, np.ones_like(ids, dtype=np.float32), weighted=True
        )


def test_projection_metrics_use_shared_neighbor_semantics() -> None:
    high = np.tile(np.arange(NEIGHBORS, dtype=np.int32), (8, 1))
    low = np.tile(np.arange(25, dtype=np.int32), (8, 1))
    metrics = projection_metrics(high, low)
    assert metrics == {"ffr": 1.0, "recall_at_10": 1.0}


def test_synthesis_accepts_only_complete_execution_valid_cells() -> None:
    cells = {
        scale: {
            "scale": scale,
            "rows": ROWS[scale],
            "held_hash": HELD_HASHES[scale],
            "n_queries": N_QUERIES,
            "neighbors": NEIGHBORS,
            "guards_passed": True,
            "aumap_inverse_distance": {"ffr": 0.5, "recall_at_10": 0.02},
            "unweighted_knn15": {"ffr": 0.4, "recall_at_10": 0.01},
        }
        for scale in SCALES
    }
    synthesis = build_synthesis(
        official_probe={
            "package": "approx-umap==0.2.0",
            "max_abs_error": 0.0,
            "passed": True,
        },
        cells=cells,
        historical_context={"200k": {}, "500k": None, "2m": {}},
    )
    assert synthesis["outcome"] == "aumap-oos-baseline-measured"
    assert synthesis["scales"]["2m"]["delta_weighted_minus_unweighted"] == {
        "ffr": pytest.approx(0.1),
        "recall_at_10": pytest.approx(0.01),
    }
    broken = dict(cells)
    broken.pop("500k")
    with pytest.raises(Round0175Error):
        build_synthesis(
            official_probe={
                "package": "approx-umap==0.2.0",
                "max_abs_error": 0.0,
                "passed": True,
            },
            cells=broken,
            historical_context={},
        )
