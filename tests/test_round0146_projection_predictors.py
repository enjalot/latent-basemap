"""CUDA-hidden tests for R0146's frozen CPU predictor analysis."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0142_jina_universality import MAP_ORDER, PROBE_ORDER
from basemap.round0146_projection_predictors import (
    PREDICTOR_ORDER,
    Round0146Error,
    correlation_table,
    geometry_predictors,
    spearman_rho,
    support_distance_predictor,
    systematic_positions,
)


def _unit_random(rows: int, seed: int) -> np.ndarray:
    values = np.random.RandomState(seed).normal(size=(rows, 768)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def test_systematic_positions_are_unique_bounded_and_seeded() -> None:
    first = systematic_positions(24_948_663, 8_192, seed=17)
    same = systematic_positions(24_948_663, 8_192, seed=17)
    other = systematic_positions(24_948_663, 8_192, seed=18)
    assert np.array_equal(first, same)
    assert not np.array_equal(first, other)
    assert len(first) == len(np.unique(first)) == 8_192
    assert first[0] >= 0 and first[-1] < 24_948_663


@pytest.mark.parametrize(
    ("length", "count"), [(0, 1), (10, 0), (10, 11)]
)
def test_systematic_positions_reject_invalid_geometry(length: int, count: int) -> None:
    with pytest.raises(Round0146Error, match="sample geometry"):
        systematic_positions(length, count, seed=1)


def test_geometry_predictors_are_finite_and_exactly_accounted() -> None:
    values = _unit_random(256, 4)
    result = geometry_predictors(
        values, source_row_ids=np.arange(10_000, 10_256), label="synthetic"
    )
    assert result["sample"]["sample_rows"] == 256
    assert result["twonn"]["valid_rows"] >= 230
    assert result["twonn"]["intrinsic_dimension"] > 0
    assert np.isfinite(result["hubness"]["skew"])
    assert result["hubness"]["mean_occurrence"] == pytest.approx(10.0)
    assert result["anisotropy"]["eigen_ratio"] >= 1.0


def test_geometry_predictors_reject_changed_row_identity() -> None:
    with pytest.raises(Round0146Error, match="source-row IDs"):
        geometry_predictors(
            _unit_random(64, 5), source_row_ids=np.arange(63), label="broken"
        )


def test_support_distance_reports_exact_nearest_cosine() -> None:
    support = np.zeros((16, 768), dtype=np.float32)
    support[np.arange(16), np.arange(16)] = 1.0
    queries = support[[0, 3, 7, 12]].copy()
    result = support_distance_predictor(queries, support, label="exact")
    assert result["query_rows"] == 4
    assert result["support_rows"] == 16
    assert result["minimum"] == pytest.approx(0.0)
    assert result["maximum"] == pytest.approx(0.0)


def test_support_distance_rejects_non_normalized_rows() -> None:
    queries = _unit_random(8, 1)
    support = _unit_random(16, 2) * 0.1
    with pytest.raises(Round0146Error, match="normalization guard"):
        support_distance_predictor(queries, support, label="bad")


def test_spearman_uses_average_tie_ranks() -> None:
    assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rho([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)


def test_correlation_table_closes_full_map_probe_matrix() -> None:
    cells = []
    for map_index, map_key in enumerate(MAP_ORDER):
        for probe_index, probe in enumerate(PROBE_ORDER):
            value = float(probe_index + 1)
            cells.append({
                "map": map_key,
                "probe": probe,
                "ffr_retention": -value - 0.1 * map_index,
                "recall10_retention": -2.0 * value - 0.1 * map_index,
                **{predictor: value for predictor in PREDICTOR_ORDER},
            })
    table = correlation_table(cells)
    assert len(table) == 2 * 3 * len(PREDICTOR_ORDER)
    assert all(
        row["spearman_rho"] == pytest.approx(-1.0)
        for row in table
        if row["scope"] != "pooled-descriptive"
    )
    assert all(row["spearman_rho"] < 0 for row in table)
    assert all(row["direction_consistent"] is True for row in table)


def test_correlation_table_rejects_missing_cell() -> None:
    cells = []
    for map_key in MAP_ORDER:
        for probe_index, probe in enumerate(PROBE_ORDER):
            value = float(probe_index + 1)
            cells.append({
                "map": map_key,
                "probe": probe,
                "ffr_retention": value,
                "recall10_retention": value,
                **{predictor: value for predictor in PREDICTOR_ORDER},
            })
    with pytest.raises(Round0146Error, match="matrix is incomplete"):
        correlation_table(cells[:-1])
