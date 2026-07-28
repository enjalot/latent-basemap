from __future__ import annotations

import numpy as np

from basemap.round0096_larger_nlist import (
    GLOBAL_MEAN_FLOOR,
    NLIST,
    PER_CORPUS_MEAN_FLOOR,
    POLICY_GRID,
    TRAIN_ROWS,
    select_cell,
)
from experiments.round0096_nodes import (
    _normalized_rows,
    _policy_metrics,
    _retained_batch,
)


def test_registered_geometry_is_one_deliberate_four_x_step() -> None:
    assert NLIST == 4 * 8_192
    assert TRAIN_ROWS == 40 * NLIST
    assert len(POLICY_GRID) == 16
    assert len(set(POLICY_GRID)) == len(POLICY_GRID)
    assert max(width for _nprobe, width in POLICY_GRID) + 1 == 2_048
    assert GLOBAL_MEAN_FLOOR == 0.90
    assert PER_CORPUS_MEAN_FLOOR == 0.84


def test_retained_batch_excludes_only_registered_rows() -> None:
    excluded = np.array([2, 5, 9, 10, 18], dtype=np.int64)
    assert _retained_batch(excluded, start=4, stop=12).tolist() == [
        4, 6, 7, 8, 11,
    ]
    assert _retained_batch(excluded, start=12, stop=18).tolist() == list(
        range(12, 18)
    )


def test_normalized_rows_are_finite_unit_vectors() -> None:
    encoded = np.array(
        [[3, 4, 0], [0, -5, 12], [1, 1, 1]], dtype=np.int8,
    )
    values = _normalized_rows(
        encoded, np.array([2, 0], dtype=np.int64),
    )
    assert values.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(values, axis=1), np.ones(2), atol=1e-6,
    )


def test_policy_metrics_require_every_corpus_not_only_global() -> None:
    sample = np.array(
        [1, 2, 50_000_001, 50_000_002, 100_000_001, 100_000_002],
        dtype=np.int64,
    )
    exact = np.tile(np.arange(15, dtype=np.int32), (6, 1))
    selected = exact.copy()
    selected[0, :4] = np.arange(100, 104)
    selected[1, :4] = np.arange(110, 114)
    metrics = _policy_metrics(
        selected,
        exact,
        sample=sample,
        unambiguous=np.ones(6, dtype=bool),
    )
    assert metrics["mean_recall_at_15_unambiguous"] > 0.90
    assert metrics["passes_global_floor"] is True
    assert metrics["by_corpus"]["fineweb"]["passes_floor"] is False
    assert metrics["passes_every_corpus_floor"] is False


def test_selector_uses_only_dual_passing_benchmarked_cells() -> None:
    cells = {
        "global-only": {
            "nprobe": 128,
            "shortlist_width": 512,
            "passes_global_floor": True,
            "passes_every_corpus_floor": False,
            "benchmark": {"median_wall_seconds_per_query": 0.001},
        },
        "passing-slow": {
            "nprobe": 256,
            "shortlist_width": 1024,
            "passes_global_floor": True,
            "passes_every_corpus_floor": True,
            "benchmark": {"median_wall_seconds_per_query": 0.003},
        },
        "passing-fast": {
            "nprobe": 512,
            "shortlist_width": 512,
            "passes_global_floor": True,
            "passes_every_corpus_floor": True,
            "benchmark": {"median_wall_seconds_per_query": 0.002},
        },
    }
    assert select_cell(cells) is cells["passing-fast"]
