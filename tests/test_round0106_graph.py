from __future__ import annotations

import numpy as np
import pytest

from basemap.round0106_graph import (
    K,
    PARTS,
    RETAINED_ROWS,
    Round0106Error,
    compact_to_global,
    global_to_compact,
    validate_part_specs,
)
from experiments.build_weighted_graph import (
    fuzzy_directed_from_knn,
    symmetrize_bucket,
)
from experiments.round0106_nodes import _validate_joined_reciprocity


def test_compact_mapping_handles_contiguous_exclusion_runs():
    excluded = np.asarray([0, 1, 5, 6, 7, 100], dtype=np.int64)
    compact = np.asarray([0, 1, 2, 3, 4, 94, 95, 96], dtype=np.int64)
    expected = np.asarray([2, 3, 4, 8, 9, 99, 101, 102], dtype=np.int64)
    observed = compact_to_global(compact, excluded)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        global_to_compact(observed, excluded), compact
    )


def test_global_to_compact_rejects_excluded_rows():
    with pytest.raises(Round0106Error, match="excluded"):
        global_to_compact(
            np.asarray([7], dtype=np.int64),
            np.asarray([7], dtype=np.int64),
        )


def test_parts_close_exact_universe_and_degree():
    validate_part_specs()
    assert sum(value["retained_rows"] for value in PARTS.values()) == RETAINED_ROWS
    assert sum(value["retained_rows"] * K for value in PARTS.values()) == (
        RETAINED_ROWS * K
    )


def test_fuzzy_directed_rows_and_tconorm_are_reciprocal():
    indices = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0],
        ],
        dtype=np.int32,
    )
    distances = np.asarray(
        [
            [0.0, 0.1, 0.4],
            [0.0, 0.2, 0.3],
            [0.0, 0.15, 0.5],
        ],
        dtype=np.float32,
    )
    sources, targets, weights, *_ = fuzzy_directed_from_knn(
        indices, distances, 3.0, local_connectivity=1.0
    )
    np.testing.assert_array_equal(sources, np.repeat(np.arange(3), 2))
    joined_sources, joined_targets, joined_weights = symmetrize_bucket(
        sources, targets, weights, 3
    )
    keys = joined_sources.astype(np.int64) * 3 + joined_targets
    reverse = joined_targets.astype(np.int64) * 3 + joined_sources
    positions = np.searchsorted(keys, reverse)
    np.testing.assert_array_equal(keys[positions], reverse)
    np.testing.assert_array_equal(joined_weights[positions], joined_weights)


def test_joined_reciprocity_full_scan(tmp_path):
    joined = tmp_path / "joined"
    joined.mkdir()
    np.savez(
        joined / "part-0000.npz",
        sources=np.asarray([0, 1, 1, 2], dtype=np.int32),
        targets=np.asarray([1, 0, 2, 1], dtype=np.int32),
        weights=np.asarray([0.5, 0.5, 0.75, 0.75], dtype=np.float32),
    )
    receipt = _validate_joined_reciprocity(
        joined=str(joined), counts=[4]
    )
    assert receipt["directed_edges_checked"] == 4
    assert receipt["every_reverse_present_once"] is True


def test_joined_reciprocity_rejects_missing_reverse(tmp_path):
    joined = tmp_path / "joined"
    joined.mkdir()
    np.savez(
        joined / "part-0000.npz",
        sources=np.asarray([0], dtype=np.int32),
        targets=np.asarray([1], dtype=np.int32),
        weights=np.asarray([0.5], dtype=np.float32),
    )
    with pytest.raises(Round0106Error, match="reverse"):
        _validate_joined_reciprocity(joined=str(joined), counts=[1])
