from __future__ import annotations

import numpy as np
import pytest

from basemap.round0106_graph import (
    K,
    MINIMUM_SHARD_SOURCES_PER_SECOND,
    PARTS,
    RETAINED_ROWS,
    Round0106Error,
    compact_to_global,
    global_to_compact,
    update_performance_streak,
    validate_part_specs,
)
from experiments.build_weighted_graph import (
    fuzzy_directed_from_knn,
    symmetrize_bucket,
)
from experiments.round0106_nodes import (
    _validate_directed_memberships,
    _validate_joined_reciprocity,
)
from experiments.prepare_round0106_queue import (
    DECISION,
    INDEX,
    INDEX_RECEIPT,
    QUALIFICATION,
    R0105_ROOT,
)


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


def test_search_inputs_bind_successful_r0105_attempt():
    assert R0105_ROOT.endswith("/round-0105/queue-attempt-3/artifacts")
    assert INDEX.startswith(R0105_ROOT + "/")
    assert INDEX_RECEIPT.startswith(R0105_ROOT + "/")
    assert QUALIFICATION.startswith(R0105_ROOT + "/")
    assert DECISION.startswith(R0105_ROOT + "/")


def test_performance_streak_ignores_warmup_and_resets_on_recovery():
    assert update_performance_streak(
        0,
        completed_new_shards=1,
        sources_per_second=1.0,
    ) == 0
    streak = update_performance_streak(
        0,
        completed_new_shards=2,
        sources_per_second=MINIMUM_SHARD_SOURCES_PER_SECOND - 1,
    )
    assert streak == 1
    assert update_performance_streak(
        streak,
        completed_new_shards=3,
        sources_per_second=MINIMUM_SHARD_SOURCES_PER_SECOND - 1,
    ) == 2
    assert update_performance_streak(
        streak,
        completed_new_shards=3,
        sources_per_second=MINIMUM_SHARD_SOURCES_PER_SECOND,
    ) == 0
    with pytest.raises(Round0106Error, match="throughput"):
        update_performance_streak(
            0,
            completed_new_shards=2,
            sources_per_second=float("nan"),
        )


def test_fuzzy_membership_closure_allows_fp32_zero_elimination():
    rows = np.asarray([10, 11], dtype=np.int64)
    all_targets = np.asarray(
        [
            np.arange(20, 20 + K),
            np.arange(40, 40 + K),
        ],
        dtype=np.int32,
    )
    sources = np.concatenate(
        [
            np.full(K - 5, 10, dtype=np.int32),
            np.full(K, 11, dtype=np.int32),
        ]
    )
    targets = np.concatenate(
        [all_targets[0, : K - 5], all_targets[1]]
    ).astype(np.int32)
    weights = np.linspace(
        np.nextafter(np.float32(0), np.float32(1)),
        np.float32(1),
        len(sources),
        dtype=np.float32,
    )
    closure = _validate_directed_memberships(
        rows=rows,
        all_targets=all_targets,
        sources=sources,
        targets=targets,
        weights=weights,
    )
    assert closure == {
        "knn_edges": 2 * K,
        "directed_edges": 2 * K - 5,
        "zero_memberships_eliminated": 5,
        "sources_with_eliminated_memberships": 1,
        "minimum_memberships_per_source": K - 5,
    }


def test_fuzzy_membership_closure_rejects_non_knn_target():
    rows = np.asarray([10], dtype=np.int64)
    all_targets = np.arange(20, 20 + K, dtype=np.int32)[None, :]
    with pytest.raises(Round0106Error, match="membership"):
        _validate_directed_memberships(
            rows=rows,
            all_targets=all_targets,
            sources=np.full(K, 10, dtype=np.int32),
            targets=np.r_[all_targets[0, :-1], 999].astype(np.int32),
            weights=np.ones(K, dtype=np.float32),
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
