from __future__ import annotations

import numpy as np
import pytest

from experiments import prepare_round0151_queue as queue_prep
from basemap.round0105_search import GROUPS
from basemap.round0151_scale_census import (
    Round0151Error,
    build_prefix_drop_mapping,
    compare_to_u12,
    inventory_group_ranges,
    largest_remainder_prefix_quotas,
)


def toy_ranges(rows_per_group: int = 4) -> dict[str, tuple[int, int]]:
    return {
        group: (index * rows_per_group, (index + 1) * rows_per_group)
        for index, group in enumerate(GROUPS)
    }


def toy_selection(rows_per_group: int = 4) -> dict[str, object]:
    ranges = toy_ranges(rows_per_group)
    datasets = [
        group if index < 3 else f"fineweb2-{group}-chunked-500-jina-v5-nano"
        for index, group in enumerate(GROUPS)
    ]
    return {
        "source_order": datasets,
        "ranges": [
            {
                "dataset": dataset,
                "global_row_start": ranges[group][0],
                "global_row_stop": ranges[group][1],
            }
            for group, dataset in zip(GROUPS, datasets, strict=True)
        ],
    }


def test_inventory_group_ranges_reconstructs_registered_order() -> None:
    assert inventory_group_ranges(toy_selection(), expected_rows=88) == toy_ranges()
    changed = toy_selection()
    changed["source_order"] = list(reversed(changed["source_order"]))
    with pytest.raises(Round0151Error, match="order changed"):
        inventory_group_ranges(changed, expected_rows=88)


def test_prefix_quotas_close_with_registered_tie_break() -> None:
    counts = {group: 4 for group in GROUPS}
    quotas = largest_remainder_prefix_quotas(counts, target=45)
    assert sum(quotas.values()) == 45
    assert [quotas[group] for group in GROUPS[:2]] == [3, 2]
    assert all(quotas[group] == 2 for group in GROUPS[2:])


def test_prefix_drop_mapping_drops_without_backfill() -> None:
    ranges = toy_ranges()
    excluded = np.asarray([1, 8, 9, 87], dtype=np.int64)
    mapping, ids, census = build_prefix_drop_mapping(
        ranges, excluded, target=44
    )
    assert census["raw_prefix_target"] == 44
    assert census["dropped_rows"] == 3
    assert census["retained_rows"] == 41
    assert census["replacement_rows"] == 0
    assert mapping.tolist()[:4] == [0, 4, 5, 12]
    assert 1 not in mapping and 8 not in mapping and 9 not in mapping
    assert census["groups"][GROUPS[-1]]["dropped_rows"] == 0
    assert ids.shape == mapping.shape
    assert np.all(mapping[1:] > mapping[:-1])


def test_prefix_drop_rejects_noncontiguous_groups() -> None:
    ranges = toy_ranges()
    start, stop = ranges[GROUPS[1]]
    ranges[GROUPS[1]] = (start + 1, stop)
    with pytest.raises(Round0151Error, match="not contiguous"):
        build_prefix_drop_mapping(ranges, np.empty(0, dtype=np.int64), target=44)


def test_u12_comparison_reports_exact_overlap_and_distinctness() -> None:
    comparison = compare_to_u12(
        np.asarray([0, 1, 4, 7], dtype=np.int64),
        np.asarray([0, 2, 4, 8], dtype=np.int64),
    )
    assert comparison == {
        "candidate_rows": 4,
        "u12_rows": 4,
        "row_count_delta": 0,
        "overlap_rows": 2,
        "candidate_only_rows": 2,
        "u12_only_rows": 2,
        "jaccard": 1 / 3,
        "byte_or_set_identical": False,
        "distinct": True,
    }


def test_queue_activation_rejects_nonpositive_r0150(monkeypatch) -> None:
    monkeypatch.setattr(queue_prep, "_accepted_review", lambda *_args: [])
    monkeypatch.setattr(
        queue_prep,
        "_read_sealed",
        lambda *_args, **_kwargs: (
            {
                "round_id": "0150",
                "capability": "jina-2m-drop-only-seed-replication-v1",
                "outcome": "drop-only-restoration-is-seed-sensitive-or-control-inconclusive",
                "drop_only_scale_candidate_released": False,
            },
            {},
        ),
    )
    with pytest.raises(RuntimeError, match="positive R0150 activation"):
        queue_prep._accepted_activation()
