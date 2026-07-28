from __future__ import annotations

import inspect

from basemap.round0086_program import (
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    RETAINED_ROWS,
    ROW_COUNT,
    select_cell,
)
from experiments import (
    prepare_round0086_queue,
    round0081_nodes,
    round0086_nodes,
)


def test_fixed_grid_starts_from_confirmed_120m_policy() -> None:
    assert POLICY_GRID == (
        (128, 256),
        (192, 256),
        (256, 256),
        (128, 384),
        (192, 384),
        (256, 384),
        (128, 512),
        (192, 512),
        (256, 512),
    )
    assert MEAN_RECALL_FLOOR == 0.90
    assert ROW_COUNT == 150_000_000
    assert RETAINED_ROWS == 147_221_757


def test_selector_uses_measured_wall_not_grid_order() -> None:
    cells = {}
    for index, (nprobe, width) in enumerate(POLICY_GRID):
        cells[f"nprobe-{nprobe}-width-{width}"] = {
            "nprobe": nprobe,
            "shortlist_width": width,
            "passes_mean_floor": index in {0, 4, 8},
            "benchmark": {
                "median_wall_seconds_per_query": {
                    0: 0.003,
                    4: 0.001,
                    8: 0.002,
                }.get(index, 1.0),
            },
        }
    selected = select_cell({"cells": cells})
    assert selected is cells["nprobe-192-width-384"]


def test_queue_is_reference_stage_filter_then_fixed_qualification() -> None:
    source = inspect.getsource(prepare_round0086_queue.prepare_round0086)
    assert source.count('action="stage"') == 1
    assert source.count('action="filter"') == 1
    assert source.count('action="qualify"') == 1
    assert "gpu_hours_cap=0.5" in source
    assert '"substrate_is_reference_only_no_payload_copy": True' in source
    assert '"r0083_does_not_change_floor_in_place": True' in source
    assert '"sample_seed": 86' in source


def test_historical_dependencies_match_actual_consumers() -> None:
    source = inspect.getsource(prepare_round0086_queue.prepare_round0086)
    for round_id in ("0025", "0033", "0049", "0080", "0082"):
        assert f'"{round_id}"' in source
    for capability in (
        "minilm-int8-shards-v1",
        "minilm-150m-row-eligibility-v1",
        "minilm-balanced-60m-candidate-quality-v1",
        "minilm-balanced-90m-120m-scale-geometry-v1",
        "minilm-balanced-120m-gpu-ivfpq-search-confirmed-v1",
    ):
        assert capability in source


def test_qualification_late_binds_only_own_producer_outputs() -> None:
    source = inspect.getsource(round0086_nodes.run_qualification)
    assert 'filter_receipt.get("release_sha")' in source
    assert 'filter_receipt.get("filtered_index") != filtered' in source
    assert '"substrate_manifest_sha256"' in source
    assert '"filtered_index_sha256"' in source
    assert "finally:" in source


def test_shared_qualification_labels_tier_dynamically() -> None:
    source = inspect.getsource(round0081_nodes.run_qualification)
    assert 'f"fixed_registered_{TIER}_universe"' in source
    assert 'f"balanced-{TIER} policy qualification failed: "' in source
