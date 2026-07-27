from __future__ import annotations

import inspect

from basemap import round0071_substrate
from experiments import prepare_round0071_queue, round0071_nodes


def test_registered_90m_intervals_are_balanced_and_disjoint() -> None:
    assert round0071_substrate.INTERVALS == (
        (0, 30_000_000),
        (50_000_000, 80_000_000),
        (100_000_000, 130_000_000),
    )
    assert sum(
        stop - start
        for start, stop in round0071_substrate.INTERVALS
    ) == round0071_substrate.ROW_COUNT
    assert round0071_substrate.ROWS_PER_CORPUS == 30_000_000


def test_registered_eligibility_accounting_closes() -> None:
    summary = round0071_substrate.ELIGIBILITY_SUMMARY
    assert (
        summary["retained_row_count"]
        + summary["excluded_row_count"]
        == round0071_substrate.ROW_COUNT
    )
    assert (
        summary["unique_nonzero_rows"]
        + summary["exact_nonzero_family_count"]
        == summary["retained_row_count"]
    )
    assert (
        summary["duplicate_copy_rows_excluded"]
        + summary["zero_row_count"]
        == summary["excluded_row_count"]
    )


def test_builder_recomputes_subset_families_and_never_trains() -> None:
    source = inspect.getsource(round0071_nodes.run_build_substrate)
    assert "write_subset_eligibility(" in source
    assert "intervals=INTERVALS" in source
    assert '"training_performed": False' in source
    assert '"optimizer_updates": 0' in source
    assert "model" not in source


def test_queue_has_one_cpu_node_and_no_scale_decision() -> None:
    source = inspect.getsource(prepare_round0071_queue.prepare_round0071)
    assert source.count('"action": "build_substrate"') == 1
    assert "gpu_hours_cap=0.0" in source
    assert '"gpu_required": False' in source
    assert '"no_scale_decision": True' in source
    assert '"no_training": True' in source
