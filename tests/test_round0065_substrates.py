from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from basemap.round0049_program import compact_to_global, global_to_compact
from basemap.round0065_substrates import (
    SUBSETS,
    Round0065Error,
    subset_spec,
)


@pytest.mark.parametrize("tier", ["45m", "120m"])
def test_registered_subset_mappings_are_balanced_and_bijective(
    tier: str,
) -> None:
    spec = subset_spec(tier)
    width = spec["first_rows_per_corpus"]
    compact = np.asarray(
        [
            0,
            width - 1,
            width,
            2 * width - 1,
            2 * width,
            3 * width - 1,
        ],
        dtype=np.int64,
    )
    global_rows = compact_to_global(
        compact,
        intervals=spec["intervals"],
    )
    assert global_rows.tolist() == [
        0,
        width - 1,
        50_000_000,
        50_000_000 + width - 1,
        100_000_000,
        100_000_000 + width - 1,
    ]
    np.testing.assert_array_equal(
        global_to_compact(
            global_rows,
            intervals=spec["intervals"],
        ),
        compact,
    )


def test_registered_eligibility_accounting_closes() -> None:
    for spec in SUBSETS.values():
        summary = spec["eligibility_summary"]
        assert (
            summary["retained_row_count"]
            + summary["excluded_row_count"]
            == spec["row_count"]
        )
        assert (
            summary["duplicate_copy_rows_excluded"]
            + summary["zero_row_count"]
            == summary["excluded_row_count"]
        )
        assert (
            summary["unique_nonzero_rows"]
            + summary["exact_nonzero_family_count"]
            == summary["retained_row_count"]
        )


def test_unknown_subset_is_rejected() -> None:
    with pytest.raises(Round0065Error):
        subset_spec("150m")


def test_round0065_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0065_substrates")
    importlib.import_module("experiments.round0065_nodes")
    importlib.import_module("experiments.prepare_round0065_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_round0065_is_cpu_only_and_does_not_choose_a_scale() -> None:
    from experiments import prepare_round0065_queue

    source = inspect.getsource(
        prepare_round0065_queue.prepare_round0065
    )
    assert "gpu_hours_cap=0.0" in source
    assert 'execution_authority="autonomous-cpu"' in source
    assert 'gpu=False' in source
    assert '"no_scale_decision": True' in source
    assert '"45m"' in source
    assert '"120m"' in source
    assert "candidate" not in "\n".join(
        line
        for line in source.splitlines()
        if '"no_candidate_search"' not in line
    )
