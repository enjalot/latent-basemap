from __future__ import annotations

import numpy as np
import pytest
import torch

from basemap.round0196_grease_batch_stable import (
    INFERENCE_CHUNK_ROWS,
    Round0196Error,
    diagnose_execution,
    fixed_chunks,
)


def _execution(
    *,
    grease: float,
    both: float,
    grease_component: float = 0.0,
    both_component: float = 0.0,
) -> dict:
    grease_passes = max(grease_component, grease) <= 1e-4
    both_passes = max(both_component, both) <= 1e-4
    return {
        "schema": "round0196-grease-batch-stable-cpu-execution-v1",
        "device": "cpu",
        "source_checkpoint_round": "0181",
        "query_rows": 20_000,
        "dimension": 768,
        "probe_rows": INFERENCE_CHUNK_ROWS,
        "reload_tolerance": 1e-4,
        "candidates": {
            "baseline": {
                "grease_batch_max_abs_error": 0.001,
                "numap_batch_max_abs_error": 0.001,
            },
            "fixed_grease": {
                "grease_batch_max_abs_error": grease_component,
                "numap_batch_max_abs_error": grease,
            },
            "fixed_grease_and_pumap": {
                "grease_batch_max_abs_error": both_component,
                "numap_batch_max_abs_error": both,
            },
        },
        "selected_patch": (
            "fixed-256-row-grease-network"
            if grease_passes
            else (
                "fixed-256-row-grease-and-pumap-networks"
                if both_passes
                else None
            )
        ),
    }


def test_fixed_chunks_make_first_probe_call_identical() -> None:
    values = torch.arange(800 * 4, dtype=torch.float32).reshape(800, 4)

    def batch_shape_sensitive(cell: torch.Tensor) -> np.ndarray:
        return (cell + len(cell) / 1000).numpy()

    full = fixed_chunks(values, batch_shape_sensitive)
    probe = fixed_chunks(values[:INFERENCE_CHUNK_ROWS], batch_shape_sensitive)
    np.testing.assert_array_equal(full[:INFERENCE_CHUNK_ROWS], probe)


def test_minimal_passing_patch_is_selected() -> None:
    grease = diagnose_execution(_execution(grease=1e-6, both=0.0))
    assert grease["passed"] is True
    assert grease["selected_patch"] == "fixed-256-row-grease-network"
    both = diagnose_execution(_execution(grease=2e-4, both=0.0))
    assert both["passed"] is True
    assert both["selected_patch"] == "fixed-256-row-grease-and-pumap-networks"


def test_failed_candidates_activate_only_negative_closure() -> None:
    decision = diagnose_execution(_execution(grease=2e-4, both=3e-4))
    assert decision["passed"] is False
    assert decision["f2_gpu_baseline_activated"] is False
    assert decision["f3_negative_closure_activated"] is True
    assert decision["additional_debug_or_f4_authorized"] is False


def test_numap_stability_cannot_hide_grease_stage_drift() -> None:
    decision = diagnose_execution(
        _execution(
            grease=0.0,
            both=0.0,
            grease_component=2e-4,
            both_component=3e-4,
        )
    )
    assert decision["passed"] is False
    assert decision["selected_patch"] is None
    assert decision["f3_negative_closure_activated"] is True


def test_unreproduced_baseline_cannot_activate_gpu() -> None:
    value = _execution(grease=0.0, both=0.0)
    value["candidates"]["baseline"]["numap_batch_max_abs_error"] = 0.0
    decision = diagnose_execution(value)
    assert decision["passed"] is False
    assert decision["baseline_failure_reproduced"] is False
    assert decision["f2_gpu_baseline_activated"] is False


def test_inconsistent_selection_fails_closed() -> None:
    value = _execution(grease=1e-6, both=0.0)
    value["selected_patch"] = "fixed-256-row-grease-and-pumap-networks"
    with pytest.raises(Round0196Error, match="minimal"):
        diagnose_execution(value)


@pytest.mark.parametrize(
    ("field", "changed"),
    [("dimension", 384), ("reload_tolerance", 1e-3)],
)
def test_execution_geometry_and_guard_are_frozen(field: str, changed: float) -> None:
    value = _execution(grease=0.0, both=0.0)
    value[field] = changed
    with pytest.raises(Round0196Error, match="contract changed"):
        diagnose_execution(value)
