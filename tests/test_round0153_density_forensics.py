"""CUDA-hidden tests for R0153 density-v2 forensics."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0153_density_forensics import (
    CURRENT_POPULATION_REFERENCES,
    HISTORICAL_ROW_CELLS,
    REGISTERED_FLOOR,
    Round0153Error,
    classify_density_branch,
    diagnostic_values,
)
from experiments import prepare_round0153_queue


def _historical(value: float) -> dict[str, float]:
    return {key: value for key in HISTORICAL_ROW_CELLS}


def _current(value: float) -> dict[str, float]:
    return {key: value for key in CURRENT_POPULATION_REFERENCES}


def test_cpu_queue_uses_independent_execution_checkout() -> None:
    assert prepare_round0153_queue.RELEASE_ROOT.endswith("latent-basemap-cpu-run")
    assert prepare_round0153_queue.RELEASE_ROOT != "/home/enjalot/code/latent-basemap-run"


def test_density_branch_activates_scale_only_on_registered_pattern() -> None:
    decision = classify_density_branch(
        _historical(REGISTERED_FLOOR),
        _current(REGISTERED_FLOOR - 0.01),
    )
    assert decision["outcome"] == "density-restores-with-row-universe"
    assert decision["track_f_activated"] is True
    assert decision["floor_changed"] is False


def test_density_branch_closes_when_all_historical_cells_fail() -> None:
    decision = classify_density_branch(
        _historical(REGISTERED_FLOOR - 0.001),
        _current(REGISTERED_FLOOR - 0.01),
    )
    assert decision["outcome"] == "density-does-not-restore"
    assert decision["track_f_activated"] is False


def test_density_branch_reports_mixed_without_activating() -> None:
    historical = _historical(REGISTERED_FLOOR + 0.01)
    historical[HISTORICAL_ROW_CELLS[-1]] = REGISTERED_FLOOR - 0.01
    decision = classify_density_branch(
        historical, _current(REGISTERED_FLOOR - 0.01)
    )
    assert decision["outcome"] == "density-mixed-owner-decision-required"
    assert decision["track_f_activated"] is False


def test_density_branch_rejects_missing_cell() -> None:
    with pytest.raises(Round0153Error, match="incomplete"):
        classify_density_branch({}, _current(0.1))


def test_diagnostic_transcription_keeps_legacy_density_label() -> None:
    result = diagnostic_values({
        "panel": {
            "ffr": 0.56,
            "recall@k": 0.01,
            "purity": {"k256": 1.02, "k1024": 0.97},
            "density": 0.57,
        },
        "projection": {"ffr": 0.53, "recall_at_10": 0.0098},
        "decision_metrics": {"ffr": 0.56},
    })
    assert result["legacy_panel_density_not_density_v2"] == pytest.approx(0.57)
    assert "density_v2" not in result
    assert np.isfinite(list(result["registered_decision_metrics"].values())).all()
