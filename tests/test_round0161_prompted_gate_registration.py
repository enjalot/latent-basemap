"""Pure preregistration tests for R0161 prompted-universe gates."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0160_prompted_seed_family import METRICS, SEEDS
from basemap.round0161_prompted_gate_registration import (
    FORMULA,
    Round0161Error,
    register_prompted_gates,
)


def _cells() -> dict[str, dict]:
    return {
        f"seed{seed}": {
            "decision_metrics": {
                metric: 0.2 + 0.01 * position + 0.001 * (seed - 42)
                for position, metric in enumerate(METRICS)
            }
        }
        for seed in SEEDS
    }


def test_gate_formula_is_exact_mean_minus_two_sample_sd() -> None:
    cells = _cells()
    result = register_prompted_gates(cells)
    assert result["registered"] is True
    assert result["formula"] == FORMULA
    assert result["raw_floor_changed"] is False
    for metric in METRICS:
        values = np.asarray([cells[f"seed{seed}"]["decision_metrics"][metric] for seed in SEEDS])
        gate = result["gates"][metric]
        assert gate["sample_sd_ddof1"] == pytest.approx(values.std(ddof=1))
        assert gate["floor"] == pytest.approx(values.mean() - 2 * values.std(ddof=1))


def test_gate_registration_rejects_an_incomplete_family() -> None:
    with pytest.raises(Round0161Error, match="incomplete"):
        register_prompted_gates({})
