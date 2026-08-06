from __future__ import annotations

import statistics

import pytest

from basemap.round0192_quarter_seed_family import GATE_METRICS, ROWS
from basemap.round0193_mixed_gate_registration import (
    FORMULA,
    SEEDS,
    Round0193Error,
    register_mixed_gates,
)
from experiments.prepare_round0193_queue import ROUND_FILE


def _family() -> dict:
    family = {
        "outcome": "mixed-quarter-three-seed-family-complete",
        "seeds": list(SEEDS),
        "rows": ROWS,
        "gate_registration_deferred_to_reviewed_cpu_round": True,
        "gate_metric_cells": {
            str(seed): {
                metric: 0.20 + index / 100 + (seed - 42) / 1000
                for index, metric in enumerate(GATE_METRICS)
            }
            for seed in SEEDS
        },
    }
    family["descriptive_summaries"] = {
        metric: {
            "values_seed42_seed43_seed44": [
                family["gate_metric_cells"][str(seed)][metric]
                for seed in SEEDS
            ],
            "mean": statistics.fmean(
                family["gate_metric_cells"][str(seed)][metric]
                for seed in SEEDS
            ),
            "sample_sd_ddof1": statistics.stdev(
                family["gate_metric_cells"][str(seed)][metric]
                for seed in SEEDS
            ),
        }
        for metric in GATE_METRICS
    }
    return family


def test_queue_preparer_targets_actual_issue_date() -> None:
    assert ROUND_FILE.endswith("/round-0193-2026-08-06.md")


def test_formula_is_exact_mean_minus_two_sample_sd() -> None:
    result = register_mixed_gates(_family())
    assert result["registered"] is True
    assert result["formula"] == FORMULA
    assert result["n"] == 3
    assert result["r0161_prompted_fineweb_floors_unchanged"] is True
    for metric in GATE_METRICS:
        gate = result["gates"][metric]
        assert gate["sample_sd_ddof1"] == pytest.approx(0.001)
        assert gate["floor"] == pytest.approx(gate["mean"] - 0.002)


def test_incomplete_or_unreviewed_family_fails_closed() -> None:
    incomplete = _family()
    incomplete["gate_metric_cells"].pop("44")
    with pytest.raises(Round0193Error, match="incomplete"):
        register_mixed_gates(incomplete)
    undeferred = _family()
    undeferred["gate_registration_deferred_to_reviewed_cpu_round"] = False
    with pytest.raises(Round0193Error, match="premise"):
        register_mixed_gates(undeferred)


def test_impossible_metric_or_inconsistent_summary_fails_closed() -> None:
    impossible = _family()
    impossible["gate_metric_cells"]["42"][GATE_METRICS[0]] = 1.01
    with pytest.raises(Round0193Error, match="invalid"):
        register_mixed_gates(impossible)
    inconsistent = _family()
    inconsistent["descriptive_summaries"][GATE_METRICS[0]]["mean"] += 1e-6
    with pytest.raises(Round0193Error, match="disagrees"):
        register_mixed_gates(inconsistent)
