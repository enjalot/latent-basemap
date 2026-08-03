"""Decision-contract tests for the conditional prompted-diverse Q3 rung."""
from __future__ import annotations

import pytest

from basemap.round0105_search import GROUPS
from basemap.round0166_prompted_8m import METRICS
from basemap.round0169_prompted_diverse import (
    RETENTION_RATIO,
    Round0169Error,
    prompted_diverse_decision,
)


def _inputs():
    baseline = {metric: 1.0 for metric in METRICS}
    return {
        "native": {metric: 1.0 for metric in METRICS},
        "matched_2m": {metric: RETENTION_RATIO for metric in METRICS},
        "baseline_2m_seed42": baseline,
        "prompted_floors": {metric: 0.9 for metric in METRICS},
        "group_ffr": {name: (0.8 if name in GROUPS[:3] else 0.32) for name in GROUPS},
        "prompted_ood": {
            "polish_recall_at_50_of_high10": 0.10,
            "in_mix_median_recall_at_50_of_high10": 0.20,
        },
        "raw_r0132_ood": {
            "polish_recall_at_50_of_high10": 0.10 / RETENTION_RATIO,
            "in_mix_median_recall_at_50_of_high10": 0.20 / RETENTION_RATIO,
        },
    }


def test_all_registered_boundaries_are_inclusive() -> None:
    decision = prompted_diverse_decision(**_inputs())
    assert decision["passed"] is True
    assert decision["outcome"] == "prompted-diverse-u12-rung-qualified"
    assert decision["language_relative_ffr"]["floor"] == pytest.approx(0.32)
    assert decision["polish_ood_gate"]["ratio"] == pytest.approx(0.5)
    assert all(cell["passed"] for cell in decision["raw_r0132_ood_retention_gates"].values())


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ("native", "density_v2"),
        ("matched_2m", "ffr"),
        ("group_ffr", GROUPS[3]),
        ("prompted_ood", "polish_recall_at_50_of_high10"),
        ("prompted_ood", "in_mix_median_recall_at_50_of_high10"),
    ],
)
def test_each_gate_stack_can_fail_the_decision(path: str, key: str) -> None:
    values = _inputs()
    values[path][key] *= 0.8
    decision = prompted_diverse_decision(**values)
    assert decision["passed"] is False
    assert decision["outcome"] == "prompted-diverse-u12-rung-not-qualified"


def test_metric_or_language_omission_is_invalid_not_a_negative() -> None:
    values = _inputs()
    del values["group_ffr"][GROUPS[-1]]
    with pytest.raises(Round0169Error, match="incomplete"):
        prompted_diverse_decision(**values)

    values = _inputs()
    del values["matched_2m"][METRICS[-1]]
    with pytest.raises(Round0169Error, match="metric set changed"):
        prompted_diverse_decision(**values)
