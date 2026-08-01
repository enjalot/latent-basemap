from __future__ import annotations

import pytest

from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    RESTORATION_FLOORS,
)
from basemap.round0147_row_policy import (
    TREATMENT,
    Round0147Error,
    build_decision,
)


def _cell(offset: float = 0.01) -> dict:
    values = {key: value + offset for key, value in RESTORATION_FLOORS.items()}
    return {
        "panel": {
            "ffr": values["ffr"],
            "purity": {
                "k256": values["purity_fidelity_k256"],
                "k1024": values["purity_fidelity_k1024"],
            },
        },
        "projection": {
            "ffr": values["projection_ffr"],
            "recall_at_10": values["ood_recall_at_10"],
        },
    }


def _summary() -> dict:
    return {
        "target_rows": 2_000_000,
        "historical_stream_rows": 8_000_000,
        "scan_rows": 2_010_427,
        "skipped_excluded_rows": 10_427,
        "raw_prefix_excluded_rows": 10_367,
        "replacement_rows_beyond_raw_prefix": 10_367,
    }


def test_eligible_historical_policy_restoration_is_narrow() -> None:
    decision = build_decision(
        {
            CURRENT_GRAPH_CURRENT_HOST: _cell(0.02),
            TREATMENT: _cell(0.01),
        },
        selection_summary=_summary(),
    )
    assert decision["outcome"] == "eligible-historical-row-policy-restores"
    assert decision["duplicate_control_compatible_with_restoration"] is True
    assert decision["duplicate_control_causal_claimed"] is False
    assert decision["diverse_scale_transfer_claimed"] is False
    assert all(
        value == pytest.approx(-0.01)
        for value in decision["paired_eligible_minus_raw_historical"].values()
    )


def test_one_failed_metric_blocks_restoration() -> None:
    treatment = _cell(0.01)
    treatment["projection"]["recall_at_10"] = (
        RESTORATION_FLOORS["ood_recall_at_10"] - 1e-6
    )
    decision = build_decision(
        {
            CURRENT_GRAPH_CURRENT_HOST: _cell(0.02),
            TREATMENT: treatment,
        },
        selection_summary=_summary(),
    )
    assert decision["outcome"] == (
        "eligible-historical-row-policy-does-not-restore"
    )
    assert decision["duplicate_control_compatible_with_restoration"] is False


def test_nonrestoring_control_cannot_activate_conditional_round() -> None:
    with pytest.raises(Round0147Error, match="activation requires"):
        build_decision(
            {
                CURRENT_GRAPH_CURRENT_HOST: _cell(-0.1),
                TREATMENT: _cell(0.01),
            },
            selection_summary=_summary(),
        )


def test_selection_cardinality_cannot_silently_shrink() -> None:
    summary = _summary()
    summary["target_rows"] -= 10_367
    with pytest.raises(Round0147Error, match="selection does not close"):
        build_decision(
            {
                CURRENT_GRAPH_CURRENT_HOST: _cell(0.02),
                TREATMENT: _cell(0.01),
            },
            selection_summary=summary,
        )
