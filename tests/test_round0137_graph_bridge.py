from __future__ import annotations

from pathlib import Path

import pytest

from basemap.round0134_functional_showdown import METRIC_ORDER
from basemap.round0137_graph_bridge import (
    CELL_ORDER,
    CONTROL,
    HISTORICAL,
    TREATMENT,
    Round0137Error,
    build_decision,
)
from experiments.prepare_round0137_queue import (
    GPU_HOURS_EXPECTED,
    GPU_HOURS_MAXIMUM,
    GPU_HOURS_MINIMUM,
    GPU_HOURS_P90,
    REVIEW_CAPABILITIES,
)


def _cell(value: float = 0.5) -> dict:
    return {
        "panel": {
            "ffr": value,
            "purity": {"k256": 1.0, "k1024": 1.0},
        },
        "projection": {"ffr": value, "recall_at_10": value},
    }


def _cells(*, historical: float, control: float, treatment: float) -> dict:
    return {
        HISTORICAL: _cell(historical),
        CONTROL: _cell(control),
        TREATMENT: _cell(treatment),
    }


def test_high_recall_graph_restoration_stops_before_sampler_bridge():
    decision = build_decision(
        _cells(historical=0.6, control=0.5, treatment=0.6)
    )
    assert decision["outcome"] == "high-recall-graph-sufficient-to-restore-function"
    assert decision["high_recall_graph_sufficient"] is True
    assert decision["sampler_bridge_authorized"] is False
    assert all(
        row["candidate_at_least_baseline"]
        for row in decision["treatment_vs_historical_target"]["metrics"].values()
    )


def test_nonrestoring_nonregressing_graph_authorizes_sampler_bridge():
    decision = build_decision(
        _cells(historical=0.6, control=0.5, treatment=0.55)
    )
    assert decision["outcome"] == "high-recall-graph-insufficient-to-restore-function"
    assert decision["sampler_bridge_authorized"] is True


def test_graph_treatment_that_regresses_control_is_called_out():
    decision = build_decision(
        _cells(historical=0.6, control=0.5, treatment=0.4)
    )
    assert decision["outcome"] == "high-recall-graph-regresses-current-control"
    assert decision["sampler_bridge_authorized"] is True


def test_cell_order_is_frozen():
    cells = _cells(historical=0.6, control=0.5, treatment=0.6)
    assert tuple(cells) == CELL_ORDER
    with pytest.raises(Round0137Error, match="reordered"):
        build_decision(dict(reversed(list(cells.items()))))


def test_registered_metrics_dependencies_and_budget_are_bounded():
    assert METRIC_ORDER == (
        "ffr",
        "purity_fidelity_k256",
        "purity_fidelity_k1024",
        "projection_ffr",
        "ood_recall_at_10",
    )
    assert set(REVIEW_CAPABILITIES) == {
        "0037",
        "0103",
        "0104",
        "0122",
        "0134",
    }
    assert (
        GPU_HOURS_MINIMUM,
        GPU_HOURS_EXPECTED,
        GPU_HOURS_P90,
        GPU_HOURS_MAXIMUM,
    ) == (1.25, 1.60, 2.10, 2.50)


def test_r0104_extension_keeps_default_selector_and_scopes_override():
    source = (
        Path(__file__).parents[1] / "experiments" / "round0104_nodes.py"
    ).read_text(encoding="utf-8")
    assert "if passed and selected is None:" in source
    assert "if forced_nprobe is not None:" in source
    assert "forced_nprobe not in GRAPH_NPROBE_GRID" in source
    assert 'list(job.get("shared_arms", ARMS))' in source
    assert 'str(job.get("shared_round_id", ROUND_ID))' in source
