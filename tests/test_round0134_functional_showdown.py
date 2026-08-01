from __future__ import annotations

import math

import pytest

from basemap.round0134_functional_showdown import (
    CELL_ORDER,
    CURRENT_R0104_SEED42,
    CURRENT_RAW_SEED42,
    CURRENT_RAW_SEED43,
    HISTORICAL_SEED42,
    HISTORICAL_SEED43,
    METRIC_ORDER,
    Round0134Error,
    build_decision,
    purity_fidelity,
)
from experiments.prepare_round0134_queue import (
    GPU_HOURS_EXPECTED,
    GPU_HOURS_MAXIMUM,
    GPU_HOURS_MINIMUM,
    GPU_HOURS_P90,
    REVIEW_CAPABILITIES,
)


def _cell(
    *,
    ffr: float = 0.6,
    purity256: float = 1.0,
    purity1024: float = 1.0,
    projection_ffr: float = 0.5,
    recall: float = 0.01,
):
    return {
        "panel": {
            "ffr": ffr,
            "purity": {"k256": purity256, "k1024": purity1024},
        },
        "projection": {"ffr": projection_ffr, "recall_at_10": recall},
    }


def _cells(**overrides):
    values = {key: _cell() for key in CELL_ORDER}
    values.update(overrides)
    return values


def test_purity_fidelity_is_symmetric_and_maximal_at_one():
    assert purity_fidelity(1.0) == pytest.approx(1.0)
    assert purity_fidelity(2.0) == pytest.approx(0.5)
    assert purity_fidelity(0.5) == pytest.approx(0.5)
    with pytest.raises(Round0134Error):
        purity_fidelity(0.0)


def test_current_equal_or_better_authorizes_density_v3():
    cells = _cells(
        **{
            CURRENT_R0104_SEED42: _cell(ffr=0.61, projection_ffr=0.51),
            CURRENT_RAW_SEED42: _cell(ffr=0.62, recall=0.011),
            CURRENT_RAW_SEED43: _cell(ffr=0.62, recall=0.011),
        }
    )
    decision = build_decision(cells)
    assert decision["outcome"] == "current-recipe-functionally-noninferior"
    assert decision["density_v3_calibration_authorized"] is True
    assert decision["fuzzy_graph_or_sampler_bridges_authorized"] is False
    assert decision["failed_cells"] == []


def test_one_historical_functional_advantage_selects_bridge_branch():
    cells = _cells(
        **{
            HISTORICAL_SEED42: _cell(projection_ffr=0.55),
            HISTORICAL_SEED43: _cell(projection_ffr=0.55),
        }
    )
    decision = build_decision(cells)
    assert decision["outcome"] == "historical-recipe-functionally-better"
    assert decision["density_v3_calibration_authorized"] is False
    assert decision["fuzzy_graph_or_sampler_bridges_authorized"] is True
    assert "pre_r0115_seed42:projection_ffr" in decision["failed_cells"]
    assert "raw_current_two_seed:projection_ffr" in decision["failed_cells"]


def test_purity_overseparation_is_not_treated_as_unbounded_improvement():
    cells = _cells(
        **{
            CURRENT_R0104_SEED42: _cell(purity256=1.2),
            CURRENT_RAW_SEED42: _cell(purity256=1.2),
            CURRENT_RAW_SEED43: _cell(purity256=1.2),
        }
    )
    decision = build_decision(cells)
    assert decision["outcome"] == "historical-recipe-functionally-better"
    assert "pre_r0115_seed42:purity_fidelity_k256" in decision["failed_cells"]


def test_cell_order_is_an_authenticated_selector_input():
    cells = _cells()
    reordered = dict(reversed(list(cells.items())))
    with pytest.raises(Round0134Error, match="reordered"):
        build_decision(reordered)


def test_registered_metrics_and_budget_match_round_design():
    assert METRIC_ORDER == (
        "ffr",
        "purity_fidelity_k256",
        "purity_fidelity_k1024",
        "projection_ffr",
        "ood_recall_at_10",
    )
    assert (
        GPU_HOURS_MINIMUM,
        GPU_HOURS_EXPECTED,
        GPU_HOURS_P90,
        GPU_HOURS_MAXIMUM,
    ) == (0.10, 0.30, 0.42, 0.50)
    assert set(REVIEW_CAPABILITIES) == {
        "0037",
        "0038",
        "0104",
        "0115",
        "0117",
        "0119",
        "0122",
    }


def test_raw_two_seed_contrast_is_seed_matched():
    cells = _cells()
    decision = build_decision(cells)
    contrast = decision["contrasts"]["raw_current_two_seed"]
    assert contrast["historical_cells"] == [HISTORICAL_SEED42, HISTORICAL_SEED43]
    assert contrast["current_cells"] == [CURRENT_RAW_SEED42, CURRENT_RAW_SEED43]
    assert decision["contrasts"]["pre_r0115_seed42"]["current_cells"] == [
        CURRENT_R0104_SEED42
    ]
