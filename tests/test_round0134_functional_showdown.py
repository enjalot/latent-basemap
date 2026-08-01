from __future__ import annotations

import math

import numpy as np
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
    SOURCE_ROWS,
)
from experiments.round0134_nodes import (
    _load_frozen_query_truth,
    _load_reference,
    _load_shared_evaluation_inputs,
)


R0037_SHARED_RECEIPT = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/"
    "shared-reference/receipt.json"
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
    assert SOURCE_ROWS == 2_000_000
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


def test_reviewed_r0037_query_truth_survives_current_builder_source_drift():
    import json

    with open(R0037_SHARED_RECEIPT, encoding="utf-8") as handle:
        shared = json.load(handle)
    truth = _load_frozen_query_truth(
        shared["query_truth"]["canonical_path"],
        expected_key=shared["query_truth_key"],
        expected_policy=shared["query_truth_exactness"],
        expected_payload_sha256=shared["query_truth_payload_sha256"],
    )
    assert truth["neighbors"].shape == (20_000, 10)
    assert truth["corpus_cardinality"] == 2_000_000
    assert truth["historical_builder_policy_authenticated"] is True

    changed = dict(shared["query_truth_exactness"])
    changed["implementation_sha256"] = "0" * 64
    with pytest.raises(Round0134Error, match="policy changed"):
        _load_frozen_query_truth(
            shared["query_truth"]["canonical_path"],
            expected_key=shared["query_truth_key"],
            expected_policy=changed,
            expected_payload_sha256=shared["query_truth_payload_sha256"],
        )


def test_real_r0037_source_query_and_reference_views_close_before_cuda():
    import json

    from basemap.panel_v2 import _resolve_reference, sample_anchors
    from experiments.round0027_nodes import _panel_config

    with open(
        "/data/latent-basemap/runs/round-0134/queue/queue.json",
        encoding="utf-8",
    ) as handle:
        job = json.load(handle)["jobs"][0]
    _source_signature, source, queries = _load_shared_evaluation_inputs(job)
    _shared, _shared_signature, reference, _truth, centroids = _load_reference(job)
    config = _panel_config()
    anchors = sample_anchors(len(source), config)
    resolved, reused = _resolve_reference(
        source, anchors, config, centroids, reference
    )
    assert reused is True
    assert resolved["key"] == reference["key"]
    assert source.dtype == np.dtype("<f4")
    assert queries.dtype == np.dtype("<f4")
