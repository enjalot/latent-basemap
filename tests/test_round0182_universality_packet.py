from __future__ import annotations

import copy

import pytest

from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0182_universality_packet import (
    PROMPTED_MAP_ORDER,
    RAW_MAP_ORDER,
    Round0182Error,
    build_packet,
    render_markdown,
)


def _correlations(maps: tuple[str, ...], pooled: float) -> list[dict]:
    return [
        {
            "scope": scope,
            "outcome": "ffr_retention",
            "predictor": "twonn_intrinsic_dimension",
            "spearman_rho": -0.5 - index * 0.01,
        }
        for index, scope in enumerate(maps)
    ] + [{
        "scope": "pooled-descriptive",
        "outcome": "ffr_retention",
        "predictor": "twonn_intrinsic_dimension",
        "spearman_rho": pooled,
    }]


def _fixtures() -> tuple[dict, dict, dict]:
    prompted_cells = []
    raw_rows = []
    for probe_index, probe in enumerate(PROBE_ORDER):
        for map_index, map_key in enumerate(PROMPTED_MAP_ORDER):
            prompted_cells.append({
                "map": map_key,
                "probe": probe,
                "ffr_retention": 0.4 + 0.1 * map_index,
                "recall10_retention": 0.3 + 0.01 * probe_index,
            })
        for map_index, map_key in enumerate(RAW_MAP_ORDER):
            raw_rows.append({
                "map": map_key,
                "probe": probe,
                "ffr_retention": 0.6 + 0.1 * map_index,
                "recall10_retention": 0.2 + 0.01 * probe_index,
            })
    prompted_rho = -0.58
    raw_rho = -0.44
    prompted = {
        "schema": "jina-prompted-universality-panel-v1",
        "round_id": "0178",
        "map_order": list(PROMPTED_MAP_ORDER),
        "probe_order": list(PROBE_ORDER),
        "diagnostic_only": True,
        "no_causal_prompt_claim": True,
        "cells": prompted_cells,
        "prompted_geometry": {
            probe: {"geometry": {"twonn": {"intrinsic_dimension": 8.0 + index}}}
            for index, probe in enumerate(PROBE_ORDER)
        },
        "twonn_correlations": _correlations(PROMPTED_MAP_ORDER, prompted_rho),
        "raw_comparison": {
            "prompted_pooled_twonn_ffr_rho": prompted_rho,
            "raw_pooled_twonn_ffr_rho": raw_rho,
        },
    }
    raw = {
        "schema": "jina-diverse-universality-panel-v1",
        "round_id": "0142",
        "probe_order": list(PROBE_ORDER),
        "rows": raw_rows,
    }
    predictors = {
        "schema": "jina-diverse-projection-loss-predictors-v1",
        "round_id": "0146",
        "correlations": _correlations(RAW_MAP_ORDER, raw_rho),
    }
    return prompted, raw, predictors


def test_build_packet_is_complete_and_preserves_direction() -> None:
    prompted, raw, predictors = _fixtures()
    packet = build_packet(prompted=prompted, raw=raw, predictors=predictors)
    assert len(packet["rows"]) == len(PROBE_ORDER)
    assert set(packet["rows"][0]["maps"]) == set(PROMPTED_MAP_ORDER + RAW_MAP_ORDER)
    assert packet["twonn_ffr_spearman"]["prompted"]["pooled-descriptive"] == -0.58
    assert packet["twonn_ffr_spearman"]["raw"]["pooled-descriptive"] == -0.44
    assert packet["rows"][0]["maps"][PROMPTED_MAP_ORDER[0]]["verdict"] == "named-failure"
    assert packet["rows"][0]["maps"][RAW_MAP_ORDER[1]]["verdict"] == "pass"
    rendered = render_markdown(packet)
    assert "Prompted/raw Jina universality readout" in rendered
    assert PROMPTED_MAP_ORDER[0] in rendered
    assert RAW_MAP_ORDER[0] in rendered


def test_duplicate_or_missing_cell_fails_closed() -> None:
    prompted, raw, predictors = _fixtures()
    broken = copy.deepcopy(prompted)
    broken["cells"][-1] = dict(broken["cells"][0])
    with pytest.raises(Round0182Error, match="duplicate or unknown"):
        build_packet(prompted=broken, raw=raw, predictors=predictors)


def test_cross_substrate_rho_binding_fails_closed() -> None:
    prompted, raw, predictors = _fixtures()
    prompted["raw_comparison"]["raw_pooled_twonn_ffr_rho"] = -0.1
    with pytest.raises(Round0182Error, match="binding changed"):
        build_packet(prompted=prompted, raw=raw, predictors=predictors)
