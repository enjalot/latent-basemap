from __future__ import annotations

import copy

import pytest

from basemap.round0195_release_proposal import METRICS, Round0195Error, build_proposal


def _inputs():
    cells = {}
    gates = {}
    for metric in METRICS:
        gates[metric] = {"floor": 0.5}
    for seed in (42, 43, 44, 45):
        cells[f"seed{seed}"] = {
            "seed": seed,
            "decision_metrics": {metric: 0.6 for metric in METRICS},
            "coordinates": {"kind": "file", "sha256": str(seed)},
            "train_receipt": {"kind": "file", "sha256": f"r{seed}"},
        }
    family = {
        "schema": "round0160-prompted-four-seed-family-evidence-v1",
        "seeds": [42, 43, 44, 45],
        "population": {"rows": 1_993_761, "embedding_convention": "Document: "},
        "cells": cells,
    }
    gate_receipt = {
        "schema": "round0161-prompted-universe-quality-gates-v1",
        "registered": True,
        "seed_family": [42, 43, 44, 45],
        "gates": gates,
    }
    maps = {
        "r0115-prompted-2m-seed42": {
            "verdict": "pass", "ffr_retention": 0.8
        },
        "r0117-prompted-2m-seed43": {
            "verdict": "amber", "ffr_retention": 0.7
        },
    }
    universality = {
        "schema": "round0182-universality-readout-packet-v1",
        "diagnostic_only": True,
        "rows": [{"maps": maps} for _ in range(11)],
    }
    methods = {
        "schema": "round0183-heldout-projection-method-table-v1",
        "rows": {"2m": {
            "corrected_parametric_standard_curve_seed42": {"ffr": 0.5785},
            "aumap_inverse_distance_k15": {"ffr": 0.53973},
        }},
    }
    scale = {"decision": {"outcome": "confirmed-2-of-3-seed-sensitive"}}
    return family, gate_receipt, universality, methods, scale


def test_builds_honest_nonpublishing_proposal() -> None:
    value = build_proposal(*_inputs())
    assert value["candidate_id"] == "basemap-jina-v5-nano-en-2m-v0"
    assert value["qualification"]["all_four_seeds_pass_all_six_commensurate_gates"]
    assert value["method_context"]["candidate_specific_contrast"] is False
    assert value["proposal"]["registry_promotion_performed"] is False
    assert value["ood_caveats"]["universal_quality_claim"] is False


def test_any_gate_failure_fails_closed() -> None:
    values = list(_inputs())
    values[0] = copy.deepcopy(values[0])
    values[0]["cells"]["seed45"]["decision_metrics"]["ffr"] = 0.4
    with pytest.raises(Round0195Error, match="does not pass"):
        build_proposal(*values)


def test_method_context_cannot_silently_change() -> None:
    values = list(_inputs())
    values[3] = copy.deepcopy(values[3])
    values[3]["rows"]["2m"]["corrected_parametric_standard_curve_seed42"]["ffr"] = 0.6
    with pytest.raises(Round0195Error, match="context changed"):
        build_proposal(*values)
