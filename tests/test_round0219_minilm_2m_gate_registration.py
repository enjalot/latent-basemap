"""Contract tests for the R0219 MiniLM mixed-2M gate registration.

These cover both halves of the round: the arithmetic (mean - 2 sample sd,
ddof=1, over four cells) and the *design constraint* that the gate covers FFR and
the two purity fidelities and nothing else. The second half matters more — the
arithmetic is three lines and the metric set is the scientific decision.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any

import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY as PANEL_CAPABILITY,
    CORPUS_SLUGS,
    DIAGNOSTIC_METRICS,
    PANEL_METRICS,
    SEEDS,
    build_family_panel_evidence,
    corpus_ffr_view,
    map_capability,
    panel_metric_view,
)
from basemap.round0219_minilm_2m_gate_registration import (
    CAPABILITY,
    EXCLUDED_METRICS,
    FORMULA,
    GATE_METRICS,
    GATE_SCHEMA,
    MULTIPLIER,
    ROUND_ID,
    Round0219Error,
    register_minilm_gates,
)
from experiments import round0219_nodes


DENSITIES = (0.601, 0.664, 0.588, 0.649)
FFRS = (0.4102, 0.4098, 0.4105, 0.4094)
K256 = (0.981, 0.977, 0.984, 0.979)
K1024 = (1.021, 1.028, 1.017, 1.025)


def _panel(index: int) -> dict[str, Any]:
    return {
        "density": DENSITIES[index],
        "ffr": FFRS[index],
        "purity": {"k256": K256[index], "k1024": K1024[index]},
        "ffr_by_group": {
            slug: {"anchors": 1_000, "ffr": round(FFRS[index] + 0.01 * position, 4)}
            for position, slug in enumerate(CORPUS_SLUGS)
        },
        "guards": {
            "coords_finite": True,
            "coords_collapsed": False,
            "emb_finite": True,
            "emb_zero_rows": 0,
        },
        "provenance": {"hiD_reference_reused": True},
    }


def _evidence() -> dict[str, Any]:
    cells = {}
    for index, seed in enumerate(SEEDS):
        panel = _panel(index)
        cells[seed] = {
            "seed": seed,
            "capability": map_capability(seed),
            "panel": panel,
            "panel_metrics": panel_metric_view(panel),
            "corpus_ffr": corpus_ffr_view(panel),
        }
    evidence = build_family_panel_evidence(cells)
    evidence["execution_checks"] = {"all_four_cells_scored": True}
    return evidence


def test_round0219_identity_and_formula() -> None:
    assert ROUND_ID == "0219"
    assert CAPABILITY == "minilm-mixed-2m-quality-gates-v1"
    assert FORMULA == "family mean - 2 * sample standard deviation (ddof=1)"
    assert MULTIPLIER == 2.0


def test_round0219_gates_ffr_and_purity_only() -> None:
    """The design constraint, asserted rather than documented."""
    assert GATE_METRICS == ("ffr", "purity_fidelity_k256", "purity_fidelity_k1024")
    assert "density_v2" not in GATE_METRICS
    assert "heldout_recall_at_10" not in GATE_METRICS
    assert set(EXCLUDED_METRICS) == {"density_v2", "heldout_recall_at_10"}
    assert not set(GATE_METRICS) & set(EXCLUDED_METRICS)
    assert set(GATE_METRICS) | set(DIAGNOSTIC_METRICS) == set(PANEL_METRICS)


def test_round0219_arithmetic_reproduces_mean_minus_two_sample_sd() -> None:
    registration = register_minilm_gates(_evidence())
    assert registration["schema"] == GATE_SCHEMA
    assert registration["registered"] is True
    assert registration["n"] == 4
    assert registration["sample_standard_deviation_ddof"] == 1
    expected = {
        "ffr": list(FFRS),
        "purity_fidelity_k256": [math.exp(-abs(math.log(v))) for v in K256],
        "purity_fidelity_k1024": [math.exp(-abs(math.log(v))) for v in K1024],
    }
    for metric, values in expected.items():
        cell = registration["gates"][metric]
        assert cell["seed_order"] == list(SEEDS)
        assert cell["values"] == pytest.approx(values, rel=1e-12)
        assert cell["mean"] == pytest.approx(statistics.fmean(values), rel=1e-12)
        assert cell["sample_sd_ddof1"] == pytest.approx(
            statistics.stdev(values), rel=1e-12
        )
        assert cell["floor"] == pytest.approx(
            statistics.fmean(values) - 2.0 * statistics.stdev(values), rel=1e-12
        )
        assert cell["direction"] == "higher-is-better"
        assert cell["floor_is_vacuous"] is False


def test_round0219_density_is_transcribed_never_gated() -> None:
    registration = register_minilm_gates(_evidence())
    assert "density_v2" not in registration["gates"]
    diagnostic = registration["diagnostic_metrics"]["density_v2"]
    assert diagnostic["registered_as_floor"] is False
    assert diagnostic["role"] == "diagnostic-only, transcribed"
    assert diagnostic["values"] == pytest.approx(list(DENSITIES), rel=1e-12)
    assert registration["density_v2_role"] == "diagnostic-only, transcribed"
    # The exclusion is empirically motivated: density's relative spread on this
    # synthetic family dwarfs FFR's, which is the R0214 pattern in miniature.
    assert diagnostic["relative_spread_of_mean"] > (
        registration["gates"]["ffr"]["relative_spread_of_mean"]
    )


def test_round0219_per_corpus_ffr_is_descriptive_not_a_floor() -> None:
    registration = register_minilm_gates(_evidence())
    assert set(registration["per_corpus_ffr"]) == set(CORPUS_SLUGS)
    for slug, cell in registration["per_corpus_ffr"].items():
        assert cell["registered_as_floor"] is False
        assert "floor" not in cell, slug


def test_round0219_leaves_every_other_universe_floor_alone() -> None:
    registration = register_minilm_gates(_evidence())
    assert registration["r0161_prompted_floors_unchanged"] is True
    assert registration["r0193_mixed_english_floors_unchanged"] is True
    assert registration["raw_universe_floors_unchanged"] is True
    assert registration["training_performed"] is False
    assert registration["evaluation_performed"] is False
    assert "never a cross-universe floor" in registration["applies_to"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"round_id": "0217"},
        {"gate_registered": True},
        {"gate_registerable_here": True},
        {"seeds": [42, 43, 44]},
        {"training_performed": True},
        {"execution_checks": {"all_four_cells_scored": False}},
    ],
)
def test_round0219_refuses_a_drifted_panel_premise(mutation: dict) -> None:
    evidence = _evidence()
    evidence.update(mutation)
    with pytest.raises(Round0219Error):
        register_minilm_gates(evidence)


def test_round0219_refuses_an_incomplete_or_invalid_cell() -> None:
    evidence = _evidence()
    evidence["panel_metric_cells"].pop("45")
    with pytest.raises(Round0219Error):
        register_minilm_gates(evidence)
    evidence = _evidence()
    evidence["panel_metric_cells"]["43"]["ffr"] = float("nan")
    with pytest.raises(Round0219Error):
        register_minilm_gates(evidence)
    evidence = _evidence()
    evidence["panel_metric_cells"]["44"]["purity_fidelity_k256"] = 1.4
    with pytest.raises(Round0219Error):
        register_minilm_gates(evidence)


def test_round0219_node_is_cpu_only_and_seals(tmp_path) -> None:
    panel_path = tmp_path / "seed-family-panel.json"
    panel = prompt_contract.seal({
        **_evidence(),
        "capabilities": [PANEL_CAPABILITY],
        "release_sha": "e" * 40,
        "evaluation_performed": True,
        "seed_invariant_sha256": "9" * 64,
    })
    panel_path.write_text(
        json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = tmp_path / "gates"
    job = {
        "action": round0219_nodes.ACTION,
        "panel_evidence": str(panel_path),
        "upstream_review_state": {"round_id": "0218", "accepted_reviews": 0},
        "outputs": [str(output)],
    }
    active = {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}}
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        round0219_nodes.run_job(active, job)
        with (output / "minilm-quality-gates.json").open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        prompt_contract.validate_seal(receipt, label="R0219 gate receipt")
        assert receipt["capabilities"] == [CAPABILITY]
        assert receipt["decision"]["gpu_used"] is False
        assert receipt["decision"]["gated_metrics"] == list(GATE_METRICS)
        assert receipt["source_panel_seed_invariant_sha256"] == "9" * 64
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        with pytest.raises(Round0219Error):
            round0219_nodes.run_job(active, {**job, "outputs": [str(tmp_path / "no")]})
        with pytest.raises(Round0219Error):
            round0219_nodes.run_job(active, {**job, "action": "something_else"})
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
