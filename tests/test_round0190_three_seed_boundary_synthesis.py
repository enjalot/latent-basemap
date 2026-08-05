from __future__ import annotations

import pytest

from basemap.round0190_three_seed_boundary_synthesis import (
    GATE_METRICS,
    Round0190Error,
    synthesize,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0190_nodes as nodes
from experiments.prepare_round0190_queue import (
    DECISION_PATHS,
    EVALUATION_PATHS,
    R0160_FAMILY,
    R0161_GATES,
)


def _view(pile: float, scale: float = 1.0) -> dict[str, float]:
    return {
        "pile_ffr": pile,
        "density_v2": 0.2 * scale,
        "ffr": 0.59 * scale,
        "purity_fidelity_k256": 0.93 * scale,
        "purity_fidelity_k1024": 0.94 * scale,
        "projection_ffr": 0.52 * scale,
        "heldout_recall_at_10": 0.008 * scale,
    }


def _gates() -> dict[str, dict[str, float]]:
    return {metric: {"floor": 0.1} for metric in GATE_METRICS}


def test_two_of_three_releases_seed_sensitive_capacity_branch() -> None:
    cells = {
        "seed42": {"half": _view(0.4556), "full": _view(0.4358)},
        "seed43": {"half": _view(0.4644), "full": _view(0.4584)},
        "seed44": {"half": _view(0.4634), "full": _view(0.4423)},
    }
    decision = synthesize(
        cells=cells,
        quarter_seed42=_view(0.4669),
        fineweb_seed42={metric: 0.5 for metric in GATE_METRICS},
        gates=_gates(),
    )
    assert decision["outcome"] == "confirmed-2-of-3-seed-sensitive"
    assert decision["positive_by_seed"] == {
        "seed42": True,
        "seed43": False,
        "seed44": True,
    }
    assert decision["positive_seed_count"] == 2
    assert decision["capacity_sibling_activated"] is True
    assert decision["retention_summary"]["mean"] == pytest.approx(
        0.9660293039374669
    )
    assert decision["retention_summary"]["sample_sd_ddof1"] == pytest.approx(
        0.018259992401468972
    )
    assert decision["width_null_noise_scale"]["value"] == pytest.approx(
        0.011634861408714734
    )


def test_one_of_three_does_not_activate_capacity() -> None:
    cells = {
        "seed42": {"half": _view(0.5), "full": _view(0.49)},
        "seed43": {"half": _view(0.5), "full": _view(0.48)},
        "seed44": {"half": _view(0.5), "full": _view(0.49)},
    }
    decision = synthesize(
        cells=cells,
        quarter_seed42=_view(0.5),
        fineweb_seed42={metric: 0.5 for metric in GATE_METRICS},
        gates=_gates(),
    )
    assert decision["outcome"] == "not-confirmed-across-three-seeds"
    assert decision["capacity_sibling_activated"] is False


def test_metric_or_seed_drift_fails_closed() -> None:
    cells = {
        "seed42": {"half": _view(0.5), "full": _view(0.49)},
        "seed43": {"half": _view(0.5), "full": _view(0.48)},
    }
    with pytest.raises(Round0190Error, match="cell set"):
        synthesize(
            cells=cells,
            quarter_seed42=_view(0.5),
            fineweb_seed42={metric: 0.5 for metric in GATE_METRICS},
            gates=_gates(),
        )


def test_all_bound_source_artifacts_are_sealed_and_expected() -> None:
    decisions = {
        key: prompt_contract.read_sealed(path, label=f"R{key} decision")
        for key, path in DECISION_PATHS.items()
    }
    assert decisions["0187"]["decision"]["outcome"] == (
        "composition-controlled-size-regression"
    )
    assert decisions["0188"]["decision"]["outcome"] == (
        "composition-controlled-size-regression-not-replicated"
    )
    assert decisions["0189"]["decision"]["outcome"] == (
        "composition-controlled-size-regression-seed44-positive"
    )
    for path in EVALUATION_PATHS.values():
        receipt = prompt_contract.read_sealed(path, label="evaluation")
        assert receipt["round_id"] in {"0187", "0188", "0189"}
        assert receipt["execution_checks"]
        assert all(receipt["execution_checks"].values())
    assert prompt_contract.read_sealed(R0160_FAMILY, label="family")["round_id"] == "0160"
    assert prompt_contract.read_sealed(R0161_GATES, label="gates")["registered"] is True


def test_unknown_action_fails_closed() -> None:
    with pytest.raises(Round0190Error, match="does not authorize"):
        nodes.run_job({"manifest": {"round_id": "0190"}}, {"action": "train"})
