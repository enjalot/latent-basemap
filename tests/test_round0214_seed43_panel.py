from __future__ import annotations

import os

import pytest

from basemap.round0166_prompted_8m import METRICS
from basemap.round0214_seed43_panel import (
    CAPABILITY,
    CELLS_REQUIRED_FOR_GATE,
    PAIRED_REFERENCE,
    ROUND_ID,
    Round0214Error,
    SEED,
    descriptive_panel_decision,
    paired_spread,
)
from experiments import round0214_nodes as nodes


HEALTHY = {
    "density_v2": 0.1787, "ffr": 0.6377, "purity_fidelity_k256": 0.95129,
    "purity_fidelity_k1024": 0.9495, "projection_ffr": 0.4465,
    "heldout_recall_at_10": 0.00845,
}
FLOORS = {m: 0.99 for m in METRICS}   # deliberately unreachable


def _inputs(**over):
    from basemap.round0105_search import GROUPS
    p = {
        "native": dict(HEALTHY), "matched_2m": dict(HEALTHY),
        "baseline_2m_seed42": dict(HEALTHY), "prompted_floors": dict(FLOORS),
        "group_ffr": {n: 0.60 for n in GROUPS},
        "prompted_ood": {"polish_recall_at_50_of_high10": 0.295,
                         "in_mix_median_recall_at_50_of_high10": 0.2534},
        "raw_r0132_ood": {"polish_recall_at_50_of_high10": 0.2582,
                          "in_mix_median_recall_at_50_of_high10": 0.239},
    }
    p.update(over)
    return p


def test_identity() -> None:
    assert ROUND_ID == "0214" and SEED == 43
    assert CAPABILITY == "jina-prompted-diverse-u12-seed43-panel-readout-v1"


def test_no_quality_metric_can_fail_this_round() -> None:
    """Every floor unreachable, yet the decision still passes: nothing is decisive."""
    PAIRED_REFERENCE.clear()
    d = descriptive_panel_decision(**_inputs())
    assert d["passed"] is True
    assert d["decisive_quality_gate_registered"] is False
    assert d["atlas_quality_claim_available"] is False
    assert d["seed_family"]["gate_registerable_here"] is False
    assert d["seed_family"]["cells_required_for_gate"] == CELLS_REQUIRED_FOR_GATE == 3
    for cell in d["descriptive_cells"]["native_absolute_cells"].values():
        assert cell["role"] == "descriptive"
        assert "passed" not in cell


def test_a_collapsed_map_still_reports_rather_than_judges() -> None:
    d = descriptive_panel_decision(**_inputs(native={m: 1e-6 for m in METRICS}))
    assert d["passed"] is True  # execution gates decide, not quality
    assert d["decisive_quality_gate_registered"] is False


def test_paired_spread_reports_difference_and_refuses_a_sigma() -> None:
    spread = paired_spread(
        this_cell={m: 1.0 for m in METRICS},
        paired_cell={m: 0.8 for m in METRICS},
    )
    assert spread["n_cells"] == 2
    assert spread["sigma_estimated"] is False
    assert "sigma" not in str(spread["cells"])
    for cell in spread["cells"].values():
        assert cell["absolute_difference"] == pytest.approx(0.2)
        assert cell["relative_difference"] == pytest.approx(0.25)
    assert spread["largest_relative_difference"] == pytest.approx(0.25)


def test_spread_is_attached_only_when_a_paired_cell_is_bound() -> None:
    PAIRED_REFERENCE.clear()
    assert "paired_native_spread" not in descriptive_panel_decision(**_inputs())
    PAIRED_REFERENCE.update(
        {"native_decision_metrics": {m: 0.5 for m in METRICS}}
    )
    try:
        d = descriptive_panel_decision(**_inputs())
        assert "paired_native_spread" in d
        assert d["paired_native_spread"]["n_cells"] == 2
    finally:
        PAIRED_REFERENCE.clear()


def test_missing_metric_in_the_paired_cell_fails_closed() -> None:
    with pytest.raises(Round0214Error):
        paired_spread(this_cell={m: 1.0 for m in METRICS}, paired_cell={"ffr": 0.5})


def test_bound_paths_point_at_the_queues_that_sealed() -> None:
    from experiments.prepare_round0214_queue import R0211_EVALUATION, TRAIN_OUTPUT

    assert "queue-cap-corrected" in TRAIN_OUTPUT
    assert "queue-correction-2" in R0211_EVALUATION
    for path in (os.path.join(TRAIN_OUTPUT, "model.pt"), R0211_EVALUATION):
        if not os.path.exists(path):
            pytest.skip(f"{path} is not present on this machine")
    assert os.path.exists(os.path.join(TRAIN_OUTPUT, "train-receipt.json"))
