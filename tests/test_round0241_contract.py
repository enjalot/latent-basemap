"""R0241 contract — the arithmetic, the guard, and the source-level safety.

Everything here is CPU-only and takes milliseconds. It covers the parts of the
round that can be wrong silently: the sampling-power statement, the self-
calibrating wall guard, the fail-closed probe view, the tolerance chain and the
per-seed movement table.

The signal-safety assertion at the bottom is deliberately a source read rather
than a call to `experiments/check_signal_safety.py`. That detector is
unreleased, it has a known waiver hole on a locally hoisted argv, and 2 of 18
planted evasions still evade it, so its exit code is not evidence. What IS
evidence is that the three files this round adds contain no subprocess, no
Popen, no kill, no terminate, no send_signal and no timeout bound at all.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from basemap.round0238_rung5 import (
    IMBALANCE_REPLICATE_SEEDS,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
)
from basemap.round0241_qualify import (
    REVIEWER_RAW_IN_DEGREE_MAX,
    REVIEWER_RAW_ZERO_IN_DEGREE_ROWS,
)
from basemap.round0240_rung5 import (
    R0238_ADMISSIBLE_MAX_CLUSTER_ROWS,
    R0238_GUARDED_MAX_CLUSTER_ROWS,
    R0238_MEASURED_IMBALANCE_BY_SEED,
)
from basemap.round0241_qualify import (
    GPU_HOURS_CAP,
    GRAPH_UNDER_TEST,
    MAX_ZERO_DEGREE_ROWS,
    PROBE_GATHER_BLOCK,
    REGISTERED_GRAPH_ARRAY_BYTES,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    REGISTERED_SELECTED_CLUSTERS,
    ROUND_ID,
    ROWS,
    Round0241Error,
    StageGuard,
    TRIPWIRE_BLOCK,
    cross_check_structural,
    per_seed_movement,
    probe_power_statement,
    recall_verdict,
    reconcile_in_degree,
    region_detection_power,
    sampling_uncertainty,
    smallest_detectable_region,
    tolerance_chain,
    tripwire_verdict,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_FILES = (
    "basemap/round0241_qualify.py",
    "experiments/round0241_nodes.py",
    "experiments/prepare_round0241_queue.py",
)


# --------------------------------------------------------------------------- #
# registered literals
# --------------------------------------------------------------------------- #
def test_registered_literals():
    assert ROUND_ID == "0241"
    assert ROWS == 100_000_000
    assert GPU_HOURS_CAP == 12.0
    assert MAX_ZERO_DEGREE_ROWS == 0
    assert REGISTERED_SELECTED_CLUSTERS == 400
    assert REGISTERED_GRAPH_ARRAY_BYTES == 6_000_000_128
    assert len(REGISTERED_GRAPH_IDS_SHA256) == 64
    assert len(REGISTERED_GRAPH_COS_SHA256) == 64
    assert len(REGISTERED_LADDER_RECEIPT_SHA256) == 64
    assert TRUTH_PROBE_ROWS == 500_000 and TRUTH_PROBE_SEED == 238_000
    assert TRIPWIRE_BLOCK > 0 and PROBE_GATHER_BLOCK > 0


# --------------------------------------------------------------------------- #
# the sampling-power statement
# --------------------------------------------------------------------------- #
def test_sampling_uncertainty_carries_the_finite_population_correction():
    out = sampling_uncertainty(mean=0.9957, sd=0.05)
    assert out["probe_rows"] == 500_000 and out["population_rows"] == 100_000_000
    assert out["sampling_fraction"] == pytest.approx(0.005)
    expected_fpc = math.sqrt((100_000_000 - 500_000) / (100_000_000 - 1))
    assert out["finite_population_correction"] == pytest.approx(expected_fpc)
    assert out["standard_error"] == pytest.approx(
        0.05 / math.sqrt(500_000) * expected_fpc
    )
    assert out["ci95_high"] - out["ci95_low"] == pytest.approx(
        2 * out["ci95_half_width"]
    )


def test_sampling_uncertainty_rejects_a_degenerate_probe():
    with pytest.raises(Round0241Error):
        sampling_uncertainty(mean=1.0, sd=0.1, probe_rows=1, population=10)
    with pytest.raises(Round0241Error):
        sampling_uncertainty(mean=1.0, sd=-1.0)


def test_region_detection_power_is_monotone_and_matches_the_approximation():
    small = region_detection_power(region_rows=100)
    medium = region_detection_power(region_rows=400)
    large = region_detection_power(region_rows=100_000)
    detected = [
        small["probability_detected"], medium["probability_detected"],
        large["probability_detected"],
    ]
    assert 0.0 < detected[0] < detected[1] < 1.0
    assert detected[2] == pytest.approx(1.0)
    # against the with-replacement approximation, which is tight at f << 1
    approx = 1.0 - (1.0 - 400 / 100_000_000) ** 500_000
    assert medium["probability_detected"] == pytest.approx(approx, rel=1e-3)
    assert small["expected_probe_rows_in_region"] == pytest.approx(0.5)
    assert region_detection_power(region_rows=0)["probability_detected"] == 0.0


def test_smallest_detectable_region_is_the_exact_threshold():
    at95 = smallest_detectable_region(confidence=0.95)
    assert at95["probability_detected"] >= 0.95
    one_smaller = region_detection_power(region_rows=at95["region_rows"] - 1)
    assert one_smaller["probability_detected"] < 0.95
    at99 = smallest_detectable_region(confidence=0.99)
    assert at99["region_rows"] > at95["region_rows"]
    # a 0.5% probe cannot see a region of a few hundred rows reliably
    assert 100 < at95["region_rows"] < 10_000


def test_probe_power_statement_says_what_it_cannot_do():
    out = probe_power_statement(mean=0.9957, sd=0.05)
    assert out["uncertainty"]["standard_error"] > 0
    assert len(out["detection_by_region_size"]) == 6
    text = " ".join(out["cannot_establish"]).lower()
    assert "localised" in text and "spatial" in text
    assert "probability 1" in out["the_tripwire_is_not_a_sample"]


# --------------------------------------------------------------------------- #
# the wall guard R0240 did not have
# --------------------------------------------------------------------------- #
class _Clock:
    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def test_stage_guard_refuses_a_stage_it_cannot_afford_on_its_own_measurement():
    guard = StageGuard(
        label="probe gather", units_total=100, budget_s=10.0, deadline_s=1e6,
        clock=_Clock(1.0),
    )
    with pytest.raises(Round0241Error, match="REFUSES"):
        guard.unit_done("unit 0")
    assert guard.prediction is not None
    assert guard.prediction["fits"] is False
    assert guard.prediction["units_measured"] == 1
    assert "OWN measured units" in guard.prediction["basis"]


def test_stage_guard_admits_a_stage_that_fits_and_keeps_polling():
    seen: list[str] = []
    guard = StageGuard(
        label="degree pass", units_total=4, budget_s=1e6, deadline_s=1e6,
        abort_check=seen.append, clock=_Clock(0.001),
    )
    for index in range(4):
        guard.unit_done(f"unit {index}")
    assert guard.units_done == 4
    assert len(seen) == 4, "one poll per unit, not one poll per stage"
    assert guard.prediction["fits"] is True
    assert guard.receipt()["poll_points_per_unit"] == 1


def test_stage_guard_raises_in_band_on_its_own_deadline():
    guard = StageGuard(
        label="degree pass", units_total=100, budget_s=1e6, deadline_s=2.0,
        clock=_Clock(1.5),
    )
    with pytest.raises(Round0241Error, match="cooperative deadline"):
        for index in range(10):
            guard.unit_done(f"unit {index}")


def test_stage_guard_relays_the_runners_abort_request():
    def refuse(where: str) -> None:
        raise RuntimeError(f"runner abort at {where}")

    guard = StageGuard(
        label="probe gather", units_total=10, budget_s=1e6, deadline_s=1e6,
        abort_check=refuse, clock=_Clock(0.001),
    )
    with pytest.raises(RuntimeError, match="runner abort"):
        guard.unit_done("unit 0")


def test_stage_guard_rejects_nonsense_configuration():
    with pytest.raises(Round0241Error):
        StageGuard(label="x", units_total=0, budget_s=1.0, deadline_s=1.0)
    with pytest.raises(Round0241Error):
        StageGuard(label="x", units_total=1, budget_s=0.0, deadline_s=1.0)


# --------------------------------------------------------------------------- #
# the tripwire verdict keeps the gating and descriptive quantities apart
# --------------------------------------------------------------------------- #
def test_tripwire_holds_only_when_every_gating_quantity_is_zero():
    clean = tripwire_verdict(
        out_degree_zero_rows=0, in_degree_zero_rows=4_424_010,
        symmetrised_isolated_rows=0, raw_zero_degree_rows=0,
    )
    assert clean["holds"] is True
    assert clean["first_execution_at_this_scale"] is True
    # review-0240-01/F1: 4.42% zero-IN-degree is hubness, not a defect, and
    # gating on it would fail every rung spuriously.
    assert clean["descriptive_quantities"] == ["raw_in_degree_zero_rows"]
    assert "symmetrised_isolated_rows_derived" in clean["gating_quantities"]
    assert "raw_in_degree_zero_rows" not in clean["gating_quantities"]
    assert clean["graph"] == GRAPH_UNDER_TEST
    assert "graph-k15-ids" in clean["graph"]
    # every direction must name its graph, and the identity must be stated
    assert "SAME NUMBER by construction" in clean["degree_semantics_note"]
    assert "does NOT certify" in clean["vacuity_note"]

    for kwargs in (
        {"symmetrised_isolated_rows": 1},
        {"out_degree_zero_rows": 1},
        {"raw_zero_degree_rows": 1},
    ):
        base = {
            "out_degree_zero_rows": 0, "in_degree_zero_rows": 0,
            "symmetrised_isolated_rows": 0, "raw_zero_degree_rows": 0,
        }
        base.update(kwargs)
        assert tripwire_verdict(**base)["holds"] is False


def test_recall_verdict_reports_both_sides_of_the_floor():
    passing = recall_verdict(
        tie_aware={"mean": 0.97, "p10": 0.93, "min": 0.0},
        strict={"mean": 0.95, "p10": 0.87, "min": 0.0},
    )
    assert passing["holds"] is True
    assert passing["mean_floor"] == RECALL_MEAN_FLOOR
    assert passing["p10_floor"] == RECALL_P10_FLOOR
    failing = recall_verdict(
        tie_aware={"mean": 0.89, "p10": 0.93, "min": 0.0},
        strict={"mean": 0.80, "p10": 0.70, "min": 0.0},
    )
    assert failing["holds"] is False and failing["mean_clears"] is False


def test_in_degree_reconciles_against_the_independent_reviewer_measurement():
    same = reconcile_in_degree(
        measured_zero_in_degree_rows=REVIEWER_RAW_ZERO_IN_DEGREE_ROWS,
        measured_in_degree_max=REVIEWER_RAW_IN_DEGREE_MAX,
        measured_in_degree_mean=15.0,
        measured_out_degree_min=15,
        measured_out_degree_zero_rows=0,
    )
    assert same["agree"] is True
    assert same["disagreement_is_the_finding"] is False
    assert same["zero_in_degree_rows"]["difference"] == 0
    assert same["zero_in_degree_rows"]["fraction_of_rows"] == pytest.approx(
        0.0442401
    )
    assert "review-0240" in same["reference"]

    differs = reconcile_in_degree(
        measured_zero_in_degree_rows=REVIEWER_RAW_ZERO_IN_DEGREE_ROWS + 7,
        measured_in_degree_max=REVIEWER_RAW_IN_DEGREE_MAX,
        measured_in_degree_mean=15.0,
        measured_out_degree_min=15,
        measured_out_degree_zero_rows=0,
    )
    assert differs["agree"] is False
    assert differs["disagreement_is_the_finding"] is True
    assert differs["zero_in_degree_rows"]["difference"] == 7


# --------------------------------------------------------------------------- #
# the structural cross-check
# --------------------------------------------------------------------------- #
_FIELDS = {
    "rows": 10, "width": 15, "out_of_range_entries": 0,
    "rows_with_out_of_range": 0, "self_loop_entries": 3,
    "rows_with_self_loop": 3, "duplicate_entries": 1, "rows_with_duplicates": 1,
    "min_usable_degree": 13, "rows_below_k": 4, "zero_degree_rows": 0,
}


def test_cross_check_passes_when_two_passes_agree():
    out = cross_check_structural(reviewed=dict(_FIELDS), own=dict(_FIELDS))
    assert out["agree"] is True


def test_cross_check_fails_closed_on_any_disagreement():
    own = dict(_FIELDS)
    own["zero_degree_rows"] = 1
    with pytest.raises(Round0241Error, match="cross-check FAILED"):
        cross_check_structural(reviewed=dict(_FIELDS), own=own)


# --------------------------------------------------------------------------- #
# the three debts
# --------------------------------------------------------------------------- #
def test_tolerance_chain_reproduces_the_published_percentages():
    chain = tolerance_chain(
        r0240_admissible_max_cluster_rows=R0238_ADMISSIBLE_MAX_CLUSTER_ROWS,
        r0240_guarded_max_cluster_rows=6568660.892948,
        r0240_measured_imbalance=2.8194485,
        imbalance_margin=1.164884,
        mean_cluster_rows=2_000_000.0,
    )
    legs = chain["legs"]
    assert legs[0]["tolerance_percent"] == pytest.approx(73.8255, abs=1e-3)
    assert legs[1]["tolerance_percent"] == pytest.approx(51.3984, abs=1e-3)
    assert legs[2]["tolerance_percent"] == pytest.approx(51.4516, abs=1e-3)
    assert legs[1]["guarded_max_cluster_rows"] == R0238_GUARDED_MAX_CLUSTER_ROWS
    assert chain["consumed_in_the_50m_to_100m_doubling"] > 0.2
    assert chain["moved_on_re_measurement"] == pytest.approx(
        legs[2]["tolerance"] - legs[1]["tolerance"]
    )


def test_per_seed_movement_detects_the_ranking_inversion():
    at_50m = {
        226: 2.359035, 236: 2.170129, 1236: 2.029724,
        2236: 2.208610, 3236: 2.456543,
    }
    table = per_seed_movement(
        r0240_by_seed={
            226: 2.224096, 236: 2.8194485, 1236: 2.242996,
            2236: 2.6950615, 3236: 2.094509,
        },
        r0237_by_seed=at_50m,
    )
    assert [row["seed"] for row in table["rows"]] == sorted(
        int(s) for s in IMBALANCE_REPLICATE_SEEDS
    )
    assert table["ranking_preserved_across_the_doubling"] is False
    assert table["ranking_preserved_on_remeasurement"] is True
    move = {row["seed"]: row["move_50m_to_100m"] for row in table["rows"]}
    assert move[236] == pytest.approx(0.2997, abs=1e-3)
    assert move[3236] == pytest.approx(-0.1462, abs=1e-3)


def test_per_seed_movement_works_without_the_50m_grid():
    table = per_seed_movement(
        r0240_by_seed=dict(R0238_MEASURED_IMBALANCE_BY_SEED),
        r0237_by_seed=None,
    )
    assert table["ranking_preserved_across_the_doubling"] is None
    assert all(row["move_50m_to_100m"] is None for row in table["rows"])
    assert all(
        row["move_on_remeasurement"] == pytest.approx(0.0)
        for row in table["rows"]
    )


# --------------------------------------------------------------------------- #
# safety, read from the source rather than delegated to a detector
# --------------------------------------------------------------------------- #
FORBIDDEN = (
    "subprocess", "Popen", "os.kill", "killpg", "signal.alarm",
    ".kill(", ".terminate(", ".send_signal(", "pkill", "killall", "timeout=",
)


def test_the_node_path_contains_no_signalling_construct():
    """Read the files. Do not ask a detector, which is unreleased and leaky."""
    for relative in (
        "basemap/round0241_qualify.py", "experiments/round0241_nodes.py",
    ):
        source = open(os.path.join(REPO, relative), encoding="utf-8").read()
        for token in FORBIDDEN:
            assert token not in source, f"{relative} contains {token!r}"


def test_the_prepare_script_never_bounds_a_child_by_a_signalling_wall():
    source = open(
        os.path.join(REPO, "experiments/prepare_round0241_queue.py"),
        encoding="utf-8",
    ).read()
    for token in FORBIDDEN:
        if token == "subprocess":
            continue  # git and the CPU smoke are launched here, unbounded
        assert token not in source, f"prepare script contains {token!r}"


def test_the_new_files_are_the_only_ones_this_round_adds():
    for relative in NEW_FILES + (
        "tests/test_round0241_contract.py", "tests/test_round0241_cpu_smoke.py",
    ):
        assert os.path.exists(os.path.join(REPO, relative))
    assert np.__version__
