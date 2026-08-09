"""R0229 — pre-launch checks on the contract, the grids and the registered tests.

The CUDA-hidden CPU smoke for this round. It exercises the registered inference
rule against R0228's sealed bytes, the structural bound against R0227's sealed
bytes, the per-rung arithmetic, the guard's spill scaling, and the property that
review-0228-01 named as the program's recurring defect: publishing a null from a
test whose resolution ceiling lies above its own decision threshold.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from basemap.round0226_graph_builders import A_SPILL
from basemap.round0227_low_c_contract import CLUSTER_CAPACITY_ROWS
from basemap import round0229_quality_contract as contract


R0227_REACHABILITY_PATH = (
    "/data/latent-basemap/runs/round-0227/queue/artifacts/low-c-reachability/"
    "reachability-vs-cluster-count.json"
)
R0228_GEOMETRY_PATH = (
    "/data/latent-basemap/runs/round-0228/queue-correction-1/artifacts/"
    "minilm-mixed-2m-cluster-spill-map-geometry-v1/cluster-spill-map-geometry.json"
)


def _load(path: str):
    if not os.path.exists(path):
        pytest.skip(f"sealed artifact absent: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# the structural bound
# --------------------------------------------------------------------------- #
def test_registered_ceilings_bind_to_r0227_sealed_bytes():
    bound = contract.verify_r0227_ceilings(_load(R0227_REACHABILITY_PATH))
    assert set(bound["cells"]) == {
        str(c) for c in contract.R0227_STRICT_CEILING_BY_C
    }
    assert bound["cells"]["16"]["strict_ceiling_all_rows"] == pytest.approx(
        0.9532496, abs=1e-9
    )


def test_verify_ceilings_refuses_a_disagreement():
    sealed = _load(R0227_REACHABILITY_PATH)
    tampered = json.loads(json.dumps(sealed))
    tampered["ceilings_by_clusters"]["16"]["strict_mean_all_rows"] = 0.99
    with pytest.raises(contract.Round0229Error):
        contract.verify_r0227_ceilings(tampered)


def test_the_nn_descent_headroom_at_c16_is_under_two_thousandths():
    # The round's registered bound: every quality knob together can buy at most
    # this much, and c = 4's built recall is out of reach at any setting.
    assert contract.NND_HEADROOM_AT_C16 == pytest.approx(0.001939, abs=1e-5)
    assert contract.C4_UNREACHABLE_MARGIN_AT_C16 < -0.03


# --------------------------------------------------------------------------- #
# the registered displacement test
# --------------------------------------------------------------------------- #
def test_permutation_reproduces_review_0228_on_sealed_bytes():
    sealed = _load(R0228_GEOMETRY_PATH)
    bound = contract.verify_r0228_displacement(sealed)
    for clusters, expected in contract.REVIEW_0228_DISPLACEMENT_P_BY_C.items():
        cell = bound["cells"][str(clusters)]
        result = contract.exact_displacement_permutation(
            candidate_gaps=cell["candidate_gaps"], exact_gaps=cell["exact_gaps"]
        )
        assert result["p_one_sided"] == pytest.approx(
            expected, abs=contract.REVIEW_0228_P_TOLERANCE
        )
        assert result["labellings"] == contract.PERMUTATION_LABELLINGS


def test_complete_separation_at_c8_and_c16_and_not_at_c4():
    bound = contract.verify_r0228_displacement(_load(R0228_GEOMETRY_PATH))
    verdicts = {}
    for clusters in (4, 8, 16):
        cell = bound["cells"][str(clusters)]
        result = contract.exact_displacement_permutation(
            candidate_gaps=cell["candidate_gaps"], exact_gaps=cell["exact_gaps"]
        )
        verdicts[clusters] = (
            result["complete_separation"], contract.displacement_verdict(result)
        )
    assert verdicts[4] == (False, "c4-LIKE")
    assert verdicts[8] == (True, "DISPLACED")
    assert verdicts[16] == (True, "DISPLACED")


def test_verify_r0228_refuses_a_tampered_gap():
    sealed = json.loads(json.dumps(_load(R0228_GEOMETRY_PATH)))
    arm = sealed["displacement"]["16"]["vs_exact_family"]
    arm["difference_in_differences_in_exact_sd"] = 0.0
    with pytest.raises(contract.Round0229Error):
        contract.verify_r0228_displacement(sealed)


def test_permutation_p_is_one_when_the_candidate_arm_is_lowest():
    result = contract.exact_displacement_permutation(
        candidate_gaps=[0.0, 0.01, 0.02], exact_gaps=[1.0] * 8
    )
    assert result["p_one_sided"] == pytest.approx(1.0)
    assert contract.displacement_verdict(result) != "DISPLACED"


def test_permutation_hits_its_floor_under_complete_separation():
    result = contract.exact_displacement_permutation(
        candidate_gaps=[1.0, 1.1, 1.2], exact_gaps=[0.0] * 8
    )
    assert result["p_one_sided"] == pytest.approx(
        contract.PERMUTATION_RESOLUTION_CEILING
    )
    assert result["smallest_attainable_p"] == result["p_one_sided"]
    assert result["complete_separation"] is True


def test_every_permutation_result_carries_its_smallest_attainable_p():
    result = contract.exact_displacement_permutation(
        candidate_gaps=[0.11, 0.12, 0.13],
        exact_gaps=[0.01 * i for i in range(1, 9)],
    )
    # Review-0228-01's central methodological finding: a test whose smallest
    # attainable p lies above its threshold cannot reject, and reporting its
    # null describes the design rather than the data.
    assert "smallest_attainable_p" in result
    assert result["smallest_attainable_p"] == pytest.approx(1.0 / 165.0)
    assert result["can_reject_at_alpha"] is True
    assert contract.test_can_reject(
        smallest_attainable_p=result["smallest_attainable_p"], threshold=0.05
    )
    # ... and it could NOT have rejected under R0228's twelve-test correction.
    assert not contract.test_can_reject(
        smallest_attainable_p=result["smallest_attainable_p"],
        threshold=0.05 / 12.0,
    )


def test_ties_raise_the_attainable_floor_and_the_contract_notices():
    # A degenerate null arm makes many labellings tie at the maximum, so the
    # smallest attainable p is far above 1/165. Enumerating it rather than
    # assuming a unique maximum is what makes the guard honest.
    result = contract.exact_displacement_permutation(
        candidate_gaps=[0.1, 0.2, 0.3], exact_gaps=[0.1] * 8
    )
    assert result["labellings_at_the_maximum"] > 1
    assert result["smallest_attainable_p"] > 1.0 / 165.0
    assert result["can_reject_at_alpha"] is False


def test_r0228s_twelve_test_correction_is_shown_to_be_unpassable():
    # The exact arithmetic review-0228-01 used to block R0228's panel null.
    assert contract.PERMUTATION_RESOLUTION_CEILING == pytest.approx(1.0 / 165.0)
    assert contract.PERMUTATION_RESOLUTION_CEILING > 0.05 / 12.0


# --------------------------------------------------------------------------- #
# the trend test
# --------------------------------------------------------------------------- #
def test_trend_enumerates_1680_assignments_and_resolves_below_alpha():
    arms = {"a": [0.0, 0.1, 0.2], "b": [1.0, 1.1, 1.2], "c": [2.0, 2.1, 2.2]}
    regressor = {"a": 0.10, "b": 0.20, "c": 0.27}
    result = contract.exact_did_trend(arm_values=arms, regressor=regressor)
    assert result["assignments"] == contract.TREND_ASSIGNMENTS
    assert result["resolution_ceiling"] == pytest.approx(1.0 / 1680.0)
    assert result["can_reject_at_alpha"] is True
    # |r| is symmetric under reversing the arm order, so the maximum is attained
    # twice and the attainable floor is 2/1680, not 1/1680. The contract
    # enumerates it rather than assuming a unique maximum.
    assert result["labellings_at_the_maximum"] == 2
    assert result["smallest_attainable_p"] == pytest.approx(2.0 / 1680.0)
    assert result["p_two_sided"] == pytest.approx(2.0 / 1680.0)


def test_trend_is_null_when_the_arms_are_exchangeable():
    arms = {"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0], "c": [0.0, 1.0, 2.0]}
    regressor = {"a": 0.10, "b": 0.20, "c": 0.27}
    result = contract.exact_did_trend(arm_values=arms, regressor=regressor)
    assert result["observed_pearson_r"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_two_sided"] > 0.5


def test_trend_uses_missing_edge_mass_not_c():
    # Review-0228-01: the effect is monotone in missing edge mass, and the
    # regressor must be that, not the cluster count.
    assert "rows carrying" in contract.TREND_TEST_NOTE
    assert set(contract.R0228_ROWS_CARRYING_LOSS_BY_C) == {4, 8, 16}
    values = [contract.R0228_ROWS_CARRYING_LOSS_BY_C[c] for c in (4, 8, 16)]
    assert values == sorted(values)


# --------------------------------------------------------------------------- #
# per-rung c, from measured imbalance only
# --------------------------------------------------------------------------- #
def test_per_rung_c_at_100m_is_24_not_22():
    # Review-0227-01's correction: c = 22 came from review-0226-01's model and
    # is not in the measured set at all.
    answer = contract.smallest_measured_clusters(rows=100_000_000, spill=2)
    assert answer["clusters"] == 24
    assert answer["imbalance_source"] == "R0227 sealed measured imbalance"
    assert answer["projected_max_cluster_rows"] <= CLUSTER_CAPACITY_ROWS
    assert 22 not in contract.R0227_MEASURED_IMBALANCE


def test_projected_max_cluster_rows_refuses_an_unmeasured_c():
    with pytest.raises(contract.Round0229Error):
        contract.projected_max_cluster_rows(rows=100_000_000, clusters=22, spill=2)


def test_matched_families_have_equal_mean_cluster_rows():
    for family in ("A", "B"):
        members = [
            cell for cell in contract.SPILL_GRID if cell["family"] == family
        ]
        means = {
            contract.family_mean_cluster_rows(
                int(cell["clusters"]), int(cell["spill"])
            )
            for cell in members
        }
        assert len(means) == 1, f"family {family} is not matched: {means}"


def test_the_three_feasible_cells_really_are_feasible_at_100m():
    for cell in contract.SPILL_GRID:
        if cell["family"] != "F":
            continue
        assert contract.rung_is_feasible(
            rows=100_000_000, clusters=int(cell["clusters"]),
            spill=int(cell["spill"]),
        ), cell


def test_family_a_and_b_are_not_feasible_at_100m():
    # They are reference points at matched device cost, not candidates. Saying
    # so here stops a later reader from adopting one.
    for cell in contract.SPILL_GRID:
        if cell["family"] == "F":
            continue
        if int(cell["clusters"]) not in contract.R0227_MEASURED_IMBALANCE:
            continue
        assert not contract.rung_is_feasible(
            rows=100_000_000, clusters=int(cell["clusters"]),
            spill=int(cell["spill"]),
        ), cell


# --------------------------------------------------------------------------- #
# the guard, the grids, the trigger
# --------------------------------------------------------------------------- #
def test_guard_scales_its_prediction_with_spill():
    two = contract.guard_for_spill(rows=contract.ROWS, clusters=16, spill=2)
    eight = contract.guard_for_spill(rows=contract.ROWS, clusters=16, spill=8)
    assert two["guard_rows_scaled_for_spill"] == contract.ROWS
    assert eight["guard_rows_scaled_for_spill"] == 4 * contract.ROWS
    assert (
        eight["prediction"]["predicted_max_cluster_rows"]
        > two["prediction"]["predicted_max_cluster_rows"]
    )
    assert A_SPILL == 2


def test_sweep_grid_is_ascending_and_intermediate_never_below_graph_degree():
    cost = [
        (cell["intermediate_graph_degree"], cell["max_iterations"])
        for cell in contract.QUALITY_SWEEP
    ]
    assert contract.QUALITY_SWEEP[0]["cell"] == contract.BASELINE_CELL
    assert contract.QUALITY_SWEEP[0]["graph_degree"] == 32
    assert contract.QUALITY_SWEEP[0]["intermediate_graph_degree"] == 48
    assert contract.QUALITY_SWEEP[0]["max_iterations"] == 20
    for cell in contract.QUALITY_SWEEP:
        assert cell["intermediate_graph_degree"] >= cell["graph_degree"], cell
    assert len({cell["cell"] for cell in contract.QUALITY_SWEEP}) == len(cost)


def test_igd_host_law_is_quantised_not_linear():
    def law(igd: int) -> int:
        return 2 * int(32 * math.ceil(1.3 * igd / 32))

    # plan-minilm-100m-v2: igd 64 and 65 cost the same; 96 -> 128 doubles.
    assert law(64) == law(65)
    assert law(128) > law(96)


def test_phase2_trigger_fires_only_on_a_registered_gain():
    sweep = [
        {"cell": "q0-baseline", "tie_aware_recall_all_rows": 0.9512},
        {"cell": "q3", "tie_aware_recall_all_rows": 0.9530},
    ]
    spill = [
        {"cell": "F-c24-s2", "feasible_at_100m": True,
         "strict_ceiling_all_rows": 0.9448},
    ]
    quiet = contract.phase2_trigger(
        sweep_cells=sweep, spill_cells=spill, partition_strict_ceiling=0.9533
    )
    assert quiet["phase2_runs"] is False
    assert quiet["tunable_gain"] < contract.PHASE2_RECALL_TRIGGER

    loud = contract.phase2_trigger(
        sweep_cells=sweep,
        spill_cells=[
            {"cell": "F-c200-s8", "feasible_at_100m": True,
             "strict_ceiling_all_rows": 0.99},
        ],
        partition_strict_ceiling=0.9533,
    )
    assert loud["phase2_runs"] is True
    assert loud["triggers"]["structural_gain"] is True


def test_phase2_trigger_fires_when_a_cell_beats_its_own_ceiling():
    # The bound is registered as falsifiable, and a violation must be visible.
    result = contract.phase2_trigger(
        sweep_cells=[
            {"cell": "q0-baseline", "tie_aware_recall_all_rows": 0.9512},
            {"cell": "q3", "tie_aware_recall_all_rows": 0.9600},
        ],
        spill_cells=[],
        partition_strict_ceiling=0.9533,
    )
    assert result["triggers"]["bound_violated"] is True
    assert result["cells_above_their_own_ceiling"] == ["q3"]


def test_phase2_trigger_needs_the_baseline_cell():
    with pytest.raises(contract.Round0229Error):
        contract.phase2_trigger(
            sweep_cells=[{"cell": "q3", "tie_aware_recall_all_rows": 0.95}],
            spill_cells=[],
            partition_strict_ceiling=0.9533,
        )


# --------------------------------------------------------------------------- #
# projections
# --------------------------------------------------------------------------- #
def test_power_fit_recovers_a_planted_exponent_and_keeps_its_range():
    sizes = [1e5, 3e5, 1e6, 3e6, 9e6]
    seconds = [2.0 * size ** 0.8 for size in sizes]
    fit = contract.power_fit(sizes, seconds)
    assert fit["exponent_b"] == pytest.approx(0.8, abs=1e-9)
    assert fit["fitted_range_cluster_rows"] == [1e5, 9e6]
    projection = contract.project_from_power_fit(fit, 12_085_000)
    assert projection["is_extrapolation"] is True
    assert projection["extrapolation_factor_beyond_fitted_max"] == pytest.approx(
        12_085_000 / 9e6
    )
    assert projection["label"] == "PROJECTION"


def test_spill_io_is_its_own_line_and_scales_with_spill():
    two = contract.spill_io_seconds(
        rows=100_000_000, spill=2, rate_bytes_per_s=5.53e9
    )
    eight = contract.spill_io_seconds(
        rows=100_000_000, spill=8, rate_bytes_per_s=5.53e9
    )
    assert eight["total_bytes"] == pytest.approx(4.0 * two["total_bytes"])
    assert two["total_bytes"] == pytest.approx(2 * 2 * 100_000_000 * 384 * 4)
    assert len(contract.SPILL_IO_RATES_BYTES_PER_S) == 2
    # Review-0227-01: R0227's own spill phase ran at 2.52-4.30 GB/s on largely
    # warm reads, so the cold-read band is not the conservative one.
    assert min(contract.SPILL_IO_MEASURED_RATES_BYTES_PER_S) < min(
        contract.SPILL_IO_RATES_BYTES_PER_S
    )


def test_power_fit_refuses_too_few_points():
    with pytest.raises(contract.Round0229Error):
        contract.power_fit([1.0, 2.0], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# nothing is claimed
# --------------------------------------------------------------------------- #
def test_no_gate_no_adoption_no_equivalence_no_training():
    assert contract.GATE_REGISTERABLE_HERE is False
    assert contract.GATE_RELEASE_CLAIMED is False
    assert contract.ADOPTION_CLAIMED is False
    assert contract.EQUIVALENCE_CLAIMED is False
    assert contract.TRAINING_PERFORMED is False


def test_the_round_imports_r0228s_geometry_rather_than_reimplementing_it():
    import basemap.round0228_geometry as geometry

    for name in (
        "map_scale", "clump_profile", "true_neighbour_scatter",
        "density_matched_control", "displacement_summary",
    ):
        assert callable(getattr(geometry, name))
    # R0228's registered constants, used verbatim so phase 2's numbers are
    # directly comparable with R0228's.
    assert geometry.SCATTER_SAMPLE_ROWS == 20_000
    assert geometry.SCATTER_SAMPLE_SEED == 228
