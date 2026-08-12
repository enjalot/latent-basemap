"""R0261 contract tests — the guards, their plants, and the registered price.

Every guard this round ships is exercised by a test that plants the defect into
the **shipped** function and asserts the shipped function refuses it, and by a
test that the clean input is accepted (a guard that refuses everything is not a
guard). review-0260-01 §D.2/§K downgraded R0260 because its central ordering
guard shipped with neither refusal branch exercised while nineteen tests covered
lesser claims; both refusal branches of R0261's ordering guard are planted here.
"""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0216_minilm_2m_substrate import Round0216Error
from basemap import round0261_four_m_graph as science


# --------------------------------------------------------------------------- #
# the registered universe
# --------------------------------------------------------------------------- #


def test_composition_is_r0216_at_exactly_two_times():
    from basemap.round0216_minilm_2m_substrate import COMPOSITION as TWO_M

    assert science.ROWS == 4_000_000
    assert sum(n for _name, n in science.COMPOSITION) == science.ROWS
    assert [name for name, _n in science.COMPOSITION] == [name for name, _n in TWO_M]
    for (name_a, n_a), (name_b, n_b) in zip(science.COMPOSITION, TWO_M):
        assert name_a == name_b
        assert n_a == 2 * n_b
    assert science.TARGET_SHARES == {name: n / 2_000_000 for name, n in TWO_M}


def test_block_geometry_matches_r0216_so_the_back_check_is_like_for_like():
    from experiments import round0216_nodes as r0216

    assert science.QUERY_BLOCK == r0216.QUERY_BLOCK
    assert science.SEARCH_BLOCK == r0216.SEARCH_BLOCK


def test_the_4m_substrate_is_not_claimed_to_be_nested():
    assert science.NESTED_IN_R0216 is False
    assert science.SELECTION_SEED != 216


# --------------------------------------------------------------------------- #
# the registered price
# --------------------------------------------------------------------------- #


def test_two_term_fit_recovers_both_points_exactly():
    fit = science.two_term_fit(rows_a=2_000_000, seconds_a=112.0,
                               rows_b=6_250_000, seconds_b=1048.0)
    assert science.predict_search_s(rows=2_000_000, **fit) == pytest.approx(112.0)
    assert science.predict_search_s(rows=6_250_000, **fit) == pytest.approx(1048.0)


def test_two_term_fit_refuses_a_degenerate_pair():
    with pytest.raises(science.Round0261Error):
        science.two_term_fit(rows_a=2_000_000, seconds_a=1.0,
                             rows_b=2_000_000, seconds_b=2.0)


def test_cost_prediction_spans_its_four_models_and_names_its_point():
    prediction = science.cost_prediction()
    models = prediction["models"]
    assert set(models) == {
        "M1_pure_quadratic_from_r0216", "M2_two_term_r0216_r0233",
        "M3_r0233_quadratic_plus_r0216_linear", "M4_half_of_r0216_is_linear"}
    low, high = prediction["interval_s"]
    assert low == min(models.values())
    assert high == max(models.values())
    assert prediction["point_estimate_s"] == models[prediction["point_estimate_model"]]
    assert low <= prediction["point_estimate_s"] <= high
    assert prediction["label"] == "prediction"
    # M1 is the pure-quadratic anchor: 4x the sealed 2M wall.
    assert models["M1_pure_quadratic_from_r0216"] == pytest.approx(
        4.0 * science.R0216_EXACT_SEARCH_S)


def test_prediction_carries_no_measured_4m_quantity():
    """It must survive the ordering guard's own second refusal branch."""
    prediction = science.cost_prediction()
    assert not [key for key in prediction if str(key).startswith("measured")]
    assert prediction["sources"].keys() == {"r0216", "r0233"}
    assert prediction["sources"]["r0216"]["rows"] == 2_000_000
    assert prediction["sources"]["r0233"]["rows"] == 6_250_000
    # Fed to the guard with a seal time, a genuine prediction passes; the same
    # dict with a measured 4M wall added does not.
    body = {"sealed_at_unix": 1.0, **prediction}
    assert science.assert_prediction_precedes_build(
        prediction=body, build_started_unix=2.0)["prediction_precedes_build"]
    with pytest.raises(science.Round0261Error, match="not a prediction"):
        science.assert_prediction_precedes_build(
            prediction={**body, "measured_exact_search_s": 434.0},
            build_started_unix=2.0)


def test_other_rungs_label_everything_except_4m_as_a_prediction():
    fit = science.cost_prediction()["m2_fit"]
    rungs = science.price_other_rungs(**fit)
    assert rungs["4000000"]["label"] == "measurement"
    for key, entry in rungs.items():
        if key != "4000000":
            assert entry["label"] == "prediction"


# --------------------------------------------------------------------------- #
# guard 1 — the R0215 degree-zero tripwire
# --------------------------------------------------------------------------- #


def test_degree_census_counts_a_planted_edgeless_row():
    sources = np.array([0, 0, 1, 1, 3, 3], dtype=np.int32)
    census = science.degree_census(sources, rows=4)
    assert census["zero_degree_rows"] == 1
    assert census["min"] == 0


def test_the_shipped_judge_refuses_one_edgeless_row():
    census = science.degree_census(np.array([0, 0, 1, 1, 3, 3], dtype=np.int32), rows=4)
    with pytest.raises(Round0216Error, match="zero edges"):
        science.validate_exact_graph(
            degrees=census,
            gating_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
            builder_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
            edges=6)


def test_the_shipped_judge_accepts_zero_edgeless_rows():
    checks = science.validate_exact_graph(
        degrees={"zero_degree_rows": 0, "min": 5, "median": 19.0,
                 "mean": 24.17, "max": 1394},
        gating_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
        builder_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
        edges=48_344_648)
    assert checks["zero_degree_rows"] == 0
    assert checks["zero_degree_tripwire"] == 0


def test_degree_census_refuses_an_out_of_range_source_id():
    with pytest.raises(science.Round0261Error):
        science.degree_census(np.array([0, 9], dtype=np.int32), rows=4)


# --------------------------------------------------------------------------- #
# guard 2 — the recall floors and the program bar
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mean,p10", [
    (science.MEAN_RECALL_FLOOR - 1e-6, 1.0),
    (1.0, science.P10_RECALL_FLOOR - 1e-6),
])
def test_the_gating_probe_below_an_exact_floor_is_refused(mean, p10):
    with pytest.raises(Round0216Error, match="below the"):
        science.validate_exact_graph(
            degrees={"zero_degree_rows": 0, "min": 5, "median": 19.0,
                     "mean": 24.17, "max": 1394},
            gating_recall={"mean_recall_at_k": mean, "p10_recall_at_k": p10},
            builder_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
            edges=48_344_648)


def test_the_builder_probe_below_the_program_bar_is_refused():
    with pytest.raises(science.Round0261Error, match="program floor"):
        science.validate_exact_graph(
            degrees={"zero_degree_rows": 0, "min": 5, "median": 19.0,
                     "mean": 24.17, "max": 1394},
            gating_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
            builder_recall={"mean_recall_at_k": science.PROGRAM_RECALL_FLOOR - 1e-6,
                            "p10_recall_at_k": 1.0},
            edges=48_344_648)


def test_the_builder_probe_is_declared_non_gating():
    assert science.GPU_PROBE_IS_INDEPENDENT is False
    checks = science.validate_exact_graph(
        degrees={"zero_degree_rows": 0, "min": 5, "median": 19.0,
                 "mean": 24.17, "max": 1394},
        gating_recall={"mean_recall_at_k": science.MEAN_RECALL_FLOOR,
                       "p10_recall_at_k": science.P10_RECALL_FLOOR},
        builder_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
        edges=48_344_648)
    assert checks["builder_probe_is_independent"] is False
    assert checks["gating_probe"] == "independent CPU brute-force pass"
    # The exact floors are applied to the gating probe, and it sits exactly on
    # them here: a builder probe of 1.0 did not rescue anything.
    assert checks["mean_recall_at_k"] == science.MEAN_RECALL_FLOOR


# --------------------------------------------------------------------------- #
# guard 3 — the selection law
# --------------------------------------------------------------------------- #


def test_a_prefix_selection_is_refused():
    with pytest.raises(science.Round0261Error, match="span"):
        science.assert_shard_span(corpus="fineweb", shards_touched=93, shards_total=99)


def test_a_full_span_is_accepted():
    span = science.assert_shard_span(corpus="fineweb", shards_touched=99,
                                     shards_total=99)
    assert span["coverage"] == 1.0


def test_a_rebalanced_composition_with_the_same_total_is_refused():
    counts = {name: n for name, n in science.COMPOSITION}
    first, second = science.COMPOSITION[0][0], science.COMPOSITION[1][0]
    counts[first] += 1
    counts[second] -= 1
    with pytest.raises(science.Round0261Error):
        science.validate_composition(counts)


def test_the_registered_composition_is_accepted():
    observed = science.validate_composition({name: n for name, n in science.COMPOSITION})
    assert observed[science.COMPOSITION[0][0]]["share"] == pytest.approx(0.40)


# --------------------------------------------------------------------------- #
# guard 4 — the ordering guard, BOTH refusal branches
# --------------------------------------------------------------------------- #


def test_a_prediction_sealed_after_the_build_started_is_refused():
    with pytest.raises(science.Round0261Error, match="not sealed before"):
        science.assert_prediction_precedes_build(
            prediction={"sealed_at_unix": 2000.0}, build_started_unix=1000.0)


def test_a_prediction_sealed_at_exactly_the_build_start_is_refused():
    with pytest.raises(science.Round0261Error, match="not sealed before"):
        science.assert_prediction_precedes_build(
            prediction={"sealed_at_unix": 1000.0}, build_started_unix=1000.0)


def test_a_prediction_that_binds_a_measured_4m_wall_is_refused():
    with pytest.raises(science.Round0261Error, match="not a prediction"):
        science.assert_prediction_precedes_build(
            prediction={"sealed_at_unix": 1.0, "measured_exact_search_s": 434.0},
            build_started_unix=1000.0)


def test_a_prediction_with_no_seal_time_is_refused():
    with pytest.raises(science.Round0261Error, match="no seal time"):
        science.assert_prediction_precedes_build(
            prediction={"interval_s": [1.0, 2.0]}, build_started_unix=1000.0)


def test_a_genuine_prediction_is_accepted():
    ordering = science.assert_prediction_precedes_build(
        prediction={"sealed_at_unix": 1000.0}, build_started_unix=1001.0)
    assert ordering["prediction_precedes_build"] is True
    assert ordering["seconds_between"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# guard 5 — the 2M back-check
# --------------------------------------------------------------------------- #


def test_the_back_check_passes_for_a_law_through_the_2m_point():
    fit = science.cost_prediction()["m2_fit"]
    check = science.back_check_at_2m(**fit)
    assert check["holds"] is True
    assert check["relative_error"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_the_back_check_fails_for_a_law_wrong_by_fifty_percent(factor):
    fit = dict(science.cost_prediction()["m2_fit"])
    fit["quadratic_s_per_pair"] *= factor
    check = science.back_check_at_2m(**fit)
    assert check["holds"] is False
    assert abs(check["relative_error"]) > science.BACK_CHECK_REL_TOL


def test_a_pure_quadratic_law_measured_at_4m_would_still_pass_at_2m():
    """The check is not so loose that any law passes, nor so tight that the
    honest alternative reading fails: a builder whose 4M wall were entirely
    quadratic implies `a = t/16e12`, and at 2M that reproduces `t/4`."""
    a = 448.7587486691773 / (4_000_000.0 ** 2)
    check = science.back_check_at_2m(quadratic_s_per_pair=a, linear_s_per_row=0.0)
    assert check["holds"] is True


# --------------------------------------------------------------------------- #
# guard 6 — the prediction scorer
# --------------------------------------------------------------------------- #


def test_score_prediction_reports_outside_in_both_directions():
    prediction = science.cost_prediction()
    low, high = prediction["interval_s"]
    assert science.score_prediction(
        prediction=prediction, measured_s=low - 1.0)["inside_the_registered_interval"] is False
    assert science.score_prediction(
        prediction=prediction, measured_s=high + 1.0)["inside_the_registered_interval"] is False
    assert science.score_prediction(
        prediction=prediction, measured_s=low)["inside_the_registered_interval"] is True


def test_score_prediction_names_the_closest_model():
    prediction = science.cost_prediction()
    scored = science.score_prediction(
        prediction=prediction,
        measured_s=prediction["models"]["M4_half_of_r0216_is_linear"])
    assert scored["closest_model"] == "M4_half_of_r0216_is_linear"


# --------------------------------------------------------------------------- #
# the independent CPU probe
# --------------------------------------------------------------------------- #


def _unit_rows(rng: np.random.RandomState, rows: int, dim: int) -> np.ndarray:
    X = rng.normal(size=(rows, dim)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_cpu_exact_topk_agrees_with_a_dense_reference_on_a_tiny_universe():
    rng = np.random.RandomState(0)
    X = _unit_rows(rng, 400, 16)
    probe = np.arange(0, 400, 37)
    ids, cos = science.cpu_exact_topk(X, probe, k=5, block=64)
    sims = X[probe] @ X.T
    sims[np.arange(probe.size), probe] = -np.inf
    reference = np.argsort(-sims, axis=1)[:, :5]
    assert (ids == reference).all()
    assert cos == pytest.approx(np.take_along_axis(sims, reference, axis=1), abs=1e-6)


def test_cpu_exact_topk_never_returns_the_query_row_itself():
    rng = np.random.RandomState(1)
    X = _unit_rows(rng, 200, 8)
    probe = np.arange(0, 200, 11)
    ids, _cos = science.cpu_exact_topk(X, probe, k=4, block=37)
    assert not (ids == probe[:, None]).any()


def test_score_cpu_probe_is_perfect_on_a_builder_that_matches_the_truth():
    rng = np.random.RandomState(2)
    X = _unit_rows(rng, 300, 12)
    probe = np.arange(0, 300, 13)
    ids, cos = science.cpu_exact_topk(X, probe, k=6, block=50)
    scored = science.score_cpu_probe(truth_ids=ids, truth_cos=cos,
                                     builder_ids=ids, builder_cos=cos, k=6)
    assert scored["strict"]["mean"] == 1.0
    assert scored["tie_aware"]["mean"] == 1.0
    assert science.gating_recall_block(scored)["mean_recall_at_k"] == 1.0


def test_score_cpu_probe_sees_a_builder_that_lost_a_neighbour():
    """The positive control for the probe itself: corrupt one neighbour per row
    with a far-away id and prove BOTH estimators drop below the exact floors."""
    rng = np.random.RandomState(3)
    X = _unit_rows(rng, 300, 12)
    probe = np.arange(0, 300, 13)
    ids, cos = science.cpu_exact_topk(X, probe, k=6, block=50)
    broken_ids = ids.copy()
    broken_cos = cos.copy()
    broken_ids[:, -1] = (broken_ids[:, -1] + 150) % 300
    broken_cos[:, -1] = -1.0
    scored = science.score_cpu_probe(truth_ids=ids, truth_cos=cos,
                                     builder_ids=broken_ids,
                                     builder_cos=broken_cos, k=6)
    assert scored["strict"]["mean"] < 1.0
    assert scored["tie_aware"]["mean"] < 1.0
    with pytest.raises(Round0216Error):
        science.validate_exact_graph(
            degrees={"zero_degree_rows": 0, "min": 5, "median": 19.0,
                     "mean": 24.17, "max": 1394},
            gating_recall=science.gating_recall_block(scored),
            builder_recall={"mean_recall_at_k": 1.0, "p10_recall_at_k": 1.0},
            edges=48_344_648)


def test_tie_forgiveness_does_not_forgive_a_genuinely_worse_neighbour():
    """A candidate whose cosine is below the truth's k-th by more than the
    registered tolerance is NOT forgiven, so tie-awareness is not a loophole."""
    truth_ids = np.array([[0, 1, 2]], dtype=np.int64)
    truth_cos = np.array([[0.9, 0.8, 0.7]], dtype=np.float64)
    builder_ids = np.array([[0, 1, 99]], dtype=np.int64)
    builder_cos = np.array([[0.9, 0.8, 0.7 - 10 * science.TIE_TOLERANCE]])
    scored = science.score_cpu_probe(truth_ids=truth_ids, truth_cos=truth_cos,
                                     builder_ids=builder_ids,
                                     builder_cos=builder_cos, k=3)
    assert scored["tie_aware"]["mean"] == pytest.approx(2.0 / 3.0)
    forgiven = science.score_cpu_probe(
        truth_ids=truth_ids, truth_cos=truth_cos, builder_ids=builder_ids,
        builder_cos=np.array([[0.9, 0.8, 0.7 - 0.1 * science.TIE_TOLERANCE]]), k=3)
    assert forgiven["tie_aware"]["mean"] == 1.0


# --------------------------------------------------------------------------- #
# the control bundle the node seals
# --------------------------------------------------------------------------- #


def test_all_controls_plants_every_guard_and_none_is_a_literal():
    controls = science.all_controls()
    assert controls["guards_shipped"] == 6
    assert controls["defects_planted"] == 19
    for name, block in controls["guards"].items():
        assert int(block["defects_planted"]) >= 2, name
    assert controls["every_guard_has_a_plant"] is True
    assert controls["guards_with_at_least_one_plant"] == controls["guards_shipped"]
    assert controls["refusals_recorded"] == controls["defects_planted"]
    ordering = controls["guards"]["ordering"]
    assert ordering["branches_exercised"] == [
        "binds_a_measured_quantity", "not_sealed_first"]
    assert ordering["both_branches_exercised"] is True
    assert len(ordering["plants"]) == 4


def test_every_recorded_plant_carries_the_shipped_functions_own_error():
    """A plant entry is only evidence if it quotes a real raise."""
    controls = science.all_controls()
    for name, block in controls["guards"].items():
        for plant in block.get("plants") or ():
            if "error" not in plant:
                continue
            assert plant["refused"] is True, (name, plant)
            assert plant["error"].startswith(("Round0261Error:", "Round0216Error:")), (
                name, plant)


def test_a_plant_that_is_accepted_raises_rather_than_being_recorded_as_refused():
    """The control harness must not silently record an accepted plant."""
    with pytest.raises(science.Round0261Error, match="ACCEPTED by the shipped path"):
        science._plant("a_guard_that_does_not_guard", lambda: {"holds": True})
