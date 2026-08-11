"""R0251 contract + CPU smoke.

Every guard this round adds ships a planted defect it catches, and every node
action has its entry path executed through `run_job`. The GPU nodes are not run
here; what is run is the dispatch, the contract functions and the models.
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap import round0251_estimator_table as table_mod
from basemap import round0251_rescore as rescore_mod
from basemap import round0251_trainer_setup as setup_mod
from basemap.round0247_registry import registered_value


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND0251_FILES = (
    "basemap/round0251_trainer_setup.py",
    "basemap/round0251_rescore.py",
    "basemap/round0251_estimator_table.py",
    "experiments/round0251_nodes.py",
    "experiments/prepare_round0251_queue.py",
    "tests/test_round0251_contract.py",
)
CEILING = registered_value("r0246_max_poll_spacing_s")


# --------------------------------------------------------------------------- #
# the release trainer's new hook
# --------------------------------------------------------------------------- #


def test_the_release_class_carries_the_five_declared_poll_sites():
    receipt = setup_mod.declared_sites_match_the_release()
    assert receipt["sites_match"] is True
    assert receipt["hook_default_is_none"] is True
    assert len(receipt["declared_sites"]) == 5


def test_a_renamed_poll_site_is_caught(monkeypatch):
    """Planted defect: the release renames a site and the module does not."""
    from basemap.pumap.parametric_umap import ParametricUMAP

    monkeypatch.setattr(
        ParametricUMAP, "ABORT_POLL_SITE_SETUP_COMPLETE", "something else", raising=True
    )
    with pytest.raises(setup_mod.Round0251SetupError):
        setup_mod.declared_sites_match_the_release()


def test_the_hook_is_a_noop_by_default_and_calls_through_when_installed():
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP()
    assert model.abort_poll is None
    model._poll_abort("nowhere")  # must not raise
    seen = []
    model.abort_poll = seen.append
    model._poll_abort(ParametricUMAP.ABORT_POLL_SITE_TRAIN_BATCH)
    assert seen == [ParametricUMAP.ABORT_POLL_SITE_TRAIN_BATCH]


def test_the_trainer_diff_is_additive_only():
    """The five call sites and the method exist, and no science line moved.

    Asserted structurally rather than by reading the diff: `fit` must contain
    exactly five `_poll_abort` calls and `_poll_abort` must have no side effect
    beyond the call.
    """
    path = os.path.join(REPO, "basemap/pumap/parametric_umap/core.py")
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    klass = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ParametricUMAP"
    )
    fit = next(
        node for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit"
    )
    calls = [
        node for node in ast.walk(fit)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_poll_abort"
    ]
    assert len(calls) == 5
    poll = next(
        node for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "_poll_abort"
    )
    assigned = [node for node in ast.walk(poll) if isinstance(node, (ast.Assign, ast.AugAssign))]
    assert len(assigned) == 1  # `poll = self.abort_poll`, nothing else


# --------------------------------------------------------------------------- #
# the recorder and the phase split
# --------------------------------------------------------------------------- #


class _FakeGate:
    def __init__(self):
        self.calls = []

    def __call__(self, where):
        self.calls.append(where)


def _clock_from(values):
    iterator = iter(values)

    def _clock():
        return next(iterator)

    return _clock


def test_the_recorder_keeps_the_whole_series_and_forwards_every_read():
    gate = _FakeGate()
    recorder = setup_mod.PollRecorder(
        gate=gate, clock=_clock_from([0.0, 1.0, 1.5, 2.0, 2.25])
    )
    recorder.anchor("start")
    recorder("pumap.fit setup complete")
    recorder(setup_mod.BATCH_POLL_SITE)
    recorder(setup_mod.BATCH_POLL_SITE)
    recorder(setup_mod.BATCH_POLL_SITE)
    assert gate.calls[0] == "start"
    assert recorder.batches == 3
    assert recorder.batch_gaps == [0.5, 0.5, 0.25]
    assert recorder.receipt()["batch_reads"] == 3


def test_phase_report_splits_setup_from_steady_state():
    records = [
        ("pumap.fit setup entered", 0.10),
        ("pumap.fit setup complete", 1.90),
        (setup_mod.BATCH_POLL_SITE, 0.40),
        (setup_mod.BATCH_POLL_SITE, 0.02),
        (setup_mod.BATCH_POLL_SITE, 0.03),
    ]
    report = setup_mod.phase_report(records, arm="unit")
    assert report["widest_setup_gap_s"] == pytest.approx(1.90)
    assert report["widest_steady_state_gap_s"] == pytest.approx(0.03)
    assert report["the_binding_phase"] == "setup"
    assert report["widest_gap_across_both_phases_s"] == pytest.approx(1.90)


def test_phase_report_fails_closed_when_no_batch_read_happened():
    """Planted defect: an arm that never reaches a batch must not be scored."""
    with pytest.raises(setup_mod.Round0251SetupError):
        setup_mod.phase_report([("pumap.fit setup entered", 0.1)], arm="unit")


def test_setup_reduction_reports_a_shortfall_rather_than_hiding_it():
    reduced = setup_mod.setup_reduction(before_gap_s=8.0, after_gap_s=0.5)
    assert reduced["setup_is_below_the_ceiling"] is True
    assert reduced["setup_is_below_the_ceiling_with_the_required_margin"] is True
    assert reduced["reduction_factor"] == pytest.approx(16.0)
    not_reduced = setup_mod.setup_reduction(before_gap_s=8.0, after_gap_s=5.0)
    assert not_reduced["setup_is_below_the_ceiling"] is False
    assert not_reduced["shortfall_over_the_ceiling_if_not"] == pytest.approx(
        5.0 / CEILING
    )


# --------------------------------------------------------------------------- #
# the tail models
# --------------------------------------------------------------------------- #


def test_fit_setup_gap_separates_the_trainer_diff_from_the_node_stage():
    report = {
        "arm": "unit",
        "setup_gaps_by_site": {
            "R0251 node substrate opened": {"widest_gap_s": 1.30},
            "pumap.fit setup edge-list prepared": {"widest_gap_s": 0.06},
            "pumap.fit train batch": {"widest_gap_s": 0.05},
        },
    }
    split = setup_mod.fit_setup_gap(report)
    assert split["widest_setup_gap_inside_fit_s"] == pytest.approx(0.06)
    assert split["widest_setup_gap_outside_fit_s"] == pytest.approx(1.30)
    assert split["the_binding_setup_interval_is_inside_fit"] is False
    assert split["widest_setup_gap_outside_fit_after"] == "R0251 node substrate opened"


def test_the_hash_projection_scales_linearly_and_labels_itself():
    projection = setup_mod.hash_bound_setup_projection(
        measured_gap_s=1.3, measured_bytes=3_072_000_128,
        target_rows=100_000_000, dimension=384,
    )
    assert projection["kind"] == "projection"
    assert projection["is_a_measurement_at_the_target_rows"] is False
    assert projection["projected_gap_s"] == pytest.approx(
        1.3 * (100_000_000 * 384 * 4) / 3_072_000_128, rel=1e-9
    )
    assert projection["projected_meets_the_registered_ceiling"] is False


def test_the_hash_projection_refuses_a_degenerate_measurement():
    with pytest.raises(setup_mod.Round0251SetupError):
        setup_mod.hash_bound_setup_projection(
            measured_gap_s=0.0, measured_bytes=1, target_rows=10, dimension=384
        )


def test_the_distribution_free_bound_is_the_rule_of_three_when_nothing_exceeds():
    gaps = [0.001] * 10_000
    bound = setup_mod.distribution_free_tail_bound(gaps, batches_at_target=4_100_000.0)
    assert bound["batches_exceeding_the_ceiling_observed"] == 0
    assert bound["upper_bound_on_the_per_batch_exceedance_probability"] == pytest.approx(
        bound["rule_of_three_approximation"], rel=0.01
    )
    assert bound["upper_bound_on_expected_exceedances_at_the_target"] > 1.0
    assert bound["is_a_measurement_at_the_target_wall"] is False


def test_the_distribution_free_bound_sees_a_planted_exceedance():
    gaps = [0.001] * 9_999 + [CEILING * 2.0]
    bound = setup_mod.distribution_free_tail_bound(gaps, batches_at_target=1_000.0)
    assert bound["batches_exceeding_the_ceiling_observed"] == 1
    assert bound["upper_bound_on_the_per_batch_exceedance_probability"] > 1.0 / 10_000


def test_the_pot_model_recovers_a_bounded_tail():
    rng = np.random.default_rng(20260811)
    gaps = rng.uniform(0.0005, 0.0015, size=20_000).tolist()
    model = setup_mod.peaks_over_threshold_tail(
        gaps, batches_at_target=4_100_000.0
    )
    assert model["tail_is_bounded"] is True
    assert model["fitted_finite_endpoint_s"] == pytest.approx(0.0015, abs=2e-4)
    assert model["return_level_meets_the_ceiling"] is True


def test_the_pot_model_refuses_too_few_observations():
    with pytest.raises(setup_mod.Round0251SetupError):
        setup_mod.peaks_over_threshold_tail([0.001] * 100, batches_at_target=10.0)


def test_tail_model_publishes_both_answers():
    rng = np.random.default_rng(7)
    gaps = (0.001 + rng.exponential(0.0002, size=20_000)).tolist()
    model = setup_mod.tail_model(gaps, arm_wall_s=90.0)
    assert model["batch_multiple_at_the_target"] == pytest.approx(400.0)
    assert set(model) >= {
        "peaks_over_threshold", "distribution_free", "threshold_sensitivity"
    }
    assert [row["threshold_quantile"] for row in model["threshold_sensitivity"]] == list(
        setup_mod.POT_THRESHOLD_LADDER
    )
    pot = model["peaks_over_threshold"]
    low, high = pot["shape_bootstrap_ci_95"]
    assert low <= pot["fitted_shape_xi"] <= high
    assert pot["return_level_if_the_tail_were_exponential_s"] > pot["threshold_s"]


def test_the_tail_verdict_refuses_an_unidentified_extreme_value_fit():
    """Planted defect: a threshold ladder that disagrees by orders of magnitude."""
    unidentified = {
        "peaks_over_threshold": {
            "observed_max_over_the_ceiling": 0.02,
            "batches_observed": 10_000,
            "return_level_over_the_ceiling": 415.0,
            "shape_bootstrap_ci_95": [0.38, 1.87],
        },
        "threshold_sensitivity": [
            {
                "return_level_over_the_ceiling": value,
                "return_level_if_the_tail_were_exponential_over_the_ceiling": 0.015,
            }
            for value in (11419.0, 415.0, 0.049)
        ],
        "distribution_free": {
            "upper_bound_on_expected_exceedances_at_the_target": 1201.0
        },
    }
    verdict = setup_mod.tail_verdict(unidentified)
    assert verdict["the_extreme_value_fit_is_identified"] is False
    assert verdict["threshold_ladder_return_level_spread"] > 1e5
    assert "NOT determined by this rung" in verdict["plain_statement"]

    identified = dict(unidentified)
    identified["threshold_sensitivity"] = [
        {
            "return_level_over_the_ceiling": value,
            "return_level_if_the_tail_were_exponential_over_the_ceiling": 0.015,
        }
        for value in (0.02, 0.025, 0.03)
    ]
    identified["peaks_over_threshold"] = {
        **unidentified["peaks_over_threshold"],
        "shape_bootstrap_ci_95": [-0.1, 0.2],
    }
    assert setup_mod.tail_verdict(identified)["the_extreme_value_fit_is_identified"] is True


# --------------------------------------------------------------------------- #
# the rescore comparison
# --------------------------------------------------------------------------- #


_SEALED = {
    "seed": 42,
    "source_round": "0218",
    "coordinates_ordered_sha256": "a" * 64,
    "panel_metrics": {
        "density_v2": 0.4377,
        "ffr": 0.3369,
        "purity_fidelity_k256": 0.9788566953797964,
        "purity_fidelity_k1024": 0.7326,
    },
    "purity_ratios": {"k256": 1.0216, "k1024": 0.7326},
    "hi_d_agreement": {"k256": 0.3828, "k1024": 0.2385},
    "corpus_ffr": {"code": {"anchors": 445, "ffr": 0.3209}},
}


def _observed(**overrides):
    payload = {
        "observed_panel_metrics": dict(_SEALED["panel_metrics"]),
        "observed_ratios": dict(_SEALED["purity_ratios"]),
        "observed_hi_d_agreement": dict(_SEALED["hi_d_agreement"]),
        "observed_corpus_ffr": {"code": {"anchors": 445, "ffr": 0.3209}},
        "observed_coordinates_sha256": _SEALED["coordinates_ordered_sha256"],
    }
    payload.update(overrides)
    return payload


def test_an_identical_rescore_reports_no_drift():
    comparison = rescore_mod.compare_rescore(sealed=_SEALED, **_observed())
    assert comparison["the_map_side_scorer_reproduces"] is True
    assert comparison["values_drifted"] == 0
    assert comparison["values_exactly_equal"] == comparison["values_compared"]
    assert comparison["coordinates_ordered_sha256_identical"] is True


def test_a_planted_drift_in_one_metric_is_caught():
    """Planted defect: the k256 ratio moves by two panel quanta."""
    ratios = dict(_SEALED["purity_ratios"])
    ratios["k256"] = 1.0218
    comparison = rescore_mod.compare_rescore(
        sealed=_SEALED, **_observed(observed_ratios=ratios)
    )
    assert comparison["the_map_side_scorer_reproduces"] is False
    assert "k256" in comparison["drifted"]


def test_a_sub_quantum_difference_is_not_called_drift():
    metrics = dict(_SEALED["panel_metrics"])
    metrics["ffr"] = 0.3369 + 5e-5
    comparison = rescore_mod.compare_rescore(
        sealed=_SEALED, **_observed(observed_panel_metrics=metrics)
    )
    assert comparison["the_map_side_scorer_reproduces"] is True
    assert comparison["values_exactly_equal"] < comparison["values_compared"]


def test_the_shift_test_reproduces_the_reviews_two_p_values():
    cells = {
        "42": {"density_v2": 0.4377, "ffr": 0.3369},
        "43": {"density_v2": 0.4406, "ffr": 0.3382},
        "44": {"density_v2": 0.4387, "ffr": 0.3258},
        "45": {"density_v2": 0.4477, "ffr": 0.3227},
        "46": {"density_v2": 0.4434, "ffr": 0.3312},
        "47": {"density_v2": 0.4400, "ffr": 0.3209},
        "48": {"density_v2": 0.4393, "ffr": 0.3344},
        "49": {"density_v2": 0.4491, "ffr": 0.3240},
        "50": {"density_v2": 0.4292, "ffr": 0.3325},
        "51": {"density_v2": 0.4506, "ffr": 0.3227},
        "52": {"density_v2": 0.4462, "ffr": 0.3329},
        "53": {"density_v2": 0.4477, "ffr": 0.3192},
        "54": {"density_v2": 0.4455, "ffr": 0.3341},
        "55": {"density_v2": 0.4360, "ffr": 0.3317},
        "56": {"density_v2": 0.4304, "ffr": 0.3399},
        "57": {"density_v2": 0.4315, "ffr": 0.3354},
    }
    ratios = {
        "42": {"k256": 1.0216, "k1024": 0.7326},
        "43": {"k256": 1.0059, "k1024": 0.7229},
        "44": {"k256": 1.0046, "k1024": 0.6980},
        "45": {"k256": 0.9929, "k1024": 0.6936},
        "46": {"k256": 1.0049, "k1024": 0.7214},
        "47": {"k256": 0.9932, "k1024": 0.6842},
        "48": {"k256": 1.0370, "k1024": 0.7266},
        "49": {"k256": 1.0099, "k1024": 0.6991},
        "50": {"k256": 1.0120, "k1024": 0.7129},
        "51": {"k256": 1.0024, "k1024": 0.7048},
        "52": {"k256": 1.0055, "k1024": 0.7168},
        "53": {"k256": 1.0065, "k1024": 0.6865},
        "54": {"k256": 1.0115, "k1024": 0.7197},
        "55": {"k256": 1.0293, "k1024": 0.7235},
        "56": {"k256": 1.0232, "k1024": 0.7121},
        "57": {"k256": 1.0259, "k1024": 0.7221},
    }
    shift = rescore_mod.shift_test(panel_metric_cells=cells, raw_purity_ratios=ratios)
    k256 = shift["series"]["ratio::k256"]
    density = shift["series"]["density_v2"]
    assert k256["new_cell_ranks_in_the_pooled_sixteen"] == [15, 13, 14]
    assert density["new_cell_ranks_in_the_pooled_sixteen"] == [4, 2, 3]
    assert k256["mann_whitney_p_exact"] == pytest.approx(0.025, abs=1e-6)
    assert k256["mann_whitney_p_asymptotic"] == pytest.approx(0.0313538, abs=1e-6)
    assert density["mann_whitney_p_exact"] == pytest.approx(0.025, abs=1e-6)
    assert k256["welch_p"] == pytest.approx(0.000279, abs=5e-6)


def test_poolability_verdict_flips_with_the_rescore():
    shift = {
        "series": {
            "ratio::k256": {"mann_whitney_p_exact": 0.025},
            "density_v2": {"mann_whitney_p_exact": 0.025},
        }
    }
    good = rescore_mod.poolability_verdict(
        rescore={"the_map_side_scorer_reproduces": True}, shift=shift
    )
    assert good["the_sixteen_cells_are_poolable_on_the_map_side"] is True
    assert good["the_k256_shift_survives_the_control"] is True
    bad = rescore_mod.poolability_verdict(
        rescore={"the_map_side_scorer_reproduces": False}, shift=shift
    )
    assert bad["the_sixteen_cells_are_poolable_on_the_map_side"] is False
    assert "MIXED" in bad["what_it_means_for_the_sixteen_cell_family"]


# --------------------------------------------------------------------------- #
# the joint table and the coupling
# --------------------------------------------------------------------------- #


_CELL = {
    "cell_id": "cluster-spill-c8-seed42",
    "family": "cluster-spill-c8",
    "values": {"ffr": 0.3075, "purity_fidelity_k256": 0.98, "purity_fidelity_k1024": 0.71},
    "ratios": {"k256": 1.01, "k1024": 0.71},
}


def test_the_coupling_column_flips_when_the_floor_crosses_the_cell():
    below = table_mod.coupling_column(
        cell=_CELL, floors={"ffr": 0.3055370}, bands={}, metric="ffr"
    )
    above = table_mod.coupling_column(
        cell=_CELL, floors={"ffr": 0.3116760}, bands={}, metric="ffr"
    )
    assert below["fails"] is False
    assert above["fails"] is True


def test_dominance_never_reads_the_coupling_columns():
    """Planted defect: a coupling column smuggled into the criteria list."""
    original = table_mod.PRE_REGISTERED_CRITERIA
    table_mod.PRE_REGISTERED_CRITERIA = original + (
        {
            "criterion": "fails_the_coupling_cell",
            "direction": "must hold",
            "source": "planted",
        },
    )
    try:
        with pytest.raises(table_mod.Round0251TableError):
            table_mod.dominance({"rows": [{"estimator": "a"}]})
    finally:
        table_mod.PRE_REGISTERED_CRITERIA = original


def test_the_pre_registered_criteria_exclude_the_coupling_columns():
    names = {item["criterion"] for item in table_mod.PRE_REGISTERED_CRITERIA}
    assert not (names & set(table_mod.EXCLUDED_FROM_DOMINANCE))
    assert "qualifies_at_n13_as_well_as_n16" in names


def test_dominance_finds_a_planted_dominator():
    rows = []
    for index, name in enumerate(("weak", "strong")):
        rows.append({
            "estimator": name,
            "requirement_1_coverage": True,
            "requirement_3_attainability": True,
            "minimum_exact_invariance_depth": 1 + index,
            "detection_power_at_minus_2_sigma": 0.2 + 0.1 * index,
            "asymptotic_breakdown_point_at_this_n": 0.5,
            "new_cell_false_fail_rate_one_sided": 0.02 - 0.01 * index,
            "qualifies_at_n13_as_well_as_n16": bool(index),
        })
    verdict = table_mod.dominance({"rows": rows})
    assert verdict["dominating_candidates"] == ["strong"]


def test_the_module_says_the_reproduction_is_not_independent_evidence():
    assert "NOT independent evidence" in table_mod.NOT_INDEPENDENT_EVIDENCE
    assert "registers no estimator" in table_mod.REGISTERS_NOTHING


# --------------------------------------------------------------------------- #
# entry paths and hygiene
# --------------------------------------------------------------------------- #


def test_run_job_dispatches_every_declared_action():
    from experiments import round0251_nodes as nodes

    seen = []
    original = {
        nodes.RESCORE_ACTION: nodes.run_rescore,
        nodes.TABLE_ACTION: nodes.run_table,
        nodes.TRAINSETUP_ACTION: nodes.run_trainsetup,
    }
    try:
        nodes.run_rescore = lambda a, j: seen.append(nodes.RESCORE_ACTION)
        nodes.run_table = lambda a, j: seen.append(nodes.TABLE_ACTION)
        nodes.run_trainsetup = lambda a, j: seen.append(nodes.TRAINSETUP_ACTION)
        for action in nodes.ACTIONS:
            nodes.run_job({"manifest": {"round_id": "0251"}}, {"action": action})
    finally:
        nodes.run_rescore = original[nodes.RESCORE_ACTION]
        nodes.run_table = original[nodes.TABLE_ACTION]
        nodes.run_trainsetup = original[nodes.TRAINSETUP_ACTION]
    assert seen == list(nodes.ACTIONS)


def test_run_job_refuses_an_unknown_action():
    from experiments import round0251_nodes as nodes

    with pytest.raises(nodes.Round0251Error):
        nodes.run_job({"manifest": {"round_id": "0251"}}, {"action": "nope"})


def test_every_handler_refuses_another_rounds_queue():
    from experiments import round0251_nodes as nodes

    other = {"manifest": {"round_id": "0250"}}
    for handler, error in (
        (nodes.run_rescore, rescore_mod.Round0251RescoreError),
        (nodes.run_table, table_mod.Round0251TableError),
        (nodes.run_trainsetup, setup_mod.Round0251SetupError),
    ):
        with pytest.raises(error):
            handler(other, {})


def test_no_round0251_file_contains_a_hidden_sigkill():
    """`subprocess.run(..., timeout=N)` is `Popen.kill()`. Banned outright."""
    for name in ROUND0251_FILES:
        with open(os.path.join(REPO, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                keyword.arg == "timeout" for keyword in node.keywords
            ):
                func = node.func
                target = getattr(func, "attr", getattr(func, "id", ""))
                assert target not in {"run", "communicate", "wait"}, (
                    f"{name} passes timeout= to {target}"
                )


def test_no_registered_bound_is_retyped_as_a_value_in_this_round():
    """Every threshold this round compares against is read from the registry.

    Prose may quote R0250's measured ceiling as motivation; a numeric CONSTANT
    equal to a registered bound is what would let a comparison drift away from
    the registry, and this walks the AST rather than the text so a docstring
    citation does not read as a re-typed bound.
    """
    banned = {
        registered_value("r0246_max_poll_spacing_s"),
        registered_value("min_binding_slope_bytes_per_s"),
    }
    for name in ROUND0251_FILES:
        if name.startswith("tests/"):
            continue  # this file names the bounds in order to ban them
        with open(os.path.join(REPO, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                assert node.value not in banned, f"{name} re-types {node.value!r}"


def test_the_page_cache_eviction_helper_touches_no_bytes(tmp_path):
    from experiments.round0251_nodes import _evict_page_cache

    target = tmp_path / "probe.bin"
    target.write_bytes(b"x" * 4096)
    before = target.read_bytes()
    receipt = _evict_page_cache([str(target)])
    assert receipt["method"] == "posix_fadvise(POSIX_FADV_DONTNEED)"
    assert receipt["files"][0]["bytes"] == 4096
    assert target.read_bytes() == before
