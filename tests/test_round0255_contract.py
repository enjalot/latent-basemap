"""R0255 contract + CPU smoke: the n = 29 MAD_n gate, and both guards' controls.

CUDA-hidden. Every test calls the SHIPPED function -- none re-implements a guard,
which is the defect review-0253-01 refused. The two guards each plant five defects
and check that the predicate they replace accepts all five.
"""
from __future__ import annotations

import math

import pytest

from basemap import round0234_calibration as calibration
from basemap.round0234_calibrated_floors import (
    CANDIDATE_ORDER,
    GATED_METRICS,
    METRICS,
    PURITY_METRICS,
    band_at,
    floor_at,
    identity_bound,
)
from basemap.round0255_gate_n29 import (
    EXACT_FAMILY_SEEDS,
    GATE_CAPABILITY,
    IDENTITY_BOUND_AT_N,
    N_EXACT,
    N_HELD_OUT,
    OWNER_RULING_ESTIMATOR,
    OWNER_RULING_N,
    RETAINED_FAMILY_SOURCES,
    THIS_FAMILY,
    attainability_and_power,
    exact_cell_id,
    falsifiability_statement,
    independence_control,
    owner_ruling_registration,
    poolability_shift_test,
)
from basemap.round0255_panel_n29 import (
    PANEL_CAPABILITY_N29,
    POOLED_CELL_SOURCES,
    Round0255PanelError,
    pool_twenty_nine_cells,
    replay_control_comparison,
)
from basemap.round0255_seed_extension_n29 import (
    OWNER_RULING_N as EXTENSION_OWNER_RULING_N,
    POOLED_SEEDS,
    REPLAY_CONTROL_CAPABILITY,
    REPLAY_CONTROL_SEED,
    R0250_POOLED_SEEDS,
    SEEDS,
    predict_cell_footprint,
)
from basemap.round0255_treatment import (
    HELD_OUT_CELL_IDS,
    Round0255FamilyError,
    Round0255TreatmentError,
    TRAIN_CLOSURE_MODULES,
    assert_family_is_2m_only,
    assert_runtime_closure_matches_seal,
    family_purity_controls,
    runtime_closure_hashes,
    treatment_closure_controls,
)


# --------------------------------------------------------------------------- #
# the family
# --------------------------------------------------------------------------- #


def test_the_thirteen_new_seeds_are_disjoint_from_the_sixteen():
    assert SEEDS == tuple(range(58, 71))
    assert len(SEEDS) == 13
    assert not set(SEEDS) & set(R0250_POOLED_SEEDS)
    assert POOLED_SEEDS == tuple(R0250_POOLED_SEEDS) + SEEDS
    assert len(POOLED_SEEDS) == len(set(POOLED_SEEDS)) == 29


def test_n_is_the_owner_ruling_n_everywhere():
    assert N_EXACT == OWNER_RULING_N == EXTENSION_OWNER_RULING_N == 29
    assert len(EXACT_FAMILY_SEEDS) == 29
    assert N_HELD_OUT == 12


def test_the_identity_bound_at_29_is_28_over_root_29():
    assert IDENTITY_BOUND_AT_N == pytest.approx(28.0 / math.sqrt(29.0), rel=0, abs=0)
    assert IDENTITY_BOUND_AT_N == identity_bound(29)
    assert 5.1994 < IDENTITY_BOUND_AT_N < 5.1995


def test_the_replay_control_is_seed_42_and_not_a_family_cell():
    assert REPLAY_CONTROL_SEED == 42
    prediction = predict_cell_footprint(REPLAY_CONTROL_SEED, replay_control=True)
    assert prediction["is_a_family_cell"] is False
    assert prediction["capability"] == REPLAY_CONTROL_CAPABILITY
    assert prediction["refused_a_priori"] is False


def test_every_family_cell_prediction_is_admissible():
    for seed in SEEDS:
        prediction = predict_cell_footprint(seed)
        assert prediction["is_a_family_cell"] is True
        assert prediction["refused_a_priori"] is False


# --------------------------------------------------------------------------- #
# guard 2 -- the family is 2M-only. Five planted defects.
# --------------------------------------------------------------------------- #


def test_the_family_purity_guard_accepts_the_honest_family():
    verdict = assert_family_is_2m_only([exact_cell_id(seed) for seed in POOLED_SEEDS])
    assert verdict["n"] == 29
    assert verdict["family_is_the_2m_universe_only"] is True


@pytest.mark.parametrize(
    "planted",
    [
        [exact_cell_id(seed) for seed in POOLED_SEEDS] + ["ladder-6250k-h2048-seed42"],
        [exact_cell_id(seed) for seed in POOLED_SEEDS] + ["cluster-spill-c8-seed42"],
        [exact_cell_id(seed) for seed in POOLED_SEEDS] + [REPLAY_CONTROL_CAPABILITY],
        [exact_cell_id(seed) for seed in POOLED_SEEDS] + [exact_cell_id(42)],
        [],
    ],
)
def test_the_family_purity_guard_refuses_each_planted_defect(planted):
    with pytest.raises(Round0255FamilyError):
        assert_family_is_2m_only(planted)


def test_the_family_purity_controls_are_run_by_the_shipped_guard():
    controls = family_purity_controls()
    assert controls["planted"] == 5
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_old_predicate_accepted_every_one"] is True
    assert controls["the_honest_family_still_passes"] is True


def test_a_short_family_is_refused_too():
    with pytest.raises(Round0255FamilyError):
        assert_family_is_2m_only([exact_cell_id(seed) for seed in POOLED_SEEDS[:-1]])


# --------------------------------------------------------------------------- #
# guard 1 -- the treatment closure. Five planted defects.
# --------------------------------------------------------------------------- #


def _closure_fixture():
    observed = runtime_closure_hashes(TRAIN_CLOSURE_MODULES)
    sealed = {
        "files": {
            name: {
                "sha256_at_release": entry["sha256"],
                "sha256_at_r0217": entry["sha256"],
            }
            for name, entry in observed.items()
        }
    }
    return sealed, observed


def test_the_closure_guard_accepts_the_honest_closure():
    sealed, observed = _closure_fixture()
    verdict = assert_runtime_closure_matches_seal(sealed=sealed, observed=observed)
    assert verdict["every_module_ran_the_sealed_bytes"] is True
    assert verdict["modules_checked"] == len(TRAIN_CLOSURE_MODULES)


def test_the_closure_guard_refuses_an_empty_closure():
    with pytest.raises(Round0255TreatmentError):
        assert_runtime_closure_matches_seal(sealed={"files": {}}, observed={}, modules=())


def test_the_closure_controls_are_run_by_the_shipped_guard():
    sealed, observed = _closure_fixture()
    controls = treatment_closure_controls(sealed=sealed, observed=observed)
    assert controls["planted"] == 5
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_old_predicate_accepted_every_one"] is True
    assert controls["the_honest_closure_still_passes"] is True
    assert {item["control"] for item in controls["controls"]} == {
        "content_drift",
        "missing_module",
        "extra_module",
        "malformed_digest",
        "empty_closure",
    }


# --------------------------------------------------------------------------- #
# the registration: the ruling, not the rule
# --------------------------------------------------------------------------- #


def _series():
    """A synthetic twenty-nine-cell family in the shape of the real one."""
    base = {
        "density_v2": 0.44,
        "ffr": 0.33,
        "purity_fidelity_k256": 0.986,
        "purity_fidelity_k1024": 0.711,
    }
    series = {
        metric: [base[metric] + 0.0005 * ((index % 7) - 3) for index in range(29)]
        for metric in METRICS
    }
    log_series = {
        metric: [math.log(1.008 + 0.001 * ((index % 5) - 2)) for index in range(29)]
        for metric in PURITY_METRICS
    }
    return series, log_series


def _selection_fixture():
    at29 = calibration.calibrate(29, families=20_000)
    at29.pop("_arrays")
    series, log_series = _series()
    candidates = {}
    for name in CANDIDATE_ORDER:
        entry = at29["candidates"][name]
        candidates[name] = {
            "estimator": name,
            "qualifies": name == OWNER_RULING_ESTIMATOR,
            "requirement_1_coverage": True,
            "requirement_2_invariance": True,
            "requirement_3_attainability": True,
            "minimum_exact_invariance_depth": 2,
            "calibrated_one_sided_multiplier": float(
                entry["one_sided"]["calibrated_multiplier"]
            ),
            "calibrated_two_sided_multiplier": float(
                entry["two_sided"]["calibrated_multiplier"]
            ),
        }
    return at29, series, log_series, {
        "candidates": candidates,
        "chosen_estimator": "median_minus_k_iqrn",
    }


def test_the_owner_ruling_is_registered_even_when_the_rule_disagrees():
    _at29, _series_values, _logs, selection = _selection_fixture()
    registration = owner_ruling_registration(selection=selection)
    assert registration["registered_estimator"] == OWNER_RULING_ESTIMATOR
    assert registration["selection_rule_would_have_chosen"] == "median_minus_k_iqrn"
    assert registration["selection_rule_agrees_with_the_ruling"] is False
    assert registration["registered_by"].startswith("owner ruling")


def test_the_owner_ruling_refuses_an_estimator_that_was_never_calibrated():
    _at29, _series_values, _logs, selection = _selection_fixture()
    with pytest.raises(Exception):
        owner_ruling_registration(selection=selection, estimator="not_an_estimator")


def test_the_calibrated_multiplier_at_29_sits_below_the_identity_bound():
    at29 = calibration.calibrate(29, families=20_000, names=(OWNER_RULING_ESTIMATOR,))
    at29.pop("_arrays")
    entry = at29["candidates"][OWNER_RULING_ESTIMATOR]
    statement = falsifiability_statement(
        estimator=OWNER_RULING_ESTIMATOR,
        multiplier_one_sided=float(entry["one_sided"]["calibrated_multiplier"]),
        multiplier_two_sided=float(entry["two_sided"]["calibrated_multiplier"]),
    )
    assert statement["one_sided_multiplier_below_the_identity_bound"] is True
    assert statement["registered_family_every_defining_cell_can_fail"] is True
    assert statement["estimator"] == OWNER_RULING_ESTIMATOR
    assert OWNER_RULING_ESTIMATOR in statement["plain_statement"]


def test_attainability_and_power_covers_every_gated_floor():
    at29, series, log_series, _selection = _selection_fixture()
    entry = at29["candidates"][OWNER_RULING_ESTIMATOR]
    k_one = float(entry["one_sided"]["calibrated_multiplier"])
    k_two = float(entry["two_sided"]["calibrated_multiplier"])
    floors = {
        metric: floor_at(OWNER_RULING_ESTIMATOR, series[metric], k_one)
        for metric in METRICS
    }
    bands = {
        metric: tuple(
            math.exp(value)
            for value in band_at(OWNER_RULING_ESTIMATOR, log_series[metric], k_two)
        )
        for metric in PURITY_METRICS
    }
    table = attainability_and_power(
        estimator=OWNER_RULING_ESTIMATOR,
        n=29,
        multiplier_one_sided=k_one,
        multiplier_two_sided=k_two,
        calibrated_entry=entry,
        floors=floors,
        bands=bands,
        series=series,
        log_series=log_series,
    )
    assert {row["metric"] for row in table["per_floor"]} == set(GATED_METRICS)
    for row in table["per_floor"]:
        assert row["detection_power_at_minus_2_sigma"] is not None
        assert row["every_defining_cell_can_fail"] is True
        assert row["multiplier_below_the_identity_bound"] is True


# --------------------------------------------------------------------------- #
# the independence control -- both halves
# --------------------------------------------------------------------------- #


def _family_cell_fixture():
    """The synthetic family, expressed as CELLS -- the repaired control's input."""
    series, _logs = _series()
    return [
        {
            "cell_id": exact_cell_id(seed),
            "family": "exact-graph",
            "values": {metric: series[metric][index] for metric in METRICS},
            "ratios": {
                "k256": 1.008 + 0.001 * ((index % 5) - 2),
                "k1024": 0.712 + 0.001 * ((index % 3) - 1),
            },
        }
        for index, seed in enumerate(POOLED_SEEDS)
    ]


def _held_out_fixture():
    return [
        {
            "cell_id": cell_id,
            "family": cell_id.rsplit("-", 1)[0],
            "values": {metric: 0.3 for metric in METRICS},
            "ratios": {"k256": 1.01, "k1024": 0.71},
        }
        for cell_id in HELD_OUT_CELL_IDS
    ]


def test_the_independence_control_holds_and_is_not_inert():
    at29, _series_values, _logs, _selection = _selection_fixture()
    entry = at29["candidates"][OWNER_RULING_ESTIMATOR]
    family = _family_cell_fixture()
    control = independence_control(
        estimator=OWNER_RULING_ESTIMATOR,
        multiplier_one_sided=float(entry["one_sided"]["calibrated_multiplier"]),
        multiplier_two_sided=float(entry["two_sided"]["calibrated_multiplier"]),
        family_cells=family,
        held_out_cells=_held_out_fixture(),
        defining_cell_ids=[cell["cell_id"] for cell in family],
    )
    assert control["the_fit_is_independent_of_every_held_out_cell"] is True
    assert control["the_fit_is_not_inert"] is True
    assert control["holds"] is True
    assert len(control["arms"]) == 4
    assert all(arm["perturbation_reaches_the_fit"] for arm in control["arms"])


# --------------------------------------------------------------------------- #
# poolability
# --------------------------------------------------------------------------- #


def test_the_poolability_test_reports_and_never_filters():
    cells = {}
    ratios = {}
    for index, seed in enumerate(POOLED_SEEDS):
        shift = 0.01 if seed in SEEDS else 0.0
        cells[str(seed)] = {
            "density_v2": 0.44 + 0.0001 * index - shift,
            "ffr": 0.33 + 0.0001 * index,
            "purity_fidelity_k256": 0.98,
            "purity_fidelity_k1024": 0.71,
        }
        ratios[str(seed)] = {"k256": 1.008 + shift, "k1024": 0.71}
    report = poolability_shift_test(
        panel_metric_cells=cells, raw_purity_ratios=ratios
    )
    assert report["cells_dropped"] == 0
    assert set(report["series"]) == {
        "density_v2", "ffr", "ratio::k256", "ratio::k1024"
    }
    for row in report["series"].values():
        assert row["prior_n"] == 16
        assert row["new_n"] == 13
        assert 0.0 <= row["mann_whitney_p_exact"] <= 1.0
    assert report["series"]["density_v2"]["direction"] == "new cells lower"


# --------------------------------------------------------------------------- #
# pooling and the replay comparison
# --------------------------------------------------------------------------- #


def _pool_fixture():
    cells = {
        str(seed): {
            "density_v2": 0.44,
            "ffr": 0.33,
            "purity_fidelity_k256": 0.98,
            "purity_fidelity_k1024": 0.71,
        }
        for seed in POOLED_SEEDS
    }
    ratios = {str(seed): {"k256": 1.008, "k1024": 0.71} for seed in POOLED_SEEDS}
    corpus = {
        str(seed): {
            slug: {"anchors": 1000, "ffr": 0.33}
            for slug in ("code", "fineweb", "pile", "redpajama")
        }
        for seed in POOLED_SEEDS
    }
    return cells, ratios, corpus


def test_pooling_twenty_nine_cells_and_refusing_the_control():
    cells, ratios, corpus = _pool_fixture()
    pooled = pool_twenty_nine_cells(
        cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
    )
    assert pooled["n"] == 29
    assert pooled["reaches_the_owner_ruling_n"] is True
    assert pooled["replay_control_is_not_a_family_cell"] is True

    cells[REPLAY_CONTROL_CAPABILITY] = dict(cells["42"])
    with pytest.raises(Round0255PanelError):
        pool_twenty_nine_cells(
            cells=cells, ratios=ratios, corpus=corpus, sources=POOLED_CELL_SOURCES
        )


def test_the_replay_comparison_flags_a_drift_and_passes_an_exact_match():
    sealed = {"ffr": 0.3369, "density_v2": 0.4377}
    ratios = {"k256": 1.0216, "k1024": 0.7326}
    same = replay_control_comparison(
        observed=dict(sealed),
        sealed_r0218=sealed,
        observed_ratios=dict(ratios),
        sealed_ratios=ratios,
        tolerance=1e-4,
    )
    assert same["the_train_side_treatment_reproduces"] is True
    assert same["values_exactly_equal"] == 4

    drifted = replay_control_comparison(
        observed={"ffr": 0.34, "density_v2": 0.4377},
        sealed_r0218=sealed,
        observed_ratios=dict(ratios),
        sealed_ratios=ratios,
        tolerance=1e-4,
    )
    assert drifted["the_train_side_treatment_reproduces"] is False


# --------------------------------------------------------------------------- #
# the joint criteria and the node entry points
# --------------------------------------------------------------------------- #


def test_the_joint_criteria_carry_five_families_newest_last():
    names = [item["family"] for item in RETAINED_FAMILY_SOURCES] + [THIS_FAMILY]
    assert len(names) == 5
    assert names[-1] == THIS_FAMILY
    assert "r0250_n16_calibrated_robust" in names


def test_the_capabilities_are_named_for_n29():
    assert GATE_CAPABILITY.endswith("n29-v1")
    assert PANEL_CAPABILITY_N29.endswith("n29-v1")


def test_every_node_action_dispatches():
    from experiments import round0255_nodes as nodes

    assert nodes.run_job.__module__ == "experiments.round0255_nodes"
    for action in (nodes.TRAIN_ACTION, nodes.PANEL_ACTION, nodes.GATE_ACTION):
        assert isinstance(action, str) and action
    with pytest.raises(Exception):
        nodes.run_job({"manifest": {"round_id": "0255"}}, {"action": "nope"})


def test_the_prepare_script_contains_no_subprocess_timeout():
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1] / "experiments"
    for name in ("prepare_round0255_queue.py", "round0255_nodes.py"):
        text = (source / name).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        assert "timeout=" not in code
