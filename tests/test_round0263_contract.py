"""R0263 contract — the restored two-sided ``purity_fidelity_k256`` band and its controls.

CUDA-hidden, no ``/data`` write, no queue. Every test runs on this round's own sealed
CONSTANTS (``round0263_nodes`` carries R0263's bitwise copy of the R0255 band, the
retained/untouched R0260 floors and the calibration multipliers) plus the frozen
canonical oracle below — so the whole file is hermetic and needs neither ``/data`` nor
the canonical criterion's home repo.

The judge-equivalence oracle (§A) is implemented with **frozen triples**, not an
``importlib`` import: the canonical ``experiments/metrics/k256_two_sided.py`` lives in a
DIFFERENT repo (``/home/enjalot/code/latent-basemap``), is absent from this run
checkout, and pulls in ``numpy``/``scipy`` at import. The 45 ``(map, ratio) -> (verdict,
direction, z_madn)`` triples below were produced from that module's ``would_be_verdicts``
(band re-fitted from the 29 sealed cells at the sealed ``k2``, which reproduces R0255's
registered band bitwise) and are asserted here as the reference oracle the local
``_judge_k256_two_sided`` must match bitwise. An OPTIONAL cross-check against the live
canonical module runs only when that module and its deps are importable, and is skipped
otherwise; the frozen oracle is the load-bearing check either way.
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

from basemap.round0234_calibrated_floors import identity_bound
from basemap.round0255_gate_n29 import N_EXACT
from experiments.round0260_nodes import _bitwise_same
from experiments.round0263_nodes import (
    RESTORED_METRIC,
    SEALED_FFR_FLOOR,
    SEALED_HELD_OUT_FLIP,
    SEALED_IDENTITY_BOUND_AT_N29,
    SEALED_K1,
    SEALED_K1024_ONE_SIDED_FLOOR,
    SEALED_K2,
    SEALED_K256_BAND,
    SEALED_K256_ONE_SIDED_FLOOR,
    SEALED_RUNG_K256,
    Round0263NodeError,
    _assert_no_descriptive_value_in_the_gate,
    _assert_registry_restores_k256_two_sided,
    _descriptive_floor_never_gates_controls,
    _descriptive_one_sided_diagnostics,
    _judge_k256_two_sided,
    _ordering_positive_control,
    _refuse_if_rung_bound,
    _registered_criteria_k256_two_sided,
    _registry_controls_k256_two_sided,
    _sidedness_controls_k256_two_sided,
)


# --------------------------------------------------------------------------- #
# The sealed band, as the local judge reads it, and the bitwise reproduction targets.
# --------------------------------------------------------------------------- #

BAND = dict(SEALED_K256_BAND)  # ratio_lower / ratio_upper / log_centre / log_scale

BAND_LOWER = 0.9697263687993266
BAND_UPPER = 1.0550731452902518
RETAINED_ONE_SIDED_FLOOR = 0.9756474051324426
K1024_FLOOR = 0.6690913806270514
FFR_FLOOR = 0.3125328635126472
IDENTITY_BOUND = 5.199469468957452


# --------------------------------------------------------------------------- #
# The frozen canonical oracle: 45 published maps, produced by the canonical
# k256_two_sided.would_be_verdicts() with the band re-fitted at the sealed k2.
# (map, ratio, two_sided_verdict, direction, z_madn) — z_madn is full precision.
# --------------------------------------------------------------------------- #

CANONICAL_ORACLE: list[tuple[str, float, str, str, float]] = [
    ("exact-seed42", 1.0216, "PASS", "inside", 0.7415420173283057),
    ("exact-seed43", 1.0059, "PASS", "inside", -0.4143493710796308),
    ("exact-seed44", 1.0046, "PASS", "inside", -0.5108674998267384),
    ("exact-seed45", 0.9929, "PASS", "inside", -1.385193912905344),
    ("exact-seed46", 1.0049, "PASS", "inside", -0.4885830043827463),
    ("exact-seed47", 0.9932, "PASS", "inside", -1.362646864114957),
    ("exact-seed48", 1.0370, "PASS", "inside", 1.8582160369131069),
    ("exact-seed49", 1.0099, "PASS", "inside", -0.11815099136317544),
    ("exact-seed50", 1.0120, "PASS", "inside", 0.0368838597919468),
    ("exact-seed51", 1.0024, "PASS", "inside", -0.6744907594765952),
    ("exact-seed52", 1.0055, "PASS", "inside", -0.44403396479700313),
    ("exact-seed53", 1.0065, "PASS", "inside", -0.3698446060339828),
    ("exact-seed54", 1.0115, "PASS", "inside", 0.0),
    ("exact-seed55", 1.0293, "PASS", "inside", 1.3019674537579604),
    ("exact-seed56", 1.0232, "PASS", "inside", 0.8583409281296036),
    ("exact-seed57", 1.0259, "PASS", "inside", 1.0550255567977598),
    ("exact-seed58", 1.0313, "PASS", "inside", 1.4468466388621959),
    ("exact-seed59", 1.0417, "PASS", "inside", 2.195718028526063),
    ("exact-seed60", 1.0171, "PASS", "inside", 0.4120617162447295),
    ("exact-seed61", 1.0187, "PASS", "inside", 0.5293769796489998),
    ("exact-seed62", 0.9988, "PASS", "inside", -0.9430140948790219),
    ("exact-seed63", 0.9885, "PASS", "inside", -1.7166689208680679),
    ("exact-seed64", 1.0152, "PASS", "inside", 0.27250990797069324),
    ("exact-seed65", 1.0208, "PASS", "inside", 0.6830739463631683),
    ("exact-seed66", 1.0096, "PASS", "inside", -0.14032514020764952),
    ("exact-seed67", 1.0180, "PASS", "inside", 0.4780742407123558),
    ("exact-seed68", 0.9979, "PASS", "inside", -1.0102961547198352),
    ("exact-seed69", 1.0239, "PASS", "inside", 0.9093830307872974),
    ("exact-seed70", 1.0063, "PASS", "inside", -0.38467657920879444),
    ("cuvs-igd48-seed42", 1.0080, "PASS", "inside", -0.2586986602956652),
    ("cuvs-igd48-seed43", 1.0191, "PASS", "inside", 0.558677005060159),
    ("cuvs-igd48-seed44", 1.0345, "PASS", "inside", 1.6780699471108411),
    ("cluster-spill-c4-seed42", 0.9946, "PASS", "inside", -1.2575172720487455),
    ("cluster-spill-c4-seed43", 0.9932, "PASS", "inside", -1.362646864114957),
    ("cluster-spill-c4-seed44", 1.0009, "PASS", "inside", -0.7862580974458578),
    ("cluster-spill-c8-seed42", 0.9893, "PASS", "inside", -1.656291133152107),
    ("cluster-spill-c8-seed43", 0.9738, "PASS", "inside", -2.834894990834103),
    ("cluster-spill-c8-seed44", 1.0271, "PASS", "inside", 1.1422748473225095),
    ("cluster-spill-c16-seed42", 0.9989, "PASS", "inside", -0.9355420531477947),
    ("cluster-spill-c16-seed43", 1.0063, "PASS", "inside", -0.38467657920879444),
    ("cluster-spill-c16-seed44", 0.9916, "PASS", "inside", -1.482976577838234),
    ("replay-control-seed42", 1.0216, "PASS", "inside", 0.7415420173283057),
    ("ladder-6250k-h2048-seed42", 1.1012, "FAIL", "over", 6.341402679742757),
    ("ladder-6250k-h2048-seed43", 1.1022, "FAIL", "over", 6.4091475230706205),
    ("ladder-6250k-h2048-seed44", 1.0982, "FAIL", "over", 6.1377984193361765),
]

CANONICAL_MODULE_PATH = (
    "/home/enjalot/code/latent-basemap/experiments/metrics/k256_two_sided.py"
)


# --------------------------------------------------------------------------- #
# A. judge equivalence — the machine-checked oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,ratio,verdict,direction,z_madn", CANONICAL_ORACLE)
def test_local_judge_matches_the_canonical_oracle_bitwise(
    name, ratio, verdict, direction, z_madn
):
    """The local `_judge_k256_two_sided` reproduces the canonical judge on every map.

    verdict + direction + z_madn, the last compared bitwise (`==`, not `approx`).
    """
    judged = _judge_k256_two_sided(ratio, BAND)
    assert judged["verdict"] == verdict, name
    assert judged["direction"] == direction, name
    assert judged["z_madn"] == z_madn, name


def test_the_oracle_covers_the_required_family_and_rung_cells():
    """29 family cells + the three 6.25M rungs + the c8-seed43 held-out flip."""
    names = {row[0] for row in CANONICAL_ORACLE}
    assert {f"exact-seed{seed}" for seed in range(42, 71)} <= names
    assert {
        "ladder-6250k-h2048-seed42",
        "ladder-6250k-h2048-seed43",
        "ladder-6250k-h2048-seed44",
    } <= names
    assert "cluster-spill-c8-seed43" in names
    assert len(CANONICAL_ORACLE) == 45


def test_the_three_rung_maps_fail_over_at_the_registered_z():
    """The pre-reg table: +6.341 / +6.409 / +6.138, FAIL/over, over the restored band."""
    expected = {
        1.1012: 6.341402679742757,
        1.1022: 6.4091475230706205,
        1.0982: 6.1377984193361765,
    }
    for ratio, z in expected.items():
        judged = _judge_k256_two_sided(ratio, BAND)
        assert judged["verdict"] == "FAIL"
        assert judged["direction"] == "over"
        assert judged["z_madn"] == z


def test_the_held_out_c8_seed43_passes_inside_at_the_registered_z():
    """The honest other half: R0260's tighter one-sided floor had failed it; the band passes."""
    ratio = float(SEALED_HELD_OUT_FLIP["k256"])
    assert ratio == 0.9738
    judged = _judge_k256_two_sided(ratio, BAND)
    assert judged["verdict"] == "PASS"
    assert judged["direction"] == "inside"
    assert judged["z_madn"] == -2.834894990834103


def test_optional_cross_check_against_the_live_canonical_module():
    """If the canonical module (a different repo, numpy/scipy) is importable, cross-check it.

    Skipped when absent — the frozen oracle above is the load-bearing equivalence check.
    """
    if not os.path.exists(CANONICAL_MODULE_PATH):
        pytest.skip("canonical k256_two_sided.py is not present in this environment")
    spec = importlib.util.spec_from_file_location("k256_canon", CANONICAL_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # numpy/scipy or sealed-data unavailable
        pytest.skip(f"canonical module not importable here: {type(error).__name__}: {error}")
    band = module.fit_band(module.SEALED_FAMILY_K256, k2=module.R0255_N29_K2)
    # The canonical fit reproduces R0255's band, which is R0263's sealed band, bitwise.
    assert band["lower"] == BAND_LOWER
    assert band["upper"] == BAND_UPPER
    for _name, ratio, _verdict, _direction, _z in CANONICAL_ORACLE:
        canonical = module.judge(ratio, band)
        local = _judge_k256_two_sided(ratio, BAND)
        assert local["verdict"] == canonical["verdict"]
        assert local["direction"] == canonical["direction"]
        assert local["z_madn"] == canonical["z_madn"]


# --------------------------------------------------------------------------- #
# B. positive controls — every new guard gets a planted defect that MUST fire
# --------------------------------------------------------------------------- #


def _honest_registry():
    criteria = _registered_criteria_k256_two_sided(
        band=BAND, k1024_floor=K1024_FLOOR, ffr_floor=FFR_FLOOR
    )
    descriptive = _descriptive_one_sided_diagnostics(
        one_sided_floor=RETAINED_ONE_SIDED_FLOOR, k1=SEALED_K1
    )
    return criteria, descriptive


def test_the_honest_registry_passes_both_inverted_guards():
    criteria, descriptive = _honest_registry()
    assert _assert_registry_restores_k256_two_sided(criteria, descriptive)["holds"]
    assert _assert_no_descriptive_value_in_the_gate(
        criteria=criteria, descriptive=descriptive
    )["holds"]
    # the restored k256 side is a two-sided band; k1024/ffr stay one-sided floors.
    assert criteria[RESTORED_METRIC]["kind"] == "two_sided_ratio_band"
    assert "ratio_upper" not in criteria["purity_fidelity_k1024"]


# --- guard 1: the ordering positive control (review-0260-01) ---------------- #


def test_the_ordering_guard_refuses_a_planted_rung_manifest_but_passes_the_real_one():
    real_job = {
        "expected_inputs": [
            {"canonical_path": "/data/latent-basemap/runs/round-0255/queue/artifacts/x/panel.json"},
            {
                "canonical_path": (
                    "/data/latent-basemap/runs/round-0260/queue/artifacts/"
                    "minilm-mixed-2m-one-sided-purity-floors-n29-v1/floors.json"
                )
            },
        ]
    }
    # The real (clean) manifest passes the same refusal the node applies to itself.
    assert _refuse_if_rung_bound(real_job)[
        "no_rung_artifact_is_bound_to_the_registration"
    ]
    # The planted-rung manifest MUST raise through the identical code path.
    planted_job = {
        "expected_inputs": [
            {
                "canonical_path": (
                    "/data/latent-basemap/runs/round-0257/queue-correction-2/"
                    "artifacts/minilm-mixed-6250k-rung-gate-verdict-v1/judge.json"
                )
            }
        ]
    }
    with pytest.raises(Round0263NodeError):
        _refuse_if_rung_bound(planted_job)
    # And the aggregate control proves the guard discriminates, not merely passes.
    ordering = _ordering_positive_control(
        real_job, planted_rung_path=planted_job["expected_inputs"][0]["canonical_path"]
    )
    assert ordering["real_manifest_passes_the_guard"]
    assert ordering["planted_manifest_is_refused"]
    assert ordering["the_ordering_guard_discriminates"]


# --- guard 2: the value-level descriptive leak guard ------------------------ #


def test_a_retired_descriptive_value_smuggled_into_a_gated_key_is_refused():
    """The realistic defect: the descriptive one-sided floor pasted into the gate as a
    k256 ratio_floor — a shape a key-name guard cannot see."""
    criteria, descriptive = _honest_registry()
    leaked_criteria = {
        **criteria,
        RESTORED_METRIC: {
            **criteria[RESTORED_METRIC],
            "ratio_floor": RETAINED_ONE_SIDED_FLOOR,
        },
    }
    with pytest.raises(Round0263NodeError):
        _assert_no_descriptive_value_in_the_gate(
            criteria=leaked_criteria, descriptive=descriptive
        )


def test_the_restored_band_edges_are_the_sole_gated_values_exempt_from_the_leak_guard():
    """A sanity face of the same guard: the two restored edges are legitimately gated."""
    criteria, descriptive = _honest_registry()
    report = _assert_no_descriptive_value_in_the_gate(
        criteria=criteria, descriptive=descriptive
    )
    assert sorted(report["restored_edges_exempted"]) == sorted([BAND_LOWER, BAND_UPPER])
    assert report["values_leaking_into_the_gate"] == []


# --- guard 3: the registry-restore guard, via the aggregate plant harness --- #


def test_every_planted_registry_defect_is_refused():
    criteria, descriptive = _honest_registry()
    controls = _registry_controls_k256_two_sided(
        criteria=criteria, descriptive=descriptive, floor=RETAINED_ONE_SIDED_FLOOR
    )
    assert controls["the_honest_registry_passes"]
    assert controls["planted_defects"] == 5
    assert controls["planted_defects_refused"] == 5
    assert controls["every_planted_defect_is_refused"]
    assert controls["holds"]


def test_leaving_k256_as_a_one_sided_floor_is_refused():
    """The round-does-nothing defect, exercised directly."""
    criteria, descriptive = _honest_registry()
    with pytest.raises(Round0263NodeError):
        _assert_registry_restores_k256_two_sided(
            {
                **criteria,
                RESTORED_METRIC: {
                    "kind": "one_sided_ratio_floor",
                    "ratio_floor": RETAINED_ONE_SIDED_FLOOR,
                    "applicable_power": "one_sided",
                },
            },
            descriptive,
        )


def test_giving_k1024_the_restored_upper_edge_is_refused():
    """This round is k256-only; k1024's retired upper edge fails a faithful map."""
    criteria, descriptive = _honest_registry()
    with pytest.raises(Round0263NodeError):
        _assert_registry_restores_k256_two_sided(
            {
                **criteria,
                "purity_fidelity_k1024": {
                    **criteria["purity_fidelity_k1024"],
                    "ratio_upper": 0.7719564268121961,
                },
            },
            descriptive,
        )


# --- guard 4: descriptive-floor inertness, both arms ------------------------ #


def test_the_descriptive_floor_is_inert_to_the_shipped_judge_and_the_plant_is_potent():
    """Move the retired floor to 1e9: the shipped judge's margins stay bitwise, while a
    planted consumer that gates on the floor must flip — so the stillness is evidence."""
    controls = _descriptive_floor_never_gates_controls(
        observations=SEALED_RUNG_K256, band=BAND, floor=RETAINED_ONE_SIDED_FLOOR
    )
    assert controls["the_shipped_judge_is_inert_to_the_descriptive_floor"]
    assert controls["shipped_margins_are_bitwise_identical"]
    assert controls["the_mutation_is_potent"]  # the planted consumer's verdict moved
    assert controls["planted_consumer_verdicts_changed"]
    assert controls["holds"]


# --- guard 5: the sidedness controls (the restoration's whole content) ------ #


def test_every_sidedness_control_holds_including_an_over_direction_fail():
    controls = _sidedness_controls_k256_two_sided(BAND)
    assert controls["every_control_held"]
    assert controls["at_least_one_control_fails_over"]  # the direction R0260 removed
    assert controls["at_least_one_control_fails_under"]


# --- guard 6: the bitwise-band comparator ----------------------------------- #


def test_a_perturbed_band_fails_the_bitwise_comparator():
    """The bitwise guard node 1 uses to require the fit is R0255's: perturb an edge, it fails."""
    honest = _bitwise_same(
        {"ratio_lower": BAND_LOWER, "ratio_upper": BAND_UPPER},
        {
            "ratio_lower": SEALED_K256_BAND["ratio_lower"],
            "ratio_upper": SEALED_K256_BAND["ratio_upper"],
        },
    )
    assert honest["every_value_is_bitwise_identical"]
    perturbed = _bitwise_same(
        {"ratio_lower": BAND_LOWER, "ratio_upper": BAND_UPPER},
        {"ratio_lower": BAND_LOWER + 1e-9, "ratio_upper": BAND_UPPER},
    )
    assert not perturbed["every_value_is_bitwise_identical"]


# --------------------------------------------------------------------------- #
# C. bitwise reproduction of every carried constant
# --------------------------------------------------------------------------- #


def test_the_restored_band_edges_reproduce_bitwise():
    assert SEALED_K256_BAND["ratio_lower"] == BAND_LOWER
    assert SEALED_K256_BAND["ratio_upper"] == BAND_UPPER
    # and the local judge derives them into the band it reads.
    criteria = _registered_criteria_k256_two_sided(
        band=BAND, k1024_floor=K1024_FLOOR, ffr_floor=FFR_FLOOR
    )
    assert criteria[RESTORED_METRIC]["ratio_lower"] == BAND_LOWER
    assert criteria[RESTORED_METRIC]["ratio_upper"] == BAND_UPPER


def test_the_retained_and_untouched_floors_reproduce_bitwise():
    assert SEALED_K256_ONE_SIDED_FLOOR == RETAINED_ONE_SIDED_FLOOR
    assert SEALED_K1024_ONE_SIDED_FLOOR == K1024_FLOOR
    assert SEALED_FFR_FLOOR == FFR_FLOOR


def test_the_identity_bound_is_28_over_sqrt_29_and_the_multiplier_sits_below_it():
    assert identity_bound(N_EXACT) == IDENTITY_BOUND
    assert SEALED_IDENTITY_BOUND_AT_N29 == IDENTITY_BOUND
    assert SEALED_K2 < IDENTITY_BOUND
    assert SEALED_K1 < SEALED_K2  # the retired floor is tighter on the under side


def test_the_calibration_multipliers_reproduce_bitwise():
    assert SEALED_K2 == 3.147763220676551
    assert SEALED_K1 == 2.6934393367626024
