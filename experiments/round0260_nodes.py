"""Execute R0260 — register the one-sided purity floors, then re-adjudicate R0257.

Two CPU nodes, in that order, and the order is the round's central claim.

**Node 1, `register_one_sided_purity_floors_n29`.** Reads the sealed twenty-nine
2M family cells and the sealed R0256 gate artifact. Re-derives the calibration at
`n = 29` restricted to the registered estimator and requires it bitwise; re-derives
the retired two-sided bands and requires them bitwise (proof the family and the
fit are the same ones, so the only change is sidedness); fits the two one-sided
purity ratio floors; publishes attainability against the identity bound, one-sided
power matched to the now-one-sided criteria, and the compounded panel false-alarm
rate. **No rung artifact is bound to this node**, and the node proves that by
reading its own declared inputs rather than asserting it.

**Node 2, `readjudicate_0257_one_sided`.** Depends on node 1. Re-scores R0257's
three sealed `6250k` maps from their sealed coordinates' panel values, publishes
the new verdict beside the old one read out of R0257's own sealed verdict
artifact, and carries the retired two-sided values beside every row as descriptive
diagnostics. It refuses to run unless node 1's registration was sealed **before**
this node started, by the producer's own done-marker timestamp.

Nothing here trains anything, touches a floor R0255/R0256 registered, or rewrites
a published result. R0257's three FAILs stand; this round adds a dated record.
"""
from __future__ import annotations

import json
import math
import os
import re
import resource
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import create_fresh_directory
from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0234_calibration as calibration
from basemap.round0234_calibrated_floors import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    METRICS,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    attainability,
    band_at,
    floor_at,
    positive_scale_witness,
)
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0255_gate_n29 import (
    EXACT_FAMILY_SEEDS,
    IDENTITY_BOUND_AT_N,
    N_EXACT,
)
from basemap.round0255_treatment import (
    assert_family_is_2m_only,
    exact_cell_id,
)
from basemap.round0256_gate_repair import (
    DEFINING_CELL_IDS,
    GATE_CAPABILITY as R0256_GATE_CAPABILITY,
    REGISTERED_ESTIMATOR_NAMES,
)
from basemap.round0257_judgement import (
    GATE_CAPABILITY as R0257_CONSUMED_GATE_CAPABILITY,
)
from basemap.round0257_rung_contract import (
    assert_no_rung_map_in_the_gate_family,
    rung_cell_ids,
    rung_family_purity_controls,
)
from basemap.round0260_one_sided_purity import (
    CONFORMITY_MARKING,
    GATE_CAPABILITY,
    GATE_SCHEMA,
    NO_TUNING_STATEMENT,
    OWNER_RULING,
    OWNER_RULING_DATE,
    OWNER_RULING_SECTION,
    ROUND_ID,
    Round0260RegistrationError,
    SUPERSEDED_PURITY_CRITERIA_IN,
    WHY_ONE_SIDED,
    assert_no_gate_reads_a_conformity_value,
    assert_registry_is_one_sided_only,
    attainability_against_the_identity_bound,
    conformity_diagnostics,
    conformity_diagnostics_never_gate_controls,
    judge_population_one_sided,
    one_sided_purity_floors,
    panel_false_alarm_rate_one_sided,
    registered_criteria_one_sided,
    registry_controls_one_sided,
    sidedness_controls,
    the_retired_band_disagrees_control,
)

from experiments.round0255_nodes import (
    SAFETY_NOTE,
    _bound_path,
    _guard_tail_reported,
    _node_gate,
    _node_guard,
    _score_gate_without_raising,
    _seal,
    _start_node,
)


REGISTER_ACTION = "register_one_sided_purity_floors_n29"
READJUDICATE_ACTION = "readjudicate_0257_one_sided"

REGISTRATION_ARTIFACT = "minilm-one-sided-purity-floors-n29.json"
READJUDICATION_CAPABILITY = "minilm-mixed-6250k-rung-readjudication-one-sided-v1"
READJUDICATION_SCHEMA = "round0260-6250k-rung-readjudication-one-sided-v1"
READJUDICATION_ARTIFACT = "readjudication-0257-one-sided.json"

CUDA_LIBRARY_PATTERN = re.compile(
    r"libcuda|libcuvs|libcudart|libcublas|libcudnn|nvidia", re.IGNORECASE
)

#: Any bound input whose path matches one of these is a rung artifact. The
#: registration node must have none: a criterion fitted with a judged map's
#: numbers in reach is retro-fitting whatever the code does with them.
RUNG_PATH_MARKERS: tuple[str, ...] = ("round-0257", "6250k", "ladder")

R0257_PUBLISHED_VERDICT = "FAIL"


class Round0260NodeError(RuntimeError):
    """The R0260 node contract changed."""


def _cuda_mappings() -> list[str]:
    """Measure, do not assert, that this node holds no GPU library."""
    with open("/proc/self/maps", "r", encoding="utf-8") as handle:
        return sorted({
            line.rsplit(" ", 1)[-1].strip()
            for line in handle
            if CUDA_LIBRARY_PATTERN.search(line)
        })


def _bitwise_same(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Every shared key compared with `==` on floats — no tolerance anywhere."""
    keys = sorted(set(left) & set(right))
    rows = {
        key: {
            "sealed": left[key],
            "rederived": right[key],
            "identical": left[key] == right[key],
            "delta": (
                float(right[key]) - float(left[key])
                if isinstance(left[key], (int, float))
                and isinstance(right[key], (int, float))
                else None
            ),
        }
        for key in keys
    }
    return {
        "compared": keys,
        "by_key": rows,
        "every_value_is_bitwise_identical": all(
            row["identical"] for row in rows.values()
        ),
    }


def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    """Resolve a reference to an artifact THIS queue produces.

    A node downstream of a producer cannot be handed a hash at prepare time,
    because the bytes do not exist yet. If the reference carries a `sha256` it is
    verified in full; otherwise the path is resolved and its signature computed
    at read time. Using `verify_signature` on a hashless intra-queue reference is
    what failed R0257's first panel attempt.
    """
    reference = dict(reference)
    if reference.get("sha256"):
        return prompt_contract.verify_signature(reference, label=label), reference
    path = str(reference.get("canonical_path") or "")
    if not path or not os.path.exists(path):
        raise Round0260NodeError(f"{label} is absent at {path!r}")
    return path, expected_input_signature(path)


def _declared_input_paths(job: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("canonical_path") or "")
        for item in job.get("expected_inputs") or []
        if isinstance(item, Mapping)
    ]


def _no_rung_artifact_is_bound(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read this node's own declared inputs and look for a judged map in them."""
    paths = _declared_input_paths(job)
    hits = sorted({
        path
        for path in paths
        for marker in RUNG_PATH_MARKERS
        if marker in path
    })
    return {
        "declared_inputs": sorted(paths),
        "declared_input_count": len(paths),
        "rung_path_markers": list(RUNG_PATH_MARKERS),
        "inputs_matching_a_rung_marker": hits,
        "no_rung_artifact_is_bound_to_the_registration": not hits,
        "why_this_is_measured_and_not_asserted": (
            "the ruling requires the criterion to be registered BEFORE any further "
            "rung map is judged. The strongest version of that is a registration "
            "node with no path to a judged map at all, which is a property of the "
            "queue manifest and is read here rather than promised."
        ),
    }


def _family_from_panel(panel: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]], dict[str, list[float]], dict[str, list[float]]
]:
    """The twenty-nine sealed 2M cells, their metric series and log-ratio series."""
    cells = panel["panel_metric_cells"]
    ratios = panel["raw_purity_ratios"]
    if tuple(sorted(int(seed) for seed in cells)) != EXACT_FAMILY_SEEDS:
        raise Round0260NodeError("R0260 sealed family is not seeds 42-70")
    family = [
        {
            "cell_id": exact_cell_id(seed),
            "values": {
                metric: float(cells[str(seed)][metric]) for metric in METRICS
            },
            "ratios": {
                key: float(ratios[str(seed)][key]) for key in ("k256", "k1024")
            },
        }
        for seed in EXACT_FAMILY_SEEDS
    ]
    if len(family) != N_EXACT:
        raise Round0260NodeError("R0260 family assembly is not 29 cells")
    series = {
        metric: [float(cell["values"][metric]) for cell in family]
        for metric in METRICS
    }
    log_series = {
        metric: [
            math.log(float(cell["ratios"][PURITY_RATIO_KEYS[metric]]))
            for cell in family
        ]
        for metric in PURITY_METRICS
    }
    return family, series, log_series


# --------------------------------------------------------------------------- #
# node 1 — the registration
# --------------------------------------------------------------------------- #


def run_register(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0260 round0260_nodes.run_register")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0260NodeError("R0260 register handler received another queue")
    node_id = str(active.get("node_id") or REGISTER_ACTION)
    label = "R0260 one-sided purity floor registration"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    started_utc = time.time()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0260 registration")

    binding = _no_rung_artifact_is_bound(job)
    if not binding["no_rung_artifact_is_bound_to_the_registration"]:
        raise Round0260NodeError(
            "R0260 registration refuses: a judged rung artifact is bound to the "
            f"node that registers the criterion: {binding['inputs_matching_a_rung_marker']}"
        )

    window = ledger.window("R0260 registration stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0260 registration stage entered")
        wrapped = window.wrap(recorder)

        panel = prompt_contract.read_sealed(
            _bound_path(job, "panel_n29", label="R0255 sealed twenty-nine-cell panel"),
            label="R0255 sealed twenty-nine-cell panel",
        )
        r0256_gate = prompt_contract.read_sealed(
            _bound_path(job, "r0256_gate", label="R0256 sealed repaired n=29 gate"),
            label="R0256 sealed repaired n=29 gate",
        )
        if (
            r0256_gate.get("capability") != R0256_GATE_CAPABILITY
            or int(r0256_gate.get("n", -1)) != N_EXACT
        ):
            raise Round0260NodeError("R0256 sealed gate receipt contract changed")
        estimator = str(r0256_gate["chosen_estimator"])
        wrapped("R0260 sealed family panel and R0256 gate read")

        # ------------------------------------------------------------------ #
        # A. the calibration, unchanged, restricted to the registered estimator
        # ------------------------------------------------------------------ #
        calibration_started = time.monotonic()
        restricted = calibration.calibrate(N_EXACT, names=REGISTERED_ESTIMATOR_NAMES)
        calibration_wall_s = time.monotonic() - calibration_started
        entry = restricted["candidates"][estimator]
        k_one = float(entry["one_sided"]["calibrated_multiplier"])
        k_two = float(entry["two_sided"]["calibrated_multiplier"])
        sealed_calibration = r0256_gate["calibration"]
        calibration_match = _bitwise_same(
            {
                "one_sided_multiplier": float(
                    sealed_calibration["one_sided_multiplier"]
                ),
                "two_sided_multiplier": float(
                    sealed_calibration["two_sided_multiplier"]
                ),
            },
            {"one_sided_multiplier": k_one, "two_sided_multiplier": k_two},
        )
        one_sided_power = {
            key: float(value)
            for key, value in entry["one_sided"]["detection_power"].items()
        }
        one_sided_false_fail = float(entry["one_sided"]["new_cell_false_fail_rate"])
        power_match = _bitwise_same(
            {
                "power_minus_2_sigma": float(
                    r0256_gate["detection_power_one_sided"]["minus_2_sigma"]
                ),
                "one_sided_false_fail": float(
                    r0256_gate["detection_power_by_criterion"]["ffr"][
                        "new_cell_false_fail_rate"
                    ]
                ),
            },
            {
                "power_minus_2_sigma": one_sided_power["minus_2_sigma"],
                "one_sided_false_fail": one_sided_false_fail,
            },
        )
        wrapped("R0260 restricted calibration reproduced bitwise")

        # ------------------------------------------------------------------ #
        # B. the family, the retired bands, and the new one-sided floors
        # ------------------------------------------------------------------ #
        family, series, log_series = _family_from_panel(panel)
        family_guard = assert_family_is_2m_only(list(DEFINING_CELL_IDS))
        family_disjointness = assert_no_rung_map_in_the_gate_family(
            list(DEFINING_CELL_IDS)
        )
        family_plants = rung_family_purity_controls(list(DEFINING_CELL_IDS))
        wrapped("R0260 family purity guard and its four plants complete")

        ffr_floor = floor_at(estimator, series["ffr"], k_one)
        retired_bands = {
            metric: band_at(estimator, log_series[metric], k_two)
            for metric in PURITY_METRICS
        }
        retired_ratio_bands = {
            metric: {
                "ratio_lower": math.exp(retired_bands[metric][0]),
                "ratio_upper": math.exp(retired_bands[metric][1]),
            }
            for metric in PURITY_METRICS
        }
        bands_match = _bitwise_same(
            {
                f"{metric}::{side}": float(
                    r0256_gate["registered_criteria"][metric][f"ratio_{side}"]
                )
                for metric in PURITY_METRICS
                for side in ("lower", "upper")
            },
            {
                f"{metric}::{side}": retired_ratio_bands[metric][f"ratio_{side}"]
                for metric in PURITY_METRICS
                for side in ("lower", "upper")
            },
        )
        ffr_match = _bitwise_same(
            {"ffr": float(r0256_gate["registered_floors"]["ffr"])},
            {"ffr": ffr_floor},
        )
        purity_floors = one_sided_purity_floors(estimator, log_series, k_one)
        wrapped("R0260 one-sided purity floors fitted from the sealed family")

        # ------------------------------------------------------------------ #
        # C. attainability, power, false-alarm rate
        # ------------------------------------------------------------------ #
        identity = attainability_against_the_identity_bound(n=N_EXACT, multiplier=k_one)
        estimator_attainability = attainability(
            estimator, n=N_EXACT, multiplier=k_one
        )
        witness = positive_scale_witness(estimator, k_one, n=N_EXACT)
        panel_far = panel_false_alarm_rate_one_sided(
            one_sided_rate=one_sided_false_fail
        )
        family_failing = {
            metric: sorted(
                cell["cell_id"]
                for cell in family
                if float(cell["ratios"][PURITY_RATIO_KEYS[metric]])
                < purity_floors[metric]["ratio_floor"]
            )
            for metric in PURITY_METRICS
        }
        family_failing["ffr"] = sorted(
            cell["cell_id"] for cell in family if float(cell["values"]["ffr"]) < ffr_floor
        )
        nearest_margin = {
            metric: min(
                (
                    math.log(float(cell["ratios"][PURITY_RATIO_KEYS[metric]]))
                    - purity_floors[metric]["log_floor"]
                )
                / purity_floors[metric]["log_scale"]
                for cell in family
            )
            for metric in PURITY_METRICS
        }
        wrapped("R0260 attainability, power and panel false-alarm rate computed")

        # ------------------------------------------------------------------ #
        # D. the registry, the conformity block, and their controls
        # ------------------------------------------------------------------ #
        criteria = registered_criteria_one_sided(
            ffr_floor=ffr_floor, purity_floors=purity_floors
        )
        scalar_floors = {"ffr": ffr_floor}
        ratio_floors = {
            metric: float(purity_floors[metric]["ratio_floor"])
            for metric in PURITY_METRICS
        }
        conformity = conformity_diagnostics(
            two_sided_bands=retired_ratio_bands, two_sided_multiplier=k_two
        )
        registry_guard = assert_registry_is_one_sided_only(
            criteria, scalar_floors, ratio_floors
        )
        conformity_guard = assert_no_gate_reads_a_conformity_value(
            criteria=criteria,
            scalar_floors=scalar_floors,
            ratio_floors=ratio_floors,
            conformity=conformity,
        )
        descriptive_floors = {
            metric: floor_at(estimator, series[metric], k_one)
            for metric in DESCRIPTIVE_METRICS
        }
        registry_plants = registry_controls_one_sided(
            criteria=criteria,
            scalar_floors=scalar_floors,
            ratio_floors=ratio_floors,
            conformity=conformity,
            descriptive_values=descriptive_floors,
        )
        wrapped("R0260 registry rebuilt and its nine plants complete")

        gate_body_core = {
            "registered_criteria": criteria,
            "registered_floors": scalar_floors,
            "registered_ratio_floors": ratio_floors,
            "descriptive_conformity_diagnostics": conformity,
            "detection_power_by_criterion": {
                metric: {
                    "criterion": "one-sided lower floor",
                    "applicable_power": "one_sided",
                    "detection_power": dict(one_sided_power),
                    "new_cell_false_fail_rate": one_sided_false_fail,
                }
                for metric in GATED_METRICS
            },
            "panel_false_alarm_rate": panel_far,
        }
        sidedness = sidedness_controls(gate_body_core)
        disagreement = the_retired_band_disagrees_control(gate_body_core)
        wrapped("R0260 sidedness controls complete")
        gate.finish("R0260 registration stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    cuda_mappings = _cuda_mappings()

    execution_checks = {
        "no_rung_artifact_is_bound_to_the_registration": bool(
            binding["no_rung_artifact_is_bound_to_the_registration"]
        ),
        "the_calibration_is_restricted_to_the_registered_estimator": (
            tuple(restricted["candidates"]) == REGISTERED_ESTIMATOR_NAMES
        ),
        "the_calibration_reproduces_r0256_bitwise": bool(
            calibration_match["every_value_is_bitwise_identical"]
        ),
        "the_one_sided_power_reproduces_r0256_bitwise": bool(
            power_match["every_value_is_bitwise_identical"]
        ),
        "the_retired_two_sided_bands_reproduce_bitwise": bool(
            bands_match["every_value_is_bitwise_identical"]
        ),
        "the_ffr_floor_is_unchanged": bool(
            ffr_match["every_value_is_bitwise_identical"]
        ),
        "the_family_is_the_2m_universe_only": bool(
            family_guard["family_is_the_2m_universe_only"]
        ),
        "the_family_and_the_judged_maps_are_disjoint": bool(
            family_disjointness["judged_and_defining_are_disjoint"]
        ),
        "every_planted_family_defect_was_refused": bool(
            family_plants["every_planted_defect_was_refused"]
        ),
        "the_multiplier_is_below_the_identity_bound": bool(
            identity["multiplier_is_below_the_identity_bound"]
        ),
        "every_defining_cell_can_fail": bool(
            estimator_attainability["every_defining_cell_can_fail"]
        ),
        "a_positive_scale_witness_fails_its_own_floor": bool(
            witness["lowest_cell_fails_its_own_floor"]
            and witness["scale_is_strictly_positive"]
        ),
        "every_gated_criterion_is_one_sided": bool(
            registry_guard["every_gated_criterion_is_one_sided"]
        ),
        "no_conformity_value_is_reachable_inside_the_gate": bool(
            conformity_guard["holds"]
        ),
        "every_planted_registry_defect_is_refused": bool(
            registry_plants["every_planted_defect_is_refused"]
        ),
        "every_sidedness_control_held": bool(sidedness["every_control_held"]),
        "the_re_registration_is_not_a_rename": bool(
            disagreement["the_re_registration_is_not_a_rename"]
        ),
        "the_panel_rate_is_above_the_per_metric_rate": (
            panel_far["panel_false_alarm_rate_under_independence"]
            > one_sided_false_fail
        ),
        "no_cuda_library_is_mapped_into_this_process": not cuda_mappings,
    }
    if not all(execution_checks.values()):
        raise Round0260NodeError(f"R0260 registration checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body: dict[str, Any] = {
        "round_id": ROUND_ID,
        "release_sha": str(active["manifest"]["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": None,
        "schema": GATE_SCHEMA,
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "node": node_id,
        "node_started_utc": started_utc,
        "abort_flag_precondition": abort_flag,
        "n": N_EXACT,
        "owner_ruling": OWNER_RULING,
        "owner_ruling_date": OWNER_RULING_DATE,
        "owner_ruling_section": OWNER_RULING_SECTION,
        "why_one_sided": WHY_ONE_SIDED,
        "no_tuning_statement": NO_TUNING_STATEMENT,
        "chosen_estimator": estimator,
        "supersedes_purity_criteria_in": SUPERSEDED_PURITY_CRITERIA_IN,
        "supersession_note": (
            "this artifact replaces the two PURITY CRITERIA of "
            f"{SUPERSEDED_PURITY_CRITERIA_IN} and nothing else. Its ffr floor is "
            "that artifact's, bitwise. Every number R0255 and R0256 published "
            "remains true as published; what changes is which side of the purity "
            "distribution is allowed to fail a map."
        ),
        "registration_precedes_any_judgement": dict(binding),
        # --- A ---
        "calibration": {
            "estimators_calibrated": list(restricted["candidates"]),
            "families_simulated": int(restricted["families_simulated"]),
            "seed": int(restricted["seed"]),
            "content": float(restricted["content"]),
            "confidence": float(restricted["confidence"]),
            "one_sided_multiplier": k_one,
            "two_sided_multiplier_retired_from_gating": k_two,
            "wall_s": calibration_wall_s,
            "reproduces_r0256_multipliers": calibration_match,
            "reproduces_r0256_one_sided_power": power_match,
            "why_unchanged": (
                "the calibration is a property of the estimator at n = 29 on a "
                "Gaussian null and reads no cell of this program. The ruling "
                "changed which tail is gated, not how the multiplier is derived, "
                "so the multiplier must come back identical -- and does."
            ),
        },
        # --- B ---
        "family": {
            "n": N_EXACT,
            "exact_family_seeds": list(EXACT_FAMILY_SEEDS),
            "defining_cell_ids": list(DEFINING_CELL_IDS),
            "guard": family_guard,
            "disjointness": family_disjointness,
            "planted_controls": family_plants,
            "rung_cell_ids_this_family_must_exclude": list(rung_cell_ids()),
            "review_0257_E2_note": (
                "review-0257-01 section E.2 found that "
                "assert_family_is_2m_only returns no_judged_map_is_in_the_family "
                "as a hardcoded literal from a function that never receives the "
                "judged set. That field is NOT quoted as evidence here. The "
                "load-bearing measurement is judged_and_defining_are_disjoint, "
                "with n_defining and n_judged, which is computed."
            ),
        },
        "registered_criteria": criteria,
        "registered_floors": scalar_floors,
        "registered_ratio_floors": ratio_floors,
        "one_sided_purity_floor_derivation": {
            metric: dict(purity_floors[metric]) for metric in PURITY_METRICS
        },
        "retired_two_sided_bands_reproduce_bitwise": bands_match,
        "ffr_floor_reproduces_bitwise": ffr_match,
        # --- C ---
        "attainability_against_the_identity_bound": identity,
        "identity_bound_at_n": IDENTITY_BOUND_AT_N,
        "estimator_attainability": estimator_attainability,
        "positive_scale_witness": witness,
        "detection_power_by_criterion": gate_body_core["detection_power_by_criterion"],
        "detection_power_one_sided": one_sided_power,
        "one_sided_new_cell_false_fail_rate": one_sided_false_fail,
        "panel_false_alarm_rate": panel_far,
        "family_cells_failing_each_criterion": family_failing,
        "nearest_family_cell_margin_in_scales": nearest_margin,
        "tolerance_content": TOLERANCE_CONTENT,
        "tolerance_confidence": TOLERANCE_CONFIDENCE,
        # --- D ---
        "descriptive_conformity_diagnostics": conformity,
        "registry_guard": registry_guard,
        "conformity_guard": conformity_guard,
        "registry_controls": registry_plants,
        "sidedness_controls": sidedness,
        "the_retired_band_disagrees": disagreement,
        "descriptive_only_never_a_gate": {
            metric: float(descriptive_floors[metric]) for metric in DESCRIPTIVE_METRICS
        },
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "execution_checks": execution_checks,
        "evaluation_performed": False,
        "training_performed": False,
        "gpu_used": bool(cuda_mappings),
        "cuda_libraries_mapped_into_this_process": cuda_mappings,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "sources": {
            "r0255_panel_n29": expected_input_signature(
                _bound_path(job, "panel_n29", label="R0255 panel")
            ),
            "r0256_repaired_gate_n29": expected_input_signature(
                _bound_path(job, "r0256_gate", label="R0256 gate")
            ),
        },
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    }
    _seal(output, REGISTRATION_ARTIFACT, body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "n": N_EXACT,
        "one_sided_multiplier": k_one,
        "identity_bound": identity["identity_bound"],
        "ffr_floor": ffr_floor,
        "ratio_floors": ratio_floors,
        "panel_false_alarm_rate": panel_far[
            "panel_false_alarm_rate_under_independence"
        ],
        "widest_gap_s": gaps.get("widest_gap_s"),
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# node 2 — the re-adjudication
# --------------------------------------------------------------------------- #


def _r0257_cells(panel: Mapping[str, Any], verdict: Mapping[str, Any]) -> dict[str, Any]:
    """R0257's three judged maps, from its own sealed panel and verdict bytes."""
    cells: dict[str, Any] = {}
    published = dict(verdict["judgement"]["verdicts"])
    by_cell_id = {
        str(cell["cell_id"]): cell for cell in dict(panel["cells"]).values()
    }
    if sorted(by_cell_id) != sorted(rung_cell_ids()):
        raise Round0260NodeError(
            "R0260 sealed R0257 panel does not carry exactly the three rung cells"
        )
    for cell_id in sorted(published):
        if cell_id not in by_cell_id:
            raise Round0260NodeError(
                f"R0260 cannot find sealed panel values for {cell_id}"
            )
        panel_cell = by_cell_id[cell_id]
        cells[cell_id] = {
            "panel_metrics": {
                metric: float(panel_cell["panel_metrics"][metric])
                for metric in METRICS
            },
            "raw_purity_ratios": {
                key: float(panel_cell["raw_purity_ratios"][key])
                for key in ("k256", "k1024")
            },
            "published_verdict": str(published[cell_id]["verdict"]),
            "published_per_metric": {
                metric: {
                    "kind": str(published[cell_id]["per_metric"][metric]["kind"]),
                    "criterion": str(
                        published[cell_id]["per_metric"][metric]["criterion"]
                    ),
                    "observed": float(
                        published[cell_id]["per_metric"][metric]["observed"]
                    ),
                    "passes": bool(published[cell_id]["per_metric"][metric]["passes"]),
                    "margin": float(published[cell_id]["per_metric"][metric]["margin"]),
                }
                for metric in GATED_METRICS
            },
        }
    if len(cells) != len(rung_cell_ids()):
        raise Round0260NodeError("R0260 re-adjudication is not the three rung maps")
    return cells


def run_readjudicate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0260 round0260_nodes.run_readjudicate")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0260NodeError("R0260 readjudicate handler received another queue")
    node_id = str(active.get("node_id") or READJUDICATE_ACTION)
    label = "R0260 R0257 re-adjudication under the one-sided criterion"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    started_utc = time.time()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0260 re-adjudication"
    )

    window = ledger.window("R0260 re-adjudication stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0260 re-adjudication stage entered")
        wrapped = window.wrap(recorder)

        registration_path, registration_signature = _intra_queue_signature(
            job["registration"], label="R0260 sealed one-sided registration"
        )
        registration = prompt_contract.read_sealed(
            registration_path, label="R0260 sealed one-sided registration"
        )
        if (
            registration.get("capability") != GATE_CAPABILITY
            or registration.get("schema") != GATE_SCHEMA
            or registration.get("gate_registered") is not True
        ):
            raise Round0260NodeError("R0260 registration artifact contract changed")

        marker_path = str(job["registration_done_marker"])
        if not os.path.exists(marker_path):
            raise Round0260NodeError(
                f"R0260 registration done-marker is absent at {marker_path}"
            )
        with open(marker_path, "rb") as handle:
            marker = json.load(handle)
        sealed_at = os.path.getmtime(registration_path)
        ordering = {
            "registration_artifact": registration_path,
            "registration_sha256": registration_signature["sha256"],
            "registration_bytes": registration_signature["bytes"],
            "registration_done_marker": marker_path,
            "registration_node": str(marker.get("node") or marker.get("id") or ""),
            "registration_sealed_at_epoch_s": sealed_at,
            "readjudication_started_at_epoch_s": started_utc,
            "seconds_between_registration_and_adjudication": started_utc - sealed_at,
            "registration_precedes_adjudication": sealed_at < started_utc,
            "the_registration_node_bound_no_rung_artifact": bool(
                registration["registration_precedes_any_judgement"][
                    "no_rung_artifact_is_bound_to_the_registration"
                ]
            ),
            "why_this_is_receipt_evidence": (
                "the runner runs this node only after the producer's done-marker "
                "exists, and the registration bytes are hashed here. Order is "
                "therefore a property of the queue's receipts, not of prose."
            ),
        }
        if not ordering["registration_precedes_adjudication"]:
            raise Round0260NodeError(
                "R0260 refuses to adjudicate: the registration was not sealed first"
            )
        if not ordering["the_registration_node_bound_no_rung_artifact"]:
            raise Round0260NodeError(
                "R0260 refuses to adjudicate: the registration node was bound to a "
                "judged rung artifact"
            )
        wrapped("R0260 registration ordering verified from receipts")

        r0257_panel = prompt_contract.read_sealed(
            _bound_path(job, "r0257_panel", label="R0257 sealed rung panel"),
            label="R0257 sealed rung panel",
        )
        r0257_verdict = prompt_contract.read_sealed(
            _bound_path(job, "r0257_verdict", label="R0257 sealed rung verdict"),
            label="R0257 sealed rung verdict",
        )
        if (
            r0257_verdict.get("round_id") != "0257"
            or str(r0257_verdict.get("gate_capability_consumed"))
            != R0257_CONSUMED_GATE_CAPABILITY
            or r0257_verdict.get("no_floor_is_written_by_this_round") is None
        ):
            raise Round0260NodeError("R0257 sealed verdict contract changed")
        cells = _r0257_cells(r0257_panel, r0257_verdict)
        wrapped("R0260 R0257 sealed panel and verdict read")

        judged = judge_population_one_sided(cells=cells, gate=registration)
        inertness = conformity_diagnostics_never_gate_controls(
            cells=cells, gate=registration
        )
        wrapped("R0260 three maps re-scored and the inertness controls complete")

        comparison: dict[str, Any] = {}
        for cell_id, payload in cells.items():
            new = judged["verdicts"][cell_id]
            comparison[cell_id] = {
                "verdict_as_published_by_r0257": payload["published_verdict"],
                "verdict_under_the_one_sided_criterion": new["verdict"],
                "verdict_changed": (
                    payload["published_verdict"] != new["verdict"]
                ),
                "per_metric": {
                    metric: {
                        "observed": new["per_metric"][metric]["observed"],
                        "observed_as_published_by_r0257": payload[
                            "published_per_metric"
                        ][metric]["observed"],
                        "observation_is_bitwise_identical": (
                            new["per_metric"][metric]["observed"]
                            == payload["published_per_metric"][metric]["observed"]
                        ),
                        "r0257_criterion": payload["published_per_metric"][metric][
                            "criterion"
                        ],
                        "r0257_kind": payload["published_per_metric"][metric]["kind"],
                        "r0257_verdict": (
                            "PASS"
                            if payload["published_per_metric"][metric]["passes"]
                            else "FAIL"
                        ),
                        "r0257_margin": payload["published_per_metric"][metric][
                            "margin"
                        ],
                        "one_sided_criterion": new["per_metric"][metric]["criterion"],
                        "one_sided_kind": new["per_metric"][metric]["kind"],
                        "one_sided_verdict": (
                            "PASS" if new["per_metric"][metric]["passes"] else "FAIL"
                        ),
                        "one_sided_margin": new["per_metric"][metric]["margin"],
                        "descriptive_conformity": new["per_metric"][metric].get(
                            "descriptive_conformity"
                        ),
                    }
                    for metric in GATED_METRICS
                },
                "descriptive_only": new["descriptive_only"],
            }
        gate.finish("R0260 re-adjudication stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    cuda_mappings = _cuda_mappings()

    observations_identical = all(
        row["per_metric"][metric]["observation_is_bitwise_identical"]
        for row in comparison.values()
        for metric in GATED_METRICS
    )
    execution_checks = {
        "the_registration_was_sealed_before_this_node_started": bool(
            ordering["registration_precedes_adjudication"]
        ),
        "the_registration_node_bound_no_rung_artifact": bool(
            ordering["the_registration_node_bound_no_rung_artifact"]
        ),
        "every_observation_is_bitwise_r0257s": bool(observations_identical),
        "three_maps_were_re_adjudicated": len(comparison) == len(rung_cell_ids()),
        "every_r0257_verdict_was_fail_as_published": all(
            row["verdict_as_published_by_r0257"] == R0257_PUBLISHED_VERDICT
            for row in comparison.values()
        ),
        "the_shipped_judge_is_inert_to_the_conformity_diagnostics": bool(
            inertness["the_shipped_judge_is_inert_to_the_diagnostics"]
        ),
        "the_inertness_mutation_is_potent": bool(inertness["the_mutation_is_potent"]),
        "no_cuda_library_is_mapped_into_this_process": not cuda_mappings,
    }
    if not all(execution_checks.values()):
        raise Round0260NodeError(
            f"R0260 re-adjudication checks failed: {execution_checks}"
        )

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body: dict[str, Any] = {
        "round_id": ROUND_ID,
        "release_sha": str(active["manifest"]["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": None,
        "schema": READJUDICATION_SCHEMA,
        "capability": READJUDICATION_CAPABILITY,
        "capabilities": [READJUDICATION_CAPABILITY],
        "node": node_id,
        "node_started_utc": started_utc,
        "abort_flag_precondition": abort_flag,
        "owner_ruling": OWNER_RULING,
        "owner_ruling_date": OWNER_RULING_DATE,
        "what_this_is": (
            "a dated re-adjudication of R0257's three 6250k rung maps under the "
            "criterion registered by this round's first node. R0257's published "
            "FAILs stand: they are true statements about the criterion in force "
            "when they were made. Nothing in R0257 is rewritten."
        ),
        "r0257_result_is_not_rewritten": True,
        "registration_ordering": ordering,
        "registration_consumed": {
            "capability": registration["capability"],
            "schema": registration["schema"],
            "sha256": registration_signature["sha256"],
            "registered_criteria": registration["registered_criteria"],
            "panel_false_alarm_rate": registration["panel_false_alarm_rate"],
        },
        "re_adjudication": comparison,
        "verdict_summary": {
            "maps_judged": judged["maps_judged"],
            "maps_passing": judged["maps_passing"],
            "maps_failing": judged["maps_failing"],
            "verdicts_changed": sorted(
                cell_id for cell_id, row in comparison.items() if row["verdict_changed"]
            ),
        },
        "verdicts": judged["verdicts"],
        "conformity_diagnostics_never_gate": inertness,
        "descriptive_conformity_diagnostics": registration[
            "descriptive_conformity_diagnostics"
        ],
        "conformity_marking": CONFORMITY_MARKING,
        "ffr_scale_dependence_warning": (
            "review-0257-01 sections D.2 and D.3: ffr is the least scale-invariant "
            "metric on this panel (k_hit fixed at 10 against a k_frac of 0.001*N; "
            "it moves 4.3-4.9 family MAD_n on a change of scoring N alone) and "
            "carries the largest family deviation of any metric here, +10.2 to "
            "+11.3 MAD_n against purity's +6.1 to +6.7. Its PASS is not quality "
            "evidence and is not offered as any."
        ),
        "what_this_does_not_establish": [
            "that the three maps are good maps: the one-sided criterion asks only "
            "that they not be MORE lossy than the 2M family's lower tolerance "
            "bound, and nothing here connects a purity ratio to a downstream use",
            "anything at 12.5M, 25M, 50M or 100M: three cells at one rung",
            "that ffr's PASS is quality evidence, for the reason recorded above",
        ],
        "gate_status": "no-gate-registered-here-the-one-sided-criterion-was-consumed-unchanged",
        "gate_registered": False,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": bool(cuda_mappings),
        "cuda_libraries_mapped_into_this_process": cuda_mappings,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "sources": {
            "r0257_rung_panel": expected_input_signature(
                _bound_path(job, "r0257_panel", label="R0257 panel")
            ),
            "r0257_rung_verdict": expected_input_signature(
                _bound_path(job, "r0257_verdict", label="R0257 verdict")
            ),
            "r0260_registration": registration_signature,
        },
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    }
    _seal(output, READJUDICATION_ARTIFACT, body)
    print(json.dumps({
        "capability": READJUDICATION_CAPABILITY,
        "maps_passing": judged["maps_passing"],
        "maps_failing": judged["maps_failing"],
        "verdicts_changed": sorted(
            cell_id for cell_id, row in comparison.items() if row["verdict_changed"]
        ),
        "registration_precedes_adjudication": ordering[
            "registration_precedes_adjudication"
        ],
        "widest_gap_s": gaps.get("widest_gap_s"),
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0260 round0260_nodes.run_job")
    action = str(job.get("action") or "")
    if action == REGISTER_ACTION:
        run_register(active, job)
        return
    if action == READJUDICATE_ACTION:
        run_readjudicate(active, job)
        return
    raise Round0260NodeError(f"R0260 unknown action {action!r}")


__all__ = [
    "READJUDICATE_ACTION",
    "READJUDICATION_ARTIFACT",
    "READJUDICATION_CAPABILITY",
    "READJUDICATION_SCHEMA",
    "REGISTER_ACTION",
    "REGISTRATION_ARTIFACT",
    "Round0260NodeError",
    "run_job",
    "run_readjudicate",
    "run_register",
]
