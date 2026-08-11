"""R0257 — apply the registered `n = 29` `MAD_n` criteria to a map from outside.

This module **judges**. It cannot fit, refit, adjust, or register a floor: it has no
code path that writes one, and every number it compares against is read out of the
sealed `minilm-mixed-2m-calibrated-madn-floors-n29-v2` artifact (R0256) and checked
against the values R0256 published, byte for byte, before any map is scored.

Three standing rules are enforced here rather than described:

* **`density_v2` is descriptive-only and may never fail a map.** The registry must
  carry it *only* under `descriptive_only_never_a_gate`; a `density_v2` entry in
  `registered_criteria` or `registered_floors` is refused. The metric is reported
  per map, and `contributes_to_verdict` is `False` on every one of those reports.
* **Power matched to sidedness.** `ffr` is a one-sided lower floor and takes the
  one-sided power; the two purity criteria are two-sided bands and take the
  two-sided power. `applicable_power` is READ from the artifact per criterion; this
  module never chooses which power to quote, and refuses an artifact whose
  `applicable_power` disagrees with the criterion's `kind`.
* **A verdict without its false-alarm rate is not interpretable.** Every map-level
  verdict carries the panel-compounded rate and every metric-level verdict carries
  its own.

Every guard here ships a positive control that plants the defect **into the
function under test**, not into the report -- review-0255's finding about
`independence_control` is the reason that distinction is spelled out.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

ROUND_ID = "0257"

GATE_CAPABILITY = "minilm-mixed-2m-calibrated-madn-floors-n29-v2"
GATE_SCHEMA = "round0256-minilm-mixed-2m-calibrated-madn-floors-n29-v2"
GATE_SOURCE_ROUNDS = ("0255", "0256")

GATED_METRICS: tuple[str, ...] = ("ffr", "purity_fidelity_k256", "purity_fidelity_k1024")
DESCRIPTIVE_METRICS: tuple[str, ...] = ("density_v2",)

#: What R0255 registered and R0256 republished bitwise unchanged. The artifact is
#: required to reproduce every one of these; a floor that has moved is a refusal,
#: not a new floor.
REGISTERED_N = 29
REGISTERED_FINGERPRINT = (
    "2f61d1ed00996b5e6b20a5712b0b0c0903eb9e4a6e9a896b2235faf635ffe020"
)
REGISTERED_FFR_FLOOR = 0.3125328635126472
REGISTERED_K256_BAND = (0.9697263687993266, 1.0550731452902518)
REGISTERED_K1024_BAND = (0.6616903134148893, 0.7719564268121961)
REGISTERED_IDENTITY_BOUND = 5.199469468957452

ONE_SIDED_POWER = {
    "minus_1_sigma": 0.08340200118107971,
    "minus_2_sigma": 0.30035090040737566,
    "minus_3_sigma": 0.6281331322475918,
}
TWO_SIDED_POWER = {
    "minus_1_sigma": 0.04529311725849311,
    "minus_2_sigma": 0.19381747812893288,
    "minus_3_sigma": 0.48275342713415287,
}
ONE_SIDED_FALSE_FAIL = 0.01222093973890843
TWO_SIDED_FALSE_FAIL = 0.01112250056202322
PANEL_FALSE_ALARM_RATE = 0.034071887878658114

POWER_BY_KIND = {
    "one_sided_lower_floor": "one_sided",
    "two_sided_ratio_band": "two_sided",
}
POWER_TABLE = {"one_sided": ONE_SIDED_POWER, "two_sided": TWO_SIDED_POWER}
FALSE_FAIL_TABLE = {
    "one_sided": ONE_SIDED_FALSE_FAIL,
    "two_sided": TWO_SIDED_FALSE_FAIL,
}

POWER_MATERIALITY = (
    "at -2 sigma the one-sided ffr floor detects a genuine regression "
    f"{ONE_SIDED_POWER['minus_2_sigma']} of the time and the two-sided purity "
    f"bands {TWO_SIDED_POWER['minus_2_sigma']} of the time. A PASS is therefore "
    "weak evidence of conformity and a FAIL is strong evidence of "
    "non-conformity: the panel-compounded false-alarm rate is "
    f"{PANEL_FALSE_ALARM_RATE}. Read every verdict in this round with both "
    "numbers beside it."
)

INDEPENDENCE_LIMITATION = (
    "KNOWN LIMITATION, carried not re-litigated (review-0256-01 §C). The shipped "
    "independence control has a blind spot: a builder that reads a held-out cell "
    "into the fit THROUGH A CLAMP into the family's observed range moves the ffr "
    "floor (0.3125328635126472 -> 0.3097872229582025) while the control still "
    "reports independent. The forward direction holds -- a cell that reaches the "
    "fit moves the floor -- but the converse does not. The floors themselves were "
    "NOT moved, and review-0256 could not move one by any other route. This round "
    "relies on the gate's independence and states the limitation at the point of "
    "reliance; it does not attempt to close it."
)

BELOW_MEDIAN_CORRECTION = (
    "Also carried: review-0256-01 refuted result-0256 §D3's below-median "
    "explanation. 4 of 14 below-median family cells do move the ffr floor, "
    "through the MAD_n scale rather than the median. The correct rule is: driving "
    "a cell moves the MAD_n floor iff the cell is above the median OR its "
    "absolute deviation is at most MAD_n."
)

FAMILY_RULE_RESTATED = (
    "The cells judged here are NOT in the family any floor was fitted on, and no "
    "code path in this module adds one. The family is the twenty-nine 2M "
    "exact-graph cells; the shipped round0255_treatment.assert_family_is_2m_only "
    "is called on it, and round0257_rung_contract."
    "assert_no_rung_map_in_the_gate_family additionally asserts disjointness from "
    "the judged set."
)


class Round0257JudgementError(RuntimeError):
    """The registered gate artifact or the judgement contract changed."""


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise Round0257JudgementError(f"R0257 {label} is not finite: {value!r}")
    return number


# --------------------------------------------------------------------------- #
# reading the sealed gate
# --------------------------------------------------------------------------- #


def validate_gate_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse anything that is not R0256's registered n = 29 gate, unmoved.

    This is the function the positive controls attack. Every check below is a
    refusal path, and each one has a plant in `judgement_controls`.
    """
    if str(artifact.get("schema")) != GATE_SCHEMA:
        raise Round0257JudgementError(
            f"R0257 gate artifact schema {artifact.get('schema')!r} is not "
            f"{GATE_SCHEMA!r}"
        )
    if str(artifact.get("capability")) != GATE_CAPABILITY:
        raise Round0257JudgementError(
            f"R0257 gate artifact capability {artifact.get('capability')!r} is not "
            f"{GATE_CAPABILITY!r}"
        )
    if int(artifact.get("n", -1)) != REGISTERED_N:
        raise Round0257JudgementError(
            f"R0257 gate artifact is at n = {artifact.get('n')!r}, registered "
            f"{REGISTERED_N}"
        )
    if str(artifact.get("registry_fingerprint")) != REGISTERED_FINGERPRINT:
        raise Round0257JudgementError(
            "R0257 gate registry fingerprint moved: "
            f"{artifact.get('registry_fingerprint')!r} != {REGISTERED_FINGERPRINT!r}"
        )

    criteria = artifact.get("registered_criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(GATED_METRICS):
        raise Round0257JudgementError(
            f"R0257 registered_criteria must be exactly {sorted(GATED_METRICS)}, "
            f"read {sorted(criteria) if isinstance(criteria, Mapping) else criteria!r}"
        )
    for metric in DESCRIPTIVE_METRICS:
        if metric in criteria:
            raise Round0257JudgementError(
                f"R0257 refuses: {metric} is descriptive-only and may not appear "
                "in registered_criteria"
            )
        floors = artifact.get("registered_floors")
        if isinstance(floors, Mapping) and metric in floors:
            raise Round0257JudgementError(
                f"R0257 refuses: {metric} is descriptive-only and may not carry a "
                "readable floor in registered_floors"
            )
        descriptive = artifact.get("descriptive_only_never_a_gate")
        if not isinstance(descriptive, Mapping) or metric not in descriptive:
            raise Round0257JudgementError(
                f"R0257 refuses: {metric} must be published under "
                "descriptive_only_never_a_gate"
            )

    expected_values: dict[str, Any] = {
        "ffr": ("floor", REGISTERED_FFR_FLOOR),
        "purity_fidelity_k256": ("band", REGISTERED_K256_BAND),
        "purity_fidelity_k1024": ("band", REGISTERED_K1024_BAND),
    }
    for metric, entry in criteria.items():
        if not isinstance(entry, Mapping) or entry.get("gates_a_map") is not True:
            raise Round0257JudgementError(
                f"R0257 criterion {metric} does not declare gates_a_map: true"
            )
        kind = str(entry.get("kind"))
        if kind not in POWER_BY_KIND:
            raise Round0257JudgementError(
                f"R0257 criterion {metric} has unknown kind {kind!r}"
            )
        shape, expected = expected_values[metric]
        if shape == "floor":
            observed = _finite(entry.get("floor"), f"{metric} floor")
            if observed != expected:
                raise Round0257JudgementError(
                    f"R0257 {metric} floor moved: {observed!r} != {expected!r}. A "
                    "floor is registered and sealed; this round does not move one."
                )
        else:
            lower = _finite(entry.get("ratio_lower"), f"{metric} lower")
            upper = _finite(entry.get("ratio_upper"), f"{metric} upper")
            if (lower, upper) != tuple(expected):
                raise Round0257JudgementError(
                    f"R0257 {metric} band moved: {(lower, upper)!r} != {expected!r}"
                )

    power = artifact.get("detection_power_by_criterion")
    if not isinstance(power, Mapping) or set(power) != set(GATED_METRICS):
        raise Round0257JudgementError(
            "R0257 detection_power_by_criterion must cover exactly the gated metrics"
        )
    for metric, entry in power.items():
        applicable = str(entry.get("applicable_power"))
        required = POWER_BY_KIND[str(criteria[metric]["kind"])]
        if applicable != required:
            raise Round0257JudgementError(
                f"R0257 refuses: {metric} is a {criteria[metric]['kind']} and takes "
                f"{required} power, but the artifact declares {applicable!r}. "
                "Quoting one-sided power beside a two-sided band overstates its "
                "sensitivity (review-0255-01)."
            )
        table = POWER_TABLE[applicable]
        observed = entry.get("detection_power")
        if not isinstance(observed, Mapping) or {
            key: _finite(value, f"{metric} power {key}")
            for key, value in observed.items()
        } != table:
            raise Round0257JudgementError(
                f"R0257 {metric} detection power moved: {observed!r} != {table!r}"
            )
        if _finite(
            entry.get("new_cell_false_fail_rate"), f"{metric} false-fail"
        ) != FALSE_FAIL_TABLE[applicable]:
            raise Round0257JudgementError(
                f"R0257 {metric} new-cell false-fail rate moved"
            )

    far = artifact.get("panel_false_alarm_rate")
    if not isinstance(far, Mapping):
        raise Round0257JudgementError(
            "R0257 refuses a gate artifact with no panel-compounded false-alarm "
            "rate: a verdict without its false-alarm rate is not interpretable"
        )
    if _finite(
        far.get("panel_false_alarm_rate_under_independence"), "panel false-alarm rate"
    ) != PANEL_FALSE_ALARM_RATE:
        raise Round0257JudgementError("R0257 panel false-alarm rate moved")
    if set(far.get("gated_criteria") or ()) != set(GATED_METRICS):
        raise Round0257JudgementError(
            "R0257 panel false-alarm rate does not compound exactly the gated set"
        )

    return {
        "capability": GATE_CAPABILITY,
        "schema": GATE_SCHEMA,
        "source_rounds": list(GATE_SOURCE_ROUNDS),
        "n": REGISTERED_N,
        "registry_fingerprint": REGISTERED_FINGERPRINT,
        "identity_bound_at_n": REGISTERED_IDENTITY_BOUND,
        "registered_criteria": {key: dict(value) for key, value in criteria.items()},
        "detection_power_by_criterion": {
            key: dict(value) for key, value in power.items()
        },
        "panel_false_alarm_rate": dict(far),
        "descriptive_only_never_a_gate": dict(
            artifact.get("descriptive_only_never_a_gate") or {}
        ),
        "every_floor_and_band_matches_what_r0255_registered": True,
        "no_descriptive_metric_is_readable_as_a_floor": True,
        "power_is_matched_to_sidedness": True,
    }


# --------------------------------------------------------------------------- #
# judging one map
# --------------------------------------------------------------------------- #


def judge_map(
    *,
    cell_id: str,
    metrics: Mapping[str, float],
    raw_ratios: Mapping[str, float],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one rung map against the registered criteria. Never fits anything."""
    criteria = dict(gate["registered_criteria"])
    power = dict(gate["detection_power_by_criterion"])
    per_metric: dict[str, Any] = {}
    for metric in GATED_METRICS:
        entry = dict(criteria[metric])
        kind = str(entry["kind"])
        applicable = POWER_BY_KIND[kind]
        if kind == "one_sided_lower_floor":
            observed = _finite(metrics.get(metric), f"{cell_id} {metric}")
            floor = float(entry["floor"])
            passes = observed >= floor
            margin = observed - floor
            criterion_text = f">= {floor}"
            comparand = "the panel metric"
        else:
            key = metric.replace("purity_fidelity_", "")
            observed = _finite(raw_ratios.get(key), f"{cell_id} raw ratio {key}")
            lower = float(entry["ratio_lower"])
            upper = float(entry["ratio_upper"])
            passes = lower <= observed <= upper
            margin = min(observed - lower, upper - observed)
            criterion_text = f"unfolded ratio in [{lower}, {upper}]"
            comparand = "the UNFOLDED purity ratio (never the folded fidelity)"
        per_metric[metric] = {
            "metric": metric,
            "kind": kind,
            "criterion": criterion_text,
            "compared_quantity": comparand,
            "observed": observed,
            "passes": bool(passes),
            "margin": margin,
            "margin_is": (
                "observed - floor" if kind == "one_sided_lower_floor"
                else "distance to the nearer band edge; negative means outside"
            ),
            "applicable_power": applicable,
            "detection_power": dict(power[metric]["detection_power"]),
            "new_cell_false_fail_rate": float(
                power[metric]["new_cell_false_fail_rate"]
            ),
        }
    descriptive: dict[str, Any] = {}
    for metric in DESCRIPTIVE_METRICS:
        if metric in metrics:
            descriptive[metric] = {
                "observed": _finite(metrics[metric], f"{cell_id} {metric}"),
                "contributes_to_verdict": False,
                "why": (
                    "descriptive-only by standing rule; one anchor of 4,000 (row "
                    "1449227, r_hd == 0) supplies about two-thirds of its value"
                ),
            }
    verdict = "PASS" if all(item["passes"] for item in per_metric.values()) else "FAIL"
    failing = sorted(
        name for name, item in per_metric.items() if not item["passes"]
    )
    return {
        "cell_id": cell_id,
        "verdict": verdict,
        "failing_criteria": failing,
        "criteria_cleared": sum(1 for item in per_metric.values() if item["passes"]),
        "criteria_total": len(per_metric),
        "per_metric": per_metric,
        "descriptive_only": descriptive,
        "panel_false_alarm_rate": PANEL_FALSE_ALARM_RATE,
        "panel_false_alarm_rate_is_an_upper_bound": True,
        "power_materiality": POWER_MATERIALITY,
        "this_cell_is_in_no_fitting_family": True,
    }


def judge_population(
    *,
    cells: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Judge every rung map and summarise, without pooling them into anything."""
    verdicts = {
        cell_id: judge_map(
            cell_id=cell_id,
            metrics=dict(payload["panel_metrics"]),
            raw_ratios=dict(payload["raw_purity_ratios"]),
            gate=gate,
        )
        for cell_id, payload in cells.items()
    }
    passing = sorted(k for k, v in verdicts.items() if v["verdict"] == "PASS")
    failing = sorted(k for k, v in verdicts.items() if v["verdict"] == "FAIL")
    by_metric: dict[str, dict[str, Any]] = {}
    for metric in GATED_METRICS:
        cleared = sorted(
            k for k, v in verdicts.items() if v["per_metric"][metric]["passes"]
        )
        by_metric[metric] = {
            "cleared": cleared,
            "failed": sorted(set(verdicts) - set(cleared)),
            "margins": {
                k: v["per_metric"][metric]["margin"] for k, v in verdicts.items()
            },
            "applicable_power": verdicts[next(iter(verdicts))]["per_metric"][metric][
                "applicable_power"
            ],
            "detection_power": verdicts[next(iter(verdicts))]["per_metric"][metric][
                "detection_power"
            ],
        }
    return {
        "verdicts": verdicts,
        "maps_judged": len(verdicts),
        "maps_passing": passing,
        "maps_failing": failing,
        "unanimous": len(passing) == 0 or len(failing) == 0,
        "by_metric": by_metric,
        "panel_false_alarm_rate": PANEL_FALSE_ALARM_RATE,
        "power_materiality": POWER_MATERIALITY,
        "family_rule": FAMILY_RULE_RESTATED,
        "independence_limitation": INDEPENDENCE_LIMITATION,
        "below_median_correction": BELOW_MEDIAN_CORRECTION,
        "a_failing_map_is_a_finding": (
            "Registered before the run: a rung map that fails is published failing, "
            "with its margin. No retrain, no other seed, no adjusted floor."
        ),
    }


# --------------------------------------------------------------------------- #
# positive controls -- every plant reaches the function under test
# --------------------------------------------------------------------------- #


def _plant_density_v2_as_a_criterion(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["registered_criteria"]["density_v2"] = {
        "floor": 0.41643957035196294,
        "gates_a_map": True,
        "kind": "one_sided_lower_floor",
        "read_as": "the metric must be at or above `floor`",
    }
    return artifact


def _plant_density_v2_floor(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact.setdefault("registered_floors", {})["density_v2"] = 0.41643957035196294
    return artifact


def _plant_one_sided_power_beside_a_band(artifact: dict[str, Any]) -> dict[str, Any]:
    entry = artifact["detection_power_by_criterion"]["purity_fidelity_k256"]
    entry["applicable_power"] = "one_sided"
    entry["detection_power"] = dict(ONE_SIDED_POWER)
    entry["new_cell_false_fail_rate"] = ONE_SIDED_FALSE_FAIL
    return artifact


def _plant_a_loosened_floor(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["registered_criteria"]["ffr"]["floor"] = 0.30
    return artifact


def _plant_a_missing_false_alarm_rate(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact.pop("panel_false_alarm_rate", None)
    return artifact


def _plant_a_widened_band(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["registered_criteria"]["purity_fidelity_k1024"]["ratio_lower"] = 0.30
    return artifact


JUDGEMENT_PLANTS = (
    (
        "density_v2_registered_as_a_criterion",
        "the descriptive metric promoted into the gated set, so it could fail a map",
        _plant_density_v2_as_a_criterion,
    ),
    (
        "density_v2_readable_as_a_floor",
        "the descriptive metric left readable in registered_floors -- R0255's own "
        "defect, which the R0161/R0193 audit found five downstream consumers of",
        _plant_density_v2_floor,
    ),
    (
        "one_sided_power_beside_a_two_sided_band",
        "the k256 band published with the one-sided power 0.30035090040737566, "
        "overstating its sensitivity 1.5496x -- review-0255-01's finding",
        _plant_one_sided_power_beside_a_band,
    ),
    (
        "the_ffr_floor_loosened",
        "the registered ffr floor lowered to 0.30, which would flip a failing map "
        "to passing -- the tuning the standing rule forbids",
        _plant_a_loosened_floor,
    ),
    (
        "the_k1024_band_widened",
        "the k1024 band's lower edge dropped to 0.30",
        _plant_a_widened_band,
    ),
    (
        "no_panel_false_alarm_rate",
        "the compounded false-alarm rate removed, leaving verdicts uninterpretable",
        _plant_a_missing_false_alarm_rate,
    ),
)


def old_gate_predicate(_artifact: Mapping[str, Any]) -> bool:
    """The construction this replaces: whatever the artifact says is the gate."""
    return True


def judgement_controls(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Plant each defect INTO `validate_gate_artifact` and require a refusal.

    The plant mutates the artifact that the function under test reads, so a
    defect that only reached a report would not register here.
    """
    controls: list[dict[str, Any]] = []
    for name, description, plant in JUDGEMENT_PLANTS:
        planted = plant(copy.deepcopy(dict(artifact)))
        refused = False
        error = None
        try:
            validate_gate_artifact(planted)
        except Round0257JudgementError as raised:
            refused = True
            error = f"{type(raised).__name__}: {raised}"
        controls.append({
            "control": name,
            "plants": description,
            "reaches_the_function_under_test": "validate_gate_artifact",
            "shipped_guard_refused": refused,
            "shipped_guard_error": error,
            "old_predicate_accepted": bool(old_gate_predicate(planted)),
        })

    # A seventh plant that attacks `judge_map` rather than the validator: a map
    # whose density_v2 is catastrophic but whose gated metrics all clear must
    # still PASS, because density_v2 cannot fail a map.
    gate = validate_gate_artifact(dict(artifact))
    conforming = {
        "ffr": REGISTERED_FFR_FLOOR + 0.01,
        "purity_fidelity_k256": 1.0,
        "purity_fidelity_k1024": 1.0,
        "density_v2": 0.0,
    }
    ratios = {"k256": 1.0, "k1024": 0.71}
    outcome = judge_map(
        cell_id="control-density-v2-cannot-fail-a-map",
        metrics=conforming,
        raw_ratios=ratios,
        gate=gate,
    )
    density_control = {
        "control": "catastrophic_density_v2_cannot_fail_a_conforming_map",
        "plants": "density_v2 = 0.0 on a map that clears all three gated criteria",
        "reaches_the_function_under_test": "judge_map",
        "verdict": outcome["verdict"],
        "held": outcome["verdict"] == "PASS",
        "density_v2_contributes_to_verdict": bool(
            outcome["descriptive_only"]["density_v2"]["contributes_to_verdict"]
        ),
    }

    # An eighth: a map placed one quantum below the ffr floor must FAIL, so the
    # judge is not vacuously passing everything.
    below = judge_map(
        cell_id="control-a-map-below-the-floor-must-fail",
        metrics={
            "ffr": REGISTERED_FFR_FLOOR - 1e-6,
            "purity_fidelity_k256": 1.0,
            "purity_fidelity_k1024": 1.0,
            "density_v2": 0.44,
        },
        raw_ratios=ratios,
        gate=gate,
    )
    falsifiability_control = {
        "control": "a_map_below_the_ffr_floor_fails",
        "plants": "ffr one part in a million below the registered floor",
        "reaches_the_function_under_test": "judge_map",
        "verdict": below["verdict"],
        "failing_criteria": below["failing_criteria"],
        "held": below["verdict"] == "FAIL" and below["failing_criteria"] == ["ffr"],
    }

    # A ninth: a purity ratio outside the band on the HIGH side must FAIL, so the
    # band is genuinely two-sided and over-separation is not silently admitted.
    over = judge_map(
        cell_id="control-over-separation-fails-the-band",
        metrics={
            "ffr": REGISTERED_FFR_FLOOR + 0.01,
            "purity_fidelity_k256": 1.0,
            "purity_fidelity_k1024": 1.0,
            "density_v2": 0.44,
        },
        raw_ratios={"k256": REGISTERED_K256_BAND[1] + 1e-6, "k1024": 0.71},
        gate=gate,
    )
    two_sided_control = {
        "control": "over_separation_fails_the_two_sided_band",
        "plants": "k256 unfolded ratio one part in a million above the band's upper "
        "edge -- the failure a folded floor cannot see",
        "reaches_the_function_under_test": "judge_map",
        "verdict": over["verdict"],
        "failing_criteria": over["failing_criteria"],
        "held": over["verdict"] == "FAIL"
        and over["failing_criteria"] == ["purity_fidelity_k256"],
    }

    honest_refused = False
    honest_error = None
    try:
        validate_gate_artifact(dict(artifact))
    except Round0257JudgementError as raised:
        honest_refused = True
        honest_error = f"{type(raised).__name__}: {raised}"

    behavioural = [density_control, falsifiability_control, two_sided_control]
    return {
        "controls": controls,
        "behavioural_controls": behavioural,
        "planted": len(controls) + len(behavioural),
        "every_planted_defect_was_refused": all(
            item["shipped_guard_refused"] for item in controls
        ),
        "every_behavioural_control_held": all(item["held"] for item in behavioural),
        "the_old_predicate_accepted_every_one": all(
            item["old_predicate_accepted"] for item in controls
        ),
        "the_honest_artifact_still_passes": not honest_refused,
        "honest_error": honest_error,
        "note": (
            "each of the six validator plants mutates the artifact that "
            "validate_gate_artifact reads, and each of the three behavioural plants "
            "changes the input judge_map scores. No control re-implements the "
            "function it tests (review-0253) and none is a dead perturbation "
            "counted but never passed (review-0255)."
        ),
    }


__all__ = [
    "BELOW_MEDIAN_CORRECTION",
    "DESCRIPTIVE_METRICS",
    "FAMILY_RULE_RESTATED",
    "GATED_METRICS",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "INDEPENDENCE_LIMITATION",
    "JUDGEMENT_PLANTS",
    "ONE_SIDED_FALSE_FAIL",
    "ONE_SIDED_POWER",
    "PANEL_FALSE_ALARM_RATE",
    "POWER_BY_KIND",
    "POWER_MATERIALITY",
    "REGISTERED_FFR_FLOOR",
    "REGISTERED_FINGERPRINT",
    "REGISTERED_IDENTITY_BOUND",
    "REGISTERED_K1024_BAND",
    "REGISTERED_K256_BAND",
    "REGISTERED_N",
    "ROUND_ID",
    "Round0257JudgementError",
    "TWO_SIDED_FALSE_FAIL",
    "TWO_SIDED_POWER",
    "judge_map",
    "judge_population",
    "judgement_controls",
    "old_gate_predicate",
    "validate_gate_artifact",
]
