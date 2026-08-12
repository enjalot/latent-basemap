"""R0260 — re-register the two purity criteria as ONE-SIDED floors.

The owner decided this on 2026-08-12 (`OWNER-DECISIONS-PENDING.md` section 5,
after review-0257-01):

> Adopt option 2 -- re-register both purity criteria as ONE-SIDED floors that
> fail under-separation only, recalibrated from the SEALED n = 29 family and
> registered BEFORE any further rung map is judged. Keep the full two-sided band
> values published beside every verdict as descriptive conformity diagnostics
> (including the k256 above-reference drift), so the discarded side is reported,
> just not gated. R0257's three FAILs stand as published; re-adjudicate them
> under the re-registered criterion in a dated addendum, never a rewrite. In
> parallel, open registered design work toward option 3.

Four things are structural in this module.

**The criterion has no free parameter.** The direction comes from the ruling; the
estimator (`median - k*MAD_n`) from the owner's section-3 ruling; the multiplier
from R0234's Gaussian-null calibration at `n = 29`, which reads no cell of this
program and has been sealed since R0255 (`2.6934393367626024`); the family is the
same twenty-nine sealed 2M cells. Nothing here was chosen after seeing a map, and
nothing here *could* have been: fixing the four inputs fixes the two numbers.

**It gates the UNFOLDED ratio, not the folded fidelity.** R0234 published a folded
purity floor that was read as a gate; folding maps `1.10` and `0.91` to the same
place and is exactly blind to the direction this ruling is about. So the purity
criteria carry their own `kind` -- `one_sided_lower_ratio_floor` -- rather than
reusing the scalar `one_sided_lower_floor` that would silently read the folded
panel value.

**Over-separation is not a defect and must not gate.** Review-0257 showed the
retired band's `k1024` upper edge `0.7719564268121961` fails a perfectly faithful
map (`ratio = 1.0`) by `+0.228`. A criterion that fails a map for being closer to
the high-D reference is measuring conformity to the 2M family, not quality.

**The discarded side is still measured.** Dropping a gate must not drop a
measurement, so the full two-sided bands, each map's position inside or outside
them, and the `k256` above-reference drift are published under
`descriptive_only_never_a_gate` -- the same marking `density_v2` carries -- and a
guard plus a planted consumer prove no shipped verdict reads them.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .round0234_calibrated_floors import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    centre_and_scale,
    identity_bound,
)


ROUND_ID = "0260"

GATE_CAPABILITY = "minilm-mixed-2m-one-sided-purity-floors-n29-v1"
GATE_SCHEMA = "round0260-minilm-mixed-2m-one-sided-purity-floors-n29-v1"

#: The artifact whose two purity CRITERIA this one replaces. Its `ffr` floor is
#: carried through unchanged, and every floor value in it stays true as published.
SUPERSEDED_PURITY_CRITERIA_IN = "minilm-mixed-2m-calibrated-madn-floors-n29-v2"

OWNER_RULING_DATE = "2026-08-12"
OWNER_RULING_SECTION = "OWNER-DECISIONS-PENDING.md section 5"
OWNER_RULING = (
    "Adopt option 2 -- re-register both purity criteria as one-sided floors that "
    "fail under-separation only, recalibrated from the sealed n = 29 family and "
    "registered before any further rung map is judged. Keep the full two-sided "
    "band values published beside every verdict as descriptive conformity "
    "diagnostics (including the k256 above-reference drift), so the discarded "
    "side is reported, just not gated. R0257's three FAILs stand as published; "
    "re-adjudicate them under the re-registered criterion in a dated addendum, "
    "never a rewrite. In parallel, open registered design work toward option 3: "
    "a reference-anchored quality criterion scoring deviation from ratio 1.0, "
    "with its acceptable-deviation calibration defined on same-universe evidence "
    "at more than one N, in place before the Phase 3 mid-scale gate family; "
    "include ffr's scale-dependence in that work -- its PASS here is not quality "
    "evidence."
)

KIND_SCALAR_FLOOR = "one_sided_lower_floor"
KIND_RATIO_FLOOR = "one_sided_lower_ratio_floor"
ONE_SIDED_KINDS: tuple[str, ...] = (KIND_SCALAR_FLOOR, KIND_RATIO_FLOOR)

#: Every gated criterion is one-sided now, so every one takes the one-sided power.
POWER_BY_KIND: dict[str, str] = {kind: "one_sided" for kind in ONE_SIDED_KINDS}

#: Keys that would re-introduce the discarded side into a gating criterion.
FORBIDDEN_CRITERION_KEYS: tuple[str, ...] = (
    "ratio_upper", "upper", "band", "ratio_band", "two_sided",
)

NO_TUNING_STATEMENT = (
    "No value in this registration was chosen after seeing a map. The direction "
    "is the owner's ruling; the estimator is the owner's section-3 ruling; the "
    "multiplier is R0234's Gaussian-null calibration at n = 29, which reads no "
    "cell of this program and has been sealed since R0255; the family is the same "
    "twenty-nine sealed 2M cells. Given those four inputs the two floors are "
    "arithmetic, not a choice. If a re-adjudicated map still fails, it is "
    "published failing."
)

WHY_ONE_SIDED = (
    "review-0257-01 section G: purity = map_agreement / hi_D_agreement, so "
    "ratio = 1.0 is a map whose neighbourhoods agree with the high-D reference "
    "exactly. The retired k1024 band's upper edge is 0.7719564268121961, so a "
    "PERFECTLY FAITHFUL map failed that criterion by +0.228. The band asked a "
    "rung map to be as lossy as a 2M map. Under-separation -- a ratio below the "
    "family's lower tolerance bound -- is the direction that can indicate a real "
    "defect, and it is the only direction gated here."
)

CONFORMITY_MARKING = "descriptive_only_never_a_gate"


class Round0260RegistrationError(RuntimeError):
    """The R0260 one-sided purity registration contract changed."""


# --------------------------------------------------------------------------- #
# A. the floors
# --------------------------------------------------------------------------- #


def one_sided_ratio_floor(
    estimator: str, logs: Sequence[float], multiplier: float
) -> dict[str, float]:
    """`exp(centre - k*scale)` on the family's unfolded purity LOG-ratios.

    The two-sided band is `exp(centre -/+ k2*scale)` on the same series, so the
    only differences between the retired criterion and this one are the
    multiplier (one-sided `k1` rather than two-sided `k2`) and the absence of an
    upper edge. The centre and the scale are the same fit on the same cells.
    """
    centre, scale = centre_and_scale(estimator, logs)
    log_floor = centre - float(multiplier) * scale
    return {
        "log_centre": centre,
        "log_scale": scale,
        "log_floor": log_floor,
        "ratio_floor": math.exp(log_floor),
        "multiplier": float(multiplier),
    }


def one_sided_purity_floors(
    estimator: str,
    log_series: Mapping[str, Sequence[float]],
    multiplier: float,
) -> dict[str, dict[str, float]]:
    return {
        metric: one_sided_ratio_floor(estimator, log_series[metric], multiplier)
        for metric in PURITY_METRICS
    }


def panel_false_alarm_rate_one_sided(
    *, one_sided_rate: float, gated_criteria: int = len(GATED_METRICS)
) -> dict[str, Any]:
    """Compound the per-metric rate across a panel that is now one-sided throughout.

    R0256 compounded `1 - (1-p1)(1-p2)^2` because two criteria were two-sided
    bands with their own, larger, false-fail rate. All three are one-sided lower
    floors here, so the panel rate is the one-sided rate compounded three times
    and is strictly smaller.
    """
    rate = float(one_sided_rate)
    count = int(gated_criteria)
    compounded = 1.0 - (1.0 - rate) ** count
    return {
        "gated_criteria": list(GATED_METRICS),
        "gated_criteria_count": count,
        "one_sided_per_metric_rate": rate,
        "panel_false_alarm_rate_under_independence": compounded,
        "expression": f"1 - (1 - {rate!r})**{count}",
        "ratio_to_the_per_metric_rate": compounded / rate if rate else math.inf,
        "this_is_an_upper_bound": True,
        "why_it_is_an_upper_bound": (
            "the three gated criteria are correlated on this family, so the true "
            "panel rate is at or below the independence product. It is still the "
            "number a rung map actually faces."
        ),
    }


def attainability_against_the_identity_bound(
    *, n: int, multiplier: float
) -> dict[str, Any]:
    """`k` beside `(n-1)/sqrt(n)`, the bound above which a floor cannot fail.

    R0219's "4/4 pass" and R0225's "0 failures at n = 8" were theorems about their
    `n`: `max|x_i - xbar|/s <= (n-1)/sqrt(n)`, so a multiplier at or above that
    bound makes every defining cell unfailable by construction. Reporting `k`
    against it is what stops a floor being published as evidence when it is an
    identity.
    """
    bound = identity_bound(int(n))
    k = float(multiplier)
    return {
        "n": int(n),
        "multiplier": k,
        "identity_bound": bound,
        "identity_bound_expression": f"(n-1)/sqrt(n) = {int(n) - 1}/sqrt({int(n)})",
        "multiplier_is_below_the_identity_bound": k < bound,
        "headroom": bound - k,
        "ratio_to_the_bound": k / bound,
        "why_this_matters": (
            "a floor whose multiplier reaches (n-1)/sqrt(n) cannot fail any cell "
            "of its own family, so '29/29 pass' would be an identity rather than "
            "a measurement. MAD_n's own attainability is stronger still and is "
            "reported separately: its scale is a middle order statistic, so "
            "sup max_i (centre - x_i)/scale is +inf and every defining cell can "
            "fail at ANY multiplier."
        ),
    }


# --------------------------------------------------------------------------- #
# B. the registry
# --------------------------------------------------------------------------- #


def registered_criteria_one_sided(
    *,
    ffr_floor: float,
    purity_floors: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Every gated criterion, one-sided, in the form it gates."""
    out: dict[str, Any] = {}
    for metric in GATED_METRICS:
        if metric in PURITY_METRICS:
            out[metric] = {
                "kind": KIND_RATIO_FLOOR,
                "ratio_floor": float(purity_floors[metric]["ratio_floor"]),
                "log_floor": float(purity_floors[metric]["log_floor"]),
                "gates_a_map": True,
                "fails_only": "under_separation",
                "applicable_power": "one_sided",
                "read_as": (
                    "the UNFOLDED purity ratio must be at or above `ratio_floor`. "
                    "There is no upper edge: a ratio above the floor passes at any "
                    "magnitude, because over-separation toward the high-D "
                    "reference is not a defect."
                ),
            }
            continue
        out[metric] = {
            "kind": KIND_SCALAR_FLOOR,
            "floor": float(ffr_floor),
            "gates_a_map": True,
            "fails_only": "under_separation",
            "applicable_power": "one_sided",
            "read_as": "the metric must be at or above `floor`",
        }
    return out


def conformity_diagnostics(
    *,
    two_sided_bands: Mapping[str, Mapping[str, float]],
    two_sided_multiplier: float,
    faithful_ratio: float = 1.0,
) -> dict[str, Any]:
    """The discarded side, kept as a measurement and marked so it cannot gate."""
    bands = {
        metric: {
            "ratio_lower": float(two_sided_bands[metric]["ratio_lower"]),
            "ratio_upper": float(two_sided_bands[metric]["ratio_upper"]),
        }
        for metric in PURITY_METRICS
    }
    faithful = {
        metric: {
            "a_perfectly_faithful_map_ratio": float(faithful_ratio),
            "inside_the_retired_band": (
                bands[metric]["ratio_lower"]
                <= float(faithful_ratio)
                <= bands[metric]["ratio_upper"]
            ),
            "distance_above_the_upper_edge": (
                float(faithful_ratio) - bands[metric]["ratio_upper"]
            ),
        }
        for metric in PURITY_METRICS
    }
    return {
        CONFORMITY_MARKING: True,
        "contributes_to_verdict": False,
        "what_these_are": (
            "the full two-sided conformity bands the owner ruling retired as "
            "gates. They remain published beside every verdict because dropping a "
            "gate must not drop the measurement. Nothing reads them."
        ),
        "two_sided_multiplier": float(two_sided_multiplier),
        "two_sided_bands": bands,
        "a_perfectly_faithful_map_against_the_retired_bands": faithful,
        "why_the_k256_drift_stays_visible": (
            "a k256 ratio above 1.0 is a map whose 2D neighbourhoods agree on the "
            "k = 256 cluster label MORE often than the high-D reference's do -- "
            "manufactured separation. It is not gated, and it is not nothing: it "
            "is the direction the owner ruling deliberately stopped failing, so "
            "it has to stay on the record to be tracked across rungs."
        ),
    }


def assert_registry_is_one_sided_only(
    criteria: Mapping[str, Any],
    scalar_floors: Mapping[str, Any],
    ratio_floors: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse a registry that gates the discarded side, or a descriptive metric."""
    problems: list[str] = []
    for metric in DESCRIPTIVE_METRICS:
        if metric in criteria:
            problems.append(f"{metric} is descriptive-only and may not be a criterion")
        if scalar_floors.get(metric) is not None:
            problems.append(f"{metric} is descriptive-only and may not carry a floor")
        if ratio_floors.get(metric) is not None:
            problems.append(
                f"{metric} is descriptive-only and may not carry a ratio floor"
            )
    for metric in GATED_METRICS:
        entry = criteria.get(metric)
        if entry is None:
            problems.append(f"{metric} gates a map and must be registered")
            continue
        kind = str(entry.get("kind"))
        if kind not in ONE_SIDED_KINDS:
            problems.append(
                f"{metric} is registered as {kind!r}; the owner ruling registers "
                "one-sided floors only"
            )
        if str(entry.get("applicable_power")) != "one_sided":
            problems.append(f"{metric} must carry one-sided power beside a one-sided floor")
        for key in FORBIDDEN_CRITERION_KEYS:
            if key in entry:
                problems.append(
                    f"{metric} carries {key!r}: the discarded side may be published "
                    "as a diagnostic, never inside a gating criterion"
                )
        if metric in PURITY_METRICS:
            if kind != KIND_RATIO_FLOOR:
                problems.append(f"{metric} must gate the UNFOLDED ratio, not a scalar")
            if ratio_floors.get(metric) is None:
                problems.append(f"{metric} ratio floor must be readable")
            if scalar_floors.get(metric) is not None:
                problems.append(
                    f"{metric} gates a ratio; a scalar floor for it is a folded misread"
                )
        else:
            if kind != KIND_SCALAR_FLOOR:
                problems.append(f"{metric} must stay a one-sided scalar floor")
            if scalar_floors.get(metric) is None:
                problems.append(f"{metric} is a scalar floor and must be readable")
    if problems:
        raise Round0260RegistrationError(
            "R0260 registry refuses: " + "; ".join(sorted(set(problems)))
        )
    return {
        "holds": True,
        "gated_criteria_registered": sorted(criteria),
        "every_gated_criterion_is_one_sided": True,
        "descriptive_metrics_absent_from_the_registry": sorted(DESCRIPTIVE_METRICS),
        "readable_scalar_floors": sorted(scalar_floors),
        "readable_ratio_floors": sorted(ratio_floors),
    }


def _numbers_reachable_from(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        out: list[float] = []
        for item in value.values():
            out.extend(_numbers_reachable_from(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_numbers_reachable_from(item))
        return out
    return []


def assert_no_gate_reads_a_conformity_value(
    *,
    criteria: Mapping[str, Any],
    scalar_floors: Mapping[str, Any],
    ratio_floors: Mapping[str, Any],
    conformity: Mapping[str, Any],
) -> dict[str, Any]:
    """No number published as a conformity diagnostic may appear in a gate.

    A value-level check rather than a key-level one: renaming `ratio_upper` to
    `floor` and pasting `0.7719564268121961` into the registry is the realistic
    way the retired side comes back, and a key-name guard would not see it.
    """
    if not conformity.get(CONFORMITY_MARKING):
        raise Round0260RegistrationError(
            f"R0260 conformity block must carry {CONFORMITY_MARKING}: true"
        )
    if conformity.get("contributes_to_verdict") is not False:
        raise Round0260RegistrationError(
            "R0260 conformity block must declare contributes_to_verdict: false"
        )
    band_values = set(_numbers_reachable_from(conformity["two_sided_bands"]))
    gate_values = set(
        _numbers_reachable_from(criteria)
        + _numbers_reachable_from(scalar_floors)
        + _numbers_reachable_from(ratio_floors)
    )
    leaked = sorted(band_values & gate_values)
    if leaked:
        raise Round0260RegistrationError(
            "R0260 refuses: retired conformity band values reachable inside the "
            f"gate registry: {leaked}"
        )
    return {
        "holds": True,
        "conformity_values_checked": len(band_values),
        "gate_values_checked": len(gate_values),
        "values_appearing_in_both": [],
        "marking": CONFORMITY_MARKING,
    }


# --------------------------------------------------------------------------- #
# C. judging one map under the re-registered criterion
# --------------------------------------------------------------------------- #


def judge_map_one_sided(
    *,
    cell_id: str,
    metrics: Mapping[str, float],
    raw_ratios: Mapping[str, float],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one map against the one-sided criteria. Never fits anything."""
    criteria = dict(gate["registered_criteria"])
    power = dict(gate["detection_power_by_criterion"])
    conformity = dict(gate["descriptive_conformity_diagnostics"])
    bands = dict(conformity["two_sided_bands"])
    per_metric: dict[str, Any] = {}
    for metric in GATED_METRICS:
        entry = dict(criteria[metric])
        kind = str(entry["kind"])
        if kind == KIND_RATIO_FLOOR:
            key = PURITY_RATIO_KEYS[metric]
            observed = float(raw_ratios[key])
            floor = float(entry["ratio_floor"])
            comparand = "the UNFOLDED purity ratio (never the folded fidelity)"
        else:
            observed = float(metrics[metric])
            floor = float(entry["floor"])
            comparand = "the panel metric"
        if not math.isfinite(observed):
            raise Round0260RegistrationError(
                f"R0260 refuses a nonfinite observation for {cell_id} {metric}"
            )
        per_metric[metric] = {
            "metric": metric,
            "kind": kind,
            "criterion": f">= {floor}",
            "compared_quantity": comparand,
            "observed": observed,
            "passes": bool(observed >= floor),
            "margin": observed - floor,
            "margin_is": "observed - floor; negative means under-separated",
            "fails_only": "under_separation",
            "applicable_power": POWER_BY_KIND[kind],
            "detection_power": dict(power[metric]["detection_power"]),
            "new_cell_false_fail_rate": float(
                power[metric]["new_cell_false_fail_rate"]
            ),
        }
        if metric in PURITY_METRICS:
            band = bands[metric]
            per_metric[metric]["descriptive_conformity"] = {
                CONFORMITY_MARKING: True,
                "contributes_to_verdict": False,
                "retired_two_sided_band": dict(band),
                "inside_the_retired_band": (
                    float(band["ratio_lower"])
                    <= observed
                    <= float(band["ratio_upper"])
                ),
                "distance_above_the_retired_upper_edge": (
                    observed - float(band["ratio_upper"])
                ),
                "above_the_high_d_reference": observed > 1.0,
            }
    descriptive: dict[str, Any] = {}
    for metric in DESCRIPTIVE_METRICS:
        if metric in metrics:
            descriptive[metric] = {
                "observed": float(metrics[metric]),
                "contributes_to_verdict": False,
                "why": (
                    "descriptive-only by standing rule; one anchor of 4,000 (row "
                    "1449227, r_hd == 0) supplies about two-thirds of its value"
                ),
            }
    verdict = "PASS" if all(item["passes"] for item in per_metric.values()) else "FAIL"
    return {
        "cell_id": cell_id,
        "verdict": verdict,
        "failing_criteria": sorted(
            name for name, item in per_metric.items() if not item["passes"]
        ),
        "criteria_cleared": sum(1 for item in per_metric.values() if item["passes"]),
        "criteria_total": len(per_metric),
        "per_metric": per_metric,
        "descriptive_only": descriptive,
        "panel_false_alarm_rate": float(
            gate["panel_false_alarm_rate"]["panel_false_alarm_rate_under_independence"]
        ),
        "panel_false_alarm_rate_is_an_upper_bound": True,
        "family_disjointness_is_measured_in_the_registration_node": (
            "round0257_rung_contract.assert_no_rung_map_in_the_gate_family, with "
            "n_defining and n_judged, runs where the family is assembled. It is "
            "not restated here as a literal."
        ),
    }


def judge_population_one_sided(
    *, cells: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    verdicts = {
        cell_id: judge_map_one_sided(
            cell_id=cell_id,
            metrics=dict(payload["panel_metrics"]),
            raw_ratios=dict(payload["raw_purity_ratios"]),
            gate=gate,
        )
        for cell_id, payload in cells.items()
    }
    passing = sorted(k for k, v in verdicts.items() if v["verdict"] == "PASS")
    failing = sorted(k for k, v in verdicts.items() if v["verdict"] == "FAIL")
    return {
        "verdicts": verdicts,
        "maps_judged": len(verdicts),
        "maps_passing": passing,
        "maps_failing": failing,
        "every_map_passes": len(failing) == 0,
    }


# --------------------------------------------------------------------------- #
# D. the planted consumer, and the controls
# --------------------------------------------------------------------------- #


def _judge_reading_the_conformity_band(
    *,
    cell_id: str,
    metrics: Mapping[str, float],
    raw_ratios: Mapping[str, float],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """PLANT: a consumer that gates on the conformity band instead of the floor.

    This is the defect the ruling's clause 4 creates the opportunity for -- the
    two-sided values stay in the artifact, so something downstream can read them
    as a gate again. It exists so the inertness of the real judge is *measured*
    against a consumer that demonstrably is not inert.
    """
    bands = dict(gate["descriptive_conformity_diagnostics"]["two_sided_bands"])
    failing = []
    for metric in PURITY_METRICS:
        observed = float(raw_ratios[PURITY_RATIO_KEYS[metric]])
        band = bands[metric]
        if not (float(band["ratio_lower"]) <= observed <= float(band["ratio_upper"])):
            failing.append(metric)
    del metrics
    return {
        "cell_id": cell_id,
        "verdict": "PASS" if not failing else "FAIL",
        "failing_criteria": sorted(failing),
    }


def _gate_with_a_tampered_conformity_band(gate: Mapping[str, Any]) -> dict[str, Any]:
    """PLANT: widen the retired bands so far that nothing could be outside them.

    Potent by construction for any finite ratio. If a shipped verdict moves under
    this, something in the decision path is reading the diagnostics.
    """
    out = {key: value for key, value in gate.items()}
    conformity = dict(out["descriptive_conformity_diagnostics"])
    conformity["two_sided_bands"] = {
        metric: {"ratio_lower": 0.0, "ratio_upper": 1.0e9}
        for metric in PURITY_METRICS
    }
    out["descriptive_conformity_diagnostics"] = conformity
    return out


def conformity_diagnostics_never_gate_controls(
    *, cells: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the diagnostics are inert, and prove the proof is not vacuous.

    Two arms on the same mutation. The shipped judge must produce byte-identical
    verdicts; the planted consumer must change at least one. An arm that only
    showed "nothing moved" would hold for a mutation that reaches nothing at all.
    """
    tampered_gate = _gate_with_a_tampered_conformity_band(gate)
    shipped_before = judge_population_one_sided(cells=cells, gate=gate)
    shipped_after = judge_population_one_sided(cells=cells, gate=tampered_gate)
    consumer_before = {
        cell_id: _judge_reading_the_conformity_band(
            cell_id=cell_id,
            metrics=dict(payload["panel_metrics"]),
            raw_ratios=dict(payload["raw_purity_ratios"]),
            gate=gate,
        )
        for cell_id, payload in cells.items()
    }
    consumer_after = {
        cell_id: _judge_reading_the_conformity_band(
            cell_id=cell_id,
            metrics=dict(payload["panel_metrics"]),
            raw_ratios=dict(payload["raw_purity_ratios"]),
            gate=tampered_gate,
        )
        for cell_id, payload in cells.items()
    }
    shipped_changed = sorted(
        cell_id
        for cell_id in cells
        if shipped_before["verdicts"][cell_id]["verdict"]
        != shipped_after["verdicts"][cell_id]["verdict"]
    )
    consumer_changed = sorted(
        cell_id
        for cell_id in cells
        if consumer_before[cell_id]["verdict"] != consumer_after[cell_id]["verdict"]
    )
    return {
        "mutation": (
            "the retired two-sided bands widened to [0.0, 1e9], which no finite "
            "ratio can fall outside"
        ),
        "shipped_verdicts_before": {
            cell_id: shipped_before["verdicts"][cell_id]["verdict"] for cell_id in cells
        },
        "shipped_verdicts_after": {
            cell_id: shipped_after["verdicts"][cell_id]["verdict"] for cell_id in cells
        },
        "shipped_margins_are_bitwise_identical": all(
            shipped_before["verdicts"][cell_id]["per_metric"][metric]["margin"]
            == shipped_after["verdicts"][cell_id]["per_metric"][metric]["margin"]
            for cell_id in cells
            for metric in GATED_METRICS
        ),
        "shipped_verdicts_changed": shipped_changed,
        "the_shipped_judge_is_inert_to_the_diagnostics": not shipped_changed,
        "planted_consumer": _judge_reading_the_conformity_band.__name__,
        "planted_consumer_verdicts_before": {
            cell_id: consumer_before[cell_id]["verdict"] for cell_id in cells
        },
        "planted_consumer_verdicts_after": {
            cell_id: consumer_after[cell_id]["verdict"] for cell_id in cells
        },
        "planted_consumer_verdicts_changed": consumer_changed,
        "the_mutation_is_potent": bool(consumer_changed),
        "holds": bool(consumer_changed) and not shipped_changed,
        "why_both_arms_exist": (
            "an inertness claim proved only by 'nothing moved' holds equally for a "
            "mutation that reaches nothing. The planted consumer is the arm that "
            "shows the mutation reaches a decision path when one reads the "
            "diagnostics -- so the shipped judge's stillness is evidence."
        ),
    }


def _synthetic_cell(
    *, ffr: float, k256: float, k1024: float, density_v2: float = 0.62
) -> dict[str, Any]:
    return {
        "panel_metrics": {
            "ffr": float(ffr),
            "purity_fidelity_k256": min(float(k256), 1.0 / float(k256)),
            "purity_fidelity_k1024": min(float(k1024), 1.0 / float(k1024)),
            "density_v2": float(density_v2),
        },
        "raw_purity_ratios": {"k256": float(k256), "k1024": float(k1024)},
    }


def sidedness_controls(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Plant each side of every criterion and require the registered outcome.

    The load-bearing pair is the first two: a map far ABOVE the floors must pass
    (that is the whole content of the ruling) and a map one part in a million
    BELOW one of them must fail on exactly that criterion (that is what stops the
    re-registration being a criterion nothing can fail).
    """
    criteria = dict(gate["registered_criteria"])
    ffr_floor = float(criteria["ffr"]["floor"])
    k256_floor = float(criteria["purity_fidelity_k256"]["ratio_floor"])
    k1024_floor = float(criteria["purity_fidelity_k1024"]["ratio_floor"])
    healthy_ffr = ffr_floor * 1.2
    epsilon = 1.0e-6

    cases: tuple[tuple[str, dict[str, Any], str, tuple[str, ...], str], ...] = (
        (
            "an_over_separating_map_passes",
            _synthetic_cell(ffr=healthy_ffr, k256=5.0, k1024=5.0),
            "PASS",
            (),
            "ratios far above the floors and above the high-D reference: the "
            "direction the owner ruling removed from the gate",
        ),
        (
            "a_perfectly_faithful_map_passes",
            _synthetic_cell(ffr=healthy_ffr, k256=1.0, k1024=1.0),
            "PASS",
            (),
            "ratio 1.0 on both purity metrics -- the map the RETIRED k1024 band "
            "failed by +0.228",
        ),
        (
            "an_under_separating_k256_map_fails_on_k256_only",
            _synthetic_cell(
                ffr=healthy_ffr, k256=k256_floor * (1.0 - epsilon), k1024=1.0
            ),
            "FAIL",
            ("purity_fidelity_k256",),
            "one part in a million below the k256 floor: the direction that is "
            "still gated",
        ),
        (
            "an_under_separating_k1024_map_fails_on_k1024_only",
            _synthetic_cell(
                ffr=healthy_ffr, k256=1.0, k1024=k1024_floor * (1.0 - epsilon)
            ),
            "FAIL",
            ("purity_fidelity_k1024",),
            "one part in a million below the k1024 floor",
        ),
        (
            "a_map_exactly_at_both_purity_floors_passes",
            _synthetic_cell(ffr=healthy_ffr, k256=k256_floor, k1024=k1024_floor),
            "PASS",
            (),
            "the floor is inclusive: `observed >= floor`",
        ),
        (
            "a_map_below_the_ffr_floor_fails_on_ffr_only",
            _synthetic_cell(ffr=ffr_floor * (1.0 - epsilon), k256=1.0, k1024=1.0),
            "FAIL",
            ("ffr",),
            "the unchanged ffr floor still gates",
        ),
    )

    results: list[dict[str, Any]] = []
    for name, cell, expected_verdict, expected_failures, why in cases:
        judged = judge_map_one_sided(
            cell_id=name,
            metrics=dict(cell["panel_metrics"]),
            raw_ratios=dict(cell["raw_purity_ratios"]),
            gate=gate,
        )
        results.append({
            "control": name,
            "plants": why,
            "expected_verdict": expected_verdict,
            "observed_verdict": judged["verdict"],
            "expected_failing_criteria": list(expected_failures),
            "observed_failing_criteria": judged["failing_criteria"],
            "held": (
                judged["verdict"] == expected_verdict
                and judged["failing_criteria"] == sorted(expected_failures)
            ),
        })
    return {
        "controls": results,
        "planted": len(results),
        "controls_that_held": sum(1 for row in results if row["held"]),
        "every_control_held": all(row["held"] for row in results),
        "at_least_one_control_fails_the_map": any(
            row["observed_verdict"] == "FAIL" for row in results
        ),
        "at_least_one_control_passes_the_map": any(
            row["observed_verdict"] == "PASS" for row in results
        ),
    }


def the_retired_band_disagrees_control(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Show the re-registration changed something, on synthetic input.

    A re-registration that produced the same verdicts as the criterion it
    replaced would be a rename. The over-separating and perfectly-faithful
    synthetic maps pass the new criterion and fail the retired one; that
    disagreement is the measurable content of the ruling.
    """
    bands = dict(gate["descriptive_conformity_diagnostics"]["two_sided_bands"])
    rows: list[dict[str, Any]] = []
    for name, ratio in (("over_separating", 5.0), ("perfectly_faithful", 1.0)):
        cell = _synthetic_cell(
            ffr=float(gate["registered_criteria"]["ffr"]["floor"]) * 1.2,
            k256=ratio,
            k1024=ratio,
        )
        judged = judge_map_one_sided(
            cell_id=name,
            metrics=dict(cell["panel_metrics"]),
            raw_ratios=dict(cell["raw_purity_ratios"]),
            gate=gate,
        )
        retired = sorted(
            metric
            for metric in PURITY_METRICS
            if not (
                float(bands[metric]["ratio_lower"])
                <= ratio
                <= float(bands[metric]["ratio_upper"])
            )
        )
        rows.append({
            "synthetic_map": name,
            "ratio_on_both_purity_metrics": ratio,
            "verdict_under_the_one_sided_criterion": judged["verdict"],
            "criteria_the_retired_band_would_fail": retired,
            "the_two_criteria_disagree": (
                judged["verdict"] == "PASS" and bool(retired)
            ),
        })
    return {
        "rows": rows,
        "the_re_registration_is_not_a_rename": all(
            row["the_two_criteria_disagree"] for row in rows
        ),
    }


def registry_controls_one_sided(
    *,
    criteria: Mapping[str, Any],
    scalar_floors: Mapping[str, Any],
    ratio_floors: Mapping[str, Any],
    conformity: Mapping[str, Any],
    descriptive_values: Mapping[str, float],
) -> dict[str, Any]:
    """Positive controls for both registry guards."""
    honest_ok = True
    honest_error = None
    try:
        assert_registry_is_one_sided_only(criteria, scalar_floors, ratio_floors)
        assert_no_gate_reads_a_conformity_value(
            criteria=criteria,
            scalar_floors=scalar_floors,
            ratio_floors=ratio_floors,
            conformity=conformity,
        )
    except Round0260RegistrationError as error:
        honest_ok = False
        honest_error = f"{type(error).__name__}: {error}"

    plants: list[dict[str, Any]] = []

    def plant(
        name: str,
        why: str,
        run: Callable[[], Any],
    ) -> None:
        refused = False
        message = None
        try:
            run()
        except Round0260RegistrationError as error:
            refused = True
            message = str(error)
        plants.append({
            "plant": name,
            "why_this_is_the_defect": why,
            "refused": refused,
            "error": message,
        })

    purity = PURITY_METRICS[0]
    band_upper = float(conformity["two_sided_bands"][purity]["ratio_upper"])
    descriptive_metric = DESCRIPTIVE_METRICS[0]

    plant(
        "a_purity_criterion_re_registered_as_a_two_sided_band",
        "the retired criterion put straight back, which is the thing the owner "
        "ruling removed",
        lambda: assert_registry_is_one_sided_only(
            {
                **dict(criteria),
                purity: {
                    "kind": "two_sided_ratio_band",
                    "ratio_lower": 0.9,
                    "ratio_upper": band_upper,
                    "applicable_power": "two_sided",
                },
            },
            scalar_floors,
            ratio_floors,
        ),
    )
    plant(
        "an_upper_edge_smuggled_into_a_one_sided_criterion",
        "the criterion keeps the one-sided `kind` and gains a `ratio_upper`, so "
        "over-separation could fail a map again through a key-name change",
        lambda: assert_registry_is_one_sided_only(
            {**dict(criteria), purity: {**dict(criteria[purity]), "ratio_upper": band_upper}},
            scalar_floors,
            ratio_floors,
        ),
    )
    plant(
        "the_retired_upper_edge_renamed_as_a_ratio_floor",
        "review-0257's k1024 upper edge pasted in under a permitted key name -- "
        "the shape a key-level guard cannot see",
        lambda: assert_no_gate_reads_a_conformity_value(
            criteria={
                **dict(criteria),
                purity: {**dict(criteria[purity]), "ratio_floor": band_upper},
            },
            scalar_floors=scalar_floors,
            ratio_floors={**dict(ratio_floors), purity: band_upper},
            conformity=conformity,
        ),
    )
    plant(
        "two_sided_power_beside_a_one_sided_floor",
        "review-0255's finding in the opposite direction: the sidedness of the "
        "power must match the sidedness of the criterion",
        lambda: assert_registry_is_one_sided_only(
            {
                **dict(criteria),
                purity: {**dict(criteria[purity]), "applicable_power": "two_sided"},
            },
            scalar_floors,
            ratio_floors,
        ),
    )
    plant(
        "density_v2_registered_as_a_criterion",
        "R0161 and R0193 both gated density_v2; the audit found five downstream "
        "artifacts consuming it as a gate",
        lambda: assert_registry_is_one_sided_only(
            {**dict(criteria), descriptive_metric: {"kind": KIND_SCALAR_FLOOR}},
            scalar_floors,
            ratio_floors,
        ),
    )
    plant(
        "density_v2_left_readable_as_a_floor",
        "R0255's shipped registry: the descriptive metric was the one populated "
        "entry in a dict named registered_floors",
        lambda: assert_registry_is_one_sided_only(
            criteria,
            {
                **dict(scalar_floors),
                descriptive_metric: float(descriptive_values[descriptive_metric]),
            },
            ratio_floors,
        ),
    )
    plant(
        "a_purity_criterion_republished_as_a_folded_scalar_floor",
        "R0234 published a folded purity floor that was read as a gate; folding "
        "maps 1.10 and 0.91 to the same place and is blind to this ruling's "
        "direction",
        lambda: assert_registry_is_one_sided_only(
            criteria,
            {**dict(scalar_floors), purity: 0.962048853686814},
            ratio_floors,
        ),
    )
    plant(
        "the_ffr_floor_made_unreadable",
        "nulling a gated scalar floor is the failure that made R0255's registry "
        "unusable in the other direction",
        lambda: assert_registry_is_one_sided_only(
            criteria,
            {key: value for key, value in scalar_floors.items() if key != "ffr"},
            ratio_floors,
        ),
    )
    plant(
        "the_conformity_block_left_unmarked",
        "an unmarked block of band values is indistinguishable from a registry",
        lambda: assert_no_gate_reads_a_conformity_value(
            criteria=criteria,
            scalar_floors=scalar_floors,
            ratio_floors=ratio_floors,
            conformity={
                key: value
                for key, value in conformity.items()
                if key != CONFORMITY_MARKING
            },
        ),
    )
    return {
        "the_honest_registry_passes": honest_ok,
        "honest_registry_error": honest_error,
        "plants": plants,
        "planted_defects": len(plants),
        "planted_defects_refused": sum(1 for row in plants if row["refused"]),
        "every_planted_defect_is_refused": all(row["refused"] for row in plants),
        "holds": bool(honest_ok and all(row["refused"] for row in plants)),
    }


__all__ = [
    "CONFORMITY_MARKING",
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "KIND_RATIO_FLOOR",
    "KIND_SCALAR_FLOOR",
    "NO_TUNING_STATEMENT",
    "ONE_SIDED_KINDS",
    "OWNER_RULING",
    "OWNER_RULING_DATE",
    "OWNER_RULING_SECTION",
    "POWER_BY_KIND",
    "ROUND_ID",
    "Round0260RegistrationError",
    "SUPERSEDED_PURITY_CRITERIA_IN",
    "WHY_ONE_SIDED",
    "assert_no_gate_reads_a_conformity_value",
    "assert_registry_is_one_sided_only",
    "attainability_against_the_identity_bound",
    "conformity_diagnostics",
    "conformity_diagnostics_never_gate_controls",
    "judge_map_one_sided",
    "judge_population_one_sided",
    "one_sided_purity_floors",
    "one_sided_ratio_floor",
    "panel_false_alarm_rate_one_sided",
    "registered_criteria_one_sided",
    "registry_controls_one_sided",
    "sidedness_controls",
    "the_retired_band_disagrees_control",
]
