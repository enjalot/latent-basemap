#!/usr/bin/env python3
"""Execute R0234 — calibrate the multiplier, then register what survives.

One node. CPU only, `0.0` GPU-h, `CUDA_VISIBLE_DEVICES=""` in the child
environment. Every input is a sealed artifact; nothing trains, scores a map, or
touches a GPU.

Order of operations, and it matters:

1. **Calibrate on the Gaussian null first**, before a single sealed cell is read.
   Six estimators, `4,000,000` families at `n = 13`, one- and two-sided. Twelve
   published external reference values — ten reviewer simulations and two closed
   forms — must reproduce inside their stated tolerances or the node aborts.
2. **Apply the pre-registered selection rule** (coverage, invariance,
   attainability; then power, materiality, breakdown point). The invariance leg
   needs the real series, so the rule is evaluated once the cells are read, but
   the rule itself and every multiplier it consumes are fixed by step 1.
3. **Fit, score, and report.** Floors and bands for every candidate on the
   thirteen exact cells; all `25` cells scored under every candidate; every
   published verdict this registration would change, named; `R0161`/`R0193`
   exposure read-only, now with the failure side quantified; and the smallest `n`
   at which the registered family would match the power of the `n = 13` sample-sd
   `95/95` gate.

Never silently: if the registered floor passes a cell that a **released** floor
failed, the round reports the reversal and **retains** the released criterion on
that metric rather than superseding it.
"""
from __future__ import annotations

import json
import math
import os
import resource
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0234_calibrated_floors import (
    CANDIDATE_CLUSTER_COUNTS,
    CANDIDATE_ORDER,
    CANDIDATE_SEEDS,
    CANDIDATES,
    COVERAGE_TARGET,
    COVERAGE_TOLERANCE,
    CUVS_FAMILY_SEEDS,
    DENSITY_V2_DEFECT,
    DESCRIPTIVE_METRICS,
    EXACT_FAMILY_SEEDS,
    EXTERNAL_CALIBRATION_TARGETS,
    GATED_METRICS,
    GATE_CAPABILITY,
    GATE_SCHEMA,
    METRICS,
    N_EXACT,
    N_HELD_OUT,
    POWER_MATERIALITY,
    POWER_SELECTION_ALTERNATIVE,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    REQUIRED_INVARIANCE_DEPTH,
    ROUND_ID,
    Round0234Error,
    SELECTION_RULE,
    attainability,
    band_at,
    centre_and_scale,
    degenerate_witness,
    floor_at,
    identity_bound,
    injection_ladder,
    positive_scale_witness,
    score_population,
    verdict_changes,
)
from basemap import round0234_calibration as calibration
from basemap import round0113_prompt_contrast as prompt_contract


GATE_ACTION = "register_calibrated_robust_floors_n13"

R0230_PANEL = (
    "/data/latent-basemap/runs/round-0230/queue/artifacts/"
    "minilm-mixed-2m-seed-family-panel-n13-v1/seed-family-panel-n13.json"
)
R0219_GATE = (
    "/data/latent-basemap/runs/round-0219/queue/artifacts/"
    "minilm-mixed-2m-quality-gates-v1/minilm-quality-gates.json"
)
R0222_GATE = (
    "/data/latent-basemap/runs/round-0222/queue/artifacts/"
    "minilm-mixed-2m-quality-gates-n8-v1/minilm-quality-gates-n8.json"
)
R0223_COMPARISON = (
    "/data/latent-basemap/runs/round-0223/queue-correction-3/artifacts/"
    "minilm-mixed-2m-cuvs-graph-map-comparison-v1/cuvs-graph-map-comparison.json"
)
R0225_GATE = (
    "/data/latent-basemap/runs/round-0225/queue/artifacts/"
    "minilm-mixed-2m-tolerance-gates-n8-v1/minilm-tolerance-gates-n8.json"
)
R0228_COMPARISON = (
    "/data/latent-basemap/runs/round-0228/queue/artifacts/"
    "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1/"
    "cluster-spill-graph-map-comparison.json"
)
R0231_GATE = (
    "/data/latent-basemap/runs/round-0231/queue/artifacts/"
    "minilm-mixed-2m-robust-floors-n13-v1/minilm-robust-floors-n13.json"
)

SOURCES = {
    "r0230_panel_n13": R0230_PANEL,
    "r0219_gate_n4": R0219_GATE,
    "r0222_gate_n8": R0222_GATE,
    "r0223_cuvs_comparison": R0223_COMPARISON,
    "r0225_tolerance_gate_n8": R0225_GATE,
    "r0228_cluster_spill_comparison": R0228_COMPARISON,
    "r0231_robust_gate_n13": R0231_GATE,
}

#: Read-only. Their files are opened and never written.
PRECEDENTS = {
    "0161": (
        "/data/latent-basemap/runs/round-0161/queue/artifacts/"
        "jina-prompted-universe-quality-gates-v1/prompted-quality-gates.json"
    ),
    "0193": (
        "/data/latent-basemap/runs/round-0193/queue/artifacts/"
        "jina-mixed-english-2m-quality-gates-v1/mixed-quality-gates.json"
    ),
}

#: Floors that a review has RELEASED. If this registration passes a cell one of
#: these failed, the released criterion is retained rather than superseded.
RELEASED_FLOOR_FAMILIES: tuple[str, ...] = ("r0225_n8_tolerance_95_95",)

POWER_LADDER_SIZES: tuple[int, ...] = tuple(range(13, 45))


def _read_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise Round0234Error(f"R0234 {label} is absent at {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _exact_cell_id(seed: int) -> str:
    return f"exact-seed{int(seed)}"


# --------------------------------------------------------------------------- #
# step 1 — the Gaussian null
# --------------------------------------------------------------------------- #


def calibrate_everything() -> dict[str, Any]:
    """Multipliers, delivered coverage, power — and the external checks."""
    at13 = calibration.calibrate(N_EXACT)
    arrays13 = at13.pop("_arrays")
    at8 = calibration.calibrate(8, families=1_000_000, two_sided=False)
    arrays8 = at8.pop("_arrays")

    nct13 = calibration.nct_one_sided_factor(N_EXACT)
    howe13 = calibration.howe_two_sided_factor(N_EXACT)
    howe8 = calibration.howe_two_sided_factor(8)

    legacy13 = {
        "mean_minus_2s": calibration.fixed_multiplier_report(
            arrays13, "mean_minus_k_sample_sd", 2.0
        ),
        "one_sided_95_95_nct": calibration.fixed_multiplier_report(
            arrays13, "mean_minus_k_sample_sd", nct13
        ),
        "median_minus_3_madn": calibration.fixed_multiplier_report(
            arrays13, "median_minus_k_madn", 3.0
        ),
        "two_sided_mean_plus_minus_2s": calibration.fixed_two_sided_report(
            arrays13, "mean_minus_k_sample_sd", 2.0
        ),
        "two_sided_median_plus_minus_3_madn": calibration.fixed_two_sided_report(
            arrays13, "median_minus_k_madn", 3.0
        ),
        "two_sided_howe": calibration.fixed_two_sided_report(
            arrays13, "mean_minus_k_sample_sd", howe13
        ),
    }
    legacy8 = {
        "mean_minus_2s": calibration.fixed_multiplier_report(
            arrays8, "mean_minus_k_sample_sd", 2.0
        ),
        "two_sided_howe": calibration.fixed_two_sided_report(
            arrays8, "mean_minus_k_sample_sd", howe8
        ),
    }

    observed = {
        "n13_mean_minus_2s_coverage": legacy13["mean_minus_2s"]["delivered_coverage"],
        "n13_mean_minus_2s_false_fail": legacy13["mean_minus_2s"][
            "new_cell_false_fail_rate"
        ],
        "n13_one_sided_95_95_coverage": legacy13["one_sided_95_95_nct"][
            "delivered_coverage"
        ],
        "n13_one_sided_95_95_false_fail": legacy13["one_sided_95_95_nct"][
            "new_cell_false_fail_rate"
        ],
        "n13_median_minus_3_madn_coverage": legacy13["median_minus_3_madn"][
            "delivered_coverage"
        ],
        "n13_median_minus_3_madn_false_fail": legacy13["median_minus_3_madn"][
            "new_cell_false_fail_rate"
        ],
        "n13_two_sided_mean_2s_false_fail": legacy13["two_sided_mean_plus_minus_2s"][
            "new_cell_false_fail_rate"
        ],
        "n13_two_sided_median_3madn_false_fail": legacy13[
            "two_sided_median_plus_minus_3_madn"
        ]["new_cell_false_fail_rate"],
        "n8_mean_minus_2s_coverage": legacy8["mean_minus_2s"]["delivered_coverage"],
        "n8_howe_two_sided_delivered": legacy8["two_sided_howe"][
            "delivered_confidence_at_content"
        ],
        "n13_calibrated_sample_sd_k_vs_nct": (
            at13["candidates"]["mean_minus_k_sample_sd"]["one_sided"][
                "calibrated_multiplier"
            ]
            - nct13
        ),
        "n13_calibrated_sample_sd_k2_vs_howe": (
            at13["candidates"]["mean_minus_k_sample_sd"]["two_sided"][
                "calibrated_multiplier"
            ]
            - howe13
        ),
    }
    checks = []
    for target in EXTERNAL_CALIBRATION_TARGETS:
        value = float(observed[target["key"]])
        delta = abs(value - float(target["value"]))
        checks.append({
            **target,
            "observed": value,
            "absolute_difference": delta,
            "passes": delta <= float(target["tolerance"]),
        })
    if not all(item["passes"] for item in checks):
        raise Round0234Error(
            "R0234 calibration harness failed an external reference: "
            + json.dumps([item for item in checks if not item["passes"]])
        )
    return {
        "n13": at13,
        "n8": at8,
        "closed_forms": {
            "nct_one_sided_95_95_at_n13": nct13,
            "howe_two_sided_95_95_at_n13": howe13,
            "howe_two_sided_95_95_at_n8": howe8,
        },
        "legacy_multipliers_at_n13": legacy13,
        "legacy_multipliers_at_n8": legacy8,
        "external_reference_checks": checks,
        "external_references_all_reproduced": True,
    }


# --------------------------------------------------------------------------- #
# step 2 — the pre-registered selection rule
# --------------------------------------------------------------------------- #


def evaluate_selection(
    *,
    calibrated: Mapping[str, Any],
    series: Mapping[str, Sequence[float]],
    log_series: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Coverage, invariance, attainability — then power, materiality, breakdown."""
    candidates: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        entry = calibrated["n13"]["candidates"][name]
        k_one = float(entry["one_sided"]["calibrated_multiplier"])
        k_two = float(entry["two_sided"]["calibrated_multiplier"])

        coverage_one = float(entry["one_sided"]["delivered_coverage"])
        coverage_two = float(entry["two_sided"]["delivered_confidence_at_content"])
        coverage_ok = (
            abs(coverage_one - COVERAGE_TARGET) <= COVERAGE_TOLERANCE
            and abs(coverage_two - COVERAGE_TARGET) <= COVERAGE_TOLERANCE
        )

        ladders: dict[str, Any] = {}
        for metric in GATED_METRICS:
            ladders[metric] = injection_ladder(
                name, series[metric], k_one, side="lower"
            )
        for metric, logs in log_series.items():
            ladders[f"log_ratio::{metric}::lower"] = injection_ladder(
                name, logs, k_two, side="lower"
            )
            ladders[f"log_ratio::{metric}::upper"] = injection_ladder(
                name, logs, k_two, side="upper"
            )
        # density_v2 is descriptive-only, so it does not decide the rule; it is
        # measured anyway so the record is complete.
        descriptive_ladders = {
            metric: injection_ladder(name, series[metric], k_one, side="lower")
            for metric in DESCRIPTIVE_METRICS
        }
        depths = {key: item["exact_invariance_depth"] for key, item in ladders.items()}
        invariance_ok = min(depths.values()) >= REQUIRED_INVARIANCE_DEPTH

        bound = attainability(name, n=N_EXACT, multiplier=k_one)
        bound_two = attainability(name, n=N_EXACT, multiplier=k_two)
        witness = positive_scale_witness(name, k_one)
        attainable_ok = (
            bool(bound["every_defining_cell_can_fail"])
            and bool(bound_two["every_defining_cell_can_fail"])
            and bool(witness["scale_is_strictly_positive"])
            and bool(witness["lowest_cell_fails_its_own_floor"])
        )

        candidates[name] = {
            "estimator": name,
            "centre": CANDIDATES[name]["centre"],
            "scale": CANDIDATES[name]["scale_name"],
            "asymptotic_breakdown_point": CANDIDATES[name]["breakdown_point"],
            "gaussian_efficiency": CANDIDATES[name]["gaussian_efficiency"],
            "calibrated_one_sided_multiplier": k_one,
            "calibrated_two_sided_multiplier": k_two,
            "delivered_one_sided_coverage": coverage_one,
            "delivered_two_sided_confidence": coverage_two,
            "new_cell_false_fail_rate_one_sided": float(
                entry["one_sided"]["new_cell_false_fail_rate"]
            ),
            "new_cell_false_fail_rate_two_sided": float(
                entry["two_sided"]["new_cell_false_fail_rate"]
            ),
            "detection_power": entry["one_sided"]["detection_power"],
            "detection_power_at_selection_alternative": float(
                entry["one_sided"]["detection_power"][
                    f"minus_{POWER_SELECTION_ALTERNATIVE:g}_sigma"
                ]
            ),
            "invariance_ladders": ladders,
            "invariance_ladders_descriptive": descriptive_ladders,
            "exact_invariance_depth_by_series": depths,
            "minimum_exact_invariance_depth": min(depths.values()),
            "attainability_one_sided": bound,
            "attainability_two_sided": bound_two,
            "positive_scale_witness": witness,
            "degenerate_witness_r0231_used": degenerate_witness(name, k_one),
            "requirement_1_coverage": coverage_ok,
            "requirement_2_invariance": invariance_ok,
            "requirement_3_attainability": attainable_ok,
            "qualifies": bool(coverage_ok and invariance_ok and attainable_ok),
        }

    qualifying = [name for name, item in candidates.items() if item["qualifies"]]
    chosen = None
    reasoning: list[str] = []
    if qualifying:
        best = max(
            candidates[name]["detection_power_at_selection_alternative"]
            for name in qualifying
        )
        tied = [
            name
            for name in qualifying
            if best - candidates[name]["detection_power_at_selection_alternative"]
            <= POWER_MATERIALITY
        ]
        reasoning.append(
            f"qualifying: {qualifying}; best detection power at "
            f"-{POWER_SELECTION_ALTERNATIVE:g} sigma is {best!r}; within the "
            f"{POWER_MATERIALITY!r} materiality band: {tied}"
        )
        top_breakdown = max(
            candidates[name]["asymptotic_breakdown_point"] for name in tied
        )
        tied = [
            name
            for name in tied
            if candidates[name]["asymptotic_breakdown_point"] == top_breakdown
        ]
        reasoning.append(f"highest asymptotic breakdown point {top_breakdown}: {tied}")
        chosen = min(
            tied, key=lambda name: candidates[name]["calibrated_one_sided_multiplier"]
        )
        reasoning.append(f"smallest calibrated multiplier among those: {chosen}")
    else:
        reasoning.append(
            "no candidate satisfies coverage AND invariance AND attainability at "
            "n = 13; nothing is registered"
        )
    return {
        "rule": SELECTION_RULE,
        "candidates": candidates,
        "qualifying": qualifying,
        "chosen_estimator": chosen,
        "reasoning": reasoning,
    }


# --------------------------------------------------------------------------- #
# published families, read from sealed bytes
# --------------------------------------------------------------------------- #


def published_families(
    *,
    r0219: Mapping[str, Any],
    r0222: Mapping[str, Any],
    r0225: Mapping[str, Any],
    r0231: Mapping[str, Any],
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    families["r0219_n4_mean_minus_2sd"] = {
        "round_id": "0219",
        "capability": r0219.get("capability"),
        "n": int(r0219["n"]),
        "multiplier": float(r0219["multiplier"]),
        "scale": "sample_sd",
        "purity_scale": "folded exp(-|log r|), one-sided floor",
        "floors": {
            metric: float(cell["floor"]) for metric, cell in r0219["gates"].items()
        },
        "bands": {},
        "defining_seeds": [int(seed) for seed in r0219["seed_family"]],
        "released": False,
    }
    families["r0222_n8_mean_minus_2sd"] = {
        "round_id": "0222",
        "capability": r0222.get("capability"),
        "n": int(r0222["n"]),
        "multiplier": float(r0222["multiplier"]),
        "scale": "sample_sd",
        "purity_scale": "folded exp(-|log r|), one-sided floor",
        "floors": {
            metric: float(cell["floor"]) for metric, cell in r0222["gates"].items()
        },
        "bands": {},
        "defining_seeds": [int(seed) for seed in r0222["seed_family"]],
        "released": False,
    }
    r0225_gates = r0225["gate"]["gates"]
    families["r0225_n8_tolerance_95_95"] = {
        "round_id": "0225",
        "capability": r0225.get("capability"),
        "n": int(r0225["gate"]["n"]),
        "multiplier": float(r0225["tolerance_factors"]["one_sided"]["k"]),
        "scale": "sample_sd",
        "purity_scale": "unfolded log-ratio, two-sided",
        "floors": {
            metric: float(cell["one_sided_tolerance_95_95"]["floor"])
            for metric, cell in r0225_gates.items()
        },
        "bands": {
            metric: (
                float(cell["two_sided_log_ratio_95_95"]["ratio_lower"]),
                float(cell["two_sided_log_ratio_95_95"]["ratio_upper"]),
            )
            for metric, cell in r0225_gates.items()
            if "two_sided_log_ratio_95_95" in cell
        },
        "defining_seeds": [int(seed) for seed in r0225["gate"]["seed_order"]],
        "released": True,
    }
    families["r0231_n13_median_minus_3_madn"] = {
        "round_id": "0231",
        "capability": r0231.get("capability"),
        "n": int(r0231["n"]),
        "multiplier": 3.0,
        "scale": "MAD_n",
        "purity_scale": "unfolded log-ratio, two-sided",
        "floors": {
            metric: float(value)
            for metric, value in r0231["registered_floors"].items()
        },
        "bands": {
            metric: (float(band["ratio_lower"]), float(band["ratio_upper"]))
            for metric, band in r0231["registered_two_sided_bands"].items()
        },
        "defining_seeds": list(EXACT_FAMILY_SEEDS),
        "released": True,
    }
    return families


def score_family(
    family: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    *,
    every_defining_cell_can_fail: bool,
) -> dict[str, Any]:
    floors = {
        metric: value
        for metric, value in family["floors"].items()
        if metric in GATED_METRICS
    }
    bands = dict(family.get("bands") or {})
    return score_population(
        cells=cells,
        floors=floors,
        bands=bands,
        metrics=GATED_METRICS,
        defining_cell_ids=[
            _exact_cell_id(seed) for seed in family.get("defining_seeds") or []
        ],
        every_defining_cell_can_fail=every_defining_cell_can_fail,
    )


# --------------------------------------------------------------------------- #
# R0161 / R0193, read-only, with the failure side quantified
# --------------------------------------------------------------------------- #


def assess_precedent_exposure() -> dict[str, Any]:
    exposure: dict[str, Any] = {}
    for round_id, path in sorted(PRECEDENTS.items()):
        artifact = _read_json(path, f"R{round_id} precedent gate")
        gates = artifact.get("gates") or {}
        n = int(artifact.get("n", -1))
        multipliers = {float(cell["multiplier"]) for cell in gates.values()}
        if len(multipliers) != 1:
            raise Round0234Error(
                f"R{round_id} gates do not share one multiplier: {multipliers}"
            )
        multiplier = multipliers.pop()
        drawn = calibration.draw_centres_and_scales(
            n, families=1_000_000, names=("mean_minus_k_sample_sd",)
        )
        delivered = calibration.summarise(
            *drawn["mean_minus_k_sample_sd"],
            multiplier,
            content=0.95,
            alternatives=(1.0, 2.0, 3.0),
        )
        nominal_k = calibration.nct_one_sided_factor(n)
        bound = identity_bound(n)
        exposure[round_id] = {
            "artifact": path,
            "sha256": expected_input_signature(path)["sha256"],
            "capability": artifact.get("capability"),
            "n": n,
            "multiplier": multiplier,
            "metrics_gated": sorted(gates),
            "floors": {
                metric: cell.get("floor") for metric, cell in sorted(gates.items())
            },
            "identity_bound": bound,
            "defining_cell_can_fail": multiplier < bound,
            "one_sided_95_95_factor_at_this_n": nominal_k,
            "factor_ratio_to_registered_multiplier": nominal_k / multiplier,
            "delivered_confidence_at_multiplier_2": delivered["delivered_coverage"],
            "new_cell_false_fail_rate_at_multiplier_2": delivered[
                "new_cell_false_fail_rate"
            ],
            "purity_scale": "folded exp(-|log r|), one-sided floor",
            "modified_by_this_round": False,
        }
    return {
        "read_only": True,
        "precedents": exposure,
        "finding": (
            "Both precedents used mean - 2*s. (1) PASS SIDE: their identity bounds "
            "are 1.5 at n = 4 (R0161) and 1.1547005383792517 at n = 3 (R0193), both "
            "BELOW the multiplier 2.0, so neither floor has ever been able to fail "
            "one of its own defining cells; every 'all cells clear' they reported "
            "is a theorem and no pass either granted was ever a test. (2) FAILURE "
            "SIDE, which R0231 did not state and review-0231-01 required: the "
            "one-sided 95/95 factor at their n is 2.5719x and 3.8280x the "
            "multiplier they used, so the floors are far tighter than a 95/95 "
            "calibration and their measured new-cell false-fail rate at k = 2.0 is "
            "recorded above. A NON-defining map could therefore have been wrongly "
            "failed by either floor, and whether one was has never been audited. "
            "(3) Both gate purity on the FOLDED scale, which cannot tell "
            "over-separation from under-separation."
        ),
        "recommendation": (
            "Re-register both before either floor judges a new treatment: from a "
            "family whose n admits a calibrated multiplier every defining cell can "
            "fail, on a robust scale whose multiplier is calibrated to a stated "
            "delivered confidence rather than inherited, with purity gated "
            "two-sidedly on the unfolded log-ratio, and mark their purity floors "
            "descriptive-only meanwhile. Additionally AUDIT every non-defining map "
            "either floor has failed, because the failure-side exposure above is "
            "real and unexamined. Neither file is modified by this round."
        ),
    }


# --------------------------------------------------------------------------- #
# the node
# --------------------------------------------------------------------------- #


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0234Error("R0234 handler received another queue")
    started = time.monotonic()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0234 calibrated robust floor registration"
    )

    # 1. the Gaussian null, before any sealed cell is opened.
    calibrated = calibrate_everything()

    panel = _read_json(R0230_PANEL, "R0230 thirteen-cell panel")
    r0219 = _read_json(R0219_GATE, "R0219 sealed n=4 gate")
    r0222 = _read_json(R0222_GATE, "R0222 sealed n=8 gate")
    r0223 = _read_json(R0223_COMPARISON, "R0223 sealed cuVS comparison")
    r0225 = _read_json(R0225_GATE, "R0225 sealed n=8 tolerance gate")
    r0228 = _read_json(R0228_COMPARISON, "R0228 sealed cluster-spill comparison")
    r0231 = _read_json(R0231_GATE, "R0231 sealed n=13 robust gate")

    exact_cells = panel["panel_metric_cells"]
    exact_ratios = panel["raw_purity_ratios"]
    if tuple(sorted(int(seed) for seed in exact_cells)) != EXACT_FAMILY_SEEDS:
        raise Round0234Error("R0234 exact family is not seeds 42-54")
    if int(panel["n"]) != N_EXACT:
        raise Round0234Error("R0234 requires R0230's thirteen-cell panel")
    if panel.get("gate_registerable_here") is not False:
        raise Round0234Error("R0230's panel claims to register a gate; it must not")

    seeds = [str(seed) for seed in EXACT_FAMILY_SEEDS]
    series = {
        metric: [float(exact_cells[seed][metric]) for seed in seeds]
        for metric in METRICS
    }
    log_series = {
        metric: [
            math.log(float(exact_ratios[seed][PURITY_RATIO_KEYS[metric]]))
            for seed in seeds
        ]
        for metric in PURITY_METRICS
    }

    # 2. the pre-registered rule.
    selection = evaluate_selection(
        calibrated=calibrated, series=series, log_series=log_series
    )
    chosen = selection["chosen_estimator"]

    # 3. every candidate's floors and bands on the thirteen cells.
    fitted: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        item = selection["candidates"][name]
        k_one = float(item["calibrated_one_sided_multiplier"])
        k_two = float(item["calibrated_two_sided_multiplier"])
        floors = {metric: floor_at(name, series[metric], k_one) for metric in METRICS}
        bands = {
            metric: band_at(name, log_series[metric], k_two) for metric in PURITY_METRICS
        }
        centres = {
            metric: dict(zip(("centre", "scale"), centre_and_scale(name, series[metric])))
            for metric in METRICS
        }
        log_centres = {
            metric: dict(
                zip(("centre", "scale"), centre_and_scale(name, log_series[metric]))
            )
            for metric in PURITY_METRICS
        }
        fitted[name] = {
            "estimator": name,
            "one_sided_multiplier": k_one,
            "two_sided_multiplier": k_two,
            "floors": floors,
            "descriptive_folded_purity_floors": {
                metric: floors[metric] for metric in PURITY_METRICS
            },
            "two_sided_ratio_bands": {
                metric: {
                    "ratio_lower": math.exp(bands[metric][0]),
                    "ratio_upper": math.exp(bands[metric][1]),
                    "log_lower": bands[metric][0],
                    "log_upper": bands[metric][1],
                    "log_centre": log_centres[metric]["centre"],
                    "log_scale": log_centres[metric]["scale"],
                    "ratio_geometric_centre": math.exp(log_centres[metric]["centre"]),
                    "quantisation_note": (
                        "panel_v2 rounds each purity ratio to four decimals inside "
                        "the scorer, so this band inherits +/- 5e-5 in r"
                    ),
                }
                for metric in PURITY_METRICS
            },
            "centre_and_scale_by_metric": centres,
            "log_centre_and_scale_by_metric": log_centres,
            "scale_over_sample_sd": {
                metric: centres[metric]["scale"] / statistics.stdev(series[metric])
                for metric in METRICS
            },
            "effective_sigma_multiplier": {
                metric: (
                    statistics.fmean(series[metric]) - floors[metric]
                ) / statistics.stdev(series[metric])
                for metric in METRICS
            },
        }

    # the 25 cells.
    exact_scoring_cells = [
        {
            "cell_id": _exact_cell_id(seed),
            "family": "exact-graph",
            "values": {
                metric: float(exact_cells[str(seed)][metric]) for metric in METRICS
            },
            "ratios": {
                key: float(exact_ratios[str(seed)][key]) for key in ("k256", "k1024")
            },
        }
        for seed in EXACT_FAMILY_SEEDS
    ]
    held_out_cells: list[dict[str, Any]] = []
    cuvs_cells = r0223["cuvs_panel_metric_cells"]
    cuvs_ratios = r0223["cuvs_purity_ratios"]
    if tuple(sorted(int(seed) for seed in cuvs_cells)) != CUVS_FAMILY_SEEDS:
        raise Round0234Error("R0234 cuVS family is not seeds 42-44")
    for seed in CUVS_FAMILY_SEEDS:
        held_out_cells.append({
            "cell_id": f"cuvs-igd48-seed{seed}",
            "family": "cuvs-igd48",
            "values": {
                metric: float(cuvs_cells[str(seed)][metric]) for metric in METRICS
            },
            "ratios": {
                key: float(cuvs_ratios[str(seed)][key]) for key in ("k256", "k1024")
            },
        })
    candidate_cells = r0228["candidate_panel_metric_cells"]
    candidate_ratios = r0228["candidate_purity_ratios"]
    if tuple(sorted(int(item) for item in candidate_cells)) != tuple(
        sorted(CANDIDATE_CLUSTER_COUNTS)
    ):
        raise Round0234Error("R0234 candidate arms are not c4/c8/c16")
    for clusters in CANDIDATE_CLUSTER_COUNTS:
        arm = candidate_cells[str(clusters)]
        arm_ratios = candidate_ratios[str(clusters)]
        if tuple(sorted(int(seed) for seed in arm)) != CANDIDATE_SEEDS:
            raise Round0234Error(f"R0234 candidate arm c{clusters} is not seeds 42-44")
        for seed in CANDIDATE_SEEDS:
            held_out_cells.append({
                "cell_id": f"cluster-spill-c{clusters}-seed{seed}",
                "family": f"cluster-spill-c{clusters}",
                "values": {
                    metric: float(arm[str(seed)][metric]) for metric in METRICS
                },
                "ratios": {
                    key: float(arm_ratios[str(seed)][key]) for key in ("k256", "k1024")
                },
            })
    if len(held_out_cells) != N_HELD_OUT:
        raise Round0234Error("R0234 held-out set is not twelve cells")
    all_cells = list(exact_scoring_cells) + list(held_out_cells)

    # every candidate, scored on all 25.
    scoring: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        item = fitted[name]
        floors = {metric: item["floors"][metric] for metric in GATED_METRICS}
        bands = {
            metric: (
                item["two_sided_ratio_bands"][metric]["ratio_lower"],
                item["two_sided_ratio_bands"][metric]["ratio_upper"],
            )
            for metric in PURITY_METRICS
        }
        can_fail = bool(
            selection["candidates"][name]["attainability_one_sided"][
                "every_defining_cell_can_fail"
            ]
        )
        scoring[name] = {
            "all_twenty_five": score_population(
                cells=all_cells,
                floors=floors,
                bands=bands,
                metrics=GATED_METRICS,
                defining_cell_ids=[
                    _exact_cell_id(seed) for seed in EXACT_FAMILY_SEEDS
                ],
                every_defining_cell_can_fail=can_fail,
            ),
            "exact_thirteen": score_population(
                cells=exact_scoring_cells,
                floors=floors,
                bands=bands,
                metrics=GATED_METRICS,
                defining_cell_ids=[
                    _exact_cell_id(seed) for seed in EXACT_FAMILY_SEEDS
                ],
                every_defining_cell_can_fail=can_fail,
            ),
            "held_out_twelve": score_population(
                cells=held_out_cells,
                floors=floors,
                bands=bands,
                metrics=GATED_METRICS,
                defining_cell_ids=[
                    _exact_cell_id(seed) for seed in EXACT_FAMILY_SEEDS
                ],
                every_defining_cell_can_fail=can_fail,
            ),
        }

    prior = published_families(r0219=r0219, r0222=r0222, r0225=r0225, r0231=r0231)
    prior_scoring = {
        name: score_family(
            family,
            all_cells,
            every_defining_cell_can_fail=(
                float(family["multiplier"]) < identity_bound(int(family["n"]))
                if family["scale"] == "sample_sd"
                else True
            ),
        )
        for name, family in prior.items()
    }

    changes = None
    retained: dict[str, Any] = {}
    registered_floors: dict[str, Any] = {}
    registered_bands: dict[str, Any] = {}
    if chosen is not None:
        changes = verdict_changes(
            chosen=scoring[chosen]["all_twenty_five"], published=prior_scoring
        )
        registered_floors = {
            metric: fitted[chosen]["floors"][metric] for metric in METRICS
        }
        registered_bands = {
            metric: fitted[chosen]["two_sided_ratio_bands"][metric]
            for metric in PURITY_METRICS
        }
        # Never let a published failure lapse by supersession.
        for item in changes["un_failed_published_failures"]:
            family_name = item["published_family"]
            if family_name not in RELEASED_FLOOR_FAMILIES:
                continue
            metric = item["metric"]
            retained[metric] = {
                "retained_from": family_name,
                "retained_capability": prior[family_name]["capability"],
                "retained_floor": prior[family_name]["floors"].get(metric),
                "retained_band": prior[family_name]["bands"].get(metric),
                "because": (
                    f"this registration passes {item['cell_id']} on {metric}, which "
                    f"{family_name} FAILED and result-0228-2026-08-09.md published "
                    "as a failure. The released criterion is RETAINED alongside the "
                    "calibrated robust floor and is NOT superseded on this metric."
                ),
            }

    ladder = calibration.power_ladder(
        "median_minus_k_madn", sizes=POWER_LADDER_SIZES
    )
    target_power = float(
        calibrated["legacy_multipliers_at_n13"]["one_sided_95_95_nct"][
            "detection_power"
        ]["minus_2_sigma"]
    )
    matching = [
        row for row in ladder if row["detection_power_at_minus_2_sigma"] >= target_power
    ]
    power_parity = {
        "reference": (
            "the n = 13 sample-sd one-sided 95/95 gate, i.e. the most powerful "
            "family at this n that also delivers nominal confidence"
        ),
        "reference_detection_power_at_minus_2_sigma": target_power,
        "ladder": ladder,
        "smallest_n_matching_reference_power": (
            min(row["n"] for row in matching) if matching else None
        ),
    }

    exposure = assess_precedent_exposure()

    execution_checks = {
        "calibration_ran_before_any_sealed_cell_was_read": True,
        "every_external_reference_reproduced": bool(
            calibrated["external_references_all_reproduced"]
        ),
        "thirteen_cells_define_the_floors": len(exact_scoring_cells) == N_EXACT,
        "twelve_held_out_cells_scored": len(held_out_cells) == N_HELD_OUT,
        "twenty_five_cells_scored_against_every_candidate": all(
            item["all_twenty_five"]["cells_scored"] == N_EXACT + N_HELD_OUT
            for item in scoring.values()
        ),
        "every_candidate_calibrated_to_nominal_coverage": all(
            item["requirement_1_coverage"]
            for item in selection["candidates"].values()
        ),
        "selection_rule_applied_as_written": (
            chosen is None or chosen in selection["qualifying"]
        ),
        "chosen_is_robust_if_any": (
            chosen is None or bool(CANDIDATES[chosen]["robust"])
        ),
        "chosen_is_exactly_invariant_if_any": (
            chosen is None
            or selection["candidates"][chosen]["requirement_2_invariance"]
        ),
        "chosen_attainable_at_positive_scale_if_any": (
            chosen is None
            or selection["candidates"][chosen]["requirement_3_attainability"]
        ),
        "purity_gated_two_sidedly_on_the_unfolded_ratio": (
            chosen is None
            or all(
                "ratio_lower" in registered_bands[metric]
                and "ratio_upper" in registered_bands[metric]
                for metric in PURITY_METRICS
            )
        ),
        "density_v2_is_not_gated": "density_v2" not in GATED_METRICS,
        "every_published_verdict_change_is_enumerated": (
            chosen is None or changes is not None
        ),
        "no_released_failure_lapses_silently": (
            chosen is None
            or all(
                item["metric"] in retained
                for item in changes["un_failed_published_failures"]
                if item["published_family"] in RELEASED_FLOOR_FAMILIES
            )
        ),
        "precedents_not_modified": all(
            item["modified_by_this_round"] is False
            for item in exposure["precedents"].values()
        ),
        "no_training_performed": True,
        "no_gpu_used": True,
    }
    if not all(execution_checks.values()):
        raise Round0234Error(f"R0234 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    receipt = prompt_contract.seal({
        "schema": GATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "outcome": (
            "2m-floors-registered-on-a-calibrated-robust-scale-at-n13"
            if chosen is not None
            else "no-estimator-satisfies-coverage-invariance-and-attainability-at-n13"
        ),
        "training_performed": False,
        "evaluation_performed": False,
        "production_or_publishing": False,
        "gpu_used": False,
        "gate_registered": chosen is not None,
        "gate_status": "registered-and-contingent-pending-review",
        "supersedes_capability": None,
        "supersession_note": (
            "R0234 supersedes NOTHING. Any released criterion this registration "
            "would un-fail is retained beside it (see retained_stricter_criteria); "
            "R0231's supersession of minilm-mixed-2m-tolerance-gates-n8-v1 should "
            "be narrowed accordingly, which is a recommendation, not an edit."
        ),
        "applies_to": (
            "byte-commensurate maps of the R0216 queue-correction-3 mixed MiniLM "
            "2M substrate under the R0217 recipe and the R0218 panel configuration "
            "only, at n = 13"
        ),
        "does_not_apply_to": (
            "jina universes, differently composed or differently sized MiniLM "
            "universes, PQ-derived graphs, or any map scored on a different panel "
            "configuration"
        ),
        "sources": {
            name: expected_input_signature(path) for name, path in SOURCES.items()
        },
        "n": N_EXACT,
        "identity_bound_at_n13": identity_bound(N_EXACT),
        "selection_rule": SELECTION_RULE,
        "calibration": calibrated,
        "selection": selection,
        "chosen_estimator": chosen,
        "registered_floors": {
            metric: (registered_floors.get(metric) if metric != "density_v2" else None)
            for metric in METRICS
        },
        "registered_floors_all_metrics_including_descriptive": registered_floors,
        "registered_two_sided_bands": registered_bands,
        "registered_floors_purity_entries_are_descriptive": (
            "the purity entries of registered_floors are NULL by design. The gate "
            "for a purity metric is the two-sided band on the unfolded ratio; the "
            "folded one-sided floor is reported as "
            "descriptive_folded_purity_floors only. Review 0231-01 found consumers "
            "would fail cells the gate passes by reading the wrong field."
        ),
        "retained_stricter_criteria": retained,
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "density_v2_defect": DENSITY_V2_DEFECT,
        "fitted_candidates": fitted,
        "scoring_by_candidate": scoring,
        "published_families": prior,
        "published_family_scoring": prior_scoring,
        "verdict_changes_versus_published": changes,
        "power_parity_ladder": power_parity,
        "held_out_cells": [cell["cell_id"] for cell in held_out_cells],
        "precedent_exposure": exposure,
        "execution_checks": execution_checks,
        "wall_seconds": time.monotonic() - started,
        "peak_host_rss_gib": peak_rss_gib,
    })
    atomic_write_new_json(
        os.path.join(output, "minilm-calibrated-robust-floors-n13.json"),
        receipt,
        immutable=True,
    )
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "chosen_estimator": chosen,
        "qualifying": selection["qualifying"],
        "registered_floors": registered_floors,
        "registered_two_sided_bands": {
            metric: [band["ratio_lower"], band["ratio_upper"]]
            for metric, band in registered_bands.items()
        },
        "retained_stricter_criteria": sorted(retained),
        "smallest_n_matching_reference_power": power_parity[
            "smallest_n_matching_reference_power"
        ],
    }, indent=2, sort_keys=True))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == GATE_ACTION:
        run_gate(active, job)
        return
    raise Round0234Error(f"R0234 unknown action {action!r}")


__all__ = [
    "GATE_ACTION",
    "POWER_LADDER_SIZES",
    "PRECEDENTS",
    "RELEASED_FLOOR_FAMILIES",
    "SOURCES",
    "assess_precedent_exposure",
    "calibrate_everything",
    "evaluate_selection",
    "published_families",
    "run_gate",
    "run_job",
    "score_family",
]
