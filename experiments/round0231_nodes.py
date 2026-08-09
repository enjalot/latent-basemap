#!/usr/bin/env python3
"""Execute R0231 — register floors a defining cell can fail, on a robust scale.

One node. CPU only, `0.0` GPU-h, `CUDA_VISIBLE_DEVICES=""` in the child
environment. Every input is a sealed artifact.

It reads R0230's thirteen-cell panel, R0223's three cuVS cells, R0228's nine
cluster-spill candidate cells, and the three prior floor registrations (R0219 at
`n = 4`, R0222 at `n = 8`, R0225's 95/95 at `n = 8`), and produces:

* **five floor families** fitted to the thirteen exact cells — R0222's
  `mean - 2s`, R0225's one-sided 95/95, and three robust candidates
  (`median - 3*MAD_n`, a 1-trimmed `mean - 2s`, a fixed margin off the median) —
  each with its measured **self-loosening** under a 1/2/3-sigma injection;
* a decision on **which estimator to register**, with the realised
  `3*MAD_n / s` ratio published per metric so a reader can see where the robust
  floor sits relative to `mean - 2s` on this family, including where it is
  stricter;
* for every family, whether a **defining cell can fail** it — the identity
  `max|z| <= (n-1)/sqrt(n) = 3.3282` at `n = 13` against its multiplier, and for
  the robust families an explicit **witness** family in which one does;
* **two-sided bands on the unfolded log-ratio** for both purity metrics, with
  lower and upper bounds, raw ratios and the direction of every deviation;
* all four prior floor families and the new ones **side by side**, scored against
  the thirteen exact cells and against the twelve held-out cells, with
  `count_is_attainable` beside every count;
* the **R0161 / R0193 exposure**, read-only, including the fact that at `n = 4`
  and `n = 3` their identity bounds are `1.5` and `1.1547`, both below their
  multiplier `2.0`, so neither floor has ever been able to fail one of its own
  defining cells.

`density_v2` floors are computed and reported and are **not** gate material.
Nothing here trains, scores a map, reads a GPU, or modifies R0161's, R0193's,
R0219's, R0222's or R0225's artifacts.
"""
from __future__ import annotations

import json
import os
import resource
import statistics
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0231_robust_floors import (
    CANDIDATE_CLUSTER_COUNTS,
    CANDIDATE_SEEDS,
    CHOSEN_ESTIMATOR,
    CHOSEN_ESTIMATOR_JUSTIFICATION,
    CUVS_FAMILY_SEEDS,
    DENSITY_V2_DEFECT,
    DESCRIPTIVE_METRICS,
    EXACT_FAMILY_SEEDS,
    FAMILY_DEFINITIONS,
    FLOOR_ESTIMATORS,
    GATED_METRICS,
    GATE_CAPABILITY,
    GATE_SCHEMA,
    LEGACY_MULTIPLIER,
    METRICS,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    ROBUST_ESTIMATORS,
    ROUND_ID,
    Round0231Error,
    defining_cell_can_fail,
    fit_floors,
    identity_bound,
    mean_minus_2sd_floor,
    one_sided_tolerance_factor,
    score_cells_against,
    witness_defining_cell_failure,
)
from basemap import round0113_prompt_contrast as prompt_contract


GATE_ACTION = "register_robust_floors_n13"

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

#: Read-only. The two precedent gates that used the same estimator at n = 4 and
#: n = 3. Their files are opened and never written.
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

SOURCES = {
    "r0230_panel_n13": R0230_PANEL,
    "r0219_gate_n4": R0219_GATE,
    "r0222_gate_n8": R0222_GATE,
    "r0223_cuvs_comparison": R0223_COMPARISON,
    "r0225_tolerance_gate_n8": R0225_GATE,
    "r0228_cluster_spill_comparison": R0228_COMPARISON,
}


def _read_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise Round0231Error(f"R0231 {label} is absent at {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _exact_cell_id(seed: int) -> str:
    return f"exact-seed{int(seed)}"


def _held_out_cells(
    cuvs: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """R0223's three cuVS cells and R0228's nine candidate cells.

    These define nothing, so they are the only cells that can genuinely test a
    floor built from the exact family.
    """
    cells: list[dict[str, Any]] = []
    cuvs_cells = cuvs["cuvs_panel_metric_cells"]
    cuvs_ratios = cuvs["cuvs_purity_ratios"]
    if tuple(sorted(int(seed) for seed in cuvs_cells)) != CUVS_FAMILY_SEEDS:
        raise Round0231Error("R0231 cuVS family is not seeds 42-44")
    for seed in CUVS_FAMILY_SEEDS:
        cells.append({
            "cell_id": f"cuvs-igd48-seed{seed}",
            "family": "cuvs-igd48",
            "seed": int(seed),
            "values": {
                metric: float(cuvs_cells[str(seed)][metric]) for metric in METRICS
            },
            "ratios": {
                key: float(cuvs_ratios[str(seed)][key]) for key in ("k256", "k1024")
            },
        })
    candidate_cells = candidate["candidate_panel_metric_cells"]
    candidate_ratios = candidate["candidate_purity_ratios"]
    if tuple(sorted(int(c) for c in candidate_cells)) != tuple(
        sorted(CANDIDATE_CLUSTER_COUNTS)
    ):
        raise Round0231Error("R0231 candidate arms are not c4/c8/c16")
    for clusters in CANDIDATE_CLUSTER_COUNTS:
        arm = candidate_cells[str(clusters)]
        arm_ratios = candidate_ratios[str(clusters)]
        if tuple(sorted(int(seed) for seed in arm)) != CANDIDATE_SEEDS:
            raise Round0231Error(
                f"R0231 candidate arm c{clusters} is not seeds 42-44"
            )
        for seed in CANDIDATE_SEEDS:
            cells.append({
                "cell_id": f"cluster-spill-c{clusters}-seed{seed}",
                "family": f"cluster-spill-c{clusters}",
                "seed": int(seed),
                "values": {
                    metric: float(arm[str(seed)][metric]) for metric in METRICS
                },
                "ratios": {
                    key: float(arm_ratios[str(seed)][key])
                    for key in ("k256", "k1024")
                },
            })
    if len(cells) != len(CUVS_FAMILY_SEEDS) + len(CANDIDATE_CLUSTER_COUNTS) * len(
        CANDIDATE_SEEDS
    ):
        raise Round0231Error("R0231 held-out set is not twelve cells")
    return cells


def _prior_families(
    *,
    r0219: Mapping[str, Any],
    r0222: Mapping[str, Any],
    r0225: Mapping[str, Any],
    exact_cells: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """The three prior registrations, read from bytes and cross-checked.

    Every floor is read out of a sealed artifact. R0222's are also **recomputed**
    from its own eight sealed cells, so the read is proved rather than trusted.
    """
    families: dict[str, Any] = {}

    r0219_floors = {
        metric: float(cell["floor"]) for metric, cell in r0219["gates"].items()
    }
    families["r0219_n4_mean_minus_2sd"] = {
        "round_id": "0219",
        "capability": r0219.get("capability"),
        "n": int(r0219["n"]),
        "formula": r0219.get("formula"),
        "multiplier": float(r0219["multiplier"]),
        "scale": "sample_sd",
        "defining_seeds": [int(seed) for seed in r0219["seed_family"]],
        "metrics_gated": sorted(r0219_floors),
        "floors": r0219_floors,
        "bands": {},
        "purity_scale": "folded exp(-|log r|), one-sided only",
    }

    r0222_floors = {
        metric: float(cell["floor"]) for metric, cell in r0222["gates"].items()
    }
    recomputed = {}
    for metric in r0222_floors:
        values = [
            float(r0222["pooled_panel_metric_cells"][str(seed)][metric])
            for seed in r0222["seed_family"]
        ]
        recomputed[metric] = mean_minus_2sd_floor(values)["floor"]
    if any(
        recomputed[metric] != r0222_floors[metric] for metric in r0222_floors
    ):
        raise Round0231Error(
            "R0231 could not reproduce R0222's sealed floors from R0222's own "
            f"sealed cells: {recomputed} != {r0222_floors}"
        )
    families["r0222_n8_mean_minus_2sd"] = {
        "round_id": "0222",
        "capability": r0222.get("capability"),
        "n": int(r0222["n"]),
        "formula": r0222.get("formula"),
        "multiplier": float(r0222["multiplier"]),
        "scale": "sample_sd",
        "defining_seeds": [int(seed) for seed in r0222["seed_family"]],
        "metrics_gated": sorted(r0222_floors),
        "floors": r0222_floors,
        "bands": {},
        "purity_scale": "folded exp(-|log r|), one-sided only",
        "recomputed_from_its_own_sealed_cells": recomputed,
    }

    r0225_gates = r0225["gate"]["gates"]
    r0225_floors = {
        metric: float(cell["one_sided_tolerance_95_95"]["floor"])
        for metric, cell in r0225_gates.items()
    }
    r0225_bands = {
        metric: {
            "ratio_lower": float(
                cell["two_sided_log_ratio_95_95"]["ratio_lower"]
            ),
            "ratio_upper": float(
                cell["two_sided_log_ratio_95_95"]["ratio_upper"]
            ),
        }
        for metric, cell in r0225_gates.items()
        if "two_sided_log_ratio_95_95" in cell
    }
    k_at_8 = float(r0225["tolerance_factors"]["one_sided"]["k"])
    derived_k_at_8 = one_sided_tolerance_factor(8)["k"]
    if abs(k_at_8 - derived_k_at_8) > 1e-9:
        raise Round0231Error(
            f"R0231 re-derived one-sided factor {derived_k_at_8} disagrees with "
            f"R0225's sealed {k_at_8}"
        )
    families["r0225_n8_tolerance_95_95"] = {
        "round_id": "0225",
        "capability": r0225.get("capability"),
        "n": int(r0225["gate"]["n"]),
        "formula": "family mean - k*s with k the one-sided 95/95 factor",
        "multiplier": k_at_8,
        "multiplier_rederived_here": derived_k_at_8,
        "scale": "sample_sd",
        "defining_seeds": list(r0225["gate"]["seed_order"]),
        "metrics_gated": sorted(r0225_floors),
        "floors": r0225_floors,
        "bands": r0225_bands,
        "purity_scale": "unfolded log-ratio, two-sided",
    }

    for name, family in families.items():
        family["identity"] = defining_cell_can_fail(
            n=family["n"],
            multiplier=family["multiplier"],
            scale=family["scale"],
        )
    # The eight-cell tables R0219 and R0222 were fitted to are a subset of the
    # thirteen scored here; assert the shared cells still carry the same values.
    for seed in r0222["seed_family"]:
        sealed = {
            key: float(value)
            for key, value in r0222["pooled_panel_metric_cells"][str(seed)].items()
        }
        mine = {
            key: float(value) for key, value in exact_cells[str(seed)].items()
        }
        if sealed != mine:
            raise Round0231Error(
                f"R0231 seed-{seed} disagrees with R0222's sealed cell; the "
                "thirteen-cell table is not an extension of the eight-cell one"
            )
    return families


def assess_precedent_exposure() -> dict[str, Any]:
    """R0161 and R0193, read-only. Their files are opened and never written."""
    exposure: dict[str, Any] = {}
    for round_id, path in sorted(PRECEDENTS.items()):
        artifact = _read_json(path, f"R{round_id} precedent gate")
        gates = artifact.get("gates") or {}
        n = int(artifact.get("n", -1))
        # These artifacts carry the multiplier per metric, not at the top level.
        multipliers = {float(cell["multiplier"]) for cell in gates.values()}
        if len(multipliers) != 1:
            raise Round0231Error(
                f"R{round_id} gates do not share one multiplier: {multipliers}"
            )
        multiplier = multipliers.pop()
        if multiplier != LEGACY_MULTIPLIER:
            raise Round0231Error(
                f"R{round_id} multiplier is {multiplier}, not the "
                f"{LEGACY_MULTIPLIER} this exposure assessment is about"
            )
        derived = one_sided_tolerance_factor(n)["k"] if n >= 3 else None
        identity = defining_cell_can_fail(
            n=n, multiplier=multiplier, scale="sample_sd"
        )
        entries = {
            metric: {
                "floor": cell.get("floor"),
                "mean": cell.get("mean"),
                "sample_sd": cell.get("sample_sd") or cell.get("sample_sd_ddof1"),
                "is_purity_ratio_metric": metric in PURITY_METRICS,
                "folded_purity_floor": metric in PURITY_METRICS,
            }
            for metric, cell in sorted(gates.items())
        }
        exposure[round_id] = {
            "artifact": path,
            "sha256": expected_input_signature(path)["sha256"],
            "capability": artifact.get("capability"),
            "formula": artifact.get("formula"),
            "n": n,
            "multiplier": multiplier,
            "seed_family": list(artifact.get("seed_family") or []),
            "metrics_gated": sorted(gates),
            "gates": entries,
            "one_sided_95_95_factor_at_this_n": derived,
            "factor_ratio_to_registered_multiplier": (
                derived / multiplier if derived and multiplier else None
            ),
            "identity": identity,
            "modified_by_this_round": False,
        }
    return {
        "read_only": True,
        "precedents": exposure,
        "finding": (
            "Both precedents used mean - 2*s, so both carry three defects at "
            "once. (1) Their identity bounds are 1.5 at n = 4 (R0161) and 1.1547 "
            "at n = 3 (R0193), BOTH below the multiplier 2.0, so neither floor "
            "has ever been able to fail one of its own defining cells: their "
            "reported 'all cells clear' results are theorems and no pass they "
            "granted was ever a test. (2) The one-sided 95/95 factor at their n "
            "is 5.143874860989298 and 7.655900133153338, i.e. 2.5719x and 3.8280x "
            "the multiplier they used, so the floors are far more punitive than "
            "the nominal calibration implies. (3) Both gate purity_fidelity on "
            "the FOLDED scale, which cannot distinguish over-separation from "
            "under-separation; this round measures both directions occurring in "
            "the same program."
        ),
        "recommendation": (
            "Re-register both before either floor is used to judge a new "
            "treatment, from a family large enough that (n-1)/sqrt(n) exceeds the "
            "chosen multiplier, on a robust scale, with purity gated two-sidedly "
            "on the unfolded log-ratio. Until then their purity floors should "
            "carry descriptive-only status. There is no urgency about maps they "
            "have already passed: the exposure is that those passes were never "
            "tests, not that any map was wrongly failed. Neither file is modified "
            "by this round."
        ),
    }


def run_gate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0231Error("R0231 handler received another queue")
    started = time.monotonic()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0231 robust floor registration"
    )

    panel = _read_json(R0230_PANEL, "R0230 thirteen-cell panel")
    r0219 = _read_json(R0219_GATE, "R0219 sealed n=4 gate")
    r0222 = _read_json(R0222_GATE, "R0222 sealed n=8 gate")
    r0223 = _read_json(R0223_COMPARISON, "R0223 sealed cuVS comparison")
    r0225 = _read_json(R0225_GATE, "R0225 sealed n=8 tolerance gate")
    r0228 = _read_json(R0228_COMPARISON, "R0228 sealed cluster-spill comparison")

    exact_cells = panel["panel_metric_cells"]
    exact_ratios = panel["raw_purity_ratios"]
    if tuple(sorted(int(seed) for seed in exact_cells)) != EXACT_FAMILY_SEEDS:
        raise Round0231Error("R0231 exact family is not seeds 42-54")
    if int(panel["n"]) != len(EXACT_FAMILY_SEEDS):
        raise Round0231Error("R0231 requires R0230's thirteen-cell panel")
    if panel.get("gate_registerable_here") is not False:
        raise Round0231Error("R0230's panel claims to register a gate; it must not")

    gate = fit_floors(cells=exact_cells, ratios=exact_ratios)

    chosen = CHOSEN_ESTIMATOR
    if chosen not in ROBUST_ESTIMATORS:
        raise Round0231Error("R0231 must register a robust estimator")
    registered_floors = {
        metric: float(gate["gates"][metric]["families"][chosen]["floor"])
        for metric in METRICS
    }
    registered_bands = {
        metric: gate["gates"][metric]["two_sided_bands"][chosen]
        for metric in PURITY_METRICS
    }

    #: The witness: a defining cell CAN fail every registered family, exhibited
    #: rather than inferred.
    witnesses = {
        name: witness_defining_cell_failure(name) for name in FLOOR_ESTIMATORS
    }

    exact_scoring_cells = [
        {
            "cell_id": _exact_cell_id(seed),
            "family": "exact-graph",
            "seed": int(seed),
            "values": {
                metric: float(exact_cells[str(seed)][metric]) for metric in METRICS
            },
            "ratios": {
                key: float(exact_ratios[str(seed)][key])
                for key in ("k256", "k1024")
            },
        }
        for seed in EXACT_FAMILY_SEEDS
    ]
    held_out_cells = _held_out_cells(r0223, r0228)

    prior = _prior_families(
        r0219=r0219, r0222=r0222, r0225=r0225, exact_cells=exact_cells
    )
    all_families: dict[str, Any] = dict(prior)
    for name in FLOOR_ESTIMATORS:
        family = gate["gates"]["ffr"]["families"][name]
        all_families[f"r0231_n13_{name}"] = {
            "round_id": ROUND_ID,
            "capability": GATE_CAPABILITY if name == chosen else None,
            "n": len(EXACT_FAMILY_SEEDS),
            "formula": name,
            "multiplier": family.get("multiplier"),
            "scale": family["scale"],
            "defining_seeds": list(EXACT_FAMILY_SEEDS),
            "metrics_gated": list(METRICS),
            "floors": {
                metric: float(gate["gates"][metric]["families"][name]["floor"])
                for metric in METRICS
            },
            "bands": {
                metric: {
                    "ratio_lower": float(
                        gate["gates"][metric]["two_sided_bands"][name]["ratio_lower"]
                    ),
                    "ratio_upper": float(
                        gate["gates"][metric]["two_sided_bands"][name]["ratio_upper"]
                    ),
                }
                for metric in PURITY_METRICS
                if name in gate["gates"][metric]["two_sided_bands"]
            },
            "purity_scale": (
                "unfolded log-ratio, two-sided"
                if name in gate["gates"]["purity_fidelity_k256"]["two_sided_bands"]
                else "one-sided floor on the folded fidelity only"
            ),
            "identity": family["defining_cell_can_fail"],
            "witness": witnesses[name],
            "registered": name == chosen,
        }

    #: Every family, scored against both populations, with attainability.
    side_by_side: dict[str, Any] = {}
    for name, family in all_families.items():
        defining_ids = [
            _exact_cell_id(seed) for seed in family["defining_seeds"]
        ]
        gated = [
            metric for metric in GATED_METRICS if metric in family["floors"]
        ]
        side_by_side[name] = {
            "family": {
                key: value
                for key, value in family.items()
                if key not in {"witness"}
            },
            "gated_metrics_scored": gated,
            "exact_family_n13": score_cells_against(
                floors={
                    metric: family["floors"][metric] for metric in gated
                },
                bands=family["bands"],
                cells=exact_scoring_cells,
                metrics=gated,
                defining_seed_ids=defining_ids,
                family_can_fail_a_defining_cell=bool(
                    family["identity"]["defining_cell_can_fail"]
                ),
            ),
            "held_out_twelve": score_cells_against(
                floors={
                    metric: family["floors"][metric] for metric in gated
                },
                bands=family["bands"],
                cells=held_out_cells,
                metrics=gated,
                defining_seed_ids=defining_ids,
                family_can_fail_a_defining_cell=bool(
                    family["identity"]["defining_cell_can_fail"]
                ),
            ),
        }

    exposure = assess_precedent_exposure()

    #: The comparison a reader needs to judge whether "robust" bought strictness
    #: or bought slack, per metric, on this family.
    strictness = {
        metric: {
            "mean_minus_2sd_floor": float(
                gate["gates"][metric]["families"]["mean_minus_2sd"]["floor"]
            ),
            "chosen_floor": registered_floors[metric],
            "chosen_minus_legacy": (
                registered_floors[metric]
                - float(gate["gates"][metric]["families"]["mean_minus_2sd"]["floor"])
            ),
            "chosen_is_stricter": (
                registered_floors[metric]
                > float(gate["gates"][metric]["families"]["mean_minus_2sd"]["floor"])
            ),
            "mad_n_over_sample_sd": gate["gates"][metric]["families"][
                "median_minus_3_mad"
            ]["mad_n_over_sample_sd"],
            "effective_sigma_multiplier": gate["gates"][metric]["families"][
                "median_minus_3_mad"
            ]["effective_sigma_multiplier"],
            "family_mean": statistics.fmean(gate["gates"][metric]["values"]),
            "family_median": statistics.median(gate["gates"][metric]["values"]),
            "family_sample_sd_ddof1": statistics.stdev(
                gate["gates"][metric]["values"]
            ),
        }
        for metric in METRICS
    }

    execution_checks = {
        "thirteen_cells_define_the_floors": gate["n"] == len(EXACT_FAMILY_SEEDS),
        "twelve_held_out_cells_scored": len(held_out_cells) == 12,
        "identity_bound_at_n13_exceeds_every_variance_multiplier": all(
            identity_bound(len(EXACT_FAMILY_SEEDS))
            > float(gate["gates"]["ffr"]["families"][name]["multiplier"])
            for name in ("mean_minus_2sd", "one_sided_tolerance_95_95")
        ),
        "a_defining_cell_can_fail_every_registered_family": all(
            bool(
                gate["gates"]["ffr"]["families"][name]["defining_cell_can_fail"][
                    "defining_cell_can_fail"
                ]
            )
            for name in FLOOR_ESTIMATORS
        ),
        "every_family_has_a_failure_witness": all(
            bool(item["lowest_cell_fails_its_own_floor"])
            for item in witnesses.values()
        ),
        "self_loosening_measured_for_every_metric": all(
            "self_loosening" in gate["gates"][metric] for metric in METRICS
        ),
        "the_chosen_estimator_is_outlier_invariant": all(
            chosen in gate["gates"][metric]["self_loosening"][
                "outlier_invariant_estimators"
            ]
            for metric in METRICS
        ),
        "both_variance_families_are_self_loosening": all(
            set(("mean_minus_2sd", "one_sided_tolerance_95_95"))
            <= set(gate["gates"][metric]["self_loosening"]["self_loosening_estimators"])
            for metric in METRICS
        ),
        "purity_gated_two_sidedly_on_the_unfolded_ratio": all(
            "ratio_lower" in registered_bands[metric]
            and "ratio_upper" in registered_bands[metric]
            for metric in PURITY_METRICS
        ),
        "density_v2_is_not_gated": "density_v2" not in GATED_METRICS,
        "r0222_floors_reproduced_from_its_own_cells": (
            "recomputed_from_its_own_sealed_cells"
            in prior["r0222_n8_mean_minus_2sd"]
        ),
        "precedents_not_modified": all(
            item["modified_by_this_round"] is False
            for item in exposure["precedents"].values()
        ),
        "no_training_performed": True,
        "no_gpu_used": True,
    }
    if not all(execution_checks.values()):
        raise Round0231Error(f"R0231 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    receipt = prompt_contract.seal({
        "schema": GATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "outcome": "2m-floors-reregistered-on-a-robust-scale-at-n13",
        "training_performed": False,
        "evaluation_performed": False,
        "production_or_publishing": False,
        "gpu_used": False,
        "gate_registered": True,
        "gate_status": "registered-and-contingent-pending-review",
        "supersedes_capability": "minilm-mixed-2m-tolerance-gates-n8-v1",
        "applies_to": (
            "byte-commensurate maps of the R0216 queue-correction-3 mixed MiniLM "
            "2M substrate under the R0217 recipe and the R0218 panel "
            "configuration only, at n = 13"
        ),
        "does_not_apply_to": (
            "jina universes, differently composed or differently sized MiniLM "
            "universes, PQ-derived graphs, or any map scored on a different panel "
            "configuration"
        ),
        "sources": {
            name: expected_input_signature(path) for name, path in SOURCES.items()
        },
        "n": len(EXACT_FAMILY_SEEDS),
        "identity_bound_at_n13": identity_bound(len(EXACT_FAMILY_SEEDS)),
        "family_definitions": FAMILY_DEFINITIONS,
        "chosen_estimator": chosen,
        "chosen_estimator_justification": CHOSEN_ESTIMATOR_JUSTIFICATION,
        "registered_floors": registered_floors,
        "registered_two_sided_bands": registered_bands,
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "density_v2_defect": DENSITY_V2_DEFECT,
        "gate": gate,
        "strictness_versus_mean_minus_2sd": strictness,
        "defining_cell_failure_witnesses": witnesses,
        "floor_families_side_by_side": side_by_side,
        "held_out_cells": [cell["cell_id"] for cell in held_out_cells],
        "precedent_exposure": exposure,
        "execution_checks": execution_checks,
        "wall_seconds": time.monotonic() - started,
        "peak_host_rss_gib": peak_rss_gib,
    })
    atomic_write_new_json(
        os.path.join(output, "minilm-robust-floors-n13.json"), receipt, immutable=True
    )
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "n": len(EXACT_FAMILY_SEEDS),
        "chosen_estimator": chosen,
        "registered_floors": registered_floors,
        "held_out_cells": len(held_out_cells),
    }, indent=2, sort_keys=True))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == GATE_ACTION:
        run_gate(active, job)
        return
    raise Round0231Error(f"R0231 unknown action {action!r}")


__all__ = [
    "GATE_ACTION",
    "PRECEDENTS",
    "SOURCES",
    "assess_precedent_exposure",
    "run_gate",
    "run_job",
]
