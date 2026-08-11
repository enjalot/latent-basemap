"""Execute R0256 — repair the evidence under R0255's registered `n = 29` gate.

One CPU node. No training, no GPU, no new cells, no new seeds. Every floor and
band is re-fitted from R0255's *sealed* twenty-nine-cell panel and required to be
**bitwise identical** to the values R0255 registered; if any one of them moves,
this node raises and the round stops, because a moved floor would mean the repair
changed the instrument rather than the evidence about it.

What it fixes, all four from review-0255:

1. **`independence_control` did not run.** Arms 1 and 2 built a `perturbed` copy
   of the held-out cells and then re-fitted the *unperturbed* family, so
   `every_floor_bitwise_identical: true` was structurally incapable of being false
   and would have held for a fit that genuinely read held-out cells. The repaired
   control derives the fit's inputs from cells through `family_series_from_cells`,
   so the perturbation reaches the fit, and this node runs three **positive
   controls** that plant "the fit reads a held-out cell" in that builder and
   require the arms to report `false`.
2. **The `64x` calibration interval.** R0255's gate node calibrated all six
   estimators -- including the two `O(n^2)` pairwise ones -- across `4,000,000`
   families, for five estimators the owner ruling forbade acting on, and paid a
   `160.83` s uninterruptible stretch for a table it was required not to use.
   This node restricts the call to the registered estimator and requires every
   multiplier, power and false-fail rate to come back byte-identical.
3. **Two-sided power beside two-sided bands.** R0234's `summarise_two_sided`
   computes no power, so both purity criteria published the `ffr` floor's
   one-sided figure. Both are published here, each labelled.
4. **The registry was backwards.** `density_v2` -- descriptive-only by standing
   rule -- was the one populated entry in `registered_floors` while the two gated
   purity criteria were nulled.
"""
from __future__ import annotations

import json
import math
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
    CANDIDATE_CLUSTER_COUNTS,
    CANDIDATE_SEEDS,
    CUVS_FAMILY_SEEDS,
    DENSITY_V2_DEFECT,
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    METRICS,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    band_at,
    floor_at,
)
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0255_gate_n29 import (
    EXACT_FAMILY_SEEDS,
    GATE_CAPABILITY as R0255_GATE_CAPABILITY,
    IDENTITY_BOUND_AT_N,
    NO_TUNING_STATEMENT,
    N_EXACT,
    N_HELD_OUT,
    OWNER_RULING,
    OWNER_RULING_ESTIMATOR,
    exact_cell_id,
    independence_control,
)
from basemap.round0255_panel_n29 import DENSITY_V2_STATUS
from basemap.round0256_gate_repair import (
    DEFINING_CELL_IDS,
    GATE_CAPABILITY,
    GATE_SCHEMA,
    REGISTERED_ESTIMATOR_NAMES,
    ROUND_ID,
    Round0256RepairError,
    SUPERSEDES_CAPABILITY,
    assert_registry_is_readable_only_as_registered,
    independence_positive_controls,
    panel_false_alarm_rate,
    registered_criteria,
    registry_controls,
    two_sided_detection_power,
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


REPAIR_ACTION = "repair_calibrated_madn_gate_evidence_n29"

#: What review-0255 recorded, so the artifact carries the before as well as the
#: after and a reader does not have to hold two documents open.
#:
#: NOTE the key names. `roundreport`'s gap census walks a node artifact for any
#: field whose terminal key is `widest_gap_s` and ranks what it finds as that
#: NODE's own measurement. Carrying R0255's number under that key made the
#: census headline this round with R0255's 160.83 s. These are another round's
#: figures for comparison and are named so they cannot be read as measurements
#: of this node.
R0255_PUBLISHED = {
    "arm_3_ffr_floor": 0.3078402410358946,
    "one_sided_power_published_beside_both_bands": 0.30035090040737566,
    "widest_uninterruptible_stretch_seconds": 160.8313321210153,
    "uninterruptible_stretch_multiple_of_the_registered_ceiling": 64.05190394580303,
    "gate_node_wall_seconds": 162.464,
    "density_v2_floor_left_readable": 0.41643957035196294,
}

UNRESOLVED = (
    {
        "item": "seeds 55-57 remain extreme on ratio::k256 at n = 29",
        "measured": (
            "review-0255 ranked them 26/23/25 among the pooled twenty-nine with an "
            "exact p of 0.0367; R0251's shift did not wash out"
        ),
        "why_it_is_not_chased_here": (
            "four metrics with no multiplicity correction gives 0.0367 x 4 = 0.147, "
            "the two flagged series are correlated at Pearson -0.404 so they are one "
            "anomaly reported twice, and refitting at n = 26 without them still fails "
            "c8-seed42. It does not move the gate and this round spends no GPU on it."
        ),
        "status": "open",
    },
    {
        "item": "detection power is structurally capped, not n-limited",
        "measured": (
            "review-0255's ladder: 95/95 power at -2 sigma asymptotes at 63.9% as "
            "n -> infinity; n = 29 gives 30.1%, n = 120 gives 48.5%, and 50% needs "
            "n ~ 130 at roughly 25 GPU-h of seeds"
        ),
        "why_it_is_not_chased_here": (
            "adding seeds is the wrong lever past the knee. The levers are the "
            "content parameter, gating on a family statistic rather than one cell, "
            "and registering the gate as a severity floor. All three are "
            "registration decisions for the manager, not measurements."
        ),
        "status": "open",
    },
)


def _cells_from_sealed(
    panel: Mapping[str, Any],
    r0223: Mapping[str, Any],
    r0228: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The twenty-nine family cells and the twelve held-out cells, from bytes."""
    exact_cells = panel["panel_metric_cells"]
    exact_ratios = panel["raw_purity_ratios"]
    if tuple(sorted(int(seed) for seed in exact_cells)) != EXACT_FAMILY_SEEDS:
        raise Round0256RepairError("R0256 exact family is not seeds 42-70")
    family = [
        {
            "cell_id": exact_cell_id(seed),
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
    held_out: list[dict[str, Any]] = []
    cuvs_cells = r0223["cuvs_panel_metric_cells"]
    cuvs_ratios = r0223["cuvs_purity_ratios"]
    if tuple(sorted(int(seed) for seed in cuvs_cells)) != CUVS_FAMILY_SEEDS:
        raise Round0256RepairError("R0256 cuVS family is not seeds 42-44")
    for seed in CUVS_FAMILY_SEEDS:
        held_out.append({
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
    for clusters in CANDIDATE_CLUSTER_COUNTS:
        arm = candidate_cells[str(clusters)]
        arm_ratios = candidate_ratios[str(clusters)]
        if tuple(sorted(int(seed) for seed in arm)) != CANDIDATE_SEEDS:
            raise Round0256RepairError(f"R0256 candidate arm c{clusters} is not 42-44")
        for seed in CANDIDATE_SEEDS:
            held_out.append({
                "cell_id": f"cluster-spill-c{clusters}-seed{seed}",
                "family": f"cluster-spill-c{clusters}",
                "values": {
                    metric: float(arm[str(seed)][metric]) for metric in METRICS
                },
                "ratios": {
                    key: float(arm_ratios[str(seed)][key])
                    for key in ("k256", "k1024")
                },
            })
    if len(family) != N_EXACT or len(held_out) != N_HELD_OUT:
        raise Round0256RepairError("R0256 cell assembly is not 29 + 12")
    return family, held_out


CUDA_LIBRARY_PATTERN = re.compile(
    r"libcuda|libcuvs|libcudart|libcublas|libcudnn|nvidia", re.IGNORECASE
)


def _cuda_mappings() -> list[str]:
    """Measure, do not assert, that this node holds no GPU library.

    R0255 published `gpu_used: false` as a literal. `vacuouscheck` calls that a
    `literal-check` -- a published verification no computation performed -- so
    the field is earned here by reading this process's own mappings.
    """
    with open("/proc/self/maps", "r", encoding="utf-8") as handle:
        return sorted({
            line.rsplit(" ", 1)[-1].strip()
            for line in handle
            if CUDA_LIBRARY_PATTERN.search(line)
        })


def _bitwise_same(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Every shared key compared with `==` on floats -- no tolerance anywhere."""
    keys = sorted(set(left) & set(right))
    rows = {
        key: {
            "r0255": left[key],
            "r0256": right[key],
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


def run_repair(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0256 round0256_nodes.run_repair")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0256RepairError("R0256 repair handler received another queue")
    node_id = str(active.get("node_id") or REPAIR_ACTION)
    label = "R0256 n=29 gate evidence repair"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0256 repair")

    window = ledger.window("R0256 repair stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0256 repair stage entered")
        wrapped = window.wrap(recorder)

        r0255_gate = prompt_contract.read_sealed(
            _bound_path(job, "r0255_gate", label="R0255 sealed n=29 gate"),
            label="R0255 sealed n=29 gate",
        )
        panel = prompt_contract.read_sealed(
            _bound_path(job, "panel_n29", label="R0255 sealed twenty-nine-cell panel"),
            label="R0255 sealed twenty-nine-cell panel",
        )
        r0223 = prompt_contract.read_sealed(
            _bound_path(job, "r0223_comparison", label="R0223 sealed cuVS comparison"),
            label="R0223 sealed cuVS comparison",
        )
        r0228 = prompt_contract.read_sealed(
            _bound_path(job, "r0228_comparison", label="R0228 sealed cluster-spill"),
            label="R0228 sealed cluster-spill comparison",
        )
        if (
            r0255_gate.get("capability") != R0255_GATE_CAPABILITY
            or r0255_gate.get("chosen_estimator") != OWNER_RULING_ESTIMATOR
            or int(r0255_gate.get("n", -1)) != N_EXACT
        ):
            raise Round0256RepairError("R0255 sealed gate receipt contract changed")
        wrapped("R0256 sealed R0255 gate, panel and held-out families read")

        # ------------------------------------------------------------------ #
        # B. the calibration, restricted to the estimator the ruling registered
        # ------------------------------------------------------------------ #
        r0255_entry = r0255_gate["calibration"]["n29"]["candidates"][
            OWNER_RULING_ESTIMATOR
        ]
        calibration_started = time.monotonic()
        restricted = calibration.calibrate(
            N_EXACT, names=REGISTERED_ESTIMATOR_NAMES
        )
        calibration_wall_s = time.monotonic() - calibration_started
        wrapped("R0256 restricted calibration at n=29 complete")
        entry = restricted["candidates"][OWNER_RULING_ESTIMATOR]
        k_one = float(entry["one_sided"]["calibrated_multiplier"])
        k_two = float(entry["two_sided"]["calibrated_multiplier"])

        one_sided_match = _bitwise_same(
            {
                "calibrated_multiplier": float(
                    r0255_entry["one_sided"]["calibrated_multiplier"]
                ),
                "delivered_coverage": float(
                    r0255_entry["one_sided"]["delivered_coverage"]
                ),
                "new_cell_false_fail_rate": float(
                    r0255_entry["one_sided"]["new_cell_false_fail_rate"]
                ),
                **{
                    f"power_{key}": float(value)
                    for key, value in r0255_entry["one_sided"][
                        "detection_power"
                    ].items()
                },
            },
            {
                "calibrated_multiplier": k_one,
                "delivered_coverage": float(entry["one_sided"]["delivered_coverage"]),
                "new_cell_false_fail_rate": float(
                    entry["one_sided"]["new_cell_false_fail_rate"]
                ),
                **{
                    f"power_{key}": float(value)
                    for key, value in entry["one_sided"]["detection_power"].items()
                },
            },
        )
        two_sided_match = _bitwise_same(
            {
                "calibrated_multiplier": float(
                    r0255_entry["two_sided"]["calibrated_multiplier"]
                ),
                "delivered_confidence_at_content": float(
                    r0255_entry["two_sided"]["delivered_confidence_at_content"]
                ),
                "new_cell_false_fail_rate": float(
                    r0255_entry["two_sided"]["new_cell_false_fail_rate"]
                ),
            },
            {
                "calibrated_multiplier": k_two,
                "delivered_confidence_at_content": float(
                    entry["two_sided"]["delivered_confidence_at_content"]
                ),
                "new_cell_false_fail_rate": float(
                    entry["two_sided"]["new_cell_false_fail_rate"]
                ),
            },
        )

        # ------------------------------------------------------------------ #
        # C. two-sided power for the two-sided bands
        # ------------------------------------------------------------------ #
        centre, scale = restricted["_arrays"][OWNER_RULING_ESTIMATOR]
        two_sided_power = two_sided_detection_power(centre, scale, k_two)
        one_sided_power = {
            key: float(value)
            for key, value in entry["one_sided"]["detection_power"].items()
        }
        two_sided_size = two_sided_detection_power(
            centre, scale, k_two, alternatives=(0.0,)
        )["minus_0_sigma"]
        power_by_criterion = {
            "ffr": {
                "criterion": "one-sided lower floor",
                "applicable_power": "one_sided",
                "detection_power": dict(one_sided_power),
                "new_cell_false_fail_rate": float(
                    entry["one_sided"]["new_cell_false_fail_rate"]
                ),
            },
        }
        for metric in PURITY_METRICS:
            power_by_criterion[metric] = {
                "criterion": "two-sided band on the unfolded purity log-ratio",
                "applicable_power": "two_sided",
                "detection_power": dict(two_sided_power),
                "new_cell_false_fail_rate": float(
                    entry["two_sided"]["new_cell_false_fail_rate"]
                ),
            }
        panel_far = panel_false_alarm_rate(
            one_sided_rate=float(entry["one_sided"]["new_cell_false_fail_rate"]),
            two_sided_rate=float(entry["two_sided"]["new_cell_false_fail_rate"]),
            two_sided_criteria=len(PURITY_METRICS),
        )
        wrapped("R0256 two-sided power and panel false-alarm rate computed")

        # ------------------------------------------------------------------ #
        # the floors must not move
        # ------------------------------------------------------------------ #
        family_cells, held_out_cells = _cells_from_sealed(panel, r0223, r0228)
        series = {
            metric: [float(cell["values"][metric]) for cell in family_cells]
            for metric in METRICS
        }
        log_series = {
            metric: [
                math.log(float(cell["ratios"][PURITY_RATIO_KEYS[metric]]))
                for cell in family_cells
            ]
            for metric in PURITY_METRICS
        }
        floors = {
            metric: floor_at(OWNER_RULING_ESTIMATOR, series[metric], k_one)
            for metric in METRICS
        }
        bands = {
            metric: band_at(OWNER_RULING_ESTIMATOR, log_series[metric], k_two)
            for metric in PURITY_METRICS
        }
        floors_match = _bitwise_same(
            {
                metric: float(value)
                for metric, value in r0255_gate[
                    "registered_floors_all_metrics_including_descriptive"
                ].items()
            },
            {metric: float(value) for metric, value in floors.items()},
        )
        bands_match = _bitwise_same(
            {
                f"{metric}::{side}": float(
                    r0255_gate["registered_two_sided_bands"][metric][f"ratio_{side}"]
                )
                for metric in PURITY_METRICS
                for side in ("lower", "upper")
            },
            {
                f"{metric}::{side}": math.exp(bands[metric][index])
                for metric in PURITY_METRICS
                for index, side in enumerate(("lower", "upper"))
            },
        )
        wrapped("R0256 floors and bands re-fitted from the sealed panel")

        # ------------------------------------------------------------------ #
        # A. the repaired independence control, and the control it never had
        # ------------------------------------------------------------------ #
        independence = independence_control(
            estimator=OWNER_RULING_ESTIMATOR,
            multiplier_one_sided=k_one,
            multiplier_two_sided=k_two,
            family_cells=family_cells,
            held_out_cells=held_out_cells,
            defining_cell_ids=DEFINING_CELL_IDS,
        )
        positive_controls = independence_positive_controls(
            estimator=OWNER_RULING_ESTIMATOR,
            multiplier_one_sided=k_one,
            multiplier_two_sided=k_two,
            family_cells=family_cells,
            held_out_cells=held_out_cells,
        )
        wrapped("R0256 independence control and its positive controls complete")

        arm3 = independence["arms"][2]
        arm3_correction = {
            "what_r0255_result_D3_said": (
                "'A single contaminated family cell also moves nothing, which is "
                "reported and not asserted: that is a depth-4 rank-order scale "
                "behaving as registered.'"
            ),
            "what_r0255_own_sealed_artifact_said": {
                "any_floor_moved": True,
                "ffr": R0255_PUBLISHED["arm_3_ffr_floor"],
            },
            "reproduced_here": {
                "any_floor_moved": bool(arm3["any_floor_moved"]),
                "ffr": arm3["floors"]["ffr"],
            },
            "baseline_ffr_floor": independence["baseline_floors"]["ffr"],
            "shift": arm3["floors"]["ffr"] - independence["baseline_floors"]["ffr"],
            "the_artifact_was_right": (
                bool(arm3["any_floor_moved"])
                and arm3["floors"]["ffr"] == R0255_PUBLISHED["arm_3_ffr_floor"]
            ),
            "why_it_moves": (
                "at n = 29, driving one cell to -inf moves the median from the 15th "
                "to the 14th order statistic, so the LOCATION moves even though the "
                "MAD_n scale is robust"
            ),
            "the_contaminated_floor_still_fails_c8_seed42": (
                float(arm3["floors"]["ffr"]) > 0.3075
            ),
        }

        # ------------------------------------------------------------------ #
        # D. the registry
        # ------------------------------------------------------------------ #
        ratio_bands = {
            metric: {
                "ratio_lower": math.exp(bands[metric][0]),
                "ratio_upper": math.exp(bands[metric][1]),
            }
            for metric in PURITY_METRICS
        }
        criteria = registered_criteria(floors=floors, bands=ratio_bands)
        scalar_floors = {"ffr": floors["ffr"]}
        registry_guard = assert_registry_is_readable_only_as_registered(
            criteria, scalar_floors
        )
        registry_plants = registry_controls(
            criteria=criteria,
            scalar_floors=scalar_floors,
            descriptive_values={
                metric: floors[metric] for metric in DESCRIPTIVE_METRICS
            },
        )
        wrapped("R0256 registry rebuilt and its positive controls complete")
        gate.finish("R0256 repair stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    cuda_mappings = _cuda_mappings()

    execution_checks = {
        "the_calibration_is_restricted_to_the_registered_estimator": (
            tuple(restricted["candidates"]) == REGISTERED_ESTIMATOR_NAMES
        ),
        "the_restricted_calibration_reproduces_r0255_one_sided_bitwise": bool(
            one_sided_match["every_value_is_bitwise_identical"]
        ),
        "the_restricted_calibration_reproduces_r0255_two_sided_bitwise": bool(
            two_sided_match["every_value_is_bitwise_identical"]
        ),
        "no_floor_moved": bool(floors_match["every_value_is_bitwise_identical"]),
        "no_band_moved": bool(bands_match["every_value_is_bitwise_identical"]),
        "the_perturbation_reaches_the_fit_in_every_arm": all(
            bool(arm["perturbation_reaches_the_fit"]) for arm in independence["arms"]
        ),
        "the_fit_is_independent_of_every_held_out_cell": bool(
            independence["the_fit_is_independent_of_every_held_out_cell"]
        ),
        "the_fit_is_not_inert": bool(independence["the_fit_is_not_inert"]),
        "every_planted_independence_defect_is_caught": bool(
            positive_controls["every_planted_defect_is_caught"]
        ),
        "the_honest_builder_still_passes": bool(
            positive_controls["the_honest_builder_still_reports_independent"]
        ),
        "arm_3_reproduces_the_r0255_artifact_not_its_prose": bool(
            arm3_correction["the_artifact_was_right"]
        ),
        "the_two_sided_power_is_below_the_one_sided_power": (
            two_sided_power["minus_2_sigma"] < one_sided_power["minus_2_sigma"]
        ),
        "the_two_sided_size_equals_the_two_sided_false_fail_rate": (
            two_sided_size == float(entry["two_sided"]["new_cell_false_fail_rate"])
        ),
        "the_panel_rate_is_above_every_per_metric_rate": (
            panel_far["panel_false_alarm_rate_under_independence"]
            > max(
                float(entry["one_sided"]["new_cell_false_fail_rate"]),
                float(entry["two_sided"]["new_cell_false_fail_rate"]),
            )
        ),
        "no_descriptive_metric_is_registered_as_a_criterion": not (
            set(DESCRIPTIVE_METRICS) & set(criteria)
        ),
        "every_gated_metric_is_registered": set(criteria) == set(GATED_METRICS),
        "every_planted_registry_defect_is_refused": bool(
            registry_plants["every_planted_defect_is_refused"]
        ),
        "no_cuda_library_is_mapped_into_this_process": not cuda_mappings,
    }
    if not all(execution_checks.values()):
        raise Round0256RepairError(f"R0256 execution checks failed: {execution_checks}")

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
        "abort_flag_precondition": abort_flag,
        "n": N_EXACT,
        "owner_ruling": OWNER_RULING,
        "chosen_estimator": OWNER_RULING_ESTIMATOR,
        "no_tuning_statement": NO_TUNING_STATEMENT,
        "identity_bound_at_n": IDENTITY_BOUND_AT_N,
        "what_this_round_repairs": (
            "the EVIDENCE under R0255's registered n = 29 MAD_n gate, not the gate. "
            "Every floor and band is required to be bitwise identical to R0255's."
        ),
        "r0255_published_values_for_comparison": dict(R0255_PUBLISHED),
        # --- B ---
        "calibration": {
            "estimators_calibrated": list(restricted["candidates"]),
            "estimators_calibrated_by_r0255": sorted(
                r0255_gate["calibration"]["n29"]["candidates"]
            ),
            "families_simulated": int(restricted["families_simulated"]),
            "seed": int(restricted["seed"]),
            "content": float(restricted["content"]),
            "confidence": float(restricted["confidence"]),
            "one_sided_multiplier": k_one,
            "two_sided_multiplier": k_two,
            "wall_s": calibration_wall_s,
            "r0255_wall_s_for_the_same_numbers": R0255_PUBLISHED[
                "widest_uninterruptible_stretch_seconds"
            ],
            "speedup_versus_r0255": (
                R0255_PUBLISHED["widest_uninterruptible_stretch_seconds"]
                / calibration_wall_s
                if calibration_wall_s
                else None
            ),
            "reproduces_r0255_one_sided": one_sided_match,
            "reproduces_r0255_two_sided": two_sided_match,
            "why_restricted": (
                "the owner ruling registered MAD_n and forbade acting on the other "
                "five candidates. R0255 nonetheless calibrated all six, including "
                "the two O(n^2) pairwise estimators, across 4,000,000 families in a "
                "single uninterruptible stretch. The sample draw precedes estimator "
                "selection, so the RNG stream and therefore every published number "
                "is unchanged -- which the two comparisons above establish rather "
                "than assert."
            ),
        },
        # --- floors, unchanged ---
        "registered_floors_are_unchanged": floors_match,
        "registered_bands_are_unchanged": bands_match,
        # --- D ---
        "registered_criteria": criteria,
        "registered_floors": scalar_floors,
        "registry_guard": registry_guard,
        "registry_controls": registry_plants,
        "registry_correction": (
            "R0255 published registered_floors with density_v2 populated at "
            f"{R0255_PUBLISHED['density_v2_floor_left_readable']!r} and BOTH gated "
            "purity entries null -- exactly backwards relative to risk. density_v2 "
            "is descriptive-only by standing rule and the R0161/R0193 audit found "
            "five downstream artifacts consuming such a value as a gate. Here "
            "registered_criteria carries every gated criterion in the form it "
            "gates, registered_floors carries only the scalar floors that gate, and "
            "no descriptive metric appears in either."
        ),
        "descriptive_only_never_a_gate": {
            metric: float(floors[metric]) for metric in DESCRIPTIVE_METRICS
        },
        "descriptive_only_note": (
            "published under a key that cannot be mistaken for a gate. "
            "density_v2 contributes NOTHING to any verdict in any family."
        ),
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "density_v2_defect": DENSITY_V2_DEFECT,
        "density_v2_status": DENSITY_V2_STATUS,
        # --- C ---
        "detection_power_by_criterion": power_by_criterion,
        "detection_power_one_sided": one_sided_power,
        "detection_power_two_sided": two_sided_power,
        "detection_power_correction": (
            "R0255's per_floor table gave BOTH two-sided purity bands the one-sided "
            f"figure {R0255_PUBLISHED['one_sided_power_published_beside_both_bands']!r} "
            "at -2 sigma while pairing it with the correctly two-sided false-fail "
            "rate. A two-sided criterion judged by one-sided power overstates its "
            "sensitivity. The ffr floor is one-sided and keeps the one-sided power; "
            "the two purity criteria are bands and take the two-sided power."
        ),
        "two_sided_power_definition": (
            "E[Phi(L + delta) + 1 - Phi(U + delta)] over the same 4,000,000 "
            "calibration families, with L = centre - k2*scale and U = centre + "
            "k2*scale. At delta = 0 it must equal the two-sided false-fail rate, "
            "which is checked."
        ),
        "two_sided_size_check": two_sided_size,
        "panel_false_alarm_rate": panel_far,
        "tolerance_content": TOLERANCE_CONTENT,
        "tolerance_confidence": TOLERANCE_CONFIDENCE,
        # --- A ---
        "independence_control": independence,
        "independence_positive_controls": positive_controls,
        "independence_correction": (
            "R0255's arms 1 and 2 built `perturbed` and never passed it: both the "
            "baseline and the 'perturbed' fit were the identical call "
            "_fit(series, log_series) on the unperturbed family. The arms could not "
            "fail and would have reported true for a fit that read held-out cells. "
            "Here every arm passes its perturbed cells through the builder into the "
            "fit, and three planted builders that DO select a held-out cell are "
            "verified to make the arms report false."
        ),
        "arm_3_correction": arm3_correction,
        # --- E ---
        "unresolved": [dict(item) for item in UNRESOLVED],
        "exact_family_seeds": list(EXACT_FAMILY_SEEDS),
        "held_out_cells": [cell["cell_id"] for cell in held_out_cells],
        "sources": {
            "r0223_cuvs_comparison": expected_input_signature(
                _bound_path(job, "r0223_comparison", label="R0223")
            ),
            "r0228_cluster_spill_comparison": expected_input_signature(
                _bound_path(job, "r0228_comparison", label="R0228")
            ),
            "r0255_panel_n29": expected_input_signature(
                _bound_path(job, "panel_n29", label="R0255 panel")
            ),
            "r0255_calibrated_madn_gate_n29": expected_input_signature(
                _bound_path(job, "r0255_gate", label="R0255 gate")
            ),
        },
        "supersedes_capability": SUPERSEDES_CAPABILITY,
        "supersession_note": (
            "supersedes the R0255 gate ARTIFACT because its independence control, "
            "its band power, its panel false-alarm rate and its registry were "
            "wrong. It supersedes no FLOOR: every floor and band here is bitwise "
            "identical to R0255's, and every criterion R0225, R0231, R0234 and "
            "R0250 registered remains in force alongside this one."
        ),
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": bool(cuda_mappings),
        "cuda_libraries_mapped_into_this_process": cuda_mappings,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    }
    _seal(output, "minilm-calibrated-madn-floors-n29-repaired.json", body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "n": N_EXACT,
        "calibration_wall_s": calibration_wall_s,
        "widest_gap_s": gaps.get("widest_gap_s"),
        "no_floor_moved": execution_checks["no_floor_moved"],
        "the_fit_is_independent_of_every_held_out_cell": execution_checks[
            "the_fit_is_independent_of_every_held_out_cell"
        ],
        "planted_defects_caught": positive_controls["planted_defects_caught"],
        "two_sided_power_minus_2_sigma": two_sided_power["minus_2_sigma"],
        "panel_false_alarm_rate": panel_far[
            "panel_false_alarm_rate_under_independence"
        ],
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0256 round0256_nodes.run_job")
    action = str(job.get("action") or "")
    if action == REPAIR_ACTION:
        run_repair(active, job)
        return
    raise Round0256RepairError(f"R0256 unknown action {action!r}")


__all__ = ["REPAIR_ACTION", "run_job", "run_repair"]
