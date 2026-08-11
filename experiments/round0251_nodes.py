#!/usr/bin/env python3
"""R0251 node handlers — three nodes, one per open item review-0250-01 named.

* `rescore_seed42_0251` (GPU) — §A. Rescore R0218's archived seed-42 checkpoint
  on this release against the same frozen panel and compare every value, then
  recompute the prior-13 vs new-3 shift test on the sealed sixteen-cell panel.
  Settles whether the map-side scorer drifted across four scoring releases.
* `estimator_table_0251` (CPU) — §B. Re-derive all six candidates at `n = 16`
  from the Gaussian null, refit each one's floors and bands on the sixteen
  cells, attach the `c8-seed42` column, and check every re-derived value against
  R0250's sealed artifact at tolerance `0`. Registers nothing.
* `trainsetup_0251` (GPU) — §C. Three arms on the release trainer with the new
  `basemap/pumap/` abort-poll hook: setup unpolled, setup polled, and a long
  rung whose whole per-batch gap series feeds the tail models.

Every node scores a real `AbortPollGate` and publishes the score whether it
holds or refuses, on R0250's reporting path: a refusal here IS the measurement.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import resource
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0234_calibration as calibration
from basemap.round0217_minilm_2m_pipeline import (
    MiniLMHostFp32EndpointArray,
    MiniLMMixedTrainingInput,
)
from basemap.round0217_minilm_2m_seed_family import (
    BATCH_SIZE,
    WARMUP_SUCCESSFUL_UPDATES,
    performance_windows,
)
from basemap.round0238_rung5 import json_safe
from basemap.round0242_locality import io_counters, json_scrub
from basemap.round0245_guard import EnforcedHostWatchdog
from basemap.round0246_guard import (
    AbortPollGate,
    Round0246Error,
    measured_slope_from_trace,
    require_abort_flag_landed,
    require_enforcement_evidence,
    require_live_sampler,
)
from basemap.round0247_registry import (
    Round0247Error,
    registered_bounds,
    registry_fingerprint,
    verify_registry,
)
from basemap.round0250_gate_n16 import (
    CANDIDATE_ORDER,
    GATED_METRICS,
    METRICS,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    band_at,
    centre_and_scale,
    floor_at,
)
from basemap.round0250_panel_n16 import (
    CENTROID_KS,
    CORPUS_SLUGS,
    DIMENSION,
    PANEL_CAPABILITY,
    PANEL_METRICS,
    POOLED_SEEDS,
    REFERENCE_MISMATCH_MESSAGE,
    ROWS,
    corpus_ffr_view,
    panel_execution_ok,
    panel_metric_view,
    raw_purity_ratios,
)
from basemap.round0250_seed_extension_n16 import (
    GRAPH_CAPABILITY,
    SEALED_DIRECTED_EDGES,
)
from basemap.round0250_trainer_loops import (
    Round0250TrainerLoopError,
    SHORT_RUNG_CONFIG_SCHEMA,
    short_rung_config,
)
from basemap.round0251_estimator_table import (
    COUPLING_CELL_CLUSTERS,
    COUPLING_CELL_ID,
    COUPLING_CELL_SEED,
    COUPLING_METRIC,
    COUPLING_STATEMENT,
    NOT_INDEPENDENT_EVIDENCE,
    PRE_REGISTERED_CRITERIA,
    REGISTERS_NOTHING,
    Round0251TableError,
    SECOND_CELL_SEED,
    TABLE_CAPABILITY,
    TABLE_SCHEMA,
    dominance,
    held_out_cell,
    joint_table,
    reproduction_checks,
)
from basemap.round0251_rescore import (
    RESCORED_SEED,
    RESCORE_CAPABILITY,
    RESCORE_SCHEMA,
    Round0251RescoreError,
    compare_rescore,
    poolability_verdict,
    sealed_seed42_cell,
    shift_test,
)
from basemap.round0251_trainer_setup import (
    BATCH_POLL_SITE,
    DECLARED_TRAINER_POLL_SITES,
    NODE_SETUP_SITES,
    NOT_A_FAMILY_CELL,
    PollRecorder,
    Round0251SetupError,
    SETUP_CAPABILITY,
    SETUP_RUNG_UPDATES,
    SETUP_SCHEMA,
    TAIL_RUNG_UPDATES,
    declared_sites_match_the_release,
    fit_setup_gap,
    hash_bound_setup_projection,
    phase_report,
    setup_reduction,
    tail_model,
    tail_verdict,
)
from experiments import round0218_nodes
from experiments.round0113_nodes import _new_model
from experiments.round0230_nodes import _open_substrate, _sealed_graph
from experiments.round0238_nodes import _check_runner_abort
from experiments.round0245_nodes import require_enforceable_abort_flag
from experiments.round0250_nodes import (
    NODE_ANON_BUDGET_BYTES,
    SAFETY_NOTE,
    _bound_path,
    _exact_cell_id,
    _load_centroids,
    _sealed_panel_evidence,
)


ROUND_ID = "0251"

RESCORE_ACTION = "round0251_rescore_seed42"
TABLE_ACTION = "round0251_estimator_table"
TRAINSETUP_ACTION = "round0251_trainer_setup"
ACTIONS: tuple[str, ...] = (RESCORE_ACTION, TABLE_ACTION, TRAINSETUP_ACTION)

ARM_SETUP_UNPOLLED = "setup_unpolled"
ARM_SETUP_POLLED = "setup_polled"
ARM_TAIL = "tail"
TRAINER_ARMS: tuple[str, ...] = (ARM_SETUP_UNPOLLED, ARM_SETUP_POLLED, ARM_TAIL)


class Round0251Error(RuntimeError):
    """The registered R0251 node contract changed."""


# --------------------------------------------------------------------------- #
# shared node scaffolding (R0250's, called rather than re-typed)
# --------------------------------------------------------------------------- #


def _start_node(label: str) -> dict[str, Any]:
    verify_registry(label=label)
    return require_enforceable_abort_flag(label=label)


def _node_guard(label: str, *, interval_s: float = 0.05) -> EnforcedHostWatchdog:
    return EnforcedHostWatchdog(
        anonymous_budget_bytes=NODE_ANON_BUDGET_BYTES,
        interval_s=float(interval_s),
        label=label,
    )


def _node_gate(label: str, *, training_performed: bool) -> AbortPollGate:
    return AbortPollGate(
        inner=_check_runner_abort,
        headroom_bytes=int(
            registered_bounds(["max_declared_headroom_bytes"])[
                "registered_max_declared_headroom_bytes"
            ]
        ),
        label=label,
        training_performed=bool(training_performed),
    )


def _guard_tail_reported(
    watchdog: EnforcedHostWatchdog, *, label: str
) -> dict[str, Any]:
    receipt = watchdog.receipt()
    tail: dict[str, Any] = {"host_watchdog": receipt}
    for key, gate in (
        ("sampler_liveness", require_live_sampler),
        ("abort_flag_landing", require_abort_flag_landed),
    ):
        try:
            tail[key] = gate(receipt, label=label)
        except (Round0246Error, Round0247Error) as error:
            tail[key] = {
                "holds": False,
                "raised": f"{type(error).__name__}: {error}",
                "reported_rather_than_aborting": True,
            }
    return tail


def _score_gate_without_raising(
    gate: AbortPollGate, tail: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """R0250's reporting path: publish the score, then publish the refusal."""
    slope = measured_slope_from_trace(
        tail["host_watchdog"]["anonymous_trace_by_second"]
    )
    verdict = gate.verdict(measured_slope_bytes_per_s=slope)
    outcome: dict[str, Any] = {
        "measured_slope_bytes_per_s": slope,
        "require_raised": False,
        "require_error": None,
    }
    try:
        required = gate.require(measured_slope_bytes_per_s=slope)
    except (Round0246Error, Round0247Error) as error:
        outcome["require_raised"] = True
        outcome["require_error"] = f"{type(error).__name__}: {error}"
        required = dict(verdict)
        required["failures"] = [
            arm
            for arm in (
                "meets_the_registered_ceiling",
                "meets_the_required_poll_count",
                "measured_spacing_is_non_zero",
            )
            if verdict.get(arm) is False
        ]
        required["holds"] = not required["failures"]
    try:
        required["enforcement_evidence"] = require_enforcement_evidence(
            required, label=label
        )
    except (Round0246Error, Round0247Error) as error:
        required["enforcement_evidence"] = {
            "holds": False,
            "error": f"{type(error).__name__}: {error}",
        }
    required["outcome"] = outcome
    return required


def _receipt_envelope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "release_sha": str(manifest["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": False,
    }


def _seal(output: str, name: str, body: Mapping[str, Any]) -> None:
    atomic_write_new_json(
        os.path.join(output, name),
        prompt_contract.seal(json_safe(json_scrub(dict(body)))),
        immutable=True,
    )


THE_INSTRUMENT_IS_DEFEATABLE = (
    "review-0249-01 §B.1 and §B.2 are OPEN and the owner has stopped guard work, "
    "so this receipt names where it rests on them rather than reusing them "
    "quietly. Every ceiling verdict in this artifact is an AbortPollGate verdict, "
    "and `AbortPollGate.max_gap_s` is a plain mutable attribute that `verdict()` "
    "reads: one assignment produces a PASSING receipt indistinguishable from an "
    "honest one, with no override record, no waived arm and no declared/effective "
    "pair for a reviewer to notice. The `EnforcedHostWatchdog` beside it has zero "
    "read-only properties. R0251 mutates no gate or watchdog attribute and its "
    "own gap series is recorded by a separate recorder whose numbers a reviewer "
    "can cross-check against the gate's, but a PASSING verdict from this "
    "instrument does not by itself license an unattended multi-hour node."
)


# --------------------------------------------------------------------------- #
# A. the seed-42 rescore
# --------------------------------------------------------------------------- #


def _authenticate_seed42(cell: Mapping[str, Any], sealed: Mapping[str, Any]) -> str:
    """R0217's archived seed-42 checkpoint, verified against the sealed bytes."""
    receipt = prompt_contract.read_sealed(
        prompt_contract.verify_signature(
            dict(cell["train_receipt"]), label="R0217 seed-42 train receipt"
        ),
        label="R0217 seed-42 train receipt",
    )
    if (
        int(receipt.get("training_seed", -1)) != RESCORED_SEED
        or receipt.get("training_performed") is not True
        or int(receipt.get("rows", -1)) != ROWS
        or int(receipt.get("dimension", -1)) != DIMENSION
        or int(receipt.get("directed_edges", -1)) != SEALED_DIRECTED_EDGES
        or receipt.get("graph_capability") != GRAPH_CAPABILITY
    ):
        raise Round0251RescoreError("R0217 seed-42 train receipt contract changed")
    if (
        dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"])
        or dict(receipt.get("graph_manifest") or {})
        != dict(sealed["manifest_signature"])
    ):
        raise Round0251RescoreError(
            "R0217 seed-42 was not trained on the substrate this panel scores"
        )
    if dict(receipt["model"]) != dict(cell["model"]):
        raise Round0251RescoreError(
            "R0218's published seed-42 checkpoint signature is not its train "
            "receipt's"
        )
    return prompt_contract.verify_signature(
        dict(receipt["model"]), label="R0217 seed-42 published map"
    )


def run_rescore(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import (
        hiD_reference_key,
        load_hiD_reference,
        reset_process_cuda_peak,
        sample_anchors,
        score_panel,
    )
    from basemap.pumap.parametric_umap import ParametricUMAP

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0251RescoreError("R0251 rescore handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0251RescoreError("R0251 map-side rescore requires CUDA")
    started = time.monotonic()
    label = "R0251 seed-42 map-side rescore"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0251 rescore")

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = _sealed_panel_evidence(job)
    pooled = prompt_contract.read_sealed(
        _bound_path(job, "panel_n16", label="R0250 sealed sixteen-cell panel"),
        label="R0250 sealed sixteen-cell panel",
    )
    target = sealed_seed42_cell(panel_evidence, pooled_panel=pooled)
    model_path = _authenticate_seed42(
        dict(panel_evidence["cells"][str(RESCORED_SEED)]), sealed
    )

    guard = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    reset_process_cuda_peak()
    with guard:
        gate.start()
        cfg = prompt_contract.panel_config()
        centroids, centroid_signatures = _load_centroids(panel_evidence)
        gate("R0251 rescore centroids loaded")
        data_identity = {
            "kind": "ordered_array",
            "shape": [ROWS, DIMENSION],
            "dtype": np.dtype("<f4").str,
            "sha256": sealed["ordered_substrate_sha256"],
        }
        reference_identity = {
            "data_identity": data_identity,
            "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        }
        reference_signature = dict(panel_evidence["shared_high_d_reference"])
        reference_path = prompt_contract.verify_signature(
            reference_signature, label="R0218 shared high-D reference"
        )
        if expected_input_signature(reference_path) != reference_signature:
            raise Round0251RescoreError(
                f"{REFERENCE_MISMATCH_MESSAGE} file signature drift"
            )
        reference = load_hiD_reference(reference_path)
        anchors = sample_anchors(ROWS, cfg)
        gate("R0251 rescore reference loaded")
        if not np.array_equal(
            np.asarray(anchors, dtype=np.int64),
            np.asarray(reference["anchor_ids"], dtype=np.int64),
        ):
            raise Round0251RescoreError(f"{REFERENCE_MISMATCH_MESSAGE} anchor drift")
        rederived_key, _parts = hiD_reference_key(
            source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
        )
        if str(rederived_key) != str(reference["key"]):
            raise Round0251RescoreError(
                f"{REFERENCE_MISMATCH_MESSAGE} re-derived key drift"
            )
        anchor_labels = round0218_nodes._anchor_corpus_labels(corpus_of_row, anchors)
        gate("R0251 rescore reference key re-derived")

        model = ParametricUMAP.load(model_path, device="cuda")
        coordinates = np.asarray(
            model.transform(source, batch_size=8192), dtype=np.float32
        )
        gate("R0251 rescore transform complete")
        if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
            raise Round0251RescoreError(
                "R0251 seed-42 transform is not a finite 2M x 2 array"
            )
        coordinates_path = os.path.join(output, f"coordinates-seed{RESCORED_SEED}.npy")
        atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
        coordinates_signature = expected_input_signature(coordinates_path)
        panel = score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            ffr_group_labels=anchor_labels,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "seed": RESCORED_SEED,
                "capability": str(target["capability"]),
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
                "reference_source_round": "0218",
                "rescore_of_round": str(target["source_round"]),
            },
        )
        gate("R0251 rescore panel scored")
        if not panel_execution_ok(panel):
            raise Round0251RescoreError("R0251 seed-42 rescored panel is collapsed")
        if not bool(panel["provenance"]["hiD_reference_reused"]):
            raise Round0251RescoreError(
                f"{REFERENCE_MISMATCH_MESSAGE} seed-42 recomputed the reference"
            )
        observed_coordinates_sha256 = ordered_array_sha256(coordinates)
        del model, coordinates
        torch.cuda.empty_cache()
        gc.collect()
        gate.finish("R0251 rescore stage end")
    tail = _guard_tail_reported(guard, label=label)
    scored = _score_gate_without_raising(gate, tail, label=label)

    comparison = compare_rescore(
        sealed=target,
        observed_panel_metrics=panel_metric_view(panel),
        observed_ratios=raw_purity_ratios(panel),
        observed_hi_d_agreement={
            key: float(panel["purity_numerators"][key]["hi_D_agreement"])
            for key in ("k256", "k1024")
        },
        observed_corpus_ffr=corpus_ffr_view(panel),
        observed_coordinates_sha256=observed_coordinates_sha256,
    )
    shift = shift_test(
        panel_metric_cells=dict(pooled["panel_metric_cells"]),
        raw_purity_ratios=dict(pooled["raw_purity_ratios"]),
    )
    verdict = poolability_verdict(rescore=comparison, shift=shift)

    execution_checks = {
        "the_reference_was_reused_not_recomputed": bool(
            panel["provenance"]["hiD_reference_reused"]
        ),
        "the_rescored_cell_is_seed_42": int(RESCORED_SEED) == 42,
        "sixteen_cells_in_the_shift_test": (
            len(shift["prior_seeds"]) + len(shift["new_seeds"]) == len(POOLED_SEEDS)
        ),
        "every_comparable_value_was_compared": comparison["values_compared"] > 0,
        "the_transform_covered_every_row": True,
    }
    if not all(execution_checks.values()):
        raise Round0251RescoreError(
            f"R0251 rescore execution checks failed: {execution_checks}"
        )

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": RESCORE_SCHEMA,
        "capability": RESCORE_CAPABILITY,
        "capabilities": [RESCORE_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "rescored_seed": RESCORED_SEED,
        "rescored_model": dict(target["model"]),
        "r0218_sealed_cell": target,
        "rescored_panel": panel,
        "rescored_coordinates": coordinates_signature,
        "rescored_coordinates_ordered_sha256": observed_coordinates_sha256,
        "comparison": comparison,
        "shift_test": shift,
        "poolability": verdict,
        "enforcement_poll_spacing": scored,
        "guard_tail": tail,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "sources": {
            "r0218_panel": panel_signature,
            "r0250_panel_n16": expected_input_signature(
                _bound_path(job, "panel_n16", label="R0250 panel")
            ),
            "centroids": centroid_signatures,
            "shared_high_d_reference": reference_signature,
        },
        "training_performed": False,
        "gate_registered": False,
        "map_decision_made": False,
        "map_quality_claimed": False,
        "published_a_map": False,
        "execution_checks": execution_checks,
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 ** 2),
        "performance": {
            "total_wall_s": time.monotonic() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
    })
    _seal(output, "seed42-map-side-rescore.json", body)
    print(json.dumps({
        "capability": RESCORE_CAPABILITY,
        "the_map_side_scorer_reproduces": comparison["the_map_side_scorer_reproduces"],
        "values_drifted": comparison["values_drifted"],
    }))


# --------------------------------------------------------------------------- #
# B. the six-candidate joint table
# --------------------------------------------------------------------------- #


def build_joint_table(
    *,
    pooled: Mapping[str, Any],
    r0228: Mapping[str, Any],
    r0234: Mapping[str, Any],
    r0250_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-derive at 16 from the null, refit on the cells, attach the coupling."""
    from experiments.round0250_nodes import calibrate_at_n16, evaluate_selection_n16

    calibrated = calibrate_at_n16(
        list(r0234["power_parity_ladder"]["ladder"]),
        dict(r0234["selection"]["candidates"]),
    )
    calibrated.pop("arrays", None)

    exact_cells = dict(pooled["panel_metric_cells"])
    exact_ratios = dict(pooled["raw_purity_ratios"])
    seeds = [str(seed) for seed in POOLED_SEEDS]
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
    selection = evaluate_selection_n16(
        calibrated=calibrated, series=series, log_series=log_series
    )

    fitted: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        item = selection["candidates"][name]
        k_one = float(item["calibrated_one_sided_multiplier"])
        k_two = float(item["calibrated_two_sided_multiplier"])
        floors = {metric: floor_at(name, series[metric], k_one) for metric in METRICS}
        bands = {
            metric: band_at(name, log_series[metric], k_two)
            for metric in PURITY_METRICS
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
            "two_sided_ratio_bands": {
                metric: {
                    "ratio_lower": math.exp(bands[metric][0]),
                    "ratio_upper": math.exp(bands[metric][1]),
                    "log_lower": bands[metric][0],
                    "log_upper": bands[metric][1],
                    "log_centre": log_centres[metric]["centre"],
                    "log_scale": log_centres[metric]["scale"],
                }
                for metric in PURITY_METRICS
            },
            "scale_over_sample_sd": {
                metric: centre_and_scale(name, series[metric])[1]
                / statistics.stdev(series[metric])
                for metric in METRICS
            },
        }

    coupling_cell = held_out_cell(
        r0228, clusters=COUPLING_CELL_CLUSTERS, seed=COUPLING_CELL_SEED
    )
    second = held_out_cell(
        r0228, clusters=COUPLING_CELL_CLUSTERS, seed=SECOND_CELL_SEED
    )
    table = joint_table(
        selection=selection,
        fitted=fitted,
        n13_selection=dict(r0234["selection"]["candidates"]),
        coupling_cell=coupling_cell,
        second_cell=second,
    )
    checks = reproduction_checks(table=table, r0250_gate=r0250_gate)
    if not all(item["reproduced"] for item in checks):
        raise Round0251TableError(
            "R0251's re-derivation does not reproduce R0250's sealed values: "
            f"{[item for item in checks if not item['reproduced']]}"
        )
    return {
        "calibration": calibrated,
        "selection": selection,
        "fitted_candidates": fitted,
        "coupling_cell_source": coupling_cell,
        "second_cell_source": second,
        "table": table,
        "dominance": dominance(table),
        "reproduction_checks": checks,
        "reproduction_checks_all_passed": True,
    }


def run_table(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0251TableError("R0251 table handler received another queue")
    started = time.monotonic()
    label = "R0251 estimator coupling joint table"
    verify_registry(label=label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0251 table")

    pooled = prompt_contract.read_sealed(
        _bound_path(job, "panel_n16", label="R0250 sealed sixteen-cell panel"),
        label="R0250 sealed sixteen-cell panel",
    )
    r0228 = prompt_contract.read_sealed(
        _bound_path(job, "r0228_comparison", label="R0228 sealed cluster-spill"),
        label="R0228 sealed cluster-spill comparison",
    )
    r0234 = prompt_contract.read_sealed(
        _bound_path(job, "r0234_gate", label="R0234 sealed calibrated gate"),
        label="R0234 sealed calibrated gate",
    )
    r0250 = prompt_contract.read_sealed(
        _bound_path(job, "r0250_gate", label="R0250 sealed n=16 gate"),
        label="R0250 sealed n=16 gate",
    )
    built = build_joint_table(pooled=pooled, r0228=r0228, r0234=r0234, r0250_gate=r0250)
    table = built["table"]

    failing = set(table["candidates_failing_the_coupling_cell"])
    passing = set(table["candidates_passing_the_coupling_cell"])
    execution_checks = {
        "six_candidates_in_the_table": len(table["rows"]) == len(CANDIDATE_ORDER),
        "every_row_carries_the_coupling_column": all(
            "fails_the_coupling_cell" in row for row in table["rows"]
        ),
        "the_coupling_column_is_not_unanimous": bool(failing and passing),
        "every_rederived_value_matches_r0250": bool(
            built["reproduction_checks_all_passed"]
        ),
        "this_round_registers_nothing": (
            table["registers_nothing"] == REGISTERS_NOTHING
        ),
        "the_dominance_test_excludes_the_coupling_column": all(
            item["criterion"] not in {"fails_the_coupling_cell", "fails_the_second_cell"}
            for item in built["dominance"]["pre_registered_criteria"]
        ),
    }
    if not all(execution_checks.values()):
        raise Round0251TableError(
            f"R0251 table execution checks failed: {execution_checks}"
        )

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": TABLE_SCHEMA,
        "capability": TABLE_CAPABILITY,
        "capabilities": [TABLE_CAPABILITY],
        "n": len(POOLED_SEEDS),
        "joint_table": table,
        "dominance": built["dominance"],
        "pre_registered_criteria": [dict(item) for item in PRE_REGISTERED_CRITERIA],
        "selection_rederived_at_n16": built["selection"],
        "fitted_candidates": built["fitted_candidates"],
        "calibration": built["calibration"],
        "coupling_cell_source": built["coupling_cell_source"],
        "second_cell_source": built["second_cell_source"],
        "reproduction_checks": built["reproduction_checks"],
        "coupling_statement": COUPLING_STATEMENT,
        "the_c8_seed42_reproduction_is_not_independent_evidence": (
            NOT_INDEPENDENT_EVIDENCE
        ),
        "registers_nothing": REGISTERS_NOTHING,
        "gate_registered": False,
        "gate_status": "registers-nothing",
        "supersedes_capability": None,
        "chosen_estimator": None,
        "the_choice_is_the_owners": (
            "R0251 does not pick. The table is the artifact; the selection is "
            "not made here and no floor in this round is registered, contingent "
            "or otherwise."
        ),
        "sources": {
            "r0250_panel_n16": expected_input_signature(
                _bound_path(job, "panel_n16", label="R0250 panel")
            ),
            "r0228_cluster_spill_comparison": expected_input_signature(
                _bound_path(job, "r0228_comparison", label="R0228")
            ),
            "r0234_calibrated_gate_n13": expected_input_signature(
                _bound_path(job, "r0234_gate", label="R0234")
            ),
            "r0250_calibrated_gate_n16": expected_input_signature(
                _bound_path(job, "r0250_gate", label="R0250")
            ),
        },
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": False,
        "production_or_publishing": False,
        "execution_checks": execution_checks,
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 ** 2),
        "wall_seconds": time.monotonic() - started,
    })
    _seal(output, "estimator-coupling-joint-table-n16.json", body)
    print(json.dumps({
        "capability": TABLE_CAPABILITY,
        "fails_c8_seed42": table["candidates_failing_the_coupling_cell"],
        "passes_c8_seed42": table["candidates_passing_the_coupling_cell"],
        "dominating": built["dominance"]["dominating_candidates"],
    }))


# --------------------------------------------------------------------------- #
# C. the trainer's setup and its batch tail
# --------------------------------------------------------------------------- #


def _evict_page_cache(paths: Sequence[str]) -> dict[str, Any]:
    """Ask the kernel to drop these files' clean pages. No root, no side effects.

    `POSIX_FADV_DONTNEED` on a read-only file is advisory and touches no bytes;
    it is the only way this round can produce a genuinely cold setup twice in
    one process without a reboot.
    """
    dropped: list[dict[str, Any]] = []
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        dropped.append({"path": path, "bytes": os.path.getsize(path)})
    return {
        "method": "posix_fadvise(POSIX_FADV_DONTNEED)",
        "files": dropped,
        "note": (
            "advisory. The kernel may retain pages another mapping still "
            "references, so 'cold' here means 'asked to be cold', and the "
            "measured setup is a lower bound on a genuinely cold one."
        ),
    }


class _FilteredPoll:
    """A poll that ignores the setup sites. Reproduces R0250's arm exactly.

    R0250 could only install a poll at `_low_dim_qs`, which is unreachable until
    the first batch is already running. Installing the release hook and dropping
    every setup site reproduces that situation on this release, so the two setup
    arms differ in exactly one thing: whether the setup reads happen.
    """

    def __init__(self, *, inner: Any) -> None:
        self._inner = inner
        self.suppressed = 0

    def __call__(self, where: str) -> None:
        if str(where) != BATCH_POLL_SITE:
            self.suppressed += 1
            return
        self._inner(where)


def _run_trainer_arm(
    *,
    arm: str,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    updates: int,
    gate: AbortPollGate,
    evict: bool,
) -> dict[str, Any]:
    """One fit with the release abort-poll hook installed, gate and series live.

    The measured stage starts BEFORE the graph is verified and loaded. R0250's
    arms started after, so the graph load — the term that dominates setup at
    100M — never entered a measured interval at all.
    """
    import torch

    recorder = PollRecorder(gate=gate, clock=time.monotonic)
    eviction = (
        _evict_page_cache([
            str(dict(job["graph_manifest_signature"])["canonical_path"]),
        ])
        if evict
        else {"method": "none", "files": [], "note": "page cache left as found"}
    )
    recorder.anchor(NODE_SETUP_SITES[0])
    started = time.monotonic()

    graph = _sealed_graph(job)
    recorder(NODE_SETUP_SITES[1])
    source, substrate_signature = _open_substrate(graph)
    recorder(NODE_SETUP_SITES[2])

    seed = int(config["seed"])
    dataset = MiniLMHostFp32EndpointArray(
        source, source_signature=substrate_signature, buffer_rows=BATCH_SIZE
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph, seed=seed)
    recorder(NODE_SETUP_SITES[3])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = min(WARMUP_SUCCESSFUL_UPDATES, max(updates // 4, 1))
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True

    filtered = _FilteredPoll(inner=recorder) if arm == ARM_SETUP_UNPOLLED else None
    model.abort_poll = filtered if filtered is not None else recorder
    try:
        model.fit(
            wrapper,
            low_memory=True,
            verbose=False,
            n_processes=6,
            random_state=seed,
            resample_negatives=False,
            precomputed_edges_path=graph["signature"]["canonical_path"],
            use_wandb=False,
        )
    finally:
        model.abort_poll = None
    wall = time.monotonic() - started

    accounting = dict(model._train_stats)
    if int(accounting.get("optimizer_steps_succeeded", -1)) != updates:
        raise Round0251SetupError(
            f"R0251 arm {arm} closed at {accounting.get('optimizer_steps_succeeded')!r} "
            f"updates, registered {updates}"
        )
    if recorder.batches != updates:
        raise Round0251SetupError(
            f"R0251 arm {arm} recorded {recorder.batches} batch reads against "
            f"{updates} updates: the per-batch poll is not one per batch"
        )
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (updates - model._bench_warmup) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
    }
    phases = phase_report(recorder.records, arm=arm)
    del model, wrapper, dataset, source, graph
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "arm": arm,
        "updates": updates,
        "page_cache": eviction,
        "stage_wall_s": wall,
        "fit_setup_seconds_reported_by_the_trainer": profiler.get("setup_seconds"),
        "steady_updates_per_s": rate,
        "train_accounting": accounting,
        "memory": memory,
        "poll_recorder": recorder.receipt(),
        "setup_reads_suppressed": (
            int(filtered.suppressed) if filtered is not None else 0
        ),
        "phase_report": phases,
        "batch_gaps": list(recorder.batch_gaps),
        "is_a_family_cell": False,
        "not_a_family_cell_note": NOT_A_FAMILY_CELL,
    }


def run_trainsetup(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0251SetupError("R0251 trainsetup handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0251SetupError("R0251 trainer setup measurement requires CUDA")
    started = time.monotonic()
    label = "R0251 trainer setup and batch tail"
    abort_flag = _start_node(label)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0251 trainsetup")

    sites = declared_sites_match_the_release()

    # The CUDA context is a one-time per-process setup cost. It is measured here
    # so it lands in exactly one arm's stage by accident rather than silently.
    cuda_t0 = time.monotonic()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    cuda_init_s = time.monotonic() - cuda_t0

    template_graph = _sealed_graph(job)
    template_source, template_signature = _open_substrate(template_graph)
    from basemap.round0217_minilm_2m_seed_family import train_config as r0217_train_config

    template, _sha = r0217_train_config(
        seed=int(job["template_seed"]),
        graph_signature=template_graph["signature"],
        graph_manifest_signature=template_graph["manifest_signature"],
        substrate_signature=template_signature,
        graph_edges=template_graph["directed_edges"],
        rows=ROWS,
    )
    del template_graph, template_source
    gc.collect()

    setup_updates = int(job.get("setup_rung_updates", SETUP_RUNG_UPDATES))
    tail_updates = int(job.get("tail_rung_updates", TAIL_RUNG_UPDATES))
    setup_config, setup_sha, setup_identity = short_rung_config(
        template, updates=setup_updates
    )
    tail_config, tail_sha, tail_identity = short_rung_config(
        template, updates=tail_updates
    )
    atomic_write_new_json(
        os.path.join(output, "short-rung-configs.json"),
        {
            "round_id": ROUND_ID,
            "capability": SETUP_CAPABILITY,
            "schema": SHORT_RUNG_CONFIG_SCHEMA,
            "setup_rung": {"config": setup_config, "sha256": setup_sha,
                           "identity": setup_identity},
            "tail_rung": {"config": tail_config, "sha256": tail_sha,
                          "identity": tail_identity},
        },
        immutable=True,
    )

    io_before = io_counters()
    arms: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    tails: dict[str, Any] = {}
    plan = (
        (ARM_SETUP_UNPOLLED, setup_config, setup_updates, True),
        (ARM_SETUP_POLLED, setup_config, setup_updates, True),
        (ARM_TAIL, tail_config, tail_updates, False),
    )
    for arm, config, updates, evict in plan:
        guard = _node_guard(f"{label} {arm}")
        gate = _node_gate(f"{label} {arm}", training_performed=True)
        with guard:
            gate.start()
            arms[arm] = _run_trainer_arm(
                arm=arm, config=config, job=job, updates=updates, gate=gate,
                evict=evict,
            )
            guard.poll(f"R0251 {arm} fit complete")
            gate.finish(f"R0251 {arm} stage end")
        tails[arm] = _guard_tail_reported(guard, label=f"{label} {arm}")
        gates[arm] = _score_gate_without_raising(
            gate, tails[arm], label=f"{label} {arm}"
        )
    io_after = io_counters()

    reduction = setup_reduction(
        before_gap_s=float(arms[ARM_SETUP_UNPOLLED]["phase_report"]["widest_setup_gap_s"]),
        after_gap_s=float(arms[ARM_SETUP_POLLED]["phase_report"]["widest_setup_gap_s"]),
    )
    fit_gaps = {
        arm: fit_setup_gap(arms[arm]["phase_report"]) for arm in TRAINER_ARMS
    }
    fit_reduction = setup_reduction(
        before_gap_s=float(fit_gaps[ARM_SETUP_UNPOLLED]["widest_setup_gap_inside_fit_s"]),
        after_gap_s=float(fit_gaps[ARM_SETUP_POLLED]["widest_setup_gap_inside_fit_s"]),
    )
    #: The binding setup interval measured here is the node-side integrity hash
    #: of the substrate, whose cost is exactly linear in the substrate bytes.
    substrate_bytes = int(dict(job["graph_manifest_signature"]).get("bytes", 0)) or None
    substrate_entry = dict(
        prompt_contract.read_sealed(
            str(dict(job["graph_manifest_signature"])["canonical_path"]),
            label="R0216 sealed substrate+graph receipt",
        )["substrate"]
    )
    hash_projection = hash_bound_setup_projection(
        measured_gap_s=float(
            dict(arms[ARM_SETUP_POLLED]["phase_report"]["setup_gaps_by_site"])[
                NODE_SETUP_SITES[2]
            ]["widest_gap_s"]
        ),
        measured_bytes=int(substrate_entry["bytes"]),
        target_rows=int(job.get("hash_projection_target_rows", 100_000_000)),
        dimension=DIMENSION,
    )
    tail = tail_model(
        arms[ARM_TAIL]["batch_gaps"],
        arm_wall_s=float(arms[ARM_TAIL]["stage_wall_s"]),
    )
    tail_conclusion = tail_verdict(tail)
    widest_by_arm = {
        arm: {
            "setup": arms[arm]["phase_report"]["widest_setup_gap_s"],
            "steady_state": arms[arm]["phase_report"]["widest_steady_state_gap_s"],
            "across_both": arms[arm]["phase_report"]["widest_gap_across_both_phases_s"],
            "across_both_over_the_ceiling": arms[arm]["phase_report"][
                "widest_gap_across_both_phases_over_the_ceiling"
            ],
            "binding_phase": arms[arm]["phase_report"]["the_binding_phase"],
        }
        for arm in TRAINER_ARMS
    }
    worst_arm = max(
        TRAINER_ARMS,
        key=lambda name: float(widest_by_arm[name]["across_both_over_the_ceiling"]),
    )

    verdict = {
        "the_release_trainer_now_reads_the_abort_flag": True,
        "declared_poll_sites": list(DECLARED_TRAINER_POLL_SITES),
        "cuda_context_init_s": cuda_init_s,
        "widest_gap_by_arm": widest_by_arm,
        "the_widest_gap_across_setup_and_steady_state_s": float(
            widest_by_arm[worst_arm]["across_both"]
        ),
        "the_widest_gap_across_setup_and_steady_state_over_the_ceiling": float(
            widest_by_arm[worst_arm]["across_both_over_the_ceiling"]
        ),
        "the_arm_it_sits_in": worst_arm,
        "the_site_it_sits_after": arms[worst_arm]["phase_report"][
            "widest_gap_across_both_phases_after"
        ],
        "setup_reduction_whole_stage": reduction,
        "setup_reduction_inside_fit": fit_reduction,
        "widest_setup_gap_by_arm_split_by_where_it_sits": fit_gaps,
        "the_binding_setup_interval_is_inside_fit": bool(
            fit_gaps[ARM_SETUP_POLLED]["the_binding_setup_interval_is_inside_fit"]
        ),
        "setup_is_below_the_ceiling_with_margin": bool(
            reduction["setup_is_below_the_ceiling_with_the_required_margin"]
        ),
        "fit_setup_is_below_the_ceiling_with_margin": bool(
            fit_reduction["setup_is_below_the_ceiling_with_the_required_margin"]
        ),
        "hash_bound_setup_projection": hash_projection,
        "tail_model": tail,
        "every_arm_meets_the_registered_ceiling": all(
            bool(gates[arm]["meets_the_registered_ceiling"]) for arm in TRAINER_ARMS
        ),
        "this_is_a_2m_rung_not_a_100m_one": (
            "the setup measured here loads a 2M substrate and a 48,344,648-edge "
            "graph. A 100M rung's setup is larger and is NOT measured; what this "
            "round establishes is that the widest setup gap is now the largest "
            "single un-polled CALL rather than the whole setup, which is the "
            "quantity that has to be re-measured at that rung."
        ),
    }

    body = dict(_receipt_envelope(active["manifest"]))
    body.update({
        "schema": SETUP_SCHEMA,
        "capability": SETUP_CAPABILITY,
        "capabilities": [SETUP_CAPABILITY],
        "abort_flag_precondition": abort_flag,
        "declared_poll_sites": sites,
        "short_rungs": {
            "setup": setup_identity,
            "tail": tail_identity,
        },
        "short_rung_config_sha256": {"setup": setup_sha, "tail": tail_sha},
        "arms": {
            arm: {key: value for key, value in item.items() if key != "batch_gaps"}
            for arm, item in arms.items()
        },
        "batch_gap_series_summary": {
            arm: {
                "count": len(arms[arm]["batch_gaps"]),
                "min_s": min(arms[arm]["batch_gaps"]) if arms[arm]["batch_gaps"] else None,
                "median_s": (
                    statistics.median(arms[arm]["batch_gaps"])
                    if arms[arm]["batch_gaps"]
                    else None
                ),
                "max_s": max(arms[arm]["batch_gaps"]) if arms[arm]["batch_gaps"] else None,
            }
            for arm in TRAINER_ARMS
        },
        "enforcement_poll_spacing": gates,
        "guard_tails": tails,
        "setup_reduction_whole_stage": reduction,
        "setup_reduction_inside_fit": fit_reduction,
        "hash_bound_setup_projection": hash_projection,
        "tail_model": tail,
        "tail_verdict": tail_conclusion,
        "verdict": verdict,
        "the_instrument_is_defeatable": THE_INSTRUMENT_IS_DEFEATABLE,
        "training_performed": True,
        "map_decision_made": False,
        "gate_registered": False,
        "map_quality_claimed": False,
        "published_a_map": False,
        "io": {
            "read_bytes": int(io_after["read_bytes"] - io_before["read_bytes"]),
            "write_bytes": int(io_after["write_bytes"] - io_before["write_bytes"]),
        },
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 ** 2),
        "performance": {
            "total_wall_s": time.monotonic() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
    })
    _seal(output, "trainer-setup-and-batch-tail.json", body)
    np.save(
        os.path.join(output, "tail-batch-gaps.npy"),
        np.asarray(arms[ARM_TAIL]["batch_gaps"], dtype=np.float64),
    )
    print(json.dumps({
        "capability": SETUP_CAPABILITY,
        "widest_gap_over_ceiling": verdict[
            "the_widest_gap_across_setup_and_steady_state_over_the_ceiling"
        ],
        "setup_below_the_ceiling_with_margin": verdict[
            "setup_is_below_the_ceiling_with_margin"
        ],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == RESCORE_ACTION:
        run_rescore(active, job)
        return
    if action == TABLE_ACTION:
        run_table(active, job)
        return
    if action == TRAINSETUP_ACTION:
        run_trainsetup(active, job)
        return
    raise Round0251Error(f"R0251 has no handler for action {action!r}")


__all__ = [
    "ACTIONS",
    "ARM_SETUP_POLLED",
    "ARM_SETUP_UNPOLLED",
    "ARM_TAIL",
    "RESCORE_ACTION",
    "ROUND_ID",
    "Round0251Error",
    "TABLE_ACTION",
    "TRAINER_ARMS",
    "TRAINSETUP_ACTION",
    "build_joint_table",
    "run_job",
    "run_rescore",
    "run_table",
    "run_trainsetup",
]
