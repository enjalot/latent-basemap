"""Execute the conditional R0131 runtime-component localization."""
from __future__ import annotations

import gc
import json
import os
import random
import re
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import InventoryFp16Array
from basemap.round0125_runtime_bridge import (
    BATCH_SIZE,
    GRAPH_EDGES,
    MATCHED_DENSITY_FLOOR,
    ROWS,
    SEED,
    STREAM_TRACE_BATCHES,
    SUCCESSFUL_UPDATES,
    validate_environment_freeze,
    validate_seal,
)
from basemap.round0131_runtime_factorial import (
    ARMS,
    CAPABILITY,
    PAIRED_BOOTSTRAP_DRAWS,
    PAIRED_BOOTSTRAP_SEED,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    PIPELINES,
    POSITIVE_R0125_OUTCOMES,
    RESIDENT_FUSED,
    RESIDENT_SEPARATE,
    ROUND_ID,
    Round0131Error,
    Round0131TrainingInput,
    new_model,
    select_outcome,
    train_config,
)
from experiments.round0104_nodes import _load_graph as _load_r0104_graph
from experiments.round0119_nodes import _load_universe
from experiments.round0125_nodes import (
    _load_shared_exact,
    _score_matched_arm,
)


TRAIN_SCHEMA = "round0131-runtime-intermediate-train-receipt-v1"
PANEL_SCHEMA = "round0131-runtime-component-density-panel-v1"
DECISION_SCHEMA = "round0131-runtime-component-decision-v1"
TRAIN_CHECK_KEYS = {
    "exact_update_closure",
    "zero_numerical_skips",
    "no_pipeline_stamp_drift",
    "endpoint_rows_match_updates",
    "bounded_stream_trace_complete",
    "initial_model_state_stamped",
}


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    validate_seal(value, label=label)
    return value


def _arm(job: Mapping[str, Any]) -> str:
    arm = str(job.get("arm") or "")
    if arm not in ARMS:
        raise Round0131Error(f"unknown R0131 arm {arm!r}")
    return arm


def _validate_environment(active: Mapping[str, Any]) -> dict[str, Any]:
    expected = (active.get("manifest") or {}).get("environment_freeze")
    if not isinstance(expected, Mapping):
        raise Round0131Error("queue lacks its environment freeze")
    return validate_environment_freeze(expected)


def _validate_positive_trigger(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    decision_signature = dict(job.get("r0125_decision") or {})
    decision_path = str(decision_signature.get("canonical_path") or "")
    if expected_input_signature(decision_path) != decision_signature:
        raise Round0131Error("R0125 decision signature changed")
    decision = _read_sealed(decision_path, label="accepted R0125 decision")
    selector = decision.get("selector") or {}
    if (
        decision.get("schema") != "round0125-device-host-runtime-decision-v1"
        or decision.get("round_id") != "0125"
        or decision.get("outcome") not in POSITIVE_R0125_OUTCOMES
        or selector.get("outcome") != decision.get("outcome")
        or selector.get("execution_valid") is not True
        or decision.get("capabilities_produced")
        != ["jina-fineweb-2m-runtime-path-density-bridge-v1"]
        or decision.get("sampler_only_cause_claimed") is not False
        or decision.get("residency_only_cause_claimed") is not False
    ):
        raise Round0131Error("R0125 did not seal the eligible positive branch")
    return decision, decision_signature


def _pipeline_mismatches(
    expected: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }


def _train_contract(train: Mapping[str, Any], *, arm: str, release_sha: str) -> bool:
    checks = train.get("train_checks")
    return bool(
        train.get("schema") == TRAIN_SCHEMA
        and train.get("round_id") == ROUND_ID
        and train.get("arm") == arm
        and train.get("release_sha") == release_sha
        and isinstance(checks, Mapping)
        and set(checks) == TRAIN_CHECK_KEYS
        and all(checks.values())
    )


def run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    import torch

    arm = _arm(job)
    environment = _validate_environment(active)
    trigger, trigger_signature = _validate_positive_trigger(job)
    shared, shared_signature = _load_shared_exact(job)
    graph = _load_r0104_graph(shared)
    config, config_sha = train_config(
        arm,
        graph_signature=shared["graph"],
        graph_manifest_signature=shared["graph_manifest"],
        graph_edges=shared["graph_edges"],
    )
    training_input = Round0131TrainingInput(
        InventoryFp16Array(0, ROWS), graph, arm=arm, device="cuda"
    )
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0131 {arm} train output"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        _seal({
            "schema": "round0131-production-config-v1",
            "round_id": ROUND_ID,
            "arm": arm,
            "config": config,
            "config_sha256": config_sha,
        }),
        immutable=True,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = new_model(config)
    model._max_train_steps = SUCCESSFUL_UPDATES
    model._bench_warmup = PERFORMANCE_WARMUP_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = PERFORMANCE_WINDOWS
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        training_input,
        low_memory=False,
        verbose=False,
        n_processes=1,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=shared["graph"]["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started
    if training_input._sampler is not None:
        training_input._sampler.close()
    accounting = dict(model._train_stats)
    runtime = training_input.runtime_stamp()
    mismatches = _pipeline_mismatches(
        config["execution"]["expected_pipeline_stamp"], runtime
    )
    exact = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": GRAPH_EDGES,
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = SUCCESSFUL_UPDATES * BATCH_SIZE
    if (
        runtime.get("endpoint_gather_calls") != SUCCESSFUL_UPDATES
        or runtime.get("source_rows_gathered") != expected_rows
        or runtime.get("destination_rows_gathered") != expected_rows
        or runtime.get("host_prefetch_consumer_batches") != SUCCESSFUL_UPDATES
        or runtime.get("host_prefetch_producer_batches")
        not in {SUCCESSFUL_UPDATES, SUCCESSFUL_UPDATES + 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows_per_side": expected_rows,
            "runtime": runtime,
        }
    trace = runtime.get("stream_trace") or {}
    if (
        trace.get("batches_hashed") != STREAM_TRACE_BATCHES
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(trace.get(key) or "")) is None
            for key in (
                "source_endpoint_ids_sha256",
                "destination_endpoint_ids_sha256",
            )
        )
    ):
        mismatches["stream_trace"] = trace
    if model.initial_model_state_sha256 is None:
        mismatches["initial_model_state_sha256"] = None
    if mismatches:
        raise Round0131Error(f"R0131 {arm} accounting failed: {mismatches}")
    for key, value in runtime.items():
        accounting[f"pipeline_{key}"] = value
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES) / model._bench_seconds
        if model._bench_seconds else 0.0
    )
    if profiler.get("aborted") is not False or rate < config["execution"]["minimum_train_upd_s"]:
        raise Round0131Error(f"R0131 {arm} performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    checks = {
        "exact_update_closure": True,
        "zero_numerical_skips": True,
        "no_pipeline_stamp_drift": True,
        "endpoint_rows_match_updates": True,
        "bounded_stream_trace_complete": True,
        "initial_model_state_stamped": True,
    }
    receipt = _seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "r0125_positive_outcome": trigger["outcome"],
        "r0125_decision": trigger_signature,
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "causal_invariant_sha256": config["causal_invariant_sha256"],
        "initial_model_state_sha256": model.initial_model_state_sha256,
        "model": expected_input_signature(model_path),
        "shared_evidence": shared_signature,
        "graph": shared["graph"],
        "graph_manifest": shared["graph_manifest"],
        "environment_freeze_sha256": environment["freeze_sha256"],
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_seconds": wall,
        "train_checks": checks,
        "memory": {
            "device_total_bytes": int(total_bytes),
            "post_train_free_bytes": int(free_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del model, training_input, graph
    torch.cuda.empty_cache()
    gc.collect()


def _authenticate_model(
    active: Mapping[str, Any], job: Mapping[str, Any], *, arm: str
):
    config_path = os.path.join(job["train_outputs"][arm], "production-config.json")
    train_path = os.path.join(job["train_outputs"][arm], "train-receipt.json")
    config_receipt = _read_sealed(config_path, label=f"R0131 {arm} config")
    train = _read_sealed(train_path, label=f"R0131 {arm} train")
    shared, shared_signature = _load_shared_exact(job)
    config, config_sha = train_config(
        arm,
        graph_signature=shared["graph"],
        graph_manifest_signature=shared["graph_manifest"],
        graph_edges=shared["graph_edges"],
    )
    expected_environment = active["manifest"]["environment_freeze"]
    if (
        config_receipt.get("schema") != "round0131-production-config-v1"
        or config_receipt.get("round_id") != ROUND_ID
        or config_receipt.get("arm") != arm
        or config_receipt.get("config") != config
        or config_receipt.get("config_sha256") != config_sha
        or train.get("production_config") != expected_input_signature(config_path)
        or train.get("production_config_sha256") != config_sha
        or train.get("shared_evidence") != shared_signature
        or train.get("environment_freeze_sha256")
        != expected_environment["freeze_sha256"]
        or not _train_contract(
            train, arm=arm, release_sha=active["manifest"]["release_sha"]
        )
    ):
        raise Round0131Error(f"R0131 {arm} model lineage changed")
    model_path = str(train["model"]["canonical_path"])
    if expected_input_signature(model_path) != train["model"]:
        raise Round0131Error(f"R0131 {arm} model bytes changed")
    from basemap.round0125_runtime_bridge import AuditedParametricUMAP

    model = AuditedParametricUMAP.load(model_path, device="cuda")
    return model, train, expected_input_signature(train_path)


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    interval = np.percentile(values, [0.5, 99.5]).astype(np.float64)
    return float(interval[0]), float(interval[1])


def run_panel(active: dict[str, Any], job: dict[str, Any]) -> None:
    _validate_environment(active)
    trigger, trigger_signature = _validate_positive_trigger(job)
    r0125_panel_signature = dict(job["r0125_panel"])
    panel_path = str(r0125_panel_signature["canonical_path"])
    if expected_input_signature(panel_path) != r0125_panel_signature:
        raise Round0131Error("R0125 matched panel bytes changed")
    r0125_panel = _read_sealed(panel_path, label="accepted R0125 matched panel")
    r0125_arrays_signature = dict(r0125_panel.get("arrays") or {})
    if (
        r0125_panel.get("schema") != "round0125-matched-runtime-density-panel-v1"
        or r0125_panel.get("round_id") != "0125"
        or expected_input_signature(r0125_arrays_signature.get("canonical_path", ""))
        != r0125_arrays_signature
    ):
        raise Round0131Error("R0125 matched panel contract changed")
    with np.load(r0125_arrays_signature["canonical_path"], allow_pickle=False) as archive:
        inherited = {
            key: np.asarray(archive[key])
            for key in (
                "anchor_compact_rows",
                "anchor_global_rows",
                "high_radius",
                "host_control__low_radius",
                "host_control__bootstrap",
                "device_treatment__low_radius",
                "device_treatment__bootstrap",
                "paired_device_minus_host_bootstrap",
            )
        }
    (
        source,
        _representatives,
        retained_global_rows,
        anchors,
        global_rows,
        high_radius,
        lineage,
        _reference,
    ) = _load_universe(job)
    if (
        not np.array_equal(anchors, inherited["anchor_compact_rows"])
        or not np.array_equal(global_rows, inherited["anchor_global_rows"])
        or not np.array_equal(high_radius, inherited["high_radius"])
    ):
        raise Round0131Error("R0125/R0131 matched universe changed")

    cells: dict[str, Any] = {
        key: dict((r0125_panel.get("cells") or {})[key])
        for key in ("host_control", "device_treatment")
    }
    arrays: dict[str, np.ndarray] = dict(inherited)
    train_signatures: dict[str, Any] = {}
    for arm in ARMS:
        model, _train, train_signature = _authenticate_model(active, job, arm=arm)
        cell, cell_arrays = _score_matched_arm(
            arm=arm,
            model=model,
            source=source,
            retained_global_rows=retained_global_rows,
            anchors=anchors,
            high_radius=high_radius,
        )
        cell["train_receipt"] = train_signature
        cells[arm] = cell
        arrays.update(cell_arrays)
        train_signatures[arm] = train_signature
        del model
        gc.collect()

    bootstraps = {
        key: np.asarray(arrays[f"{key}__bootstrap"], dtype=np.float64)
        for key in (
            "host_control", RESIDENT_FUSED, RESIDENT_SEPARATE, "device_treatment"
        )
    }
    if any(value.shape != (PAIRED_BOOTSTRAP_DRAWS,) for value in bootstraps.values()):
        raise Round0131Error("paired bootstrap geometry changed")
    deltas = {
        "residency": bootstraps[RESIDENT_FUSED] - bootstraps["host_control"],
        "endpoint_forward": (
            bootstraps[RESIDENT_SEPARATE] - bootstraps[RESIDENT_FUSED]
        ),
        "sampler_rng_epoch": (
            bootstraps["device_treatment"] - bootstraps[RESIDENT_SEPARATE]
        ),
    }
    for key, value in deltas.items():
        arrays[f"adjacent__{key}__bootstrap"] = value
    overall = bootstraps["device_treatment"] - bootstraps["host_control"]
    if not np.array_equal(overall, arrays["paired_device_minus_host_bootstrap"]):
        raise Round0131Error("R0125 inherited overall bootstrap did not reproduce")
    correlations = {
        key: float(cells[key]["density_v2"]["correlation"])
        for key in (
            "host_control", RESIDENT_FUSED, RESIDENT_SEPARATE, "device_treatment"
        )
    }
    adjacent_ci = {key: _percentile_interval(value) for key, value in deltas.items()}
    output = create_fresh_directory(
        job["outputs"][0], label="R0131 runtime component panel"
    )
    arrays_path = os.path.join(output, "runtime-component-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = _seal({
        "schema": PANEL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "r0125_positive_outcome": trigger["outcome"],
        "r0125_decision": trigger_signature,
        "r0125_matched_panel": r0125_panel_signature,
        "r0125_matched_arrays": r0125_arrays_signature,
        "lineage": lineage,
        "universe": {
            "source_rows": len(source),
            "representative_rows": len(retained_global_rows),
            "anchors": len(anchors),
            "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
            "anchor_global_rows_sha256": ordered_array_sha256(global_rows),
            "high_radius_sha256": ordered_array_sha256(high_radius),
        },
        "scorer": {
            "metric": "density-v2 radius correlation",
            "k": 15,
            "low_dim_search": "exact",
            "registered_floor": MATCHED_DENSITY_FLOOR,
            "paired_bootstrap_draws": PAIRED_BOOTSTRAP_DRAWS,
            "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
            "paired_interval": "99% percentile [0.5, 99.5]",
        },
        "cells": cells,
        "correlations": correlations,
        "adjacent_ci99": {key: list(value) for key, value in adjacent_ci.items()},
        "train_receipts": train_signatures,
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "runtime-component-panel.json"), receipt, immutable=True
    )


def run_decision(active: dict[str, Any], job: dict[str, Any]) -> None:
    _validate_environment(active)
    trigger, trigger_signature = _validate_positive_trigger(job)
    panel_path = os.path.join(job["panel_output"], "runtime-component-panel.json")
    panel = _read_sealed(panel_path, label="R0131 runtime component panel")
    panel_signature = expected_input_signature(panel_path)
    expected_train_signatures = {
        arm: expected_input_signature(
            os.path.join(job["train_outputs"][arm], "train-receipt.json")
        )
        for arm in ARMS
    }
    trains = {
        arm: _read_sealed(
            os.path.join(job["train_outputs"][arm], "train-receipt.json"),
            label=f"R0131 {arm} train",
        )
        for arm in ARMS
    }
    execution_checks = {
        "train_receipt_contracts": all(
            _train_contract(
                trains[arm], arm=arm, release_sha=active["manifest"]["release_sha"]
            )
            for arm in ARMS
        ),
        "panel_contract": (
            panel.get("schema") == PANEL_SCHEMA
            and panel.get("round_id") == ROUND_ID
            and panel.get("release_sha") == active["manifest"]["release_sha"]
            and panel.get("r0125_decision") == trigger_signature
            and panel.get("train_receipts") == expected_train_signatures
        ),
        "identical_initial_model_state": (
            trains[RESIDENT_FUSED].get("initial_model_state_sha256")
            == trains[RESIDENT_SEPARATE].get("initial_model_state_sha256")
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(trains[RESIDENT_FUSED].get("initial_model_state_sha256") or ""),
            ) is not None
        ),
        "identical_causal_invariant": (
            trains[RESIDENT_FUSED].get("causal_invariant_sha256")
            == trains[RESIDENT_SEPARATE].get("causal_invariant_sha256")
        ),
        "identical_environment": (
            trains[RESIDENT_FUSED].get("environment_freeze_sha256")
            == trains[RESIDENT_SEPARATE].get("environment_freeze_sha256")
            == active["manifest"]["environment_freeze"]["freeze_sha256"]
        ),
        "identical_bounded_numpy_stream": (
            (trains[RESIDENT_FUSED].get("exact_execution_receipt") or {}).get(
                "stream_trace"
            )
            == (trains[RESIDENT_SEPARATE].get("exact_execution_receipt") or {}).get(
                "stream_trace"
            )
        ),
        "only_forward_stamp_differs_between_new_arms": (
            (trains[RESIDENT_FUSED].get("exact_execution_receipt") or {}).get(
                "endpoint_forward"
            ) == "fused-source-destination"
            and (trains[RESIDENT_SEPARATE].get("exact_execution_receipt") or {}).get(
                "endpoint_forward"
            ) == "separate-source-destination"
        ),
    }
    selector = select_outcome(
        r0125_outcome=trigger["outcome"],
        correlations=panel["correlations"],
        adjacent_ci99={
            key: tuple(value) for key, value in panel["adjacent_ci99"].items()
        },
        execution_valid=all(execution_checks.values()),
    )
    output = create_fresh_directory(
        job["outputs"][0], label="R0131 runtime component decision"
    )
    receipt = _seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "r0125_decision": trigger_signature,
        "runtime_component_panel": panel_signature,
        "execution_checks": execution_checks,
        "selector": selector,
        "outcome": selector["outcome"],
        "capabilities_produced": [CAPABILITY] if all(execution_checks.values()) else [],
        "adjacent_path": [
            "host_control",
            RESIDENT_FUSED,
            RESIDENT_SEPARATE,
            "device_treatment",
        ],
        "native_intermediate_quality_tested": False,
        "single_mechanism_universal_cause_claimed": False,
        "production_runtime_adopted": False,
        "training_performed": True,
    })
    atomic_write_new_json(os.path.join(output, "decision.json"), receipt, immutable=True)


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0131Error("R0131 node received another queue")
    selected = job if job is not None else active.get("job") or {}
    action = str(selected.get("action") or "")
    if action == "train":
        run_train(active, selected)
    elif action == "panel":
        run_panel(active, selected)
    elif action == "decide":
        run_decision(active, selected)
    else:
        raise Round0131Error(f"unknown R0131 action {action!r}")

