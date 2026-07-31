"""Execute the self-contained R0125 device-versus-host runtime bridge."""
from __future__ import annotations

import gc
import json
import math
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
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import (
    HostFp16MaterializedArray,
    InventoryFp16Array,
    L2NormalizedArray,
    panel_config,
    preprocessing_stamp,
    validate_substrate_manifest,
    verify_signature,
)
from basemap.round0125_runtime_bridge import (
    ARMS,
    BATCH_SIZE,
    CAPABILITY,
    DECISION_METRICS,
    DEVICE_ARM,
    DIMENSION,
    GRAPH_EDGES,
    HOST_ARM,
    MATCHED_DENSITY_FLOOR,
    N_EPOCHS,
    ORIGINAL_RELEASE_SHA,
    PAIRED_BOOTSTRAP_DRAWS,
    PAIRED_BOOTSTRAP_SEED,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    QUERY_ROWS,
    QUERY_START,
    R0104_ACCEPTED_METRICS,
    R0104_GRAPH_MANIFEST_SHA256,
    R0104_GRAPH_SHA256,
    R0104_HIGH_D_REFERENCE_SHA256,
    R0104_QUERY_TRUTH_KEY,
    R0104_QUERY_TRUTH_PRODUCER_BACKEND,
    R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256,
    R0104_QUERY_TRUTH_SHA256,
    R0104_SHARED_RECEIPT_SHA256,
    R0122_FP16_MATCHED_DENSITY,
    R0122_PANEL_SHA256,
    ROUND_ID,
    ROWS,
    SEED,
    STREAM_TRACE_BATCHES,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    AuditedParametricUMAP,
    Round0125DeviceTrainingInput,
    Round0125Error,
    Round0125HostTrainingInput,
    expected_device_endpoint_accounting,
    seal,
    select_outcome,
    train_config,
    validate_environment_freeze,
    validate_seal,
)
from experiments.round0085_nodes import density_v2_calibration
from experiments.round0104_nodes import (
    _data_identity,
    _load_graph as _load_r0104_graph,
    _load_shared as _load_r0104_shared,
    _recall,
    _without_self,
)
from experiments.round0108_nodes import _panel_config as _density_panel_config
from experiments.round0119_nodes import (
    K_DENSITY,
    REPRESENTATIVE_ROWS,
    SOURCE_ROWS,
    TRANSFORM_BATCH_ROWS,
    _load_universe,
)


NATIVE_SCORE_SCHEMA = "round0125-native-runtime-arm-score-v1"
MATCHED_SCORE_SCHEMA = "round0125-matched-runtime-density-panel-v1"
DECISION_SCHEMA = "round0125-device-host-runtime-decision-v1"
MATCHED_COORDINATE_STD_MIN = 1e-8
TRAIN_CHECK_KEYS = {
    "exact_update_closure",
    "zero_numerical_skips",
    "no_pipeline_stamp_drift",
    "endpoint_rows_match_registered_path",
    "bounded_stream_trace_complete",
    "initial_model_state_stamped",
}
NATIVE_GATE_KEYS = {
    "finite_noncollapsed_coordinates",
    "transductive_recall50_gt_recall10",
    "projection_recall50_gt_recall10",
    "exact_update_closure",
    "zero_numerical_skips",
    "no_pipeline_stamp_drift",
}


def _schema(stem: str) -> str:
    return f"round0125-{stem}-v1"


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    validate_seal(value, label=label)
    return value


def _validate_job_environment(active: Mapping[str, Any]) -> dict[str, Any]:
    expected = (active.get("manifest") or {}).get("environment_freeze")
    if not isinstance(expected, Mapping):
        raise Round0125Error("queue lacks the shared environment freeze")
    return validate_environment_freeze(expected)


def _arm(job: Mapping[str, Any]) -> str:
    arm = str(job.get("arm") or "")
    if arm not in ARMS:
        raise Round0125Error(f"unknown R0125 arm {arm!r}")
    return arm


def _load_shared_exact(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    shared, signature = _load_r0104_shared(job)
    proof = shared.get("source_prefix_proof") or {}
    if (
        signature.get("sha256") != R0104_SHARED_RECEIPT_SHA256
        or shared.get("graph_edges") != GRAPH_EDGES
        or shared.get("graph", {}).get("sha256") != R0104_GRAPH_SHA256
        or shared.get("graph_manifest", {}).get("sha256")
        != R0104_GRAPH_MANIFEST_SHA256
        or shared.get("high_d_reference", {}).get("sha256")
        != R0104_HIGH_D_REFERENCE_SHA256
        or shared.get("query_truth", {}).get("sha256")
        != R0104_QUERY_TRUTH_SHA256
        or shared.get("query_truth_key") != R0104_QUERY_TRUTH_KEY
        or proof.get("payload_sha256")
        != "f4a0050e81a3755de84ba73405ba6823fa387f09a15d3ad299083fa60093f069"
    ):
        raise Round0125Error("accepted R0104 shared evidence changed")
    return shared, signature


def _load_accepted_r0104_query_truth(
    shared: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate, without relabelling, R0104's reviewed truth producer."""
    from basemap.panel_v2 import load_query_truth

    truth = load_query_truth(
        verify_signature(
            shared["query_truth"], label="accepted R0104 query truth"
        ),
        expected_key=R0104_QUERY_TRUTH_KEY,
        expected_candidate_compute_backend=R0104_QUERY_TRUTH_PRODUCER_BACKEND,
        expected_producer_implementation_sha256=(
            R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256
        ),
    )
    policy = (truth.get("key_parts") or {}).get("policy") or {}
    if (
        policy.get("implementation_sha256")
        != R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256
        or policy.get("candidate_compute_backend")
        != R0104_QUERY_TRUTH_PRODUCER_BACKEND
    ):
        raise Round0125Error("accepted R0104 query truth producer changed")
    return truth


def _new_model(config: Mapping[str, Any], *, device: str = "cuda"):
    model = config["model"]
    optimizer = config["optimizer"]
    execution = config["execution"]
    invariant = config["causal_invariant"]
    return AuditedParametricUMAP(
        n_components=model["output_dimension"],
        hidden_dim=model["hidden_dimension"],
        n_layers=model["hidden_layers"],
        n_neighbors=invariant["graph_k"],
        a=model["a"],
        b=model["b"],
        low_dim_kernel=model["low_dim_kernel"],
        correlation_weight=optimizer["correlation_weight"],
        learning_rate=optimizer["learning_rate"],
        n_epochs=optimizer["n_epochs"],
        batch_size=optimizer["batch_size"],
        device=device,
        use_batchnorm=model["use_batchnorm"],
        use_dropout=model["use_dropout"],
        clip_grad_norm=optimizer["clip_grad_norm"],
        clip_grad_value=None,
        pos_ratio=optimizer["positive_ratio"],
        architecture=model["architecture"],
        correlation_distance_transform="raw",
        lr_schedule="cosine",
        warmup_steps=optimizer["warmup_successful_updates"],
        total_steps_estimate=optimizer["successful_positive_lr_updates"],
        require_full_budget=True,
        require_graph_manifest=True,
        required_input_pipeline=execution["required_pipeline"],
        use_amp=optimizer["use_amp"],
        positive_target_mode=optimizer["positive_target_mode"],
        reject_neighbors=optimizer["reject_neighbors"],
        anchored_init="none",
        anchor_hold_weight=0.0,
        midnear_enabled=False,
        mn_pairs_per_batch=0,
        weighted_edge_sampling=optimizer["weighted_edge_sampling"],
        gpu_resident_data=execution["gpu_resident_data"],
        gpu_resident_vram_budget_gb=execution["gpu_resident_vram_budget_gb"],
        graph_manifest_path=invariant["graph_manifest"]["canonical_path"],
        graph_manifest_sha256=invariant["graph_manifest"]["sha256"],
    )


def _pipeline_mismatches(
    expected: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }


def _matched_axis_standard_deviation(coordinates: np.ndarray) -> np.ndarray:
    standard_deviation = np.asarray(
        coordinates.std(axis=0), dtype=np.float64
    )
    if (
        standard_deviation.shape != (2,)
        or not np.isfinite(standard_deviation).all()
        or np.any(standard_deviation <= MATCHED_COORDINATE_STD_MIN)
    ):
        raise Round0125Error("R0125 matched coordinates collapsed")
    return standard_deviation


def _train_receipt_contract(
    train: Mapping[str, Any], *, arm: str, release_sha: str
) -> bool:
    checks = train.get("train_checks")
    return bool(
        train.get("schema") == _schema("runtime-arm-train-receipt")
        and train.get("round_id") == ROUND_ID
        and train.get("arm") == arm
        and train.get("release_sha") == release_sha
        and isinstance(checks, Mapping)
        and set(checks) == TRAIN_CHECK_KEYS
        and all(checks.values())
    )


def _prior_artifact_release_sha(
    active: Mapping[str, Any], job: Mapping[str, Any], *, field: str
) -> str:
    """Return the authenticated release for an immutable completed artifact.

    The correction queue may consume only R0125 artifacts sealed by the exact
    original release.  Ordinary execution continues to require the active
    release, preserving the original queue's behavior.
    """
    active_release = str(active["manifest"]["release_sha"])
    observed = job.get(field)
    if observed is None:
        return active_release
    if observed != ORIGINAL_RELEASE_SHA:
        raise Round0125Error(
            f"R0125 {field} may name only the exact original release"
        )
    return ORIGINAL_RELEASE_SHA


def _open_training_input(
    arm: str, graph: Mapping[str, Any], *, device: str = "cuda"
) -> tuple[Any, dict[str, Any]]:
    substrate = validate_substrate_manifest(verify_payloads=False)
    source = InventoryFp16Array(0, ROWS)
    if arm == DEVICE_ARM:
        return (
            Round0125DeviceTrainingInput(
                source, graph, device=device, expected_rows=ROWS
            ),
            substrate,
        )
    dataset = HostFp16MaterializedArray(
        source, device=device, buffer_rows=BATCH_SIZE
    )
    return Round0125HostTrainingInput(dataset, graph), substrate


def run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    import torch

    arm = _arm(job)
    environment = _validate_job_environment(active)
    shared, shared_signature = _load_shared_exact(job)
    graph = _load_r0104_graph(shared)
    config, config_sha = train_config(
        arm,
        graph_signature=shared["graph"],
        graph_manifest_signature=shared["graph_manifest"],
        graph_edges=shared["graph_edges"],
    )
    training_input, substrate = _open_training_input(arm, graph)
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0125 {arm} train output"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        seal({
            "schema": _schema("production-config"),
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
    model = _new_model(config)
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
        low_memory=arm == HOST_ARM,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=shared["graph"]["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started
    if arm == HOST_ARM and training_input._last_sampler is not None:
        # The successful-update horizon can stop between epoch boundaries while
        # the one-slot-ahead host producer has already submitted the next fill.
        # Join it explicitly so its accounting is stable and no worker thread
        # survives into checkpoint sealing.
        training_input._last_sampler.close()
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
    expected_rows = (
        expected_device_endpoint_accounting()["endpoint_rows_per_side"]
        if arm == DEVICE_ARM else SUCCESSFUL_UPDATES * BATCH_SIZE
    )
    expected_calls = SUCCESSFUL_UPDATES * 2
    observed_calls = (
        runtime.get("endpoint_gather_calls")
        if arm == HOST_ARM else runtime.get("endpoint_gather_calls")
    )
    if (
        runtime.get("source_rows_gathered") != expected_rows
        or runtime.get("destination_rows_gathered") != expected_rows
        or observed_calls != (
            SUCCESSFUL_UPDATES if arm == HOST_ARM else expected_calls
        )
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows_per_side": expected_rows,
            "expected_gather_calls": (
                SUCCESSFUL_UPDATES if arm == HOST_ARM else expected_calls
            ),
            "runtime": runtime,
        }
    if arm == HOST_ARM and (
        runtime.get("host_prefetch_consumer_batches") != SUCCESSFUL_UPDATES
        or runtime.get("host_prefetch_producer_batches")
        not in {SUCCESSFUL_UPDATES, SUCCESSFUL_UPDATES + 1}
    ):
        mismatches["host_prefetch_accounting"] = {
            "expected_consumers": SUCCESSFUL_UPDATES,
            "expected_producers": [
                SUCCESSFUL_UPDATES,
                SUCCESSFUL_UPDATES + 1,
            ],
            "runtime": runtime,
        }
    trace = runtime.get("stream_trace")
    if (
        not isinstance(trace, Mapping)
        or trace.get("batches_hashed") != STREAM_TRACE_BATCHES
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(trace.get(key) or "")) is None
            for key in (
                "source_endpoint_ids_sha256",
                "destination_endpoint_ids_sha256",
            )
        )
    ):
        mismatches["stream_trace"] = {
            "expected_batches": STREAM_TRACE_BATCHES,
            "observed": trace,
        }
    if model.initial_model_state_sha256 is None:
        mismatches["initial_model_state_sha256"] = {
            "expected": "64-hex digest",
            "observed": None,
        }
    if mismatches:
        raise Round0125Error(f"R0125 {arm} train accounting failed: {mismatches}")
    for key, value in runtime.items():
        accounting[f"pipeline_{key}"] = value
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
        / model._bench_seconds
        if model._bench_seconds else 0.0
    )
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        raise Round0125Error(f"R0125 {arm} performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    checks = {
        "exact_update_closure": True,
        "zero_numerical_skips": True,
        "no_pipeline_stamp_drift": True,
        "endpoint_rows_match_registered_path": True,
        "bounded_stream_trace_complete": True,
        "initial_model_state_stamped": True,
    }
    receipt = seal({
        "schema": _schema("runtime-arm-train-receipt"),
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "causal_invariant_sha256": config["causal_invariant_sha256"],
        "initial_model_state_sha256": model.initial_model_state_sha256,
        "model": expected_input_signature(model_path),
        "shared_evidence": shared_signature,
        "graph": shared["graph"],
        "graph_manifest": shared["graph_manifest"],
        "substrate": substrate["signature"],
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
        "retry_count": 0,
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del model, training_input, graph
    torch.cuda.empty_cache()
    gc.collect()


def _authenticate_model(
    active: Mapping[str, Any], job: Mapping[str, Any], *, device: str = "cuda"
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    arm = _arm(job)
    shared, shared_signature = _load_shared_exact(job)
    config, config_sha = train_config(
        arm,
        graph_signature=shared["graph"],
        graph_manifest_signature=shared["graph_manifest"],
        graph_edges=shared["graph_edges"],
    )
    config_path = os.path.join(str(job["train_output"]), "production-config.json")
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    config_receipt = _read_sealed(config_path, label=f"R0125 {arm} config")
    train = _read_sealed(train_path, label=f"R0125 {arm} train")
    train_release_sha = _prior_artifact_release_sha(
        active, job, field="train_release_sha"
    )
    expected_environment = (active.get("manifest") or {}).get("environment_freeze")
    if (
        config_receipt.get("schema") != _schema("production-config")
        or config_receipt.get("round_id") != ROUND_ID
        or config_receipt.get("arm") != arm
        or config_receipt.get("config") != config
        or config_receipt.get("config_sha256") != config_sha
        or train.get("arm") != arm
        or train.get("release_sha") != train_release_sha
        or train.get("production_config") != expected_input_signature(config_path)
        or train.get("production_config_sha256") != config_sha
        or train.get("shared_evidence") != shared_signature
        or train.get("causal_invariant_sha256")
        != config["causal_invariant_sha256"]
        or train.get("environment_freeze_sha256")
        != expected_environment.get("freeze_sha256")
        or not _train_receipt_contract(
            train,
            arm=arm,
            release_sha=train_release_sha,
        )
    ):
        raise Round0125Error(f"R0125 {arm} train/config lineage changed")
    model_path = verify_signature(train["model"], label=f"R0125 {arm} model")
    model = AuditedParametricUMAP.load(model_path, device=device)
    expected = config["model"]
    observed = {
        "architecture": model.architecture,
        "input_dimension": model.input_dim,
        "hidden_dimension": model.hidden_dim,
        "hidden_layers": model.n_layers,
        "output_dimension": model.n_components,
        "use_batchnorm": model.use_batchnorm,
        "use_dropout": model.use_dropout,
        "low_dim_kernel": model.low_dim_kernel,
        "a": model.a,
        "b": model.b,
    }
    if observed != expected:
        raise Round0125Error(f"R0125 {arm} model architecture changed")
    return model, train, expected_input_signature(train_path), shared, config_sha


def run_transform(active: dict[str, Any], job: dict[str, Any]) -> None:
    arm = _arm(job)
    _validate_job_environment(active)
    model, train, train_signature, _shared, config_sha = _authenticate_model(
        active, job
    )
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0125 {arm} native transform"
    )
    X = InventoryFp16Array(0, ROWS)
    Xq = InventoryFp16Array(QUERY_START, QUERY_START + QUERY_ROWS)
    started = time.monotonic()
    coordinates = np.asarray(model.transform(X, batch_size=BATCH_SIZE), dtype=np.float32)
    query_coordinates = np.asarray(
        model.transform(Xq, batch_size=BATCH_SIZE), dtype=np.float32
    )
    if (
        coordinates.shape != (ROWS, 2)
        or query_coordinates.shape != (QUERY_ROWS, 2)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(query_coordinates).all()
    ):
        raise Round0125Error(f"R0125 {arm} native transform is invalid")
    coordinates_path = os.path.join(output, "coordinates.npy")
    query_path = os.path.join(output, "oos-query-coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    receipt = seal({
        "schema": _schema("native-transform-receipt"),
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": config_sha,
        "train_receipt": train_signature,
        "model": train["model"],
        "input_preprocessing": preprocessing_stamp("fp16_control"),
        "training_rows": [0, ROWS],
        "query_rows": [QUERY_START, QUERY_START + QUERY_ROWS],
        "coordinates": expected_input_signature(coordinates_path),
        "query_coordinates": expected_input_signature(query_path),
        "wall_seconds": time.monotonic() - started,
        "finite": True,
    })
    atomic_write_new_json(
        os.path.join(output, "transform-receipt.json"), receipt, immutable=True
    )


def run_native_score(active: dict[str, Any], job: dict[str, Any]) -> None:
    from basemap.panel_v2 import (
        cross_knn,
        ffr_from_neighbors,
        load_hiD_reference,
        recall_at_k_from_neighbors,
        score_panel,
    )

    arm = _arm(job)
    _validate_job_environment(active)
    shared, shared_signature = _load_shared_exact(job)
    config, config_sha = train_config(
        arm,
        graph_signature=shared["graph"],
        graph_manifest_signature=shared["graph_manifest"],
        graph_edges=shared["graph_edges"],
    )
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    transform_path = os.path.join(
        str(job["transform_output"]), "transform-receipt.json"
    )
    train = _read_sealed(train_path, label=f"R0125 {arm} train")
    transform = _read_sealed(transform_path, label=f"R0125 {arm} transform")
    train_release_sha = _prior_artifact_release_sha(
        active, job, field="train_release_sha"
    )
    transform_release_sha = _prior_artifact_release_sha(
        active, job, field="transform_release_sha"
    )
    if (
        not _train_receipt_contract(
            train,
            arm=arm,
            release_sha=train_release_sha,
        )
        or train.get("production_config_sha256") != config_sha
        or train.get("shared_evidence") != shared_signature
        or transform.get("schema") != _schema("native-transform-receipt")
        or transform.get("round_id") != ROUND_ID
        or transform.get("arm") != arm
        or transform.get("release_sha") != transform_release_sha
        or transform.get("production_config_sha256") != config_sha
        or transform.get("train_receipt") != expected_input_signature(train_path)
        or transform.get("model") != train.get("model")
        or transform.get("training_rows") != [0, ROWS]
        or transform.get("query_rows") != [
            QUERY_START,
            QUERY_START + QUERY_ROWS,
        ]
        or transform.get("finite") is not True
    ):
        raise Round0125Error(f"R0125 {arm} native score inputs changed")
    Z = np.load(verify_signature(transform["coordinates"], label="coordinates"),
                mmap_mode="r", allow_pickle=False)
    Zq = np.load(verify_signature(transform["query_coordinates"], label="queries"),
                 mmap_mode="r", allow_pickle=False)
    X = L2NormalizedArray(InventoryFp16Array(0, ROWS))
    cfg = panel_config()
    reference = load_hiD_reference(
        shared["high_d_reference"]["canonical_path"],
        expected_key=shared["high_d_reference_key"],
    )
    truth = _load_accepted_r0104_query_truth(shared)
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0125 {arm} native score"
    )
    started = time.monotonic()
    panel = score_panel(
        X, Z, config=cfg, centroids_by_k=None,
        hiD_reference=reference,
        reference_identity={
            "data_identity": _data_identity(shared["source_prefix_proof"]),
            "convention": {
                "row_order": "R0103 inventory first 2M rows",
                "distance": "cosine via fp32-L2-normalized squared L2",
                "self_exclusion": True,
                "anchor_namespace": "zero-based R0103 row IDs",
            },
        },
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "arm": arm,
            "release_sha": active["manifest"]["release_sha"],
            "train_receipt": expected_input_signature(train_path),
            "transform_receipt": expected_input_signature(transform_path),
            "shared_evidence": shared_signature,
        },
    )
    k_fraction = max(cfg.k_hit, int(math.ceil(cfg.frac * ROWS)))
    low_fraction = cross_knn(Zq, Z, k_fraction, cfg, hi_dim=False)
    low10 = low_fraction[:, :cfg.k_hit]
    high10 = np.asarray(truth["neighbors"], dtype=np.int64)[:, :cfg.k_hit]
    projection_ffr = ffr_from_neighbors(high10, low_fraction, cfg.k_hit)
    projection_recall = recall_at_k_from_neighbors(high10, low10, cfg.k_hit)
    low51 = cross_knn(
        np.asarray(Z[reference["anchor_ids"]], dtype=np.float32),
        Z, 51, cfg, hi_dim=False,
    )
    low50 = _without_self(low51, reference["anchor_ids"], 50)
    recall50 = _recall(reference["hi_hit"], low50, cfg.k_hit)
    query_low50 = cross_knn(Zq, Z, 50, cfg, hi_dim=False)
    query_recall50 = _recall(high10, query_low50, cfg.k_hit)
    metrics = {
        "ffr": float(panel["ffr"]),
        "density": float(panel["density"]),
        "recall_at_10": float(panel["recall@k"]),
        "oos_proj_ffr": float(projection_ffr),
        "oos_proj_recall_at_10": float(projection_recall),
    }
    guards = panel.get("guards") or {}
    execution_gates = {
        "finite_noncollapsed_coordinates": bool(
            guards.get("coords_finite") is True
            and guards.get("coords_collapsed") is False
            and guards.get("emb_finite") is True
            and guards.get("emb_zero_rows") == 0
        ),
        "transductive_recall50_gt_recall10": recall50 > metrics["recall_at_10"],
        "projection_recall50_gt_recall10": (
            query_recall50 > metrics["oos_proj_recall_at_10"]
        ),
        "exact_update_closure": bool(
            (train.get("train_checks") or {}).get("exact_update_closure")
        ),
        "zero_numerical_skips": bool(
            (train.get("train_checks") or {}).get("zero_numerical_skips")
        ),
        "no_pipeline_stamp_drift": bool(
            (train.get("train_checks") or {}).get("no_pipeline_stamp_drift")
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise Round0125Error(f"R0125 {arm} native metrics are nonfinite")
    receipt = seal({
        "schema": NATIVE_SCORE_SCHEMA,
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "train_receipt": expected_input_signature(train_path),
        "transform_receipt": expected_input_signature(transform_path),
        "shared_evidence": shared_signature,
        "high_d_reference": shared["high_d_reference"],
        "query_truth": shared["query_truth"],
        "query_truth_producer_policy": truth["key_parts"]["policy"],
        "panel": panel,
        "projection": {
            "ffr": projection_ffr,
            "recall_at_10": projection_recall,
            "recall_at_50_of_high10": query_recall50,
            "queries": QUERY_ROWS,
            "k_fraction": k_fraction,
        },
        "transductive_recall_at_50_of_high10": recall50,
        "metrics": metrics,
        "execution_gates": execution_gates,
        "score_wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "score.json"), receipt, immutable=True)


def _score_matched_arm(
    *, arm: str, model: Any, source: np.ndarray,
    retained_global_rows: np.ndarray, anchors: np.ndarray,
    high_radius: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from basemap.panel_v2 import _self_knn

    transformed = np.asarray(
        model.transform(source, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
    )
    if transformed.shape != (SOURCE_ROWS, 2) or not np.isfinite(transformed).all():
        raise Round0125Error(f"R0125 {arm} matched full transform is invalid")
    coordinates = np.asarray(transformed[retained_global_rows], dtype=np.float32)
    del transformed
    axis_standard_deviation = _matched_axis_standard_deviation(coordinates)
    config = _density_panel_config(anchors=len(anchors))
    _, distances, guard = _self_knn(
        coordinates, anchors, K_DENSITY, config,
        hi_dim=False, want_dist=True, exact=True,
    )
    low_radius = np.asarray(distances.mean(1), dtype=np.float64)
    summary, bootstrap, null = density_v2_calibration(
        high_radius, low_radius,
        bootstrap_draws=PAIRED_BOOTSTRAP_DRAWS,
        bootstrap_seed=PAIRED_BOOTSTRAP_SEED,
        null_draws=1_000,
        null_seed=10_802,
    )
    return ({
        "arm": arm,
        "transform_input_rows": SOURCE_ROWS,
        "transform_selection_after_transform": True,
        "transform_selected_rows": REPRESENTATIVE_ROWS,
        "coordinates": {
            "rows": len(coordinates),
            "ordered_sha256": ordered_array_sha256(coordinates),
            "axis_standard_deviation": axis_standard_deviation.tolist(),
            "finite": True,
            "noncollapsed": True,
            "minimum_axis_standard_deviation": MATCHED_COORDINATE_STD_MIN,
        },
        "density_v2": summary,
        "clears_unchanged_registered_floor": (
            float(summary["correlation"]) >= MATCHED_DENSITY_FLOOR
        ),
        "low_dim_exact_search_guard": guard,
    }, {
        f"{arm}__low_radius": low_radius,
        f"{arm}__bootstrap": bootstrap,
        f"{arm}__permuted_null": null,
    })


def run_matched_density(active: dict[str, Any], job: dict[str, Any]) -> None:
    _validate_job_environment(active)
    r0122_path = str(job["r0122_panel"]["canonical_path"])
    r0122_panel = _read_sealed(r0122_path, label="accepted R0122 panel")
    prior_cell = (r0122_panel.get("new_cells") or {}).get(
        "r0104_fp16_seed42_full_transform"
    ) or {}
    if (
        expected_input_signature(r0122_path) != dict(job["r0122_panel"])
        or job["r0122_panel"].get("sha256") != R0122_PANEL_SHA256
        or r0122_panel.get("schema")
        != "round0122-jina-density-provenance-bridge-panel-v1"
        or float((prior_cell.get("density_v2") or {}).get("correlation", -1.0))
        != R0122_FP16_MATCHED_DENSITY
        or prior_cell.get("clears_unchanged_registered_floor") is not False
    ):
        raise Round0125Error("accepted R0122 matched baseline changed")
    (
        source, _representatives, retained_global_rows, anchors,
        global_rows, high_radius, lineage, _reference,
    ) = _load_universe(job)
    output = create_fresh_directory(
        job["outputs"][0], label="R0125 matched density panel"
    )
    started = time.monotonic()
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "anchor_compact_rows": anchors,
        "anchor_global_rows": global_rows,
        "high_radius": high_radius,
    }
    train_signatures: dict[str, Any] = {}
    for arm in ARMS:
        model, _train, train_signature, _shared, _config_sha = _authenticate_model(
            active,
            {
                "arm": arm,
                "shared_output": job["shared_output"],
                "train_output": job["train_outputs"][arm],
                "train_release_sha": job.get("train_release_sha"),
            },
        )
        cell, cell_arrays = _score_matched_arm(
            arm=arm, model=model, source=source,
            retained_global_rows=retained_global_rows,
            anchors=anchors, high_radius=high_radius,
        )
        cell["train_receipt"] = train_signature
        cells[arm] = cell
        arrays.update(cell_arrays)
        train_signatures[arm] = train_signature
        del model
        gc.collect()
    paired_delta = (
        arrays[f"{DEVICE_ARM}__bootstrap"]
        - arrays[f"{HOST_ARM}__bootstrap"]
    )
    ci99 = np.percentile(paired_delta, [0.5, 99.5]).astype(np.float64)
    arrays["paired_device_minus_host_bootstrap"] = paired_delta
    arrays_path = os.path.join(output, "matched-density-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": MATCHED_SCORE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "accepted_r0122_panel": dict(job["r0122_panel"]),
        "accepted_r0122_fp16_density": R0122_FP16_MATCHED_DENSITY,
        "lineage": lineage,
        "universe": {
            "source_rows": SOURCE_ROWS,
            "representative_rows": REPRESENTATIVE_ROWS,
            "anchors": len(anchors),
            "anchor_compact_rows_sha256": ordered_array_sha256(anchors),
            "anchor_global_rows_sha256": ordered_array_sha256(global_rows),
            "high_radius_sha256": ordered_array_sha256(high_radius),
        },
        "scorer": {
            "metric": "density-v2 radius correlation",
            "k": K_DENSITY,
            "low_dim_search": "exact",
            "registered_floor": MATCHED_DENSITY_FLOOR,
            "absolute_bootstrap_draws": PAIRED_BOOTSTRAP_DRAWS,
            "paired_bootstrap_draws": PAIRED_BOOTSTRAP_DRAWS,
            "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
            "paired_interval": "99% percentile [0.5, 99.5]",
        },
        "cells": cells,
        "paired_device_minus_host_density": (
            float(cells[DEVICE_ARM]["density_v2"]["correlation"])
            - float(cells[HOST_ARM]["density_v2"]["correlation"])
        ),
        "paired_device_minus_host_density_ci99": ci99.tolist(),
        "train_receipts": train_signatures,
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "matched-density-panel.json"), receipt, immutable=True
    )
    del source, retained_global_rows
    gc.collect()


def run_decision(active: dict[str, Any], job: dict[str, Any]) -> None:
    _validate_job_environment(active)
    output = create_fresh_directory(
        job["outputs"][0], label="R0125 runtime bridge decision"
    )
    native: dict[str, Any] = {}
    native_signatures: dict[str, Any] = {}
    trains: dict[str, Any] = {}
    for arm in ARMS:
        score_path = os.path.join(job["native_score_outputs"][arm], "score.json")
        train_path = os.path.join(job["train_outputs"][arm], "train-receipt.json")
        native[arm] = _read_sealed(score_path, label=f"R0125 {arm} native score")
        trains[arm] = _read_sealed(train_path, label=f"R0125 {arm} train")
        native_signatures[arm] = expected_input_signature(score_path)
    matched_path = os.path.join(job["matched_output"], "matched-density-panel.json")
    matched = _read_sealed(matched_path, label="R0125 matched density panel")
    matched_signature = expected_input_signature(matched_path)
    expected_train_signatures = {
        arm: expected_input_signature(
            os.path.join(job["train_outputs"][arm], "train-receipt.json")
        )
        for arm in ARMS
    }
    execution_checks = {
        "train_receipt_contracts": all(
            _train_receipt_contract(
                trains[arm],
                arm=arm,
                release_sha=_prior_artifact_release_sha(
                    active, job, field="train_release_sha"
                ),
            )
            for arm in ARMS
        ),
        "native_score_schemas": all(
            native[arm].get("schema") == NATIVE_SCORE_SCHEMA
            and native[arm].get("round_id") == ROUND_ID
            and native[arm].get("arm") == arm
            and native[arm].get("release_sha")
            == active["manifest"]["release_sha"]
            and native[arm].get("train_receipt")
            == expected_train_signatures[arm]
            and set(native[arm].get("metrics") or {}) == set(DECISION_METRICS)
            for arm in ARMS
        ),
        "native_execution_gates": all(
            isinstance(native[arm].get("execution_gates"), Mapping)
            and set(native[arm]["execution_gates"]) == NATIVE_GATE_KEYS
            and all(native[arm]["execution_gates"].values())
            for arm in ARMS
        ),
        "matched_schema_and_cells": (
            matched.get("schema") == MATCHED_SCORE_SCHEMA
            and matched.get("round_id") == ROUND_ID
            and matched.get("release_sha")
            == active["manifest"]["release_sha"]
            and tuple(matched.get("cells") or {}) == ARMS
            and matched.get("train_receipts") == expected_train_signatures
            and all(
                (matched.get("cells") or {}).get(arm, {}).get("train_receipt")
                == expected_train_signatures[arm]
                for arm in ARMS
            )
            and all(
                ((matched.get("cells") or {}).get(arm, {}).get("coordinates") or {}).get(
                    "finite"
                ) is True
                and ((matched.get("cells") or {}).get(arm, {}).get("coordinates") or {}).get(
                    "noncollapsed"
                ) is True
                for arm in ARMS
            )
        ),
        "identical_initial_model_state": (
            trains[DEVICE_ARM].get("initial_model_state_sha256")
            == trains[HOST_ARM].get("initial_model_state_sha256")
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(trains[DEVICE_ARM].get("initial_model_state_sha256") or ""),
            ) is not None
        ),
        "identical_causal_invariant": (
            trains[DEVICE_ARM].get("causal_invariant_sha256")
            == trains[HOST_ARM].get("causal_invariant_sha256")
        ),
        "identical_environment_freeze": (
            trains[DEVICE_ARM].get("environment_freeze_sha256")
            == trains[HOST_ARM].get("environment_freeze_sha256")
            == active["manifest"]["environment_freeze"]["freeze_sha256"]
        ),
        "distinct_registered_paths": (
            (trains[DEVICE_ARM].get("exact_execution_receipt") or {}).get(
                "sampler_class"
            ) == "DeviceEdgeSampler"
            and (trains[HOST_ARM].get("exact_execution_receipt") or {}).get(
                "sampler_class"
            ) == "PairedHostWeightedJinaSampler"
        ),
        "stream_traces_complete": all(
            ((trains[arm].get("exact_execution_receipt") or {}).get(
                "stream_trace"
            ) or {}).get("batches_hashed") == STREAM_TRACE_BATCHES
            for arm in ARMS
        ),
    }
    execution_valid = all(execution_checks.values())
    host_density = float(
        matched["cells"][HOST_ARM]["density_v2"]["correlation"]
    )
    device_density = float(
        matched["cells"][DEVICE_ARM]["density_v2"]["correlation"]
    )
    selector = select_outcome(
        host_metrics=native[HOST_ARM]["metrics"],
        device_metrics=native[DEVICE_ARM]["metrics"],
        host_matched_density=host_density,
        device_matched_density=device_density,
        paired_delta_ci99=tuple(matched["paired_device_minus_host_density_ci99"]),
        execution_valid=execution_valid,
    )
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "native_scores": native_signatures,
        "matched_density_panel": matched_signature,
        "execution_checks": execution_checks,
        "selector": selector,
        "outcome": selector["outcome"],
        "capabilities_produced": (
            [CAPABILITY]
            if execution_valid else []
        ),
        "causal_scope": (
            "one seed-42 comparison of the complete legacy device sampler/runtime "
            "bundle against the R0104 host sampler/runtime bundle"
        ),
        "sampler_only_cause_claimed": False,
        "residency_only_cause_claimed": False,
        "rng_streams_equal_claimed": False,
        "production_runtime_adopted": False,
        "map_registry_state_changed": False,
        "training_performed": True,
    })
    atomic_write_new_json(
        os.path.join(output, "decision.json"), receipt, immutable=True
    )


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0125Error("R0125 node received another queue")
    selected = job if job is not None else active.get("job") or {}
    action = str(selected.get("action") or "")
    if action == "train":
        run_train(active, selected)
    elif action == "transform":
        run_transform(active, selected)
    elif action == "native_score":
        run_native_score(active, selected)
    elif action == "matched_density":
        run_matched_density(active, selected)
    elif action == "decide":
        run_decision(active, selected)
    else:
        raise Round0125Error(f"unknown R0125 action {action!r}")
