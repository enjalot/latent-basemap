"""Train the retained 25M diverse-Jina atlas."""
from __future__ import annotations

import gc
import json
import os
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0106_graph import N_NEIGHBORS, RETAINED_ROWS
from basemap.round0107_training import (
    BATCH_SIZE,
    DIMENSION,
    PERFORMANCE_WARMUP_UPDATES,
    PIPELINE,
    ROUND_ID,
    SEED,
    TRAIN_MINIMUM_UPDATES_PER_S,
    CompactHostInt8MaterializedArray,
    Round0107Error,
    Round0107TrainingInput,
    load_graph_manifest,
    performance_windows,
    seal,
    synchronize_runtime_counters,
    train_config,
)


def _new_model(config: Mapping[str, Any]):
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = config["model"]
    optimizer = config["optimizer"]
    graph = config["graph"]
    execution = config["execution"]
    return ParametricUMAP(
        n_components=model["output_dimension"],
        hidden_dim=model["hidden_dimension"],
        n_layers=model["hidden_layers"],
        n_neighbors=graph["n_neighbors_including_self"],
        a=model["a"],
        b=model["b"],
        low_dim_kernel=model["low_dim_kernel"],
        correlation_weight=optimizer["correlation_weight"],
        learning_rate=optimizer["learning_rate"],
        n_epochs=2,
        batch_size=optimizer["batch_size"],
        device="cuda",
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
        graph_manifest_path=graph["manifest"]["canonical_path"],
        graph_manifest_sha256=graph["manifest"]["sha256"],
    )


def _graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    graph = load_graph_manifest(
        str(job["graph_manifest"]),
        expected_sha256=str(job["graph_manifest_sha256"]),
    )
    manifest = graph["manifest"]
    if (
        manifest.get("release_sha") != str(job["graph_release_sha"])
        or active["manifest"]["release_sha"] != str(job["release_sha"])
    ):
        raise Round0107Error("R0107 release lineage changed")
    return graph


def run_train(active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    import torch

    graph = _graph(active, job)
    manifest = graph["manifest"]
    config, config_sha256 = train_config(
        graph_manifest=manifest,
        graph_signature=graph["signature"],
    )
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    if updates != int(graph["successful_updates"]):
        raise Round0107Error("R0107 graph-derived horizon disagrees")
    arrays = graph["arrays"]
    dataset = CompactHostInt8MaterializedArray(
        mapping=arrays["mapping"],
        buffer_rows=BATCH_SIZE,
    )
    graph_view = {
        "signature": graph["signature"],
        "graph_signatures": {
            **manifest["outputs"],
            "manifest": graph["signature"],
        },
        "mapping_signature": manifest["compact_mapping"],
        "sources": arrays["sources"],
        "targets": arrays["targets"],
        "weights": arrays["weights"],
    }
    wrapper = Round0107TrainingInput(
        dataset, graph_view, required_pipeline=PIPELINE
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0107 diverse-Jina train output"
    )
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": "round0107-production-config-v1",
            "round_id": ROUND_ID,
            "config": config,
            "config_sha256": config_sha256,
        },
        immutable=True,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = _new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = PERFORMANCE_WARMUP_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    wall_seconds = time.monotonic() - started
    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    exact = {
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": int(manifest["directed_edge_count"]),
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = updates * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != updates
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    expected_positive_draws = updates * int(
        config["optimizer"]["positive_rows_per_update"]
    )
    if (
        int(runtime["weight_acceptances"]) < expected_positive_draws
        or int(runtime["weight_proposals"]) < int(runtime["weight_acceptances"])
        or not 0 < float(runtime["weight_acceptance_rate"]) <= 1
    ):
        mismatches["weighted_rejection_accounting"] = {
            "expected_positive_draws": expected_positive_draws,
            "runtime": runtime,
        }
    if mismatches:
        raise Round0107Error(f"R0107 train accounting failed: {mismatches}")
    synchronize_runtime_counters(accounting, runtime)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    steady_updates_per_s = (
        (updates - PERFORMANCE_WARMUP_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if (
        profiler.get("aborted") is not False
        or steady_updates_per_s < TRAIN_MINIMUM_UPDATES_PER_S
    ):
        raise Round0107Error("R0107 performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    receipt = seal({
        "schema": "round0107-diverse-jina-train-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": config_sha256,
        "model": expected_input_signature(model_path),
        "graph_manifest": graph["signature"],
        "graph_outputs": manifest["outputs"],
        "compact_mapping": manifest["compact_mapping"],
        "substrate": dataset.substrate["signature"],
        "update_derivation": {
            "directed_fuzzy_edges": int(manifest["directed_edge_count"]),
            "positive_rows_per_update": int(
                config["optimizer"]["positive_rows_per_update"]
            ),
            "rule": "ceil(directed_fuzzy_edges/409)",
            "successful_updates": updates,
            "expected_positive_draws": expected_positive_draws,
        },
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
        "steady_updates_per_s": steady_updates_per_s,
        "train_wall_seconds": wall_seconds,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
        },
        "memory": {
            "device_total_bytes": int(total_bytes),
            "post_train_free_bytes": int(free_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
        "retry_count": 0,
        "training_performed": True,
        "optimizer_updates": updates,
        "evaluation_performed": False,
        "map_decision_made": False,
    })
    receipt_path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del model, wrapper, dataset, graph, arrays
    torch.cuda.empty_cache()
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0107Error("R0107 handler requires its exact round/job")
    if job.get("action") != "train_diverse_jina":
        raise Round0107Error(f"unknown R0107 action: {job.get('action')!r}")
    return run_train(active, job)
