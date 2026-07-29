"""Execute the self-contained paired fp16/int8 Round 0104."""
from __future__ import annotations

import gc
import json
import math
import os
import random
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
    ARMS,
    DECISION_METRICS,
    DIMENSION,
    GRAPH_K,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE_GRID,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_QUALITY_ROWS,
    GRAPH_QUALITY_SEED,
    GRAPH_TRAIN_ROWS,
    GRAPH_TRAIN_SEED,
    L2NormalizedArray,
    PANEL_ANCHORS,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    PIPELINE,
    QUERY_ROWS,
    QUERY_START,
    ROUND_ID,
    ROWS,
    SEED,
    SUCCESSFUL_UPDATES,
    InventoryFp16Array,
    Round0104Error,
    Round0104TrainingInput,
    paired_decision,
    panel_config,
    preprocessing_stamp,
    seal,
    source_prefix_proof,
    synchronize_runtime_counters,
    train_config,
    transform_array,
    validate_seal,
    verify_signature,
    open_training_dataset,
)


def _schema(active: Mapping[str, Any], stem: str) -> str:
    return f"round{active['manifest']['round_id']}-{stem}-v2"


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    validate_seal(value, label=label)
    return value


def _materialize_normalized(source: Any) -> np.ndarray:
    values = np.empty((len(source), DIMENSION), dtype=np.float32)
    for start in range(0, len(source), 65_536):
        stop = min(start + 65_536, len(source))
        block = np.asarray(source[start:stop], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if not np.isfinite(block).all() or not np.isfinite(norms).all() or np.any(
            norms <= 0
        ):
            raise Round0104Error("graph source has zero/nonfinite rows")
        values[start:stop] = block / norms
    return values


def _faiss_gpu_options(faiss: Any) -> Any:
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    return options


def _without_self(rows: np.ndarray, ids: np.ndarray, width: int) -> np.ndarray:
    out = np.empty((len(rows), width), dtype=np.int64)
    for index, row in enumerate(np.asarray(rows, dtype=np.int64)):
        kept = row[row != int(ids[index])]
        if len(kept) < width or len(np.unique(kept[:width])) != width:
            raise Round0104Error("search did not return enough unique nonself rows")
        out[index] = kept[:width]
    return out


def _recall_rows(observed: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.isin(observed[index], truth[index]).sum() / truth.shape[1]
            for index in range(len(truth))
        ],
        dtype=np.float64,
    )


def _data_identity(proof: Mapping[str, Any]) -> dict[str, Any]:
    shards = []
    for position, segment in enumerate(proof["segments"]):
        signature = segment["shard"]
        shards.append(
            {
                "position": position,
                "name": os.path.basename(signature["canonical_path"]),
                "bytes": int(signature["bytes"]),
                "sha256": str(signature["sha256"]),
            }
        )
    return {
        "kind": "ordered_shards",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("float32").str,
        "shards": shards,
    }


def run_build_shared(active: dict[str, Any], job: dict[str, Any]) -> None:
    import faiss
    import umap.umap_ as umap_api
    from basemap.panel_v2 import (
        build_hiD_reference,
        build_query_truth,
        sample_anchors,
        save_hiD_reference,
        save_query_truth,
    )

    output = create_fresh_directory(
        job["outputs"][0], label="R0104 shared graph/reference output"
    )
    started = time.monotonic()
    proof = source_prefix_proof()
    source = InventoryFp16Array(0, ROWS)
    X = _materialize_normalized(source)
    materialize_seconds = time.monotonic() - started

    rng = np.random.RandomState(GRAPH_TRAIN_SEED)
    train_rows = np.sort(
        rng.choice(ROWS, GRAPH_TRAIN_ROWS, replace=False).astype(np.int64)
    )
    quantizer = faiss.IndexFlatIP(DIMENSION)
    cpu_index = faiss.IndexIVFFlat(
        quantizer, DIMENSION, GRAPH_NLIST, faiss.METRIC_INNER_PRODUCT
    )
    cpu_index.cp.seed = GRAPH_TRAIN_SEED
    cpu_index.cp.niter = 25
    cpu_index.cp.spherical = True
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(1 << 30)
    index = faiss.index_cpu_to_gpu(
        resource, 0, cpu_index, _faiss_gpu_options(faiss)
    )
    train_started = time.monotonic()
    index.train(np.ascontiguousarray(X[train_rows]))
    train_seconds = time.monotonic() - train_started
    add_started = time.monotonic()
    for start in range(0, ROWS, 100_000):
        index.add(np.ascontiguousarray(X[start : min(start + 100_000, ROWS)]))
    add_seconds = time.monotonic() - add_started
    if int(index.ntotal) != ROWS:
        raise Round0104Error("R0104 IVF-Flat index row count changed")

    quality_ids = np.sort(
        np.random.RandomState(GRAPH_QUALITY_SEED)
        .choice(ROWS, GRAPH_QUALITY_ROWS, replace=False)
        .astype(np.int64)
    )
    exact_cpu = faiss.IndexFlatIP(DIMENSION)
    exact = faiss.index_cpu_to_gpu(
        resource, 0, exact_cpu, _faiss_gpu_options(faiss)
    )
    for start in range(0, ROWS, 100_000):
        exact.add(np.ascontiguousarray(X[start : min(start + 100_000, ROWS)]))
    _truth_dist, truth_raw = exact.search(
        np.ascontiguousarray(X[quality_ids]), GRAPH_K
    )
    truth = _without_self(truth_raw, quality_ids, GRAPH_K - 1)
    cells: dict[str, Any] = {}
    selected: int | None = None
    for nprobe in GRAPH_NPROBE_GRID:
        index.nprobe = nprobe
        cell_started = time.monotonic()
        _distances, raw = index.search(
            np.ascontiguousarray(X[quality_ids]), GRAPH_K
        )
        wall = time.monotonic() - cell_started
        observed = _without_self(raw, quality_ids, GRAPH_K - 1)
        recalls = _recall_rows(observed, truth)
        mean = float(recalls.mean())
        p10 = float(np.percentile(recalls, 10))
        passed = (
            mean >= GRAPH_MEAN_RECALL_FLOOR
            and p10 >= GRAPH_P10_RECALL_FLOOR
        )
        cells[str(nprobe)] = {
            "nprobe": nprobe,
            "mean_recall_at_49": mean,
            "p10_recall_at_49": p10,
            "queries": GRAPH_QUALITY_ROWS,
            "wall_seconds": wall,
            "queries_per_second": GRAPH_QUALITY_ROWS / wall,
            "passed": passed,
        }
        if passed and selected is None:
            selected = nprobe
    del exact
    if selected is None:
        raise Round0104Error("no registered IVF-Flat graph cell passed recall")

    index.nprobe = selected
    neighbors = np.empty((ROWS, GRAPH_K), dtype=np.int32)
    distances = np.empty((ROWS, GRAPH_K), dtype=np.float32)
    search_started = time.monotonic()
    for start in range(0, ROWS, 16_384):
        stop = min(start + 16_384, ROWS)
        sims, ids = index.search(np.ascontiguousarray(X[start:stop]), GRAPH_K)
        if np.any(ids < 0) or np.any(ids >= ROWS):
            raise Round0104Error("full graph search returned invalid row IDs")
        neighbors[start:stop] = ids.astype(np.int32, copy=False)
        distances[start:stop] = np.maximum(
            0.0, 1.0 - sims
        ).astype(np.float32, copy=False)
    search_seconds = time.monotonic() - search_started

    fuzzy_started = time.monotonic()
    graph, sigmas, rhos = umap_api.fuzzy_simplicial_set(
        X,
        n_neighbors=GRAPH_K,
        random_state=np.random.RandomState(SEED),
        metric="cosine",
        knn_indices=neighbors,
        knn_dists=distances,
    )
    coo = graph.tocoo()
    sources = np.asarray(coo.row, dtype=np.int32)
    targets = np.asarray(coo.col, dtype=np.int32)
    weights = np.asarray(coo.data, dtype=np.float32)
    fuzzy_seconds = time.monotonic() - fuzzy_started
    if (
        len(sources) <= ROWS * (GRAPH_K - 1)
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise Round0104Error("fuzzy graph geometry/weights are invalid")
    graph_path = os.path.join(output, "edges-k50-fuzzy.npz")
    atomic_save_new_npz(
        graph_path,
        immutable=True,
        sources=sources,
        targets=targets,
        weights=weights,
        n_nodes=np.asarray(ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    graph_signature = expected_input_signature(graph_path)
    probe_rng = np.random.RandomState(106)
    probe_ids = probe_rng.choice(len(sources), 20_000, replace=False)
    endpoint_cosines = np.einsum(
        "ij,ij->i", X[sources[probe_ids]], X[targets[probe_ids]]
    )
    random_targets = probe_rng.randint(0, ROWS, size=len(probe_ids))
    random_cosines = np.einsum(
        "ij,ij->i", X[sources[probe_ids]], X[random_targets]
    )
    endpoint_probe = {
        "rows": len(probe_ids),
        "edge_cosine_mean": float(endpoint_cosines.mean()),
        "random_cosine_mean": float(random_cosines.mean()),
        "margin": float(endpoint_cosines.mean() - random_cosines.mean()),
    }
    if endpoint_probe["margin"] <= 0.15:
        raise Round0104Error("graph endpoint cosine margin is too small")

    graph_manifest = {
        "schema": "graph_manifest.v2",
        "n_nodes": ROWS,
        "n_edges": len(sources),
        "source_min": int(sources.min()),
        "source_max": int(sources.max()),
        "target_min": int(targets.min()),
        "target_max": int(targets.max()),
        "node_namespace": "contiguous_0..n_nodes",
        "directed": True,
        "k": GRAPH_K,
        "metric": "cosine",
        "metric_input": "R0103 exact fp16 source normalized in fp32",
        "weight_semantics": "fuzzy_simplicial_set(k50)",
        "graph_path": os.path.basename(graph_path),
        "graph_sha256": graph_signature["sha256"],
        "graph_bytes": graph_signature["bytes"],
        "data_len": ROWS,
        "data_shard_records": proof["segments"],
        "graph_construction_truth": {
            "source_prefix_proof": proof,
            "search": {
                "index": "GPU IndexIVFFlat/IP",
                "nlist": GRAPH_NLIST,
                "selected_nprobe": selected,
                "quality_cells": cells,
                "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
                "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
                "self_inclusive_k": GRAPH_K,
            },
            "graph_shared_by_both_arms": True,
            "graph_built_from_fp16_control": True,
        },
        "endpoint_cosine": endpoint_probe,
        "post_hoc_identity_verified": True,
        "verified_by": "round0104-self-contained-paired-builder-v2",
    }
    manifest_path = os.path.join(output, "graph-manifest.json")
    atomic_write_new_json(manifest_path, graph_manifest, immutable=True)
    graph_manifest_signature = expected_input_signature(manifest_path)

    cfg = panel_config()
    anchors = np.sort(
        np.random.RandomState(PANEL_SEED)
        .choice(ROWS, PANEL_ANCHORS, replace=False)
        .astype(np.int64)
    )
    reference = build_hiD_reference(
        X,
        anchors,
        cfg,
        centroids_by_k=None,
        data_identity=_data_identity(proof),
        convention={
            "row_order": "R0103 inventory first 2M rows",
            "distance": "cosine via fp32-L2-normalized squared L2",
            "self_exclusion": True,
            "anchor_namespace": "zero-based R0103 row IDs",
        },
    )
    reference_path = os.path.join(output, "high-d-reference.npz")
    save_hiD_reference(reference, reference_path)

    query_source = InventoryFp16Array(QUERY_START, QUERY_START + QUERY_ROWS)
    Xq = _materialize_normalized(query_source)
    query_identity = {
        "schema": "round0104-heldout-fineweb-query-identity-v2",
        "global_rows": [QUERY_START, QUERY_START + QUERY_ROWS],
        "ordered_fp32_normalized_sha256": ordered_array_sha256(Xq),
        "segments": query_source.segments,
        "disjoint_from_training": True,
    }
    truth = build_query_truth(
        Xq,
        X,
        cfg=cfg,
        corpus_identity=_data_identity(proof),
        query_identity=query_identity,
        k=10,
    )
    truth_path = os.path.join(output, "oos-query-truth-k10.npz")
    save_query_truth(truth, truth_path)
    query_path = os.path.join(output, "oos-query-fp16.npy")
    atomic_save_new_npy(
        query_path,
        np.asarray(query_source[:], dtype=np.float16),
        immutable=True,
    )
    receipt_body = {
        "schema": _schema(active, "paired-shared-evidence"),
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "source_prefix_proof": proof,
        "graph": graph_signature,
        "graph_manifest": graph_manifest_signature,
        "graph_edges": len(sources),
        "search_qualification": {
            "selected_nprobe": selected,
            "cells": cells,
            "quality_row_ids_sha256": ordered_array_sha256(quality_ids),
            "training_row_ids_sha256": ordered_array_sha256(train_rows),
        },
        "high_d_reference": expected_input_signature(reference_path),
        "high_d_reference_key": reference["key"],
        "high_d_reference_content_sha256": reference["content_sha256"],
        "query_fp16": expected_input_signature(query_path),
        "query_identity": query_identity,
        "query_truth": expected_input_signature(truth_path),
        "query_truth_key": truth["key"],
        "query_truth_payload_sha256": truth["payload_sha256"],
        "shared_across_arms": list(ARMS),
        "performance": {
            "materialize_seconds": materialize_seconds,
            "ivf_train_seconds": train_seconds,
            "ivf_add_seconds": add_seconds,
            "full_search_seconds": search_seconds,
            "fuzzy_seconds": fuzzy_seconds,
            "total_wall_seconds": time.monotonic() - started,
        },
    }
    receipt_path = os.path.join(output, "receipt.json")
    atomic_write_new_json(receipt_path, seal(receipt_body), immutable=True)
    del X, neighbors, distances, sources, targets, weights, graph, coo
    gc.collect()


def _load_shared(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.join(str(job["shared_output"]), "receipt.json")
    receipt = _read_sealed(path, label="R0104 shared evidence")
    if (
        receipt.get("round_id") != ROUND_ID
        or receipt.get("shared_across_arms") != list(ARMS)
        or int(receipt.get("graph_edges", -1)) <= 0
    ):
        raise Round0104Error("R0104 shared evidence content changed")
    for key in ("graph", "graph_manifest", "high_d_reference", "query_truth"):
        verify_signature(receipt.get(key), label=f"R0104 shared {key}")
    return receipt, expected_input_signature(path)


def _load_graph(shared: Mapping[str, Any]) -> dict[str, Any]:
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    graph_path = verify_signature(shared["graph"], label="R0104 paired graph")
    manifest_path = verify_signature(
        shared["graph_manifest"], label="R0104 graph manifest"
    )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("schema") != "graph_manifest.v2"
        or manifest.get("n_nodes") != ROWS
        or manifest.get("n_edges") != shared["graph_edges"]
        or manifest.get("graph_sha256") != shared["graph"]["sha256"]
        or not (manifest.get("graph_construction_truth") or {}).get(
            "graph_shared_by_both_arms"
        )
    ):
        raise Round0104Error("R0104 graph manifest changed")
    sources, targets, weights, n_nodes = load_edge_arrays(
        graph_path, load_weights=True
    )
    if (
        weights is None
        or int(n_nodes) != ROWS
        or len(sources) != shared["graph_edges"]
    ):
        raise Round0104Error("R0104 graph arrays changed")
    return {
        "signature": dict(shared["graph"]),
        "manifest_signature": dict(shared["graph_manifest"]),
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
    }


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
        n_neighbors=graph["k"],
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
        graph_manifest_path=graph["manifest_path"],
        graph_manifest_sha256=graph["manifest_sha256"],
    )


def _arm(job: Mapping[str, Any]) -> str:
    arm = str(job.get("arm") or "")
    if arm not in ARMS:
        raise Round0104Error(f"unknown R0104 arm {arm!r}")
    return arm


def run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    import torch

    arm = _arm(job)
    shared, shared_signature = _load_shared(job)
    graph = _load_graph(shared)
    config, config_sha = train_config(
        arm,
        graph_signature=shared["graph"],
        graph_manifest_signature=shared["graph_manifest"],
        graph_edges=shared["graph_edges"],
    )
    dataset, substrate = open_training_dataset(
        arm,
        verify_payloads=arm == "int8_treatment",
        buffer_rows=config["optimizer"]["batch_size"],
    )
    wrapper = Round0104TrainingInput(
        dataset, graph, arm=arm, required_pipeline=PIPELINE
    )
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0104 {arm} train output"
    )
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": _schema(active, "production-config"),
            "arm": arm,
            "config": config,
            "config_sha256": config_sha,
        },
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
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=shared["graph"]["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started
    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
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
        "n_pos_edges": shared["graph_edges"],
    }
    mismatches.update(
        {
            key: {"expected": value, "observed": accounting.get(key)}
            for key, value in exact.items()
            if accounting.get(key) != value
        }
    )
    expected_rows = SUCCESSFUL_UPDATES * config["optimizer"]["batch_size"]
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        runtime["source_rows_gathered"] != expected_rows
        or runtime["destination_rows_gathered"] != expected_rows
        or runtime["host_prefetch_consumer_batches"] != SUCCESSFUL_UPDATES
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    if mismatches:
        raise Round0104Error(f"R0104 {arm} train accounting failed: {mismatches}")
    synchronize_runtime_counters(accounting, runtime)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        raise Round0104Error(f"R0104 {arm} performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    checks = {
        "exact_update_closure": True,
        "zero_numerical_skips": True,
        "no_pipeline_stamp_drift": True,
        "endpoint_rows_match_updates": True,
    }
    receipt = {
        "schema": _schema(active, "paired-train-receipt"),
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": config_sha,
        "model": expected_input_signature(model_path),
        "shared_evidence": shared_signature,
        "graph": shared["graph"],
        "graph_manifest": shared["graph_manifest"],
        "substrate": substrate["signature"],
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
    }
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), seal(receipt), immutable=True
    )
    del model, wrapper, dataset, graph
    torch.cuda.empty_cache()
    gc.collect()


def _authenticate_model(job: Mapping[str, Any]):
    arm = _arm(job)
    shared, shared_signature = _load_shared(job)
    config, config_sha = train_config(
        arm,
        graph_signature=shared["graph"],
        graph_manifest_signature=shared["graph_manifest"],
        graph_edges=shared["graph_edges"],
    )
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train = _read_sealed(train_path, label=f"R0104 {arm} train receipt")
    if (
        train.get("arm") != arm
        or train.get("production_config_sha256") != config_sha
        or train.get("shared_evidence") != shared_signature
    ):
        raise Round0104Error("R0104 train receipt/config changed")
    model_path = verify_signature(train["model"], label=f"R0104 {arm} model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
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
        raise Round0104Error("R0104 model architecture changed")
    return model, train, expected_input_signature(train_path), shared, config_sha


def run_transform(active: dict[str, Any], job: dict[str, Any]) -> None:
    arm = _arm(job)
    model, train, train_signature, shared, config_sha = _authenticate_model(job)
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0104 {arm} transform output"
    )
    X, substrate = transform_array(arm, 0, ROWS)
    Xq, _ = transform_array(arm, QUERY_START, QUERY_START + QUERY_ROWS)
    started = time.monotonic()
    coordinates = np.asarray(model.transform(X, batch_size=8192), dtype=np.float32)
    query_coordinates = np.asarray(
        model.transform(Xq, batch_size=8192), dtype=np.float32
    )
    wall = time.monotonic() - started
    if (
        coordinates.shape != (ROWS, 2)
        or query_coordinates.shape != (QUERY_ROWS, 2)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(query_coordinates).all()
    ):
        raise Round0104Error(f"R0104 {arm} transform is invalid")
    coordinates_path = os.path.join(output, "coordinates.npy")
    queries_path = os.path.join(output, "oos-query-coordinates.npy")
    atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
    atomic_save_new_npy(queries_path, query_coordinates, immutable=True)
    receipt = {
        "schema": _schema(active, "paired-transform-receipt"),
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": config_sha,
        "train_receipt": train_signature,
        "model": train["model"],
        "shared_evidence": train["shared_evidence"],
        "input_preprocessing": preprocessing_stamp(arm),
        "substrate": substrate["signature"],
        "training_rows": [0, ROWS],
        "query_rows": [QUERY_START, QUERY_START + QUERY_ROWS],
        "coordinates": expected_input_signature(coordinates_path),
        "query_coordinates": expected_input_signature(queries_path),
        "wall_seconds": wall,
        "finite": True,
    }
    atomic_write_new_json(
        os.path.join(output, "transform-receipt.json"),
        seal(receipt),
        immutable=True,
    )


def _recall(high: np.ndarray, low: np.ndarray, k: int) -> float:
    return float(
        np.mean(
            [
                len(np.intersect1d(high[index, :k], low[index])) / k
                for index in range(len(high))
            ]
        )
    )


def run_score(active: dict[str, Any], job: dict[str, Any]) -> None:
    from basemap.panel_v2 import (
        cross_knn,
        ffr_from_neighbors,
        load_hiD_reference,
        load_query_truth,
        recall_at_k_from_neighbors,
        score_panel,
    )

    arm = _arm(job)
    shared, shared_signature = _load_shared(job)
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
    train = _read_sealed(train_path, label=f"R0104 {arm} train receipt")
    transform = _read_sealed(
        transform_path, label=f"R0104 {arm} transform receipt"
    )
    if (
        train.get("arm") != arm
        or transform.get("arm") != arm
        or train.get("production_config_sha256") != config_sha
        or transform.get("production_config_sha256") != config_sha
        or train.get("shared_evidence") != shared_signature
        or transform.get("shared_evidence") != shared_signature
    ):
        raise Round0104Error("R0104 score inputs changed")
    Z = np.load(
        verify_signature(transform["coordinates"], label=f"R0104 {arm} coordinates"),
        mmap_mode="r",
        allow_pickle=False,
    )
    Zq = np.load(
        verify_signature(
            transform["query_coordinates"],
            label=f"R0104 {arm} query coordinates",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    X = L2NormalizedArray(InventoryFp16Array(0, ROWS))
    cfg = panel_config()
    reference = load_hiD_reference(
        shared["high_d_reference"]["canonical_path"],
        expected_key=shared["high_d_reference_key"],
    )
    truth = load_query_truth(
        shared["query_truth"]["canonical_path"],
        expected_key=shared["query_truth_key"],
        expected_candidate_compute_backend="cuda",
    )
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0104 {arm} score output"
    )
    started = time.monotonic()
    panel = score_panel(
        X,
        Z,
        config=cfg,
        centroids_by_k=None,
        hiD_reference=reference,
        reference_identity={
            "data_identity": _data_identity(
                shared["source_prefix_proof"]
            ),
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
    low10 = low_fraction[:, : cfg.k_hit]
    high10 = np.asarray(truth["neighbors"], dtype=np.int64)[:, : cfg.k_hit]
    projection_ffr = ffr_from_neighbors(high10, low_fraction, cfg.k_hit)
    projection_recall = recall_at_k_from_neighbors(high10, low10, cfg.k_hit)
    low51 = cross_knn(
        np.asarray(Z[reference["anchor_ids"]], dtype=np.float32),
        Z,
        51,
        cfg,
        hi_dim=False,
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
        "transductive_recall50_gt_recall10": recall50
        > metrics["recall_at_10"],
        "projection_recall50_gt_recall10": query_recall50
        > metrics["oos_proj_recall_at_10"],
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
        raise Round0104Error("R0104 score metrics are nonfinite")
    body = {
        "schema": _schema(active, "paired-score"),
        "round_id": ROUND_ID,
        "arm": arm,
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": config_sha,
        "train_receipt": expected_input_signature(train_path),
        "transform_receipt": expected_input_signature(transform_path),
        "shared_evidence": shared_signature,
        "high_d_reference": shared["high_d_reference"],
        "query_truth": shared["query_truth"],
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
    }
    atomic_write_new_json(
        os.path.join(output, "score.json"), seal(body), immutable=True
    )


def run_decision(active: dict[str, Any], job: dict[str, Any]) -> None:
    output = create_fresh_directory(
        job["outputs"][0], label="R0104 paired decision output"
    )
    reports: dict[str, Any] = {}
    signatures: dict[str, Any] = {}
    for arm in ARMS:
        path = os.path.join(str(job["score_outputs"][arm]), "score.json")
        report = _read_sealed(path, label=f"R0104 {arm} score")
        if report.get("arm") != arm or set(report.get("metrics") or {}) != set(
            DECISION_METRICS
        ):
            raise Round0104Error("R0104 paired score content changed")
        reports[arm] = report
        signatures[arm] = expected_input_signature(path)
    if (
        reports["fp16_control"]["shared_evidence"]
        != reports["int8_treatment"]["shared_evidence"]
    ):
        raise Round0104Error("R0104 arms did not use identical shared evidence")
    decision = paired_decision(
        control=reports["fp16_control"],
        treatment=reports["int8_treatment"],
    )
    body = {
        "schema": _schema(active, "paired-decision"),
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "scores": signatures,
        "shared_evidence": reports["fp16_control"]["shared_evidence"],
        "registered_decision": decision,
        "capabilities_produced": (
            ["jina-full768-host-int8-training-validation-v1"]
            if decision["passed"]
            else []
        ),
        "terminal_negative": not decision["passed"],
    }
    atomic_write_new_json(
        os.path.join(output, "decision.json"), seal(body), immutable=True
    )


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0104 node received another queue")
    selected = job if job is not None else active.get("job") or {}
    action = selected.get("action")
    if action == "build_shared":
        run_build_shared(active, selected)
    elif action == "train":
        run_train(active, selected)
    elif action == "transform":
        run_transform(active, selected)
    elif action == "score":
        run_score(active, selected)
    elif action == "decide":
        run_decision(active, selected)
    else:
        raise RuntimeError(f"unknown R0104 action: {action!r}")
