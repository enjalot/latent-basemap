"""Execute the R0124 2M FineWeb graph-degree bridge."""
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
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import _self_knn, load_hiD_reference, sample_anchors
from basemap.round0113_prompt_contrast import (
    BATCH_SIZE,
    DIMENSION,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    POSITIVE_ROWS_PER_UPDATE,
    RETAINED_ROWS,
    SEED,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    HostFp16EndpointArray,
    read_sealed,
    seal,
    synchronize_runtime_counters,
    train_config as r0115_train_config,
    verify_signature,
)
from basemap.round0124_degree_bridge import (
    ARM,
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECISION_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    GRAPH_DEGREE,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE,
    GRAPH_NPROBE_GRID,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_QUALITY_ROWS,
    GRAPH_QUALITY_SEED,
    GRAPH_SCHEMA,
    GRAPH_SEARCH_NEIGHBORS,
    GRAPH_TRAIN_ROWS,
    GRAPH_TRAIN_SEED,
    MATERIAL_DENSITY_DEGRADATION,
    NATIVE_ANCHOR_SEED,
    NATIVE_DENSITY_ANCHORS,
    NATIVE_DENSITY_SCHEMA,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_MATERIAL,
    OUTCOME_NOT_MATERIAL,
    PRODUCTION_CONFIG_SCHEMA,
    ROUND_ID,
    TRAIN_RECEIPT_SCHEMA,
    DegreeBridgeTrainingInput,
    Round0124Error,
    classify_degree_bridge,
    graph_degree_stamp,
    load_graph,
    paired_density_bootstrap,
    train_config,
    verify_retry_provenance,
)
from experiments import round0113_nodes as prompt_nodes


def _execution_round_id(active: Mapping[str, Any]) -> str:
    round_id = str((active.get("manifest") or {}).get("round_id", ""))
    if round_id != ROUND_ID:
        raise Round0124Error("R0124 handler received another queue")
    return round_id


def _reused_graph(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    retry = verify_retry_provenance(job.get("retry_provenance"))
    reused = retry["reused_graph"]
    manifest = reused["manifest"]
    graph_path = str(job.get("graph_manifest") or "")
    if graph_path != manifest["canonical_path"]:
        raise Round0124Error("R0124 retry graph path changed")
    graph = load_graph(
        graph_path,
        expected_manifest_signature=manifest,
        expected_graph_signature=reused["graph"],
        expected_topology_probe_signature=reused["topology_probe"],
        expected_release_sha=reused["source_release_sha"],
    )
    return graph, retry


def _verify_train_accounting(
    *,
    accounting: Mapping[str, Any],
    runtime: Mapping[str, Any],
    expected_stamp: Mapping[str, Any],
    expected_edges: int,
    label: str,
) -> None:
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
        "n_pos_edges": expected_edges,
    }
    mismatches = {
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    }
    mismatches.update(
        {
            f"runtime.{key}": {
                "expected": value,
                "observed": runtime.get(key),
            }
            for key, value in expected_stamp.items()
            if runtime.get(key) != value
        }
    )
    if accounting.get("pipeline_runtime") != dict(runtime):
        mismatches["pipeline_runtime"] = "does not mirror exact execution"
    pipeline_mismatches = {
        key: {
            "expected": value,
            "observed": accounting.get(f"pipeline_{key}"),
        }
        for key, value in runtime.items()
        if accounting.get(f"pipeline_{key}") != value
    }
    if pipeline_mismatches:
        mismatches["pipeline_fields"] = pipeline_mismatches
    expected_rows = SUCCESSFUL_UPDATES * BATCH_SIZE
    producer_delta = (
        int(runtime.get("host_prefetch_producer_batches", -1))
        - int(runtime.get("host_prefetch_consumer_batches", -1))
    )
    if (
        runtime.get("endpoint_gather_calls") != SUCCESSFUL_UPDATES
        or runtime.get("source_rows_gathered") != expected_rows
        or runtime.get("destination_rows_gathered") != expected_rows
        or runtime.get("host_prefetch_consumer_batches")
        != SUCCESSFUL_UPDATES
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "producer_delta": producer_delta,
        }
    weighted = prompt_nodes._weighted_rejection_accounting_mismatch(
        runtime,
        producer_delta=producer_delta,
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        raise Round0124Error(f"{label} train accounting changed: {mismatches}")


def _explicit_self_knn(
    ids: np.ndarray,
    similarities: np.ndarray,
    source_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize one self plus exactly 15 ranked nonself neighbors."""
    rows = np.asarray(source_rows, dtype=np.int64)
    neighbors = np.asarray(ids, dtype=np.int64)
    sims = np.asarray(similarities, dtype=np.float32)
    if (
        neighbors.shape != (len(rows), GRAPH_SEARCH_NEIGHBORS)
        or sims.shape != neighbors.shape
        or np.any(neighbors < 0)
        or np.any(neighbors >= RETAINED_ROWS)
    ):
        raise Round0124Error("R0124 search result geometry changed")
    self_mask = neighbors == rows[:, None]
    if not np.all(self_mask.sum(1) == 1):
        raise Round0124Error("R0124 search did not return exactly one self")
    nonself = ~self_mask
    selected = neighbors[nonself].reshape(len(rows), GRAPH_DEGREE)
    selected_sims = sims[nonself].reshape(len(rows), GRAPH_DEGREE)
    if np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
        raise Round0124Error("R0124 search returned duplicate nonself rows")
    canonical_ids = np.empty(
        (len(rows), GRAPH_SEARCH_NEIGHBORS), dtype=np.int32
    )
    canonical_distances = np.zeros_like(canonical_ids, dtype=np.float32)
    canonical_ids[:, 0] = rows.astype(np.int32)
    canonical_ids[:, 1:] = selected.astype(np.int32)
    canonical_distances[:, 1:] = np.maximum(
        0.0, 1.0 - selected_sims
    ).astype(np.float32)
    return canonical_ids, canonical_distances, selected


def _validate_control_topology_prefix(
    *,
    quality_ids: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    control_anchor_ids: np.ndarray,
    control_exact: np.ndarray,
    control_ann: np.ndarray,
) -> None:
    if (
        control_anchor_ids.shape != (GRAPH_QUALITY_ROWS,)
        or control_exact.shape[0] != GRAPH_QUALITY_ROWS
        or control_ann.shape != control_exact.shape
        or control_exact.shape[1] < GRAPH_DEGREE
        or not np.array_equal(control_anchor_ids, quality_ids)
        or not np.array_equal(control_exact[:, :GRAPH_DEGREE], truth)
        or not np.array_equal(control_ann[:, :GRAPH_DEGREE], observed)
    ):
        raise Round0124Error(
            "R0124 k15 topology is not the exact R0115 ranked prefix"
        )


def _control_graph(
    job: Mapping[str, Any],
    *,
    assembly_signature: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = str(job["r0115_control_graph_manifest"])
    signature = expected_input_signature(path)
    control = read_sealed(path, label="R0115 raw k50 control graph")
    if (
        control.get("schema") != "round0113-prompt-arm-fuzzy-graph-v1"
        or control.get("round_id") != "0115"
        or control.get("release_sha") != job.get("r0115_release_sha")
        or control.get("arm") != ARM
        or int(control.get("retained_rows", -1)) != RETAINED_ROWS
        or int(control.get("dimension", -1)) != DIMENSION
        or int(control.get("k", -1)) != 50
        or control.get("assembly") != dict(assembly_signature)
    ):
        raise Round0124Error("R0115 control graph identity changed")
    verify_signature(
        control["high_d_reference"], label="R0115 raw high-D reference"
    )
    verify_signature(
        control["query_training_copy_mask"],
        label="R0115 raw query copy mask",
    )
    verify_signature(
        control["polish_query_training_copy_mask"],
        label="R0115 raw Polish copy mask",
    )
    return control, signature


def run_build_graph(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss
    import umap.umap_ as umap_api

    _execution_round_id(active)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0124 raw k15 graph"
    )
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    query, query_signature = prompt_nodes._load_query_reserve(job)
    control, control_signature = _control_graph(
        job, assembly_signature=assembly_signature
    )
    source = prompt_nodes._open_compact(assembly, ARM)
    started = time.monotonic()
    materialize_started = time.monotonic()
    X = prompt_nodes._materialize_normalized(source)
    materialize_seconds = time.monotonic() - materialize_started

    train_rows = np.sort(
        np.random.RandomState(GRAPH_TRAIN_SEED)
        .choice(RETAINED_ROWS, GRAPH_TRAIN_ROWS, replace=False)
        .astype(np.int64)
    )
    quantizer = faiss.IndexFlatIP(DIMENSION)
    cpu_index = faiss.IndexIVFFlat(
        quantizer, DIMENSION, GRAPH_NLIST, faiss.METRIC_INNER_PRODUCT
    )
    cpu_index.cp.seed = GRAPH_TRAIN_SEED
    cpu_index.cp.niter = 25
    cpu_index.cp.spherical = True
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    index = faiss.index_cpu_to_gpu(
        resources, 0, cpu_index, prompt_nodes._faiss_gpu_options(faiss)
    )
    graph_train_started = time.monotonic()
    index.train(np.ascontiguousarray(X[train_rows]))
    graph_train_seconds = time.monotonic() - graph_train_started
    add_started = time.monotonic()
    for start in range(0, RETAINED_ROWS, 100_000):
        index.add(
            np.ascontiguousarray(
                X[start : min(start + 100_000, RETAINED_ROWS)]
            )
        )
    add_seconds = time.monotonic() - add_started
    if int(index.ntotal) != RETAINED_ROWS:
        raise Round0124Error("R0124 IVF row count changed")

    quality_ids = np.sort(
        np.random.RandomState(GRAPH_QUALITY_SEED)
        .choice(RETAINED_ROWS, GRAPH_QUALITY_ROWS, replace=False)
        .astype(np.int64)
    )
    control_search = control.get("search_qualification") or {}
    if (
        control_search.get("training_rows_sha256")
        != ordered_array_sha256(train_rows)
        or control_search.get("quality_rows_sha256")
        != ordered_array_sha256(quality_ids)
        or int(control_search.get("selected_nprobe", -1)) != GRAPH_NPROBE
    ):
        raise Round0124Error(
            "R0124 IVF seeds/panel/nprobe differ from the R0115 control"
        )
    exact = faiss.index_cpu_to_gpu(
        resources,
        0,
        faiss.IndexFlatIP(DIMENSION),
        prompt_nodes._faiss_gpu_options(faiss),
    )
    for start in range(0, RETAINED_ROWS, 100_000):
        exact.add(
            np.ascontiguousarray(
                X[start : min(start + 100_000, RETAINED_ROWS)]
            )
        )
    truth_sims, truth_ids = exact.search(
        np.ascontiguousarray(X[quality_ids]), GRAPH_SEARCH_NEIGHBORS
    )
    _truth_full, _truth_distances, truth = _explicit_self_knn(
        truth_ids, truth_sims, quality_ids
    )
    cells: dict[str, Any] = {}
    selected_observed: np.ndarray | None = None
    for nprobe in GRAPH_NPROBE_GRID:
        index.nprobe = nprobe
        cell_started = time.monotonic()
        sims, ids = index.search(
            np.ascontiguousarray(X[quality_ids]),
            GRAPH_SEARCH_NEIGHBORS,
        )
        wall = time.monotonic() - cell_started
        _full, _distances, observed = _explicit_self_knn(
            ids, sims, quality_ids
        )
        recalls = prompt_nodes._recall_rows(observed, truth)
        passed = bool(
            recalls.mean() >= GRAPH_MEAN_RECALL_FLOOR
            and np.percentile(recalls, 10) >= GRAPH_P10_RECALL_FLOOR
        )
        cells[str(nprobe)] = {
            "mean_recall_at_15": float(recalls.mean()),
            "p10_recall_at_15": float(np.percentile(recalls, 10)),
            "wall_s": wall,
            "queries_per_s": GRAPH_QUALITY_ROWS / wall,
            "passed": passed,
        }
        if nprobe == GRAPH_NPROBE:
            selected_observed = observed.copy()
    del exact
    fixed = cells.get(str(GRAPH_NPROBE)) or {}
    if fixed.get("passed") is not True or selected_observed is None:
        raise Round0124Error("R0124 fixed-nprobe k15 graph did not qualify")
    control_probe_path = verify_signature(
        control["topology_probe"], label="R0115 raw topology probe"
    )
    with np.load(control_probe_path, allow_pickle=False) as archive:
        control_anchor_ids = np.asarray(
            archive["anchor_compact_ids"], dtype=np.int64
        )
        control_exact = np.asarray(
            archive["exact_neighbors"], dtype=np.int64
        )
        control_ann = np.asarray(
            archive["qualified_ann_neighbors"], dtype=np.int64
        )
    _validate_control_topology_prefix(
        quality_ids=quality_ids,
        truth=truth,
        observed=selected_observed,
        control_anchor_ids=control_anchor_ids,
        control_exact=control_exact,
        control_ann=control_ann,
    )

    index.nprobe = GRAPH_NPROBE
    neighbors = np.empty(
        (RETAINED_ROWS, GRAPH_SEARCH_NEIGHBORS), dtype=np.int32
    )
    distances = np.empty_like(neighbors, dtype=np.float32)
    search_started = time.monotonic()
    for start in range(0, RETAINED_ROWS, 16_384):
        stop = min(start + 16_384, RETAINED_ROWS)
        rows = np.arange(start, stop, dtype=np.int64)
        sims, ids = index.search(
            np.ascontiguousarray(X[start:stop]),
            GRAPH_SEARCH_NEIGHBORS,
        )
        canonical_ids, canonical_distances, _nonself = _explicit_self_knn(
            ids, sims, rows
        )
        neighbors[start:stop] = canonical_ids
        distances[start:stop] = canonical_distances
    search_seconds = time.monotonic() - search_started

    fuzzy_started = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        X,
        n_neighbors=GRAPH_SEARCH_NEIGHBORS,
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
        len(sources) <= RETAINED_ROWS * GRAPH_DEGREE
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or np.any(weights > 1)
    ):
        raise Round0124Error("R0124 fuzzy graph arrays are invalid")
    graph_path = os.path.join(output, "edges-k15-fuzzy.npz")
    atomic_save_new_npz(
        graph_path,
        immutable=True,
        compressed=False,
        sources=sources,
        targets=targets,
        weights=weights,
        n_nodes=np.asarray(RETAINED_ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_DEGREE, dtype=np.int64),
        n_neighbors_including_self=np.asarray(
            GRAPH_SEARCH_NEIGHBORS, dtype=np.int64
        ),
    )
    topology_path = os.path.join(output, "topology-probe.npz")
    atomic_save_new_npz(
        topology_path,
        immutable=True,
        compressed=False,
        anchor_compact_ids=quality_ids,
        exact_neighbors=truth,
        qualified_ann_neighbors=selected_observed,
    )
    body = {
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "arm": ARM,
        "retained_rows": RETAINED_ROWS,
        "dimension": DIMENSION,
        "degree": graph_degree_stamp(),
        "directed_edge_count": int(len(sources)),
        "graph": expected_input_signature(graph_path),
        "assembly": assembly_signature,
        "compact_mapping": assembly["mapping"],
        "source": assembly["outputs"][ARM],
        "substrate": assembly["substrate"],
        "query_reserve": query_signature,
        "query_training_copy_mask": control["query_training_copy_mask"],
        "query_training_copy_audit": control["query_training_copy_audit"],
        "polish_query_training_copy_mask": (
            control["polish_query_training_copy_mask"]
        ),
        "polish_query_training_copy_audit": (
            control["polish_query_training_copy_audit"]
        ),
        "search_qualification": {
            "index": "GPU IndexIVFFlat/IP",
            "selected_nprobe": GRAPH_NPROBE,
            "selection_policy": (
                "fixed R0115 nprobe; grid diagnostic only; k15 qualified "
                "independently"
            ),
            "cells": cells,
            "training_rows_sha256": ordered_array_sha256(train_rows),
            "quality_rows_sha256": ordered_array_sha256(quality_ids),
            "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
            "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
        },
        "topology_probe": expected_input_signature(topology_path),
        "high_d_reference": control["high_d_reference"],
        "high_d_reference_key": control["high_d_reference_key"],
        "high_d_reference_content_sha256": (
            control["high_d_reference_content_sha256"]
        ),
        "causal_control_graph": control_signature,
        "control_topology_prefix_audit": {
            "control_probe": control["topology_probe"],
            "anchor_ids_equal": True,
            "exact_first_15_equal": True,
            "qualified_ann_first_15_equal": True,
            "control_width": int(control_exact.shape[1]),
            "treatment_width": GRAPH_DEGREE,
        },
        "causal_change": {
            "changed": "fuzzy graph neighbor degree",
            "control": {
                "label": "R0115 k50",
                "search_neighbors_including_self": 50,
                "nonself_neighbors": 49,
            },
            "treatment": graph_degree_stamp(),
            "all_other_builder_parameters_and_seeds_fixed": True,
        },
        "performance": {
            "materialize_s": materialize_seconds,
            "ivf_train_s": graph_train_seconds,
            "ivf_add_s": add_seconds,
            "full_search_s": search_seconds,
            "fuzzy_s": fuzzy_seconds,
            "total_wall_s": time.monotonic() - started,
        },
        "training_performed": False,
    }
    manifest = seal(body)
    path = os.path.join(output, "graph-manifest.json")
    atomic_write_new_json(path, manifest, immutable=True)
    del X, graph, coo, neighbors, distances, sources, targets, weights, source
    gc.collect()
    return {**manifest, "receipt": expected_input_signature(path)}


def _new_model(config: Mapping[str, Any]):
    model = prompt_nodes._new_model(config)
    model.n_epochs = int(config["execution"]["training_loop_plan"]["n_epochs"])
    return model


def run_train(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    _execution_round_id(active)
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    graph, retry = _reused_graph(job)
    config, config_sha = train_config(
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=graph["n_nodes"],
    )
    source = prompt_nodes._open_compact(assembly, ARM)
    dataset = HostFp16EndpointArray(
        source,
        arm=ARM,
        source_signature=assembly["outputs"][ARM],
        mapping_signature=assembly["mapping"],
        buffer_rows=BATCH_SIZE,
    )
    wrapper = DegreeBridgeTrainingInput(dataset, graph)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0124 k15 treatment train"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "arm": ARM,
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
        precomputed_edges_path=graph["signature"]["canonical_path"],
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
        "n_pos_edges": len(graph["sources"]),
    }
    mismatches.update(
        {
            key: {"expected": value, "observed": accounting.get(key)}
            for key, value in exact.items()
            if accounting.get(key) != value
        }
    )
    expected_rows = SUCCESSFUL_UPDATES * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != SUCCESSFUL_UPDATES
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    weighted = prompt_nodes._weighted_rejection_accounting_mismatch(
        runtime, producer_delta=producer_delta
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        raise Round0124Error(f"R0124 train accounting failed: {mismatches}")
    synchronize_runtime_counters(accounting, runtime)
    _verify_train_accounting(
        accounting=accounting,
        runtime=runtime,
        expected_stamp=expected_stamp,
        expected_edges=len(graph["sources"]),
        label="R0124 treatment pre-seal",
    )
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES)
        / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        raise Round0124Error("R0124 treatment performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    body = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "arm": ARM,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "model": expected_input_signature(model_path),
        "assembly": assembly_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "retry_provenance": retry,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
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
        "causal_change": "graph-degree-k50-to-k15-only",
        "training_performed": True,
        "optimizer_updates": SUCCESSFUL_UPDATES,
        "map_decision_made": False,
    }
    receipt = seal(body)
    path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del model, wrapper, dataset, source, graph
    torch.cuda.empty_cache()
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _authenticate_treatment_model(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _execution_round_id(active)
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    graph, retry = _reused_graph(job)
    config, config_sha = train_config(
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=graph["n_nodes"],
    )
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train = read_sealed(train_path, label="R0124 treatment train receipt")
    config_path = verify_signature(
        train.get("production_config"),
        label="R0124 treatment production config",
    )
    with open(config_path, encoding="utf-8") as handle:
        config_receipt = json.load(handle)
    runtime = train.get("exact_execution_receipt")
    accounting = train.get("train_accounting")
    checks = train.get("train_checks")
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    if (
        train.get("schema") != TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != ROUND_ID
        or train.get("arm") != ARM
        or train.get("release_sha") != active["manifest"]["release_sha"]
        or train.get("production_config_sha256") != config_sha
        or config_receipt.get("schema") != PRODUCTION_CONFIG_SCHEMA
        or config_receipt.get("round_id") != ROUND_ID
        or config_receipt.get("arm") != ARM
        or config_receipt.get("config") != config
        or config_receipt.get("config_sha256") != config_sha
        or train.get("assembly") != assembly_signature
        or train.get("graph_manifest") != graph["manifest_signature"]
        or train.get("graph") != graph["signature"]
        or train.get("retry_provenance") != retry
        or train.get("optimizer_updates") != SUCCESSFUL_UPDATES
        or not isinstance(runtime, Mapping)
        or any(runtime.get(key) != value for key, value in expected_stamp.items())
        or not isinstance(accounting, Mapping)
        or any(
            accounting.get(key) != value
            for key, value in {
                "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
                "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
                "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
                "scheduler_steps": SUCCESSFUL_UPDATES,
                "amp_overflow_skips": 0,
                "nonfinite_loss_skips": 0,
                "nonfinite_gradient_skips": 0,
                "budget_satisfied": True,
            }.items()
        )
        or not isinstance(checks, Mapping)
        or any(
            checks.get(key) is not True
            for key in (
                "exact_update_closure",
                "zero_numerical_skips",
                "no_pipeline_stamp_drift",
                "endpoint_rows_match_updates",
                "weighted_rejection_accounting_closes",
            )
        )
    ):
        raise Round0124Error("R0124 treatment train/config binding changed")
    _verify_train_accounting(
        accounting=accounting,
        runtime=runtime,
        expected_stamp=expected_stamp,
        expected_edges=len(graph["sources"]),
        label="R0124 treatment",
    )
    model_path = verify_signature(train["model"], label="R0124 treatment model")
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
        raise Round0124Error("R0124 treatment architecture changed")
    return model, train, assembly, graph


def run_diagnostics(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the reviewed R0115 panel implementation with R0124 auth."""
    _execution_round_id(active)
    originals = {
        "_authenticate_model": prompt_nodes._authenticate_model,
        "_execution_round_id": prompt_nodes._execution_round_id,
        "_schema": prompt_nodes._schema,
    }
    prompt_nodes._authenticate_model = _authenticate_treatment_model
    prompt_nodes._execution_round_id = _execution_round_id
    prompt_nodes._schema = (
        lambda stem: DIAGNOSTIC_SCHEMA
        if stem == "prompt-arm-score"
        else f"round0124-{stem}-v1"
    )
    try:
        result = prompt_nodes.run_evaluate(dict(active), dict(job))
    finally:
        for name, value in originals.items():
            setattr(prompt_nodes, name, value)
    if result.get("schema") != DIAGNOSTIC_SCHEMA:
        raise Round0124Error("R0124 diagnostic schema changed")
    return result


def _r0115_native_evidence(
    job: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    graph, graph_signature = _control_graph(
        job,
        assembly_signature=assembly_signature,
    )
    train_path = str(job["r0115_control_train_receipt"])
    score_path = str(job["r0115_control_score"])
    train_signature = expected_input_signature(train_path)
    score_signature = expected_input_signature(score_path)
    train = read_sealed(train_path, label="R0115 raw control train receipt")
    score = read_sealed(score_path, label="R0115 raw native score")
    control_config, control_config_sha = r0115_train_config(
        ARM,
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=int(graph["retained_rows"]),
    )
    production_config_path = verify_signature(
        train.get("production_config"),
        label="R0115 raw control production config",
    )
    with open(production_config_path, encoding="utf-8") as handle:
        production_config = json.load(handle)
    accounting = train.get("train_accounting")
    runtime = train.get("exact_execution_receipt")
    checks = train.get("train_checks")
    coordinates = (score.get("coordinates") or {}).get("training")
    panel = score.get("panel")
    expected_runtime = control_config["execution"]["expected_pipeline_stamp"]
    if (
        train.get("schema") != "round0113-train-receipt-v1"
        or train.get("round_id") != "0115"
        or train.get("release_sha") != job.get("r0115_release_sha")
        or train.get("arm") != ARM
        or train.get("assembly") != assembly_signature
        or train.get("graph_manifest") != graph_signature
        or train.get("graph") != graph["graph"]
        or train.get("production_config_sha256") != control_config_sha
        or train.get("optimizer_updates") != SUCCESSFUL_UPDATES
        or train.get("training_performed") is not True
        or production_config.get("schema")
        != "round0113-production-config-v1"
        or production_config.get("round_id") != "0115"
        or production_config.get("arm") != ARM
        or production_config.get("config") != control_config
        or production_config.get("config_sha256") != control_config_sha
        or not isinstance(accounting, Mapping)
        or not isinstance(runtime, Mapping)
        or any(
            runtime.get(key) != value
            for key, value in expected_runtime.items()
        )
        or not isinstance(checks, Mapping)
        or any(
            checks.get(key) is not True
            for key in (
                "exact_update_closure",
                "zero_numerical_skips",
                "no_pipeline_stamp_drift",
                "endpoint_rows_match_updates",
                "weighted_rejection_accounting_closes",
            )
        )
        or score.get("schema") != "round0113-prompt-arm-score-v1"
        or score.get("round_id") != "0115"
        or score.get("release_sha") != job.get("r0115_release_sha")
        or score.get("arm") != ARM
        or score.get("graph_manifest") != graph_signature
        or score.get("train_receipt") != train_signature
        or score.get("high_d_reference") != graph["high_d_reference"]
        or not isinstance(coordinates, Mapping)
        or not isinstance(panel, Mapping)
        or panel.get("n") != RETAINED_ROWS
        or panel.get("n_anchors") != NATIVE_DENSITY_ANCHORS
        or panel.get("k_density") != GRAPH_DEGREE
        or panel.get("density") != 0.2304
    ):
        raise Round0124Error("R0115 native control evidence changed")
    _verify_train_accounting(
        accounting=accounting,
        runtime=runtime,
        expected_stamp=expected_runtime,
        expected_edges=int(graph["directed_edge_count"]),
        label="R0115 raw control",
    )
    verify_signature(train["model"], label="R0115 raw control model")
    verify_signature(coordinates, label="R0115 raw native coordinates")
    verify_signature(
        graph["high_d_reference"],
        label="R0115 raw native high-D reference",
    )
    return (
        graph,
        graph_signature,
        train,
        train_signature,
        score,
        score_signature,
    )


def _context_evidence(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_signature = dict(job["r0106_context_graph"])
    core_signature = dict(job["r0108_context_core"])
    graph = read_sealed(
        verify_signature(
            graph_signature,
            label="R0106 25M k15 context graph",
        ),
        label="R0106 25M k15 context graph",
    )
    core = read_sealed(
        verify_signature(
            core_signature,
            label="R0108 25M native-density context",
        ),
        label="R0108 25M native-density context",
    )
    density = ((core.get("metrics") or {}).get("density_v2") or {})
    if (
        graph.get("schema")
        != "round0106-jina-diverse-25m-fuzzy-graph-v1"
        or graph.get("round_id") != "0106"
        or graph.get("k_real") != GRAPH_DEGREE
        or graph.get("n_neighbors_including_self")
        != GRAPH_SEARCH_NEIGHBORS
        or (graph.get("knn_topology") or {}).get(
            "distinct_nonself_neighbors_per_source"
        )
        != GRAPH_DEGREE
        or core.get("schema")
        != "round0108-diverse-jina-core-geometry-v1"
        or core.get("round_id") != "0108"
        or core.get("graph_manifest") != graph_signature
        or density.get("correlation") != 0.15773929111469354
    ):
        raise Round0124Error("R0106/R0108 context evidence changed")
    return graph_signature, core_signature


def run_native_density(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    _execution_round_id(active)
    retry = verify_retry_provenance(job.get("retry_provenance"))
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0124 native density contrast"
    )
    (
        control_graph,
        control_graph_signature,
        _control_train,
        control_train_signature,
        control_score,
        control_score_signature,
    ) = _r0115_native_evidence(job)
    context_graph_signature, context_core_signature = _context_evidence(job)
    diagnostic_path = os.path.join(
        str(job["diagnostic_output"]),
        "score.json",
    )
    diagnostics = read_sealed(
        diagnostic_path,
        label="R0124 treatment native diagnostics",
    )
    treatment_coordinates = (
        (diagnostics.get("coordinates") or {}).get("training") or {}
    )
    treatment_train_signature = expected_input_signature(
        os.path.join(str(job["train_output"]), "train-receipt.json")
    )
    treatment_graph_signature = retry["reused_graph"]["manifest"]
    if str(job.get("graph_manifest") or "") != treatment_graph_signature[
        "canonical_path"
    ]:
        raise Round0124Error("R0124 retry graph path changed")
    if (
        diagnostics.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("round_id") != ROUND_ID
        or diagnostics.get("release_sha")
        != active["manifest"]["release_sha"]
        or diagnostics.get("arm") != ARM
        or diagnostics.get("train_receipt") != treatment_train_signature
        or diagnostics.get("graph_manifest") != treatment_graph_signature
        or diagnostics.get("high_d_reference")
        != control_graph["high_d_reference"]
        or not isinstance(treatment_coordinates, Mapping)
    ):
        raise Round0124Error("R0124 treatment native diagnostics changed")
    control_coordinates = control_score["coordinates"]["training"]
    control_path = verify_signature(
        control_coordinates,
        label="R0115 raw native coordinates",
    )
    treatment_path = verify_signature(
        treatment_coordinates,
        label="R0124 k15 native coordinates",
    )
    reference_path = verify_signature(
        control_graph["high_d_reference"],
        label="R0115 raw native high-D reference",
    )
    reference = load_hiD_reference(
        reference_path,
        expected_key=str(control_graph["high_d_reference_key"]),
    )
    config = prompt_nodes.panel_config()
    anchors = np.asarray(reference["anchor_ids"], dtype=np.int64)
    high_radius = np.asarray(reference["r_hd"], dtype=np.float64)
    if (
        config.n_anchors != NATIVE_DENSITY_ANCHORS
        or config.anchor_seed != NATIVE_ANCHOR_SEED
        or config.k_density != GRAPH_DEGREE
        or anchors.shape != (NATIVE_DENSITY_ANCHORS,)
        or high_radius.shape != anchors.shape
        or not np.array_equal(
            anchors,
            sample_anchors(RETAINED_ROWS, config),
        )
    ):
        raise Round0124Error("R0115 native density anchor contract changed")
    control_coordinates_array = np.load(
        control_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    treatment_coordinates_array = np.load(
        treatment_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        control_coordinates_array.shape != (RETAINED_ROWS, 2)
        or treatment_coordinates_array.shape != (RETAINED_ROWS, 2)
        or not np.isfinite(control_coordinates_array).all()
        or not np.isfinite(treatment_coordinates_array).all()
    ):
        raise Round0124Error("R0124 native coordinate geometry changed")
    _control_neighbors, control_distances, _control_guard = _self_knn(
        control_coordinates_array,
        anchors,
        GRAPH_DEGREE,
        config,
        hi_dim=False,
        want_dist=True,
    )
    _treatment_neighbors, treatment_distances, _treatment_guard = _self_knn(
        treatment_coordinates_array,
        anchors,
        GRAPH_DEGREE,
        config,
        hi_dim=False,
        want_dist=True,
    )
    control_low_radius = np.asarray(control_distances).mean(1)
    treatment_low_radius = np.asarray(treatment_distances).mean(1)
    bootstrap = paired_density_bootstrap(
        high_radius=high_radius,
        control_low_radius=control_low_radius,
        treatment_low_radius=treatment_low_radius,
    )
    bootstrap_deltas = np.asarray(
        bootstrap.pop("bootstrap_deltas"),
        dtype=np.float64,
    )
    selector = classify_degree_bridge(
        control_density=float(bootstrap["control_density"]),
        treatment_density=float(bootstrap["treatment_density"]),
        delta_ci_low=float(bootstrap["paired_bootstrap_delta_ci"][0]),
        delta_ci_high=float(bootstrap["paired_bootstrap_delta_ci"][1]),
    )
    bootstrap_summary = {
        key: value
        for key, value in bootstrap.items()
        if key not in selector
    }
    if (
        round(float(bootstrap["control_density"]), 4)
        != control_score["metrics"]["density"]
        or round(float(bootstrap["treatment_density"]), 4)
        != diagnostics["metrics"]["density"]
    ):
        raise Round0124Error(
            "R0124 native density does not reproduce panel scores"
        )
    arrays_path = os.path.join(output, "native-density-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        anchor_compact_rows=anchors,
        high_radius=high_radius,
        control_low_radius=control_low_radius,
        treatment_low_radius=treatment_low_radius,
        paired_bootstrap_deltas=bootstrap_deltas,
    )
    body = {
        "schema": NATIVE_DENSITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "retry_provenance": retry,
        "control": {
            "role": "exact R0115 raw seed-42 k50 native re-score",
            "score": control_score_signature,
            "train_receipt": control_train_signature,
            "graph_manifest": control_graph_signature,
            "coordinates": control_coordinates,
            "density": bootstrap["control_density"],
            "recorded_panel_density": control_score["metrics"]["density"],
        },
        "treatment": {
            "role": "R0124 raw seed-42 k15 native score",
            "diagnostics": expected_input_signature(diagnostic_path),
            "train_receipt": treatment_train_signature,
            "graph_manifest": treatment_graph_signature,
            "coordinates": treatment_coordinates,
            "density": bootstrap["treatment_density"],
            "recorded_panel_density": diagnostics["metrics"]["density"],
        },
        "native_reference": {
            "high_d_reference": control_graph["high_d_reference"],
            "anchor_count": len(anchors),
            "anchor_seed": config.anchor_seed,
            "k_density": config.k_density,
            "low_d_search": (
                "panel-v2 exact global chunked top-k; mean k15 radius"
            ),
        },
        "registered_selector": selector,
        "bootstrap_diagnostics": bootstrap_summary,
        "arrays": expected_input_signature(arrays_path),
        "context_only": {
            "r0106_25m_k15_graph": context_graph_signature,
            "r0108_25m_seed42_native_density": context_core_signature,
            "context_can_change_selector": False,
        },
        "changed_factor": "fuzzy graph neighbor degree only",
        "core_and_ood_diagnostics_registered_role": "diagnostic-only",
        "legacy_density_floor_used": False,
        "training_performed_in_this_node": False,
    }
    score = seal(body)
    path = os.path.join(output, "native-density-score.json")
    atomic_write_new_json(path, score, immutable=True)
    del control_coordinates_array, treatment_coordinates_array
    gc.collect()
    return {**score, "receipt": expected_input_signature(path)}


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    _execution_round_id(active)
    retry = verify_retry_provenance(job.get("retry_provenance"))
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0124 degree-bridge decision"
    )
    density_path = os.path.join(
        str(job["density_output"]), "native-density-score.json"
    )
    diagnostic_path = os.path.join(
        str(job["diagnostic_output"]), "score.json"
    )
    density = read_sealed(density_path, label="R0124 density score")
    diagnostics = read_sealed(
        diagnostic_path, label="R0124 core/OOD diagnostics"
    )
    expected_gates = {
        "finite_noncollapsed_coordinates",
        "transductive_recall50_gt_recall10",
        "matched_projection_recall50_gt_recall10",
        "exact_update_closure",
        "zero_numerical_skips",
        "no_pipeline_stamp_drift",
    }
    gates = diagnostics.get("execution_gates")
    if (
        density.get("schema") != NATIVE_DENSITY_SCHEMA
        or density.get("round_id") != ROUND_ID
        or density.get("release_sha")
        != active["manifest"]["release_sha"]
        or density.get("retry_provenance") != retry
        or diagnostics.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("round_id") != ROUND_ID
        or diagnostics.get("release_sha")
        != active["manifest"]["release_sha"]
        or diagnostics.get("arm") != ARM
        or density.get("treatment", {}).get("diagnostics")
        != expected_input_signature(diagnostic_path)
        or not isinstance(gates, Mapping)
        or set(gates) != expected_gates
        or not all(gates.values())
    ):
        raise Round0124Error("R0124 decision evidence changed")
    observed_selector = density.get("registered_selector") or {}
    interval = observed_selector.get("paired_bootstrap_delta_ci") or []
    if len(interval) != 2:
        raise Round0124Error("R0124 paired density interval is missing")
    selector = classify_degree_bridge(
        control_density=float(density["control"]["density"]),
        treatment_density=float(density["treatment"]["density"]),
        delta_ci_low=float(interval[0]),
        delta_ci_high=float(interval[1]),
    )
    if selector != observed_selector:
        raise Round0124Error("R0124 registered selector receipt changed")
    causal_wording = {
        OUTCOME_MATERIAL: (
            "materially reduces native density by at least the registered "
            "0.03 margin"
        ),
        OUTCOME_NOT_MATERIAL: (
            "does not materially reduce native density by the registered "
            "0.03 margin"
        ),
        OUTCOME_INCONCLUSIVE: (
            "has an inconclusive native-density effect at the registered "
            "0.03 margin"
        ),
    }[selector["outcome"]]
    body = {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "retry_provenance": retry,
        "cumulative_attempt_accounting": retry[
            "cumulative_attempt_accounting"
        ],
        "density_score": expected_input_signature(density_path),
        "diagnostics": expected_input_signature(diagnostic_path),
        "registered_selector": selector,
        "causal_claim": (
            "within the exact R0115 raw 2M seed-42 recipe, changing only "
            "the fuzzy graph degree from the R0115 k50 tuple to 15 nonself "
            f"neighbors {causal_wording}"
        ),
        "diagnostic_metrics": diagnostics["metrics"],
        "polish_ood": diagnostics["ood"]["pol_Latn"],
        "diagnostics_can_rescue_or_fail_selector": False,
        "capabilities_produced": [
            "jina-fineweb-2m-native-k15-degree-bridge-v1"
        ],
        "legacy_density_floor_used": False,
        "r0123_representation_transfer_claim_made": False,
        "scale_contribution_excluded": True,
        "training_performed": True,
        "optimizer_updates": SUCCESSFUL_UPDATES,
        "production_ready": False,
    }
    receipt = seal(body)
    path = os.path.join(output, "decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: Mapping[str, Any],
    job: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = job if job is not None else active.get("job") or {}
    action = selected.get("action")
    if action == "build_k15_graph":
        return run_build_graph(active, selected)
    if action == "train_k15_treatment":
        return run_train(active, selected)
    if action == "evaluate_core_ood":
        return run_diagnostics(active, selected)
    if action == "score_native_density":
        return run_native_density(active, selected)
    if action == "decide_degree_bridge":
        return run_decision(active, selected)
    raise Round0124Error(f"unknown R0124 action: {action!r}")
