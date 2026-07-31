"""Execute the R0121 2M FineWeb graph-degree bridge."""
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
from basemap.round0108_evaluation import validate_seal
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
    verify_signature,
)
from basemap.round0121_degree_bridge import (
    ARM,
    DECISION_SCHEMA,
    DENSITY_SCHEMA,
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
    LOCALIZATION_OUTCOME,
    PRODUCTION_CONFIG_SCHEMA,
    REGISTERED_DENSITY_FLOOR,
    ROUND_ID,
    TRAIN_RECEIPT_SCHEMA,
    DegreeBridgeTrainingInput,
    Round0121Error,
    classify_degree_bridge,
    graph_degree_stamp,
    load_graph,
    train_config,
)
from experiments import round0113_nodes as prompt_nodes
from experiments import round0119_nodes as localization_nodes


def _execution_round_id(active: Mapping[str, Any]) -> str:
    round_id = str((active.get("manifest") or {}).get("round_id", ""))
    if round_id != ROUND_ID:
        raise Round0121Error("R0121 handler received another queue")
    return round_id


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
        raise Round0121Error("R0121 search result geometry changed")
    self_mask = neighbors == rows[:, None]
    if not np.all(self_mask.sum(1) == 1):
        raise Round0121Error("R0121 search did not return exactly one self")
    nonself = ~self_mask
    selected = neighbors[nonself].reshape(len(rows), GRAPH_DEGREE)
    selected_sims = sims[nonself].reshape(len(rows), GRAPH_DEGREE)
    if np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
        raise Round0121Error("R0121 search returned duplicate nonself rows")
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
        or control.get("arm") != ARM
        or int(control.get("retained_rows", -1)) != RETAINED_ROWS
        or int(control.get("dimension", -1)) != DIMENSION
        or int(control.get("k", -1)) != 50
        or control.get("assembly") != dict(assembly_signature)
    ):
        raise Round0121Error("R0115 control graph identity changed")
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
        str(job["outputs"][0]), label="R0121 raw k15 graph"
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
        raise Round0121Error("R0121 IVF row count changed")

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
        raise Round0121Error(
            "R0121 IVF seeds/panel/nprobe differ from the R0115 control"
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
        raise Round0121Error("R0121 fixed-nprobe k15 graph did not qualify")

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
        raise Round0121Error("R0121 fuzzy graph arrays are invalid")
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
    return prompt_nodes._new_model(config)


def run_train(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    _execution_round_id(active)
    assembly, assembly_signature = prompt_nodes._load_assembly(job)
    graph_path = str(job["graph_manifest"])
    graph_signature = expected_input_signature(graph_path)
    graph = load_graph(
        graph_path, expected_sha256=graph_signature["sha256"]
    )
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
        str(job["outputs"][0]), label="R0121 k15 treatment train"
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
        raise Round0121Error(f"R0121 train accounting failed: {mismatches}")
    synchronize_runtime_counters(accounting, runtime)
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
        raise Round0121Error("R0121 treatment performance admission failed")
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
    assembly, _assembly_signature = prompt_nodes._load_assembly(job)
    graph_path = str(job["graph_manifest"])
    graph_signature = expected_input_signature(graph_path)
    graph = load_graph(
        graph_path, expected_sha256=graph_signature["sha256"]
    )
    config, config_sha = train_config(
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=graph["n_nodes"],
    )
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train = read_sealed(train_path, label="R0121 treatment train receipt")
    if (
        train.get("schema") != TRAIN_RECEIPT_SCHEMA
        or train.get("round_id") != ROUND_ID
        or train.get("arm") != ARM
        or train.get("production_config_sha256") != config_sha
        or train.get("graph_manifest") != graph["manifest_signature"]
        or train.get("optimizer_updates") != SUCCESSFUL_UPDATES
    ):
        raise Round0121Error("R0121 treatment train/config binding changed")
    model_path = verify_signature(train["model"], label="R0121 treatment model")
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
        raise Round0121Error("R0121 treatment architecture changed")
    return model, train, assembly, graph


def run_diagnostics(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the reviewed R0115 panel implementation with R0121 auth."""
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
        else f"round0121-{stem}-v1"
    )
    try:
        result = prompt_nodes.run_evaluate(dict(active), dict(job))
    finally:
        for name, value in originals.items():
            setattr(prompt_nodes, name, value)
    if result.get("schema") != DIAGNOSTIC_SCHEMA:
        raise Round0121Error("R0121 diagnostic schema changed")
    return result


def _r0119_evidence(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    panel_path = str(job["r0119_panel"])
    decision_path = str(job["r0119_decision"])
    panel_signature = expected_input_signature(panel_path)
    decision_signature = expected_input_signature(decision_path)
    control_train_signature = expected_input_signature(
        str(job["r0115_control_train_receipt"])
    )
    with open(panel_path, encoding="utf-8") as handle:
        panel = json.load(handle)
    with open(decision_path, encoding="utf-8") as handle:
        decision = json.load(handle)
    validate_seal(panel, label="R0119 density localization panel")
    validate_seal(decision, label="R0119 density localization decision")
    control = (panel.get("cells") or {}).get("current_2m_seed42")
    if (
        panel.get("schema") != localization_nodes.SCORE_SCHEMA
        or panel.get("round_id") != "0119"
        or decision.get("schema") != localization_nodes.DECISION_SCHEMA
        or decision.get("round_id") != "0119"
        or decision.get("score") != panel_signature
        or decision.get("outcome") != LOCALIZATION_OUTCOME
        or not isinstance(control, Mapping)
        or control.get("seed") != 42
        or control.get("group") != "current_2m"
        or control.get("train_receipt") != control_train_signature
        or control.get("clears_unchanged_registered_floor") is not True
    ):
        raise Round0121Error("R0119 localization prerequisite changed")
    return panel, panel_signature, decision, decision_signature


def run_density(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    _execution_round_id(active)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0121 matched density score"
    )
    panel, panel_signature, decision, decision_signature = _r0119_evidence(job)
    (
        source,
        representatives,
        retained_global_rows,
        anchors,
        global_rows,
        high_radius,
        lineage,
        reference,
    ) = localization_nodes._load_universe(job)
    if lineage["registered_floor"] != REGISTERED_DENSITY_FLOOR:
        raise Round0121Error("R0121 registered density floor changed")
    model, train, _assembly, graph = _authenticate_treatment_model(active, job)
    bundle = {
        "model": model,
        "train": expected_input_signature(
            os.path.join(str(job["train_output"]), "train-receipt.json")
        ),
        "production_config": train["production_config"],
        "model_signature": train["model"],
        "seed": SEED,
        "group": "current_2m",
        "training_population": (
            "R0115 exact 1,993,761 raw FineWeb representatives"
        ),
        "training_graph": "variable-symmetric fuzzy k15",
        "training_dose": "500,000 successful positive-LR updates",
        "training_representation": "raw host fp16",
        "training_dequantization": "device fp32 from exact fp16",
        "authenticated_training_semantics": {
            "population_rows": RETAINED_ROWS,
            "graph_neighbors": GRAPH_DEGREE,
            "graph_neighbors_including_self": GRAPH_SEARCH_NEIGHBORS,
            "successful_updates": SUCCESSFUL_UPDATES,
            "pipeline": train["exact_execution_receipt"]["pipeline"],
            "sampler_class": train["exact_execution_receipt"]["sampler_class"],
            "positive_sampling": train["exact_execution_receipt"][
                "positive_sampling"
            ],
            "multiplicity_policy": train["exact_execution_receipt"][
                "multiplicity_policy"
            ],
            "feature_residency": train["exact_execution_receipt"][
                "feature_residency"
            ],
            "source_representation": train["exact_execution_receipt"][
                "source_representation"
            ],
            "dequantization": train["exact_execution_receipt"][
                "device_conversion"
            ],
        },
    }
    cell, arrays = localization_nodes._score_cell(
        key="treatment_k15_seed42",
        bundle=bundle,
        source=source,
        representatives=representatives,
        retained_global_rows=retained_global_rows,
        anchors=anchors,
        high_radius=high_radius,
        reference=reference,
    )
    cell["clears_unchanged_registered_floor"] = (
        float(cell["density_v2"]["correlation"])
        >= REGISTERED_DENSITY_FLOOR
    )
    arrays_path = os.path.join(output, "k15-density-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        anchor_compact_rows=anchors,
        anchor_global_rows=global_rows,
        high_radius=high_radius,
        **arrays,
    )
    control = panel["cells"]["current_2m_seed42"]
    body = {
        "schema": DENSITY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "r0119_panel": panel_signature,
        "r0119_decision": decision_signature,
        "r0119_outcome": decision["outcome"],
        "lineage": lineage,
        "universe": panel["universe"],
        "scorer": panel["scorer"],
        "control_reuse": {
            "role": "exact reused R0115 seed-42 k50 control score",
            "cell_key": "current_2m_seed42",
            "cell": control,
            "source_panel": panel_signature,
            "score_recomputed_in_r0121": False,
        },
        "treatment": cell,
        "arrays": expected_input_signature(arrays_path),
        "changed_factor": "fuzzy graph neighbor degree only",
        "core_and_ood_diagnostics_registered_role": "diagnostic-only",
        "training_performed_in_this_node": False,
    }
    score = seal(body)
    path = os.path.join(output, "density-score.json")
    atomic_write_new_json(path, score, immutable=True)
    del model, source, representatives, retained_global_rows, graph
    gc.collect()
    return {**score, "receipt": expected_input_signature(path)}


def run_decision(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    _execution_round_id(active)
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0121 degree-bridge decision"
    )
    density_path = os.path.join(
        str(job["density_output"]), "density-score.json"
    )
    diagnostic_path = os.path.join(
        str(job["diagnostic_output"]), "score.json"
    )
    density = read_sealed(density_path, label="R0121 density score")
    diagnostics = read_sealed(
        diagnostic_path, label="R0121 core/OOD diagnostics"
    )
    if (
        density.get("schema") != DENSITY_SCHEMA
        or diagnostics.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostics.get("round_id") != ROUND_ID
        or diagnostics.get("arm") != ARM
        or not all((diagnostics.get("execution_gates") or {}).values())
    ):
        raise Round0121Error("R0121 decision evidence changed")
    control_density = float(
        density["control_reuse"]["cell"]["density_v2"]["correlation"]
    )
    treatment_density = float(
        density["treatment"]["density_v2"]["correlation"]
    )
    selector = classify_degree_bridge(
        localization_outcome=str(density["r0119_outcome"]),
        control_density=control_density,
        treatment_density=treatment_density,
        registered_floor=float(density["scorer"]["registered_floor"]),
    )
    body = {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "density_score": expected_input_signature(density_path),
        "diagnostics": expected_input_signature(diagnostic_path),
        "registered_selector": selector,
        "causal_claim": (
            "within the exact R0115 raw 2M seed-42 recipe, changing only "
            "the fuzzy graph degree from the R0115 k50 tuple to 15 nonself "
            "neighbors "
            + (
                "is sufficient to cross the frozen density-v2 floor"
                if selector["k15_alone_sufficient"]
                else "is not sufficient to cross the frozen density-v2 floor"
            )
        ),
        "diagnostic_metrics": diagnostics["metrics"],
        "polish_ood": diagnostics["ood"]["pol_Latn"],
        "diagnostics_can_rescue_or_fail_selector": False,
        "capabilities_produced": ["jina-fineweb-2m-k15-degree-bridge-v1"],
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
    if action == "score_matched_density":
        return run_density(active, selected)
    if action == "decide_degree_bridge":
        return run_decision(active, selected)
    raise Round0121Error(f"unknown R0121 action: {action!r}")
