"""Execute the fixed-row Jina subsystem bisection for Round 0140."""
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
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0027_program import (
    GRAPH_BYTES as HISTORICAL_GRAPH_BYTES,
    GRAPH_EDGES as HISTORICAL_GRAPH_EDGES,
    GRAPH_PATH as HISTORICAL_GRAPH_PATH,
    GRAPH_SHA256 as HISTORICAL_GRAPH_SHA256,
    TRAIN_BYTES,
    TRAIN_PATH,
    TRAIN_SHA256,
)
from basemap.round0104_training import HostFp16MaterializedArray
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import (
    CAPABILITY,
    CURRENT_GRAPH_CURRENT_HOST,
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
    HISTORICAL_GRAPH_CURRENT_HOST,
    HISTORICAL_GRAPH_DEVICE_REPRO,
    NEW_CELLS,
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    ROUND_ID,
    ROWS,
    SEED,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    Round0140Error,
    build_decision,
    historical_preprocessing_stamp,
    host_train_config,
    metric_view,
)
from experiments import round0104_nodes as r0104
from experiments.round0027_nodes import _panel_config
from experiments.round0134_nodes import (
    _load_reference,
    _load_shared_evaluation_inputs,
    _projection_metrics,
)


TRANSFORM_BATCH_ROWS = 8_192
RENDER_ROWS = 100_000
RENDER_SEED = 13_400
ARTIFACT_SCHEMA_PREFIX = "round0140"


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = str(expected.get("canonical_path") or "")
    actual = expected_input_signature(path)
    if actual != dict(expected):
        raise Round0140Error(f"{label} bytes changed")
    return actual


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0140Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value


class HistoricalFp16Array:
    """Exact lazy view of the R0037 2M source in its original row order."""

    def __init__(self) -> None:
        signature = expected_input_signature(TRAIN_PATH)
        if signature != {
            "canonical_path": os.path.realpath(TRAIN_PATH),
            "kind": "file",
            "bytes": TRAIN_BYTES,
            "sha256": TRAIN_SHA256,
        }:
            raise Round0140Error("R0037 source bytes changed")
        self.array = np.load(TRAIN_PATH, mmap_mode="r", allow_pickle=False)
        if (
            self.array.shape != (ROWS, DIMENSION)
            or self.array.dtype != np.dtype("<f2")
            or not self.array.flags.c_contiguous
        ):
            raise Round0140Error("R0037 source geometry changed")
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.segments = [{
            "global_row_start": 0,
            "global_row_stop": ROWS,
            "dataset": "jina-en-2M-nested",
            "shard": signature,
            "shard_rows": ROWS,
            "shard_row_start": 0,
            "shard_row_stop": ROWS,
        }]

    def __len__(self) -> int:
        return ROWS

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.array[key]


class HistoricalHostFp16Array(HostFp16MaterializedArray):
    """Current host endpoint transport over the exact historical rows."""

    def execution_stamp(self) -> dict[str, Any]:
        return {
            **super().execution_stamp(),
            "row_universe": "R0037-jina-en-2M-nested-exact-order",
            "source_sha256": TRAIN_SHA256,
        }


def _source_proof() -> dict[str, Any]:
    source = HistoricalFp16Array()
    return {
        "schema": f"round{ROUND_ID}-r0037-exact-source-proof-v1",
        "rows": ROWS,
        "dimension": DIMENSION,
        "dtype": "<f2",
        "file": source.segments[0]["shard"],
        "segments": source.segments,
        "row_order": "byte-exact R0037 jina-en-2M-nested order",
    }


def run_build_current_graph(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    """Build current R0104-style graph semantics on exact R0037 rows."""
    import faiss
    import umap.umap_ as umap_api

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0140 current graph on historical rows"
    )
    started = time.monotonic()
    proof = _source_proof()
    X = r0104._materialize_normalized(HistoricalFp16Array())
    materialize_seconds = time.monotonic() - started

    train_rows = np.sort(
        np.random.RandomState(GRAPH_TRAIN_SEED)
        .choice(ROWS, GRAPH_TRAIN_ROWS, replace=False)
        .astype(np.int64)
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
        resource, 0, cpu_index, r0104._faiss_gpu_options(faiss)
    )
    stage = time.monotonic()
    index.train(np.ascontiguousarray(X[train_rows]))
    train_seconds = time.monotonic() - stage
    stage = time.monotonic()
    for start in range(0, ROWS, 100_000):
        index.add(np.ascontiguousarray(X[start : min(start + 100_000, ROWS)]))
    add_seconds = time.monotonic() - stage
    if int(index.ntotal) != ROWS:
        raise Round0140Error("current graph index row count changed")

    quality_ids = np.sort(
        np.random.RandomState(GRAPH_QUALITY_SEED)
        .choice(ROWS, GRAPH_QUALITY_ROWS, replace=False)
        .astype(np.int64)
    )
    exact = faiss.index_cpu_to_gpu(
        resource,
        0,
        faiss.IndexFlatIP(DIMENSION),
        r0104._faiss_gpu_options(faiss),
    )
    for start in range(0, ROWS, 100_000):
        exact.add(np.ascontiguousarray(X[start : min(start + 100_000, ROWS)]))
    _dist, truth_raw = exact.search(np.ascontiguousarray(X[quality_ids]), GRAPH_K)
    truth = r0104._without_self(truth_raw, quality_ids, GRAPH_K - 1)
    cells: dict[str, Any] = {}
    selected: int | None = None
    for nprobe in GRAPH_NPROBE_GRID:
        index.nprobe = nprobe
        stage = time.monotonic()
        _dist, raw = index.search(np.ascontiguousarray(X[quality_ids]), GRAPH_K)
        wall = time.monotonic() - stage
        observed = r0104._without_self(raw, quality_ids, GRAPH_K - 1)
        recalls = r0104._recall_rows(observed, truth)
        mean = float(recalls.mean())
        p10 = float(np.percentile(recalls, 10))
        passed = mean >= GRAPH_MEAN_RECALL_FLOOR and p10 >= GRAPH_P10_RECALL_FLOOR
        cells[str(nprobe)] = {
            "nprobe": nprobe,
            f"mean_recall_at_{GRAPH_K - 1}": mean,
            f"p10_recall_at_{GRAPH_K - 1}": p10,
            "queries": GRAPH_QUALITY_ROWS,
            "wall_seconds": wall,
            "passed": passed,
        }
        if passed and selected is None:
            selected = nprobe
    del exact
    if selected is None:
        raise Round0140Error("no current graph nprobe cell passed qualification")

    index.nprobe = selected
    neighbors = np.empty((ROWS, GRAPH_K), dtype=np.int32)
    distances = np.empty((ROWS, GRAPH_K), dtype=np.float32)
    stage = time.monotonic()
    for start in range(0, ROWS, 16_384):
        stop = min(start + 16_384, ROWS)
        sims, ids = index.search(np.ascontiguousarray(X[start:stop]), GRAPH_K)
        if np.any(ids < 0) or np.any(ids >= ROWS):
            raise Round0140Error("current graph search returned invalid row IDs")
        neighbors[start:stop] = ids.astype(np.int32, copy=False)
        distances[start:stop] = np.maximum(0.0, 1.0 - sims).astype(
            np.float32, copy=False
        )
    search_seconds = time.monotonic() - stage

    stage = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
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
    fuzzy_seconds = time.monotonic() - stage
    if (
        len(sources) <= ROWS * (GRAPH_K - 1)
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise Round0140Error("current fuzzy graph arrays are invalid")
    graph_path = os.path.join(output, f"edges-k{GRAPH_K}-fuzzy.npz")
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
    probe = np.random.RandomState(106)
    probe_ids = probe.choice(len(sources), 20_000, replace=False)
    edge_cos = np.einsum("ij,ij->i", X[sources[probe_ids]], X[targets[probe_ids]])
    random_targets = probe.randint(0, ROWS, size=len(probe_ids))
    random_cos = np.einsum("ij,ij->i", X[sources[probe_ids]], X[random_targets])
    endpoint = {
        "rows": len(probe_ids),
        "edge_cosine_mean": float(edge_cos.mean()),
        "random_cosine_mean": float(random_cos.mean()),
        "margin": float(edge_cos.mean() - random_cos.mean()),
    }
    if endpoint["margin"] <= 0.15:
        raise Round0140Error("current graph endpoint cosine margin is too small")
    manifest = {
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
        "metric_input": "exact R0037 fp16 source normalized in fp32",
        "weight_semantics": f"fuzzy_simplicial_set(k{GRAPH_K})",
        "graph_path": os.path.basename(graph_path),
        "graph_sha256": graph_signature["sha256"],
        "graph_bytes": graph_signature["bytes"],
        "data_len": ROWS,
        "data_shard_records": proof["segments"],
        "input_preprocessing": historical_preprocessing_stamp(),
        "graph_construction_truth": {
            "source_proof": proof,
            "search": {
                "index": "GPU IndexIVFFlat/IP",
                "nlist": GRAPH_NLIST,
                "selected_nprobe": selected,
                "quality_cells": cells,
                "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
                "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
                "self_inclusive_k": GRAPH_K,
            },
            "semantics": (
                f"current IVF/fuzzy-k{GRAPH_K} builder on R0037 exact rows"
            ),
        },
        "endpoint_cosine": endpoint,
        "post_hoc_identity_verified": True,
        "verified_by": (
            f"round{ROUND_ID}-current-graph-fixed-row-builder-v1"
        ),
    }
    manifest_path = os.path.join(output, "graph-manifest.json")
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    receipt = seal({
        "schema": f"{ARTIFACT_SCHEMA_PREFIX}-current-graph-fixed-row-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "source_proof": proof,
        "graph": graph_signature,
        "graph_manifest": expected_input_signature(manifest_path),
        "graph_edges": len(sources),
        "search_qualification": {"selected_nprobe": selected, "cells": cells},
        "endpoint_cosine": endpoint,
        "performance": {
            "materialize_seconds": materialize_seconds,
            "ivf_train_seconds": train_seconds,
            "ivf_add_seconds": add_seconds,
            "full_search_seconds": search_seconds,
            "fuzzy_seconds": fuzzy_seconds,
            "total_wall_seconds": time.monotonic() - started,
        },
    })
    atomic_write_new_json(os.path.join(output, "receipt.json"), receipt, immutable=True)
    del X, neighbors, distances, sources, targets, weights, graph, coo
    gc.collect()


def _graph_bundle(job: Mapping[str, Any]) -> dict[str, Any]:
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import load_edge_arrays

    kind = str(job.get("graph_kind") or "")
    if kind == "current-fixed-row":
        receipt_path = os.path.join(str(job["graph_output"]), "receipt.json")
        receipt = _read_sealed(
            expected_input_signature(receipt_path), label="R0140 current graph receipt"
        )
        graph_signature = _signature(receipt["graph"], label="R0140 current graph")
        manifest_signature = _signature(
            receipt["graph_manifest"], label="R0140 current graph manifest"
        )
        edges = int(receipt["graph_edges"])
    elif kind == "historical-byte-exact":
        graph_signature = _signature(job["graph"], label="R0037 historical graph")
        manifest_signature = _signature(
            job["graph_manifest"], label="R0140 historical graph adapter"
        )
        if (
            graph_signature["sha256"] != HISTORICAL_GRAPH_SHA256
            or graph_signature["bytes"] != HISTORICAL_GRAPH_BYTES
        ):
            raise Round0140Error("historical graph bytes changed")
        edges = HISTORICAL_GRAPH_EDGES
    else:
        raise Round0140Error(f"unknown graph kind: {kind}")
    with open(manifest_signature["canonical_path"], encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("schema") != "graph_manifest.v2"
        or manifest.get("graph_sha256") != graph_signature["sha256"]
        or manifest.get("n_nodes") != ROWS
        or manifest.get("n_edges") != edges
        or manifest.get("k") != GRAPH_K
    ):
        raise Round0140Error("graph manifest content changed")
    sources, targets, weights, n_nodes = load_edge_arrays(
        graph_signature["canonical_path"], load_weights=True
    )
    if weights is None or int(n_nodes) != ROWS or len(sources) != edges:
        raise Round0140Error("graph arrays changed")
    return {
        "signature": graph_signature,
        "manifest_signature": manifest_signature,
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
        "edges": edges,
        "kind": kind,
        "k": int(manifest.get("k", -1)),
    }


def run_host_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    cell = str(job.get("cell") or "")
    if cell not in {CURRENT_GRAPH_CURRENT_HOST, HISTORICAL_GRAPH_CURRENT_HOST}:
        raise Round0140Error("R0140 host cell changed")
    graph = _graph_bundle(job)
    config, config_sha = host_train_config(
        cell=cell,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=graph["edges"],
    )
    source = HistoricalFp16Array()
    dataset = HistoricalHostFp16Array(
        source, device="cuda", buffer_rows=config["optimizer"]["batch_size"]
    )
    wrapper = r0104.Round0104TrainingInput(
        dataset,
        graph,
        arm="fp16_control",
        required_pipeline=r0104.PIPELINE,
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0140 {cell} train output"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        seal({
            "schema": f"{ARTIFACT_SCHEMA_PREFIX}-host-production-config-v1",
            "round_id": ROUND_ID,
            "cell": cell,
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
    model = r0104._new_model(config)
    loader_supply = config["execution"].get("loader_supply")
    if loader_supply is not None:
        if (
            not isinstance(loader_supply, Mapping)
            or int(loader_supply.get("loader_supply_epochs", 0)) <= 0
            or int(loader_supply.get("loader_batches_per_epoch", 0)) <= 0
            or int(loader_supply.get("planned_loop_iters", 0))
            < SUCCESSFUL_UPDATES
            or int(loader_supply.get("successful_update_horizon", -1))
            != SUCCESSFUL_UPDATES
        ):
            raise Round0140Error("registered loader-supply plan is invalid")
        model.n_epochs = int(loader_supply["loader_supply_epochs"])
    model._max_train_steps = SUCCESSFUL_UPDATES
    model._bench_warmup = PERFORMANCE_WARMUP_UPDATES
    model._perf_profile = True
    model._perf_floor = TRAIN_MINIMUM_UPDATES_PER_S
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
    expected = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected.items()
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
        "n_pos_edges": graph["edges"],
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    if loader_supply is not None:
        expected_loader_accounting = {
            "loader_batches_per_epoch": int(
                loader_supply["loader_batches_per_epoch"]
            ),
            "planned_loop_iters": int(loader_supply["planned_loop_iters"]),
        }
        mismatches.update({
            key: {"expected": value, "observed": accounting.get(key)}
            for key, value in expected_loader_accounting.items()
            if accounting.get(key) != value
        })
    expected_rows = SUCCESSFUL_UPDATES * config["optimizer"]["batch_size"]
    producer_delta = int(runtime["host_prefetch_producer_batches"]) - int(
        runtime["host_prefetch_consumer_batches"]
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
    r0104.synchronize_runtime_counters(accounting, runtime)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES) / model._bench_seconds
        if model._bench_seconds else 0.0
    )
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        mismatches["performance"] = {
            "floor": TRAIN_MINIMUM_UPDATES_PER_S,
            "rate": rate,
            "aborted": profiler.get("aborted"),
        }
    if mismatches:
        raise Round0140Error(f"R0140 {cell} accounting failed: {mismatches}")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    receipt = seal({
        "schema": f"{ARTIFACT_SCHEMA_PREFIX}-host-train-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "cell": cell,
        "causal_matrix": config["causal_matrix"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "model": expected_input_signature(model_path),
        "source": expected_input_signature(TRAIN_PATH),
        "graph": graph["signature"],
        "graph_manifest": graph["manifest_signature"],
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_seconds": wall,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
        },
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
    del model, wrapper, dataset, graph
    torch.cuda.empty_cache()
    gc.collect()


def _authenticate_new_model(
    *, cell: str, train_output: str, release_sha: str, device: str = "cuda"
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    path = os.path.join(train_output, "train-receipt.json")
    receipt = _read_sealed(expected_input_signature(path), label=f"R0140 {cell} train")
    if receipt.get("round_id") != ROUND_ID or receipt.get("release_sha") != release_sha:
        raise Round0140Error(f"R0140 {cell} train lineage changed")
    if cell == HISTORICAL_GRAPH_DEVICE_REPRO:
        if (
            receipt.get("cell") != "d768_s42"
            or (receipt.get("actual_pipeline") or {}).get("pipeline") != "device"
        ):
            raise Round0140Error("historical device reproduction stamp changed")
    elif (
        receipt.get("cell") != cell
        or (receipt.get("exact_execution_receipt") or {}).get("pipeline")
        != "host_weighted_jina_paired"
    ):
        raise Round0140Error(f"R0140 {cell} host stamp changed")
    model_signature = _signature(receipt["model"], label=f"R0140 {cell} model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device=device)
    if (
        model.input_dim != DIMENSION
        or model.n_components != 2
        or model.hidden_dim != 2048
        or model.n_layers != 3
        or model.low_dim_kernel != "legacy_lp"
    ):
        raise Round0140Error(f"R0140 {cell} model architecture changed")
    return model, receipt, expected_input_signature(path)


def _render(
    *, output: str, cells: Mapping[str, Mapping[str, Any]], labels: np.ndarray
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = create_fresh_directory(os.path.join(output, "renders"), label="R0140 render")
    sample = np.sort(
        np.random.RandomState(RENDER_SEED).choice(ROWS, RENDER_ROWS, replace=False)
    )
    sample_path = os.path.join(root, "sample-row-ids.npy")
    atomic_save_new_npy(sample_path, sample, immutable=True)
    order = (
        "historical_r0037_seed42",
        "current_r0104_fp16_seed42",
        *NEW_CELLS,
    )
    titles = {
        "historical_r0037_seed42": "accepted historical R0037",
        "current_r0104_fp16_seed42": "accepted current R0104 (other rows)",
        CURRENT_GRAPH_CURRENT_HOST: (
            f"fixed rows: current fuzzy-k{GRAPH_K} graph + host"
        ),
        HISTORICAL_GRAPH_CURRENT_HOST: "fixed rows: historical graph + host",
        HISTORICAL_GRAPH_DEVICE_REPRO: "fixed rows: historical graph + device",
    }
    figure, axes = plt.subplots(1, len(order), figsize=(25, 5), dpi=140)
    color = np.asarray(labels[sample] % 20, dtype=np.int16)
    limits: dict[str, Any] = {}
    for axis, key in zip(axes, order, strict=True):
        coords = np.load(
            _signature(cells[key]["coordinates"], label=f"{key} coordinates")[
                "canonical_path"
            ],
            mmap_mode="r",
            allow_pickle=False,
        )
        points = np.asarray(coords[sample], dtype=np.float32)
        low = np.quantile(points, 0.001, axis=0)
        high = np.quantile(points, 0.999, axis=0)
        pad = np.maximum((high - low) * 0.03, 1.0e-6)
        axis.scatter(
            points[:, 0], points[:, 1], c=color, cmap="tab20", s=0.18,
            alpha=0.35, linewidths=0, rasterized=True,
        )
        axis.set_xlim(float(low[0] - pad[0]), float(high[0] + pad[0]))
        axis.set_ylim(float(low[1] - pad[1]), float(high[1] + pad[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(titles[key], fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        limits[key] = {"quantile_low": low.tolist(), "quantile_high": high.tolist()}
    figure.tight_layout()
    path = os.path.join(root, "subsystem-bisection.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    os.chmod(path, 0o444)
    receipt = seal({
        "schema": f"{ARTIFACT_SCHEMA_PREFIX}-subsystem-bisection-render-v1",
        "round_id": ROUND_ID,
        "sample": expected_input_signature(sample_path),
        "sample_seed": RENDER_SEED,
        "sample_rows": RENDER_ROWS,
        "color": "frozen R0037 k256 label modulo 20",
        "axes": "per-cell 0.1%-99.9% robust axes; diagnostic only",
        "limits": limits,
        "render": expected_input_signature(path),
    })
    manifest = os.path.join(root, "render-manifest.json")
    atomic_write_new_json(manifest, receipt, immutable=True)
    return {**receipt, "manifest": expected_input_signature(manifest)}


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0140 fixed-row functional panel"
    )
    started = time.monotonic()
    r0134 = _read_sealed(job["r0134_panel"], label="accepted R0134 panel")
    baseline = r0134.get("cells")
    if not isinstance(baseline, Mapping):
        raise Round0140Error("R0134 baseline cells are missing")
    source_signature, source, queries = _load_shared_evaluation_inputs(job)
    shared, shared_signature, reference, truth, centroids = _load_reference(job)
    from basemap.panel_v2 import score_panel

    cells: dict[str, Any] = {}
    for key in ("historical_r0037_seed42", "current_r0104_fp16_seed42"):
        cell = baseline.get(key)
        if not isinstance(cell, Mapping):
            raise Round0140Error(f"R0134 baseline missing {key}")
        _signature(cell["coordinates"], label=f"{key} coordinates")
        _signature(cell["query_coordinates"], label=f"{key} query coordinates")
        cells[key] = dict(cell)

    train_release_shas = job.get("train_release_shas")
    if train_release_shas is None:
        train_release_shas = {
            cell: active["manifest"]["release_sha"] for cell in NEW_CELLS
        }
    if (
        not isinstance(train_release_shas, Mapping)
        or set(train_release_shas) != set(NEW_CELLS)
        or any(
            not isinstance(train_release_shas[cell], str)
            or len(train_release_shas[cell]) != 40
            for cell in NEW_CELLS
        )
    ):
        raise Round0140Error("R0140 train-release lineage is malformed")

    for cell in NEW_CELLS:
        model, train, train_signature = _authenticate_new_model(
            cell=cell,
            train_output=str(job["train_outputs"][cell]),
            release_sha=str(train_release_shas[cell]),
        )
        coordinates = np.asarray(
            model.transform(source, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
        )
        query_coordinates = np.asarray(
            model.transform(queries, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
        )
        if (
            coordinates.shape != (ROWS, 2)
            or query_coordinates.shape != (20_000, 2)
            or not np.isfinite(coordinates).all()
            or not np.isfinite(query_coordinates).all()
        ):
            raise Round0140Error(f"R0140 {cell} transform is malformed")
        cell_root = create_fresh_directory(
            os.path.join(output, cell), label=f"R0140 {cell} coordinates"
        )
        coordinate_path = os.path.join(cell_root, "coordinates.npy")
        query_path = os.path.join(cell_root, "query-coordinates.npy")
        atomic_save_new_npy(coordinate_path, coordinates, immutable=True)
        atomic_save_new_npy(query_path, query_coordinates, immutable=True)
        coordinate_signature = expected_input_signature(coordinate_path)
        query_signature = expected_input_signature(query_path)
        panel = score_panel(
            source,
            coordinates,
            config=_panel_config(),
            centroids_by_k=centroids,
            hiD_reference=reference,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "cell": cell,
                "release_sha": active["manifest"]["release_sha"],
                "source": source_signature,
                "coordinates": coordinate_signature,
                "shared_reference_receipt": shared_signature,
            },
        )
        projection = _projection_metrics(
            coordinates=coordinates,
            query_coordinates=query_coordinates,
            truth=truth,
        )
        if (
            panel.get("guards", {}).get("coords_finite") is not True
            or panel.get("guards", {}).get("coords_collapsed") is not False
            or panel.get("purity", {}).get("k256") is None
            or panel.get("purity", {}).get("k1024") is None
        ):
            raise Round0140Error(f"R0140 {cell} panel guards failed")
        cells[cell] = {
            "seed": SEED,
            "role": "fixed-row-subsystem-cell",
            "training": {
                "train": train_signature,
                "model": train["model"],
                "release_sha": str(train_release_shas[cell]),
                "actual_pipeline": (
                    train.get("exact_execution_receipt")
                    or train.get("actual_pipeline")
                ),
                "train_accounting": train["train_accounting"],
            },
            "coordinates": coordinate_signature,
            "query_coordinates": query_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "query_coordinates_ordered_sha256": ordered_array_sha256(query_coordinates),
            "panel": panel,
            "projection": projection,
            "decision_metrics": metric_view({"panel": panel, "projection": projection}),
        }
        del model, coordinates, query_coordinates
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    render = _render(
        output=output,
        cells=cells,
        labels=np.asarray(reference["labels"][256], dtype=np.int32),
    )
    receipt = seal({
        "schema": f"{ARTIFACT_SCHEMA_PREFIX}-fixed-row-functional-panel-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "source": source_signature,
        "source_rows": ROWS,
        "same_ordered_training_rows_across_new_cells": True,
        "cross_round_training_row_equivalence_claimed": False,
        "shared_reference_receipt": shared_signature,
        "high_d_reference": job["high_d_reference"],
        "query_truth": job["query_truth"],
        "query_embeddings": job["query_embeddings"],
        "r0134_context_panel": job["r0134_panel"],
        "cells": cells,
        "render": render,
        "density_role": "diagnostic only against 0.17589; never selector input",
        "wall_seconds": time.monotonic() - started,
        "map_registry_state_changed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "functional-bisection.json"), receipt, immutable=True
    )


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0140 fixed-row bisection decision"
    )
    panel_path = os.path.join(str(job["panel_output"]), "functional-bisection.json")
    panel = _read_sealed(
        expected_input_signature(panel_path), label="R0140 functional panel"
    )
    if panel.get("round_id") != ROUND_ID:
        raise Round0140Error("R0140 panel identity changed")
    cells = {key: panel["cells"][key] for key in NEW_CELLS}
    decision = seal({
        **build_decision(cells),
        "release_sha": active["manifest"]["release_sha"],
        "panel": expected_input_signature(panel_path),
        "capability": CAPABILITY,
    })
    atomic_write_new_json(os.path.join(output, "decision.json"), decision, immutable=True)


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0140Error("R0140 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "build_current_graph":
        return run_build_current_graph(active, job)
    if action == "train_host":
        return run_host_train(active, job)
    if action == "historical_device_canary":
        from experiments.round0037_nodes import run_sampler_canary

        return run_sampler_canary(active, job)
    if action == "historical_device_train":
        from experiments.round0037_nodes import run_train

        return run_train(active, job)
    if action == "functional_panel":
        return run_panel(active, job)
    if action == "decide":
        return run_decision(active, job)
    raise Round0140Error(f"unknown R0140 action: {action!r}")
