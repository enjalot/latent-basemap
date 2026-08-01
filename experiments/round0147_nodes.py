"""Execute the conditional R0147 historical row-policy bridge."""
from __future__ import annotations

import gc
import json
import os
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.jina_historical_selection import (
    IndexedInventoryFp16Array,
    derive_first_eligible_historical_rows,
    load_historical_provenance,
    map_historical_positions,
    materialize_indexed_fp16_npy,
    verify_full_embedding_array,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0027_program import TRAIN_PATH
from basemap.round0104_training import HostFp16MaterializedArray
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import (
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
    PERFORMANCE_WARMUP_UPDATES,
    PERFORMANCE_WINDOWS,
    RESTORATION_FLOORS,
    SEED,
    SUCCESSFUL_UPDATES,
    TRAIN_MINIMUM_UPDATES_PER_S,
    metric_view,
)
from basemap.round0142_jina_universality import (
    COMMON_CORPUS_ROWS,
    PROBE_ORDER,
    canonical_representatives,
    fixed_separate_split,
    fixed_single_array_split,
    shape_matched_control_split,
)
from basemap.round0147_row_policy import (
    CAPABILITY,
    ROUND_ID,
    ROWS,
    TREATMENT,
    Round0147Error,
    build_decision,
    treatment_preprocessing_stamp,
    treatment_train_config,
)
from experiments import round0104_nodes as r0104
from experiments import round0142_nodes as r0142
from experiments.round0027_nodes import _panel_config
from experiments.round0134_nodes import (
    _load_reference,
    _load_shared_evaluation_inputs,
    _projection_metrics,
)


TRANSFORM_BATCH_ROWS = 8_192
RENDER_ROWS = 100_000
RENDER_SEED = 14_700


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0147Error(f"{label} bytes changed")
    return actual


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0147Error(f"JSON object required: {path}")
    return value


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    value = _read_json(signature["canonical_path"])
    validate_seal(value, label=label)
    return value


def _selection_receipt(output: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.join(output, "selection-receipt.json")
    signature = expected_input_signature(path)
    receipt = _read_sealed(signature, label="R0147 row-policy selection")
    if (
        receipt.get("round_id") != ROUND_ID
        or receipt.get("target_rows") != ROWS
        or receipt.get("size_preserving") is not True
    ):
        raise Round0147Error("R0147 selection receipt changed")
    return receipt, signature


class StagedTreatmentArray:
    """Exact contiguous fp16 treatment source materialized by the CPU node."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.signature = _signature(
            receipt["staged_source"], label="R0147 staged treatment source"
        )
        self.array = np.load(
            self.signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        if (
            self.array.shape != (ROWS, DIMENSION)
            or self.array.dtype != np.dtype("<f2")
            or not self.array.flags.c_contiguous
        ):
            raise Round0147Error("R0147 staged treatment geometry changed")
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.selection_signature = dict(receipt["selection_arrays"])
        self.segments = [{
            "global_row_start": 0,
            "global_row_stop": ROWS,
            "dataset": "r0147-eligible-historical-staged",
            "shard": self.signature,
            "shard_rows": ROWS,
            "shard_row_start": 0,
            "shard_row_stop": ROWS,
            "selection": self.selection_signature,
        }]

    def __len__(self) -> int:
        return ROWS

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.array[key]


class TreatmentHostFp16Array(HostFp16MaterializedArray):
    """Current host transport with an explicit R0147 population stamp."""

    def __init__(
        self,
        source: StagedTreatmentArray,
        *,
        device: str,
        buffer_rows: int,
    ) -> None:
        super().__init__(source, device=device, buffer_rows=buffer_rows)
        self._source = source

    def execution_stamp(self) -> dict[str, Any]:
        return {
            **super().execution_stamp(),
            "row_universe": (
                "first-2m-r0087-eligible-rows-in-r0037-historical-shuffle-order"
            ),
            "source_sha256": self._source.signature["sha256"],
            "selection_sha256": self._source.selection_signature["sha256"],
        }


def run_materialize_selection(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0147 historical eligibility selection"
    )
    started = time.monotonic()
    provenance_signature = _signature(
        job["historical_provenance"], label="historical provenance"
    )
    inventory_signature = _signature(job["inventory"], label="R0087 inventory")
    eligibility_signature = _signature(
        job["eligibility"], label="R0087 eligibility"
    )
    provenance = load_historical_provenance(provenance_signature["canonical_path"])
    inventory = _read_json(inventory_signature["canonical_path"])
    if (
        inventory.get("round_id") != "0087"
        or inventory.get("capability") != "jina-diverse-25m-inventory-v1"
    ):
        raise Round0147Error("R0087 inventory identity changed")
    with np.load(eligibility_signature["canonical_path"], allow_pickle=False) as archive:
        excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
        duplicate_excluded = np.asarray(
            archive["duplicate_excluded_rows"], dtype=np.int64
        )
        zero_rows = np.asarray(archive["zero_rows"], dtype=np.int64)
        nonfinite_rows = np.asarray(archive["nonfinite_rows"], dtype=np.int64)
    if (
        len(excluded) != 51_337
        or not np.array_equal(excluded, duplicate_excluded)
        or len(zero_rows)
        or len(nonfinite_rows)
    ):
        raise Round0147Error("R0087 eligibility semantics changed")

    raw_positions = np.arange(ROWS, dtype=np.int64)
    raw_mapping = map_historical_positions(provenance, inventory, raw_positions)
    raw_source = IndexedInventoryFp16Array(
        raw_mapping["global_rows"], inventory, dimension=DIMENSION
    )
    raw_mapping_proof = verify_full_embedding_array(TRAIN_PATH, raw_source)

    selected = derive_first_eligible_historical_rows(
        provenance, inventory, excluded, target_rows=ROWS
    )
    arrays = selected["arrays"]
    selection_path = os.path.join(output, "selection.npz")
    atomic_save_new_npz(selection_path, immutable=True, **arrays)
    selection_signature = expected_input_signature(selection_path)
    source = IndexedInventoryFp16Array(
        arrays["global_rows"], inventory, dimension=DIMENSION
    )
    staged_path = os.path.join(output, "eligible-historical-2m.f16.npy")
    staged_signature = materialize_indexed_fp16_npy(staged_path, source)
    staged = np.load(staged_path, mmap_mode="r", allow_pickle=False)
    if (
        staged.shape != (ROWS, DIMENSION)
        or staged.dtype != np.dtype("<f2")
        or not np.isfinite(np.asarray(staged[::100_000], dtype=np.float32)).all()
    ):
        raise Round0147Error("R0147 staged source failed geometry guards")
    receipt = seal({
        "schema": "round0147-historical-eligibility-selection-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "historical_provenance": provenance_signature,
        "inventory": inventory_signature,
        "eligibility": eligibility_signature,
        "eligibility_counts": {
            "excluded_rows": len(excluded),
            "duplicate_excluded_rows": len(duplicate_excluded),
            "zero_rows": len(zero_rows),
            "nonfinite_rows": len(nonfinite_rows),
        },
        "raw_historical_mapping_proof": raw_mapping_proof,
        "selection_summary": selected["summary"],
        "selection_arrays": selection_signature,
        "staged_source": staged_signature,
        "source_shards": source.segments,
        "target_rows": ROWS,
        "dimension": DIMENSION,
        "dtype": "<f2",
        "size_preserving": True,
        "historical_order_preserved": True,
        "excluded_rows_absent": True,
        "row_policy_includes_induced_graph_change": True,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "selection-receipt.json"), receipt, immutable=True
    )


def _source_proof(selection_output: str) -> tuple[StagedTreatmentArray, dict[str, Any]]:
    receipt, receipt_signature = _selection_receipt(selection_output)
    source = StagedTreatmentArray(receipt)
    proof = {
        "schema": "round0147-eligible-historical-source-proof-v1",
        "rows": ROWS,
        "dimension": DIMENSION,
        "dtype": "<f2",
        "staged_source": source.signature,
        "selection_receipt": receipt_signature,
        "selection_arrays": source.selection_signature,
        "selection_summary": receipt["selection_summary"],
        "segments": source.segments,
        "row_order": (
            "historical R0037 shuffle order after R0087 eligibility filtering and "
            "size-preserving replacement"
        ),
    }
    return source, proof


def run_build_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import faiss
    import umap.umap_ as umap_api

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0147 current graph on eligible rows"
    )
    started = time.monotonic()
    source, proof = _source_proof(str(job["selection_output"]))
    X = r0104._materialize_normalized(source)
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
        raise Round0147Error("R0147 graph index row count changed")

    quality_ids = np.sort(
        np.random.RandomState(GRAPH_QUALITY_SEED)
        .choice(ROWS, GRAPH_QUALITY_ROWS, replace=False)
        .astype(np.int64)
    )
    exact = faiss.index_cpu_to_gpu(
        resource, 0, faiss.IndexFlatIP(DIMENSION), r0104._faiss_gpu_options(faiss)
    )
    for start in range(0, ROWS, 100_000):
        exact.add(np.ascontiguousarray(X[start : min(start + 100_000, ROWS)]))
    _dist, truth_raw = exact.search(np.ascontiguousarray(X[quality_ids]), GRAPH_K)
    truth = r0104._without_self(truth_raw, quality_ids, GRAPH_K - 1)
    cells: dict[str, Any] = {}
    selected_nprobe: int | None = None
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
            "mean_recall_at_49": mean,
            "p10_recall_at_49": p10,
            "queries": GRAPH_QUALITY_ROWS,
            "wall_seconds": wall,
            "passed": passed,
        }
        if passed and selected_nprobe is None:
            selected_nprobe = nprobe
    del exact
    if selected_nprobe is None:
        raise Round0147Error("no R0147 graph nprobe cell passed qualification")

    index.nprobe = selected_nprobe
    neighbors = np.empty((ROWS, GRAPH_K), dtype=np.int32)
    distances = np.empty((ROWS, GRAPH_K), dtype=np.float32)
    stage = time.monotonic()
    for start in range(0, ROWS, 16_384):
        stop = min(start + 16_384, ROWS)
        sims, ids = index.search(np.ascontiguousarray(X[start:stop]), GRAPH_K)
        if np.any(ids < 0) or np.any(ids >= ROWS):
            raise Round0147Error("R0147 graph search returned invalid row IDs")
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
        raise Round0147Error("R0147 fuzzy graph arrays are invalid")
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
    probe = np.random.RandomState(147)
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
        raise Round0147Error("R0147 graph endpoint cosine margin is too small")
    preprocessing = treatment_preprocessing_stamp(
        source_sha256=source.signature["sha256"],
        selection_sha256=source.selection_signature["sha256"],
    )
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
        "metric_input": "exact staged R0147 fp16 treatment normalized in fp32",
        "weight_semantics": "fuzzy_simplicial_set(k50)",
        "graph_path": os.path.basename(graph_path),
        "graph_sha256": graph_signature["sha256"],
        "graph_bytes": graph_signature["bytes"],
        "data_len": ROWS,
        "data_shard_records": proof["segments"],
        "input_preprocessing": preprocessing,
        "graph_construction_truth": {
            "source_proof": proof,
            "search": {
                "index": "GPU IndexIVFFlat/IP",
                "nlist": GRAPH_NLIST,
                "selected_nprobe": selected_nprobe,
                "quality_cells": cells,
                "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
                "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
                "self_inclusive_k": GRAPH_K,
            },
            "semantics": "R0104 current builder on R0147 treatment rows",
            "row_policy_includes_induced_graph_change": True,
        },
        "endpoint_cosine": endpoint,
        "post_hoc_identity_verified": True,
        "verified_by": "round0147-current-graph-eligible-historical-builder-v1",
    }
    manifest_path = os.path.join(output, "graph-manifest.json")
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    receipt = seal({
        "schema": "round0147-current-graph-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "source_proof": proof,
        "graph": graph_signature,
        "graph_manifest": expected_input_signature(manifest_path),
        "graph_edges": len(sources),
        "search_qualification": {
            "selected_nprobe": selected_nprobe,
            "cells": cells,
        },
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


def _graph_bundle(output: str) -> dict[str, Any]:
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import load_edge_arrays

    receipt_path = os.path.join(output, "receipt.json")
    receipt = _read_sealed(
        expected_input_signature(receipt_path), label="R0147 graph receipt"
    )
    if receipt.get("round_id") != ROUND_ID:
        raise Round0147Error("R0147 graph receipt identity changed")
    graph_signature = _signature(receipt["graph"], label="R0147 graph")
    manifest_signature = _signature(
        receipt["graph_manifest"], label="R0147 graph manifest"
    )
    manifest = _read_json(manifest_signature["canonical_path"])
    edges = int(receipt["graph_edges"])
    if (
        manifest.get("schema") != "graph_manifest.v2"
        or manifest.get("graph_sha256") != graph_signature["sha256"]
        or manifest.get("n_nodes") != ROWS
        or manifest.get("n_edges") != edges
    ):
        raise Round0147Error("R0147 graph manifest content changed")
    sources, targets, weights, n_nodes = load_edge_arrays(
        graph_signature["canonical_path"], load_weights=True
    )
    if weights is None or int(n_nodes) != ROWS or len(sources) != edges:
        raise Round0147Error("R0147 graph arrays changed")
    return {
        "signature": graph_signature,
        "manifest_signature": manifest_signature,
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
        "edges": edges,
    }


def training_accounting_mismatches(
    *,
    accounting: Mapping[str, Any],
    runtime: Mapping[str, Any],
    expected_pipeline: Mapping[str, Any],
    graph_edges: int,
    batch_size: int,
    profiler: Mapping[str, Any],
    rate: float,
) -> dict[str, Any]:
    """Pure post-fit closure used by runtime, tests, and CPU smoke."""
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_pipeline.items()
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
        "n_pos_edges": graph_edges,
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = SUCCESSFUL_UPDATES * batch_size
    producer_delta = int(runtime.get("host_prefetch_producer_batches", -2)) - int(
        runtime.get("host_prefetch_consumer_batches", -4)
    )
    if (
        runtime.get("source_rows_gathered") != expected_rows
        or runtime.get("destination_rows_gathered") != expected_rows
        or runtime.get("host_prefetch_consumer_batches") != SUCCESSFUL_UPDATES
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": dict(runtime),
        }
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        mismatches["performance"] = {
            "floor": TRAIN_MINIMUM_UPDATES_PER_S,
            "rate": rate,
            "aborted": profiler.get("aborted"),
        }
    return mismatches


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    graph = _graph_bundle(str(job["graph_output"]))
    selection, selection_signature = _selection_receipt(str(job["selection_output"]))
    source = StagedTreatmentArray(selection)
    config, config_sha = treatment_train_config(
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=graph["edges"],
        source_sha256=source.signature["sha256"],
        selection_sha256=source.selection_signature["sha256"],
    )
    dataset = TreatmentHostFp16Array(
        source, device="cuda", buffer_rows=config["optimizer"]["batch_size"]
    )
    wrapper = r0104.Round0104TrainingInput(
        dataset,
        graph,
        arm="fp16_control",
        required_pipeline=r0104.PIPELINE,
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0147 eligible historical train output"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        seal({
            "schema": "round0147-host-production-config-v1",
            "round_id": ROUND_ID,
            "cell": TREATMENT,
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
    r0104.synchronize_runtime_counters(accounting, runtime)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    mismatches = training_accounting_mismatches(
        accounting=accounting,
        runtime=runtime,
        expected_pipeline=config["execution"]["expected_pipeline_stamp"],
        graph_edges=graph["edges"],
        batch_size=config["optimizer"]["batch_size"],
        profiler=profiler,
        rate=rate,
    )
    if mismatches:
        raise Round0147Error(f"R0147 training accounting failed: {mismatches}")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    receipt = seal({
        "schema": "round0147-host-train-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "cell": TREATMENT,
        "causal_matrix": config["causal_matrix"],
        "selection_receipt": selection_signature,
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "model": expected_input_signature(model_path),
        "source": source.signature,
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


def _authenticate_model(
    *, train_output: str, release_sha: str, device: str = "cuda"
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    path = os.path.join(train_output, "train-receipt.json")
    signature = expected_input_signature(path)
    receipt = _read_sealed(signature, label="R0147 train receipt")
    exact = receipt.get("exact_execution_receipt") or {}
    if (
        receipt.get("round_id") != ROUND_ID
        or receipt.get("release_sha") != release_sha
        or receipt.get("cell") != TREATMENT
        or exact.get("pipeline") != "host_weighted_jina_paired"
        or exact.get("row_universe")
        != "first-2m-r0087-eligible-rows-in-r0037-historical-shuffle-order"
    ):
        raise Round0147Error("R0147 train execution stamp changed")
    model_signature = _signature(receipt["model"], label="R0147 model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device=device)
    if (
        model.input_dim != DIMENSION
        or model.n_components != 2
        or model.hidden_dim != 2048
        or model.n_layers != 3
        or model.low_dim_kernel != "legacy_lp"
    ):
        raise Round0147Error("R0147 model architecture changed")
    return model, receipt, signature


def _render_pair(
    *, output: str, cells: Mapping[str, Mapping[str, Any]], labels: np.ndarray
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = create_fresh_directory(os.path.join(output, "renders"), label="R0147 render")
    sample = np.sort(
        np.random.RandomState(RENDER_SEED).choice(ROWS, RENDER_ROWS, replace=False)
    )
    sample_path = os.path.join(root, "sample-row-ids.npy")
    atomic_save_new_npy(sample_path, sample, immutable=True)
    order = (CURRENT_GRAPH_CURRENT_HOST, TREATMENT)
    titles = {
        CURRENT_GRAPH_CURRENT_HOST: "raw historical rows: current graph + host",
        TREATMENT: "eligible historical rows: current graph + host",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
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
            points[:, 0],
            points[:, 1],
            c=color,
            cmap="tab20",
            s=0.18,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlim(float(low[0] - pad[0]), float(high[0] + pad[0]))
        axis.set_ylim(float(low[1] - pad[1]), float(high[1] + pad[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(titles[key], fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        limits[key] = {"quantile_low": low.tolist(), "quantile_high": high.tolist()}
    figure.tight_layout()
    path = os.path.join(root, "row-policy-bridge.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    os.chmod(path, 0o444)
    receipt = seal({
        "schema": "round0147-row-policy-render-v1",
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


def run_functional_panel(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0147 fixed functional panel"
    )
    started = time.monotonic()
    r0140 = _read_sealed(job["r0140_panel"], label="accepted R0140 panel")
    if r0140.get("round_id") != "0140":
        raise Round0147Error("accepted R0140 panel identity changed")
    control = r0140.get("cells", {}).get(CURRENT_GRAPH_CURRENT_HOST)
    if not isinstance(control, Mapping):
        raise Round0147Error("R0140 restoring control is absent")
    control_metrics = metric_view(control)
    if not all(control_metrics[key] >= RESTORATION_FLOORS[key] for key in RESTORATION_FLOORS):
        raise Round0147Error("R0140 activation control no longer restores")
    source_signature, source, queries = _load_shared_evaluation_inputs(job)
    _shared, shared_signature, reference, truth, centroids = _load_reference(job)
    from basemap.panel_v2 import score_panel

    model, train, train_signature = _authenticate_model(
        train_output=str(job["train_output"]),
        release_sha=active["manifest"]["release_sha"],
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
        raise Round0147Error("R0147 treatment transform is malformed")
    cell_root = create_fresh_directory(
        os.path.join(output, TREATMENT), label="R0147 treatment coordinates"
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
            "cell": TREATMENT,
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
        raise Round0147Error("R0147 treatment panel guards failed")
    treatment = {
        "seed": SEED,
        "role": "size-preserving-historical-eligibility-treatment",
        "training": {
            "train": train_signature,
            "model": train["model"],
            "release_sha": active["manifest"]["release_sha"],
            "actual_pipeline": train["exact_execution_receipt"],
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
    cells = {CURRENT_GRAPH_CURRENT_HOST: dict(control), TREATMENT: treatment}
    render = _render_pair(
        output=output,
        cells=cells,
        labels=np.asarray(reference["labels"][256], dtype=np.int32),
    )
    selection, selection_signature = _selection_receipt(str(job["selection_output"]))
    receipt = seal({
        "schema": "round0147-historical-row-policy-functional-panel-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "source": source_signature,
        "source_rows": ROWS,
        "evaluation_universe": "exact R0037 fixed functional universe",
        "training_population_differs_by_registered_row_policy": True,
        "row_policy_includes_induced_graph_change": True,
        "selection_receipt": selection_signature,
        "selection_summary": selection["selection_summary"],
        "shared_reference_receipt": shared_signature,
        "r0140_context_panel": job["r0140_panel"],
        "cells": cells,
        "render": render,
        "density_role": "diagnostic only against 0.17589; never selector input",
        "wall_seconds": time.monotonic() - started,
        "map_registry_state_changed": False,
    })
    atomic_write_new_json(
        os.path.join(output, "functional-panel.json"), receipt, immutable=True
    )


def run_universality_panel(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    map_key = str(job.get("map_key") or "")
    if map_key not in {CURRENT_GRAPH_CURRENT_HOST, TREATMENT}:
        raise Round0147Error("unknown R0147 universality map")
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0147 universality score {map_key}"
    )
    started = time.monotonic()
    if map_key == TREATMENT:
        model, train, _train_signature = _authenticate_model(
            train_output=str(job["train_output"]),
            release_sha=active["manifest"]["release_sha"],
        )
        model_signature = dict(train["model"])
    else:
        model_signature = _signature(job["model"], label="R0140 control model")
        from basemap.pumap.parametric_umap import ParametricUMAP

        model = ParametricUMAP.load(model_signature["canonical_path"], device="cuda")
    control_signature = _signature(
        job["control_embeddings"], label="FineWeb heldout control embeddings"
    )
    control = np.load(
        control_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if control.shape != (60_000, DIMENSION) or not np.isfinite(control).all():
        raise Round0147Error("FineWeb OOD control geometry changed")
    control_representatives, control_duplicate_control = canonical_representatives(
        control
    )
    reports: dict[str, Any] = {}
    for index, name in enumerate(PROBE_ORDER):
        if name in COMMON_CORPUS_ROWS:
            values, inputs = r0142._common_probe(str(job["common_outputs"][name]), name)
            corpus_rows, query_rows, split = fixed_single_array_split(values, name=name)
            corpus = np.asarray(values[corpus_rows])
            queries = np.asarray(values[query_rows])
            corpus_ids = corpus_rows
            query_ids = 1_000_000_000 + query_rows
        elif name == "dadabase":
            source = _signature(job["dadabase"], label="Dadabase embeddings")
            texts = _signature(job["dadabase_texts"], label="Dadabase texts")
            values = np.load(source["canonical_path"], mmap_mode="r", allow_pickle=False)
            corpus_rows, query_rows, split = fixed_single_array_split(values, name=name)
            corpus = np.asarray(values[corpus_rows])
            queries = np.asarray(values[query_rows])
            corpus_ids = corpus_rows
            query_ids = 1_000_000_000 + query_rows
            inputs = {
                "embeddings": source,
                "texts": texts,
                "prompt_semantics": "legacy raw Jina-v5 diagnostic artifact",
                "production_prompt_compatibility_claimed": False,
            }
        else:
            corpus_signature = _signature(
                job["beir"][name]["corpus"], label=f"{name} corpus"
            )
            query_signature = _signature(
                job["beir"][name]["queries"], label=f"{name} queries"
            )
            corpus_ids_signature = _signature(
                job["beir"][name]["corpus_ids"], label=f"{name} corpus IDs"
            )
            query_ids_signature = _signature(
                job["beir"][name]["query_ids"], label=f"{name} query IDs"
            )
            source_corpus = np.load(
                corpus_signature["canonical_path"], mmap_mode="r", allow_pickle=False
            )
            source_queries = np.load(
                query_signature["canonical_path"], mmap_mode="r", allow_pickle=False
            )
            corpus_rows, query_rows, split = fixed_separate_split(
                source_corpus, source_queries, name=name
            )
            corpus = np.asarray(source_corpus[corpus_rows])
            queries = np.asarray(source_queries[query_rows])
            corpus_ids = corpus_rows
            query_ids = 1_000_000_000 + query_rows
            inputs = {
                "corpus_embeddings": corpus_signature,
                "query_embeddings": query_signature,
                "corpus_ids": corpus_ids_signature,
                "query_ids": query_ids_signature,
                "prompt_semantics": "legacy pooled Jina-v5 diagnostic artifact",
                "production_prompt_compatibility_claimed": False,
            }
        control_corpus, control_queries, control_split = shape_matched_control_split(
            control,
            name=name,
            corpus_rows=len(corpus_rows),
            query_rows=len(query_rows),
            representatives=control_representatives,
            duplicate_control=control_duplicate_control,
        )
        inputs = {**inputs, "control_embeddings": control_signature}
        reports[name] = r0142._score_one(
            name=name,
            corpus=corpus,
            queries=queries,
            corpus_ids=corpus_ids,
            query_ids=query_ids,
            control=control,
            control_corpus_rows=control_corpus,
            control_query_rows=control_queries,
            model=model,
            output=output,
            inputs=inputs,
            selection={"probe": split, "control": control_split},
        )
        print(
            f"R0147 {map_key} {index + 1}/{len(PROBE_ORDER)} {name}: "
            f"retention={reports[name]['metrics']['ffr_retention']:.4f}",
            flush=True,
        )
        del corpus, queries
        gc.collect()
    panel = seal({
        "schema": "round0147-row-policy-universality-panel-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "map_key": map_key,
        "model": model_signature,
        "probe_order": list(PROBE_ORDER),
        "probes": reports,
        "thresholds": {"pass_at_least": 0.70, "failure_below": 0.50},
        "role": "diagnostic-only; never part of the restoration selector",
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "universality-panel.json"), panel, immutable=True
    )


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0147 row-policy decision"
    )
    functional_path = os.path.join(
        str(job["functional_output"]), "functional-panel.json"
    )
    functional = _read_sealed(
        expected_input_signature(functional_path), label="R0147 functional panel"
    )
    if functional.get("round_id") != ROUND_ID:
        raise Round0147Error("R0147 functional panel identity changed")
    selection, selection_signature = _selection_receipt(str(job["selection_output"]))
    cells = {
        key: functional["cells"][key]
        for key in (CURRENT_GRAPH_CURRENT_HOST, TREATMENT)
    }
    decision = build_decision(
        cells, selection_summary=selection["selection_summary"]
    )
    ood: dict[str, Any] = {}
    for key, path in job["universality_outputs"].items():
        panel_path = os.path.join(str(path), "universality-panel.json")
        panel = _read_sealed(
            expected_input_signature(panel_path), label=f"R0147 {key} universality"
        )
        if (
            panel.get("round_id") != ROUND_ID
            or panel.get("map_key") != key
            or panel.get("probe_order") != list(PROBE_ORDER)
        ):
            raise Round0147Error("R0147 universality panel identity changed")
        ood[key] = {
            "panel": expected_input_signature(panel_path),
            "metrics": {
                name: panel["probes"][name]["metrics"] for name in PROBE_ORDER
            },
        }
    ood_deltas = {
        name: {
            metric: (
                float(ood[TREATMENT]["metrics"][name][metric])
                - float(ood[CURRENT_GRAPH_CURRENT_HOST]["metrics"][name][metric])
            )
            for metric in ("ffr_retention", "recall10_retention")
            if ood[TREATMENT]["metrics"][name].get(metric) is not None
            and ood[CURRENT_GRAPH_CURRENT_HOST]["metrics"][name].get(metric) is not None
        }
        for name in PROBE_ORDER
    }
    receipt = seal({
        **decision,
        "release_sha": active["manifest"]["release_sha"],
        "functional_panel": expected_input_signature(functional_path),
        "selection_receipt": selection_signature,
        "capability": CAPABILITY,
        "universality_diagnostic": ood,
        "universality_treatment_minus_control": ood_deltas,
        "universality_used_for_selector": False,
        "row_policy_includes_induced_graph_change": True,
    })
    atomic_write_new_json(os.path.join(output, "decision.json"), receipt, immutable=True)


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0147Error("R0147 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "materialize_selection":
        return run_materialize_selection(active, job)
    if action == "build_graph":
        return run_build_graph(active, job)
    if action == "train":
        return run_train(active, job)
    if action == "functional_panel":
        return run_functional_panel(active, job)
    if action == "universality_panel":
        return run_universality_panel(active, job)
    if action == "decide":
        return run_decision(active, job)
    raise Round0147Error(f"unknown R0147 action: {action!r}")
