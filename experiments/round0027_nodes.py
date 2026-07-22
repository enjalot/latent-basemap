"""Fresh-process node handlers for the Round 0027 six-cell MRL queue."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
    sha256_file,
)
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0027_program import (
    CANARY_UPDATES,
    CELL_LABELS,
    CENTROIDS,
    DIMENSIONS,
    GRAPH_EDGES,
    GRAPH_K,
    GRAPH_PATH,
    GRAPH_SHA256,
    PANEL_ANCHORS,
    PANEL_FRACTION,
    PANEL_SEED,
    PREFIX_PAYLOAD_SHA256,
    QUERY_ROWS,
    ROWS,
    SOURCE_4M_PATH,
    SOURCE_4M_SHA256,
    SOURCE_DIMENSION,
    SUCCESSFUL_UPDATES,
    TRAIN_PATH,
    TRAIN_SHA256,
    build_registered_decision,
    cosine_truth_array,
    input_array,
    parse_cell,
    preprocessing_for_dimension,
    query_row_ids,
    validate_job_cell,
    x_only_row_ceiling,
)


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _read_sealed(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items()
            if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise RuntimeError(f"sealed Round 0027 receipt changed: {path}")
    return value


def _verify_file(path: str, sha256: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature.get("sha256") != sha256:
        raise RuntimeError(
            f"Round 0027 static input changed: {path}: "
            f"{signature.get('sha256')} != {sha256}")
    return signature


def _verified_signature_path(signature: Any, *, label: str) -> str:
    if not isinstance(signature, dict):
        raise RuntimeError(f"Round 0027 {label} signature is missing")
    path = signature.get("canonical_path", "")
    if not path or expected_input_signature(path) != signature:
        raise RuntimeError(f"Round 0027 {label} content changed")
    return path


def _npy_payload_sha256(
    path: str, *, rows: int, known_file_sha256: str | None = None
) -> dict[str, Any]:
    """Hash exactly ``rows`` C-order payload rows, excluding the NPY header."""
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NumPy header version {version}")
        if (len(shape) != 2 or shape[0] < rows or shape[1] != SOURCE_DIMENSION or
                fortran or np.dtype(dtype) != np.dtype("float16")):
            raise ValueError(f"unexpected prefix-proof source header: {path} {shape}")
        header_bytes = handle.tell()
        remaining = rows * SOURCE_DIMENSION * np.dtype("float16").itemsize
        digest = hashlib.sha256()
        while remaining:
            chunk = handle.read(min(remaining, 16 << 20))
            if not chunk:
                raise ValueError(f"truncated NumPy payload: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return {
        "path": path,
        "file_sha256": known_file_sha256 or sha256_file(path),
        "declared_shape": [int(value) for value in shape],
        "dtype": np.dtype(dtype).str,
        "header_bytes": int(header_bytes),
        "prefix_rows": int(rows),
        "payload_bytes_hashed": int(
            rows * SOURCE_DIMENSION * np.dtype("float16").itemsize),
        "payload_sha256": digest.hexdigest(),
    }


def _panel_config():
    from basemap.panel_v2 import PanelV2Config

    return PanelV2Config(
        frac=PANEL_FRACTION,
        n_anchors=PANEL_ANCHORS,
        anchor_seed=PANEL_SEED,
        k_hit=10,
        k_density=15,
        k_clust=(256, 1024),
        corpus_chunk=500_000,
    )


def _new_model(config: dict[str, Any]):
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = config["model"]
    train = config["optimizer"]
    execution = config["execution"]
    graph = config["graph"]
    return ParametricUMAP(
        n_components=model["output_dimension"],
        hidden_dim=model["hidden_dimension"],
        n_layers=model["hidden_layers"],
        n_neighbors=GRAPH_K,
        a=model["a"],
        b=model["b"],
        low_dim_kernel=model["low_dim_kernel"],
        correlation_weight=train["correlation_weight"],
        learning_rate=train["learning_rate"],
        n_epochs=2,
        batch_size=train["batch_size"],
        device="cuda",
        use_batchnorm=model["use_batchnorm"],
        use_dropout=model["use_dropout"],
        clip_grad_norm=train["clip_grad_norm"],
        clip_grad_value=None,
        pos_ratio=train["positive_ratio"],
        architecture=model["architecture"],
        correlation_distance_transform="raw",
        lr_schedule="cosine",
        warmup_steps=train["warmup_successful_updates"],
        total_steps_estimate=train["successful_positive_lr_updates"],
        require_full_budget=True,
        require_graph_manifest=True,
        required_input_pipeline=execution["required_pipeline"],
        use_amp=train["use_amp"],
        positive_target_mode=train["positive_target_mode"],
        reject_neighbors=train["reject_neighbors"],
        anchored_init="none",
        anchor_hold_weight=0.0,
        midnear_enabled=False,
        mn_pairs_per_batch=0,
        weighted_edge_sampling=train["weighted_edge_sampling"],
        gpu_resident_data=execution["gpu_resident_data"],
        gpu_resident_vram_budget_gb=execution[
            "gpu_resident_vram_budget_gb"],
        graph_manifest_path=graph["manifest_path"],
        graph_manifest_sha256=graph["manifest_sha256"],
    )


def _pipeline_mismatches(stats: dict[str, Any], config: dict[str, Any]) \
        -> dict[str, dict[str, Any]]:
    expected = config["execution"]["expected_pipeline_stamp"]
    return {
        key: {"expected": value, "observed": stats.get(f"pipeline_{key}")}
        for key, value in expected.items()
        if stats.get(f"pipeline_{key}") != value
    }


def _configure_runtime(model, *, output: str, floor: float) -> None:
    model._max_train_steps = SUCCESSFUL_UPDATES
    model._bench_warmup = 200
    model._perf_profile = True
    model._perf_floor = float(floor)
    model._perf_warn_rate = max(float(floor), 200.0)
    model._perf_subfloor_patience = 2
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")


def run_sampler_canary(active: dict[str, Any], job: dict[str, Any]) -> None:
    cell = validate_job_cell(job)
    if cell["label"] != "d768_s42":
        raise RuntimeError("Round 0027 canary must use the largest control input")
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0027 canary output")
    config = cell["train_config"]
    _verify_file(TRAIN_PATH, TRAIN_SHA256)

    import torch

    torch.cuda.reset_peak_memory_stats("cuda")
    np.random.seed(cell["seed"])
    torch.manual_seed(cell["seed"])
    torch.cuda.manual_seed_all(cell["seed"])
    X = input_array(cell["dimension"])
    model = _new_model(config)
    model._max_train_steps = CANARY_UPDATES
    model._bench_warmup = 50
    model._perf_profile = True
    model._perf_floor = 200.0
    model._perf_warn_rate = 200.0
    model._perf_subfloor_patience = 2
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        X,
        low_memory=False,
        verbose=False,
        n_processes=6,
        random_state=cell["seed"],
        resample_negatives=False,
        precomputed_edges_path=GRAPH_PATH,
        use_wandb=False,
    )
    wall = time.monotonic() - started
    stats = dict(model._train_stats)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    mismatches = _pipeline_mismatches(stats, config)
    exact = {
        "positive_lr_optimizer_steps": CANARY_UPDATES,
        "optimizer_steps_succeeded": CANARY_UPDATES,
        "stop_reason": "bench_cap",
        "budget_satisfied": True,
        "n_pos_edges": GRAPH_EDGES,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
    }
    mismatches.update({
        key: {"expected": value, "observed": stats.get(key)}
        for key, value in exact.items()
        if stats.get(key) != value
    })
    if stats.get("verified_hashes", {}).get("graph_sha256") != GRAPH_SHA256:
        mismatches["verified_graph_sha256"] = {
            "expected": GRAPH_SHA256,
            "observed": stats.get("verified_hashes", {}).get("graph_sha256"),
        }
    # The weighted device sampler builds one full epoch of inverse-CDF draws
    # before its first update.  That is real setup cost and is retained in
    # ``wall_seconds``/``stats``, but it is not the registered steady-state
    # throughput window.  Use the trainer's synchronized post-warmup interval,
    # matching the existing canary contract in ``run_experiment.py``.
    bench_seconds = model._bench_seconds
    steady_updates = CANARY_UPDATES - model._bench_warmup
    rate = (
        round(steady_updates / bench_seconds, 1)
        if isinstance(bench_seconds, (int, float)) and bench_seconds > 0
        else 0.0
    )
    if profiler.get("aborted") is not False:
        mismatches["performance_profile_aborted"] = {
            "expected": False,
            "observed": profiler.get("aborted"),
        }
    if int(profiler.get("n_windows") or 0) < 5:
        mismatches["performance_profile_windows"] = {
            "expected": ">=5",
            "observed": profiler.get("n_windows"),
        }
    if rate < 200.0:
        mismatches["steady_updates_per_s"] = {
            "expected": ">=200.0", "observed": rate}
    peak_allocated = int(torch.cuda.max_memory_allocated("cuda"))
    peak_reserved = int(torch.cuda.max_memory_reserved("cuda"))
    minimum_headroom = int(1.5 * 1024 ** 3)
    conservative_headroom = min(
        int(free_bytes), max(0, int(total_bytes) - peak_reserved))
    if conservative_headroom < minimum_headroom:
        mismatches["canary_peak_headroom_bytes"] = {
            "expected": f">={minimum_headroom}",
            "observed": conservative_headroom,
        }
    if mismatches:
        raise RuntimeError(f"Round 0027 sampler canary failed: {mismatches}")
    evidence = {
        "schema": "round0027-mrl-sampler-canary-v1",
        "round_id": "0027",
        "cell": cell["label"],
        "production_config_sha256": cell["train_config_sha256"],
        "training_performed": True,
        "throwaway_optimizer_updates": CANARY_UPDATES,
        "no_model_published": True,
        "stats": stats,
        "pipeline": {
            key: stats[f"pipeline_{key}"]
            for key in config["execution"]["expected_pipeline_stamp"]
        },
        "wall_seconds": wall,
        "steady_updates_per_s": rate,
        "steady_window": {
            "warmup_updates": model._bench_warmup,
            "measured_updates": steady_updates,
            "bench_seconds": bench_seconds,
        },
        "performance_profile": profiler,
        "post_canary_memory": {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "minimum_required_free_bytes": minimum_headroom,
            "conservative_peak_headroom_bytes": conservative_headroom,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "admission": expected_input_signature(
            os.path.join(output, "admission.json")),
    }
    evidence_path = os.path.join(output, "evidence.json")
    atomic_write_new_json(evidence_path, _seal(evidence), immutable=True)
    verdict = {
        "schema": "round0027-mrl-sampler-canary-verdict-v1",
        "round_id": "0027",
        "passed": True,
        "cell": cell["label"],
        "throwaway_optimizer_updates": CANARY_UPDATES,
        "minimum_updates_per_s": 200.0,
        "observed_steady_updates_per_s": rate,
        "minimum_free_bytes": minimum_headroom,
        "observed_conservative_peak_headroom_bytes": conservative_headroom,
        "evidence": expected_input_signature(evidence_path),
    }
    atomic_write_new_json(
        os.path.join(output, "verdict.json"), _seal(verdict), immutable=True)
    del model, X
    torch.cuda.empty_cache()


def run_shared_reference(active: dict[str, Any], job: dict[str, Any]) -> None:
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0027 shared reference output")
    train_signature = _verify_file(TRAIN_PATH, TRAIN_SHA256)
    source_signature = _verify_file(SOURCE_4M_PATH, SOURCE_4M_SHA256)
    centroid_signatures = {
        str(k): _verify_file(item["path"], item["sha256"])
        for k, item in CENTROIDS.items()
    }
    ids = np.load(job["query_ids_path"], allow_pickle=False)
    expected_ids = query_row_ids()
    if (ids.dtype != np.dtype("int64") or ids.shape != (QUERY_ROWS,) or
            not np.array_equal(ids, expected_ids)):
        raise RuntimeError("Round 0027 held-out query IDs changed")

    left_proof = _npy_payload_sha256(
        TRAIN_PATH, rows=ROWS, known_file_sha256=train_signature["sha256"])
    right_proof = _npy_payload_sha256(
        SOURCE_4M_PATH, rows=ROWS,
        known_file_sha256=source_signature["sha256"])
    if (left_proof["payload_sha256"] != PREFIX_PAYLOAD_SHA256 or
            right_proof["payload_sha256"] != PREFIX_PAYLOAD_SHA256):
        raise RuntimeError("Round 0027 2M/4M literal-prefix identity failed")
    prefix_body = {
        "schema": "round0027-literal-prefix-proof-v1",
        "byte_identical": True,
        "shared_payload_sha256": PREFIX_PAYLOAD_SHA256,
        "two_million_file": left_proof,
        "four_million_prefix": right_proof,
    }
    prefix_path = os.path.join(output, "literal-prefix-proof.json")
    atomic_write_new_json(prefix_path, _seal(prefix_body), immutable=True)

    source_4m = np.load(SOURCE_4M_PATH, mmap_mode="r", allow_pickle=False)
    Xq = np.asarray(source_4m[ids], dtype=np.float32)
    if (Xq.shape != (QUERY_ROWS, SOURCE_DIMENSION) or
            not np.isfinite(Xq).all()):
        raise RuntimeError("Round 0027 held-out query matrix is invalid")
    query_path = os.path.join(output, "oos-query-embeddings-768.npy")
    atomic_save_new_npy(query_path, Xq, immutable=True)

    from basemap.panel_v2 import (
        build_hiD_reference,
        build_query_truth,
        sample_anchors,
        save_hiD_reference,
        save_query_truth,
    )

    X = input_array(768)
    cfg = _panel_config()
    centroids = {
        k: np.load(item["path"], mmap_mode="r", allow_pickle=False)
        for k, item in CENTROIDS.items()
    }
    anchors = sample_anchors(len(X), cfg)
    reference = build_hiD_reference(X, anchors, cfg, centroids)
    reference_path = os.path.join(output, "high-d-reference.npz")
    save_hiD_reference(reference, reference_path)
    query_identity = {
        "ordered_query_ids_sha256": ordered_array_sha256(ids),
        "ordered_query_embeddings_sha256": ordered_array_sha256(Xq),
        "source_4m": source_signature,
        "literal_prefix_proof": expected_input_signature(prefix_path),
    }
    corpus_identity = {
        "train": train_signature,
        "shape": [ROWS, SOURCE_DIMENSION],
        "dtype": "<f2-source/<f4-scoring",
        "literal_prefix_payload_sha256": PREFIX_PAYLOAD_SHA256,
    }
    X_cosine = cosine_truth_array()
    from basemap.data_loader import PrefixL2NormalizedArray
    Xq_cosine = PrefixL2NormalizedArray(
        Xq,
        source_dimension=SOURCE_DIMENSION,
        output_dimension=SOURCE_DIMENSION,
        normalize=True,
        source_paths=[query_path],
    )
    query_identity["cosine_truth_preprocessing"] = \
        Xq_cosine.execution_preprocessing_stamp
    corpus_identity["cosine_truth_preprocessing"] = \
        X_cosine.execution_preprocessing_stamp
    truth = build_query_truth(
        Xq_cosine,
        X_cosine,
        cfg=cfg,
        corpus_identity=corpus_identity,
        query_identity=query_identity,
        k=10,
    )
    truth_path = os.path.join(output, "oos-query-truth-k10.npz")
    save_query_truth(truth, truth_path)
    receipt = {
        "schema": "round0027-shared-score-reference-v1",
        "round_id": "0027",
        "release_sha": active["manifest"]["release_sha"],
        "train": train_signature,
        "source_4m": source_signature,
        "literal_prefix_proof": expected_input_signature(prefix_path),
        "query_ids": expected_input_signature(job["query_ids_path"]),
        "query_embeddings": expected_input_signature(query_path),
        "ordered_query_embeddings_sha256": ordered_array_sha256(Xq),
        "centroids": centroid_signatures,
        "high_d_reference": expected_input_signature(reference_path),
        "high_d_reference_key": reference["key"],
        "high_d_reference_content_sha256": reference["content_sha256"],
        "query_truth": expected_input_signature(truth_path),
        "query_truth_key": truth["key"],
        "query_truth_payload_sha256": truth["payload_sha256"],
        "query_truth_exactness": truth["key_parts"]["policy"],
        "shared_across_all_six_cells": True,
    }
    atomic_write_new_json(
        os.path.join(output, "receipt.json"), _seal(receipt), immutable=True)


def _validate_canary(path: str) -> dict[str, Any]:
    verdict_path = os.path.join(path, "verdict.json")
    verdict = _read_sealed(verdict_path)
    evidence = verdict.get("evidence")
    if (verdict.get("passed") is not True or
            verdict.get("throwaway_optimizer_updates") != CANARY_UPDATES or
            not isinstance(evidence, dict) or
            expected_input_signature(evidence.get("canonical_path", "")) != evidence):
        raise RuntimeError("Round 0027 training lacks its passing sampler canary")
    return verdict


def _publish_model(model, path: str) -> str:
    return atomic_build_new_file(path, model.save, immutable=True)


def run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    cell = validate_job_cell(job)
    _validate_canary(job["canary_output"])
    output = create_fresh_directory(
        job["outputs"][0], label=f"Round 0027 {cell['label']} train output")
    config = cell["train_config"]
    atomic_write_new_json(
        os.path.join(output, "production-config.json"),
        {
            "schema": "round0027-production-config-receipt-v1",
            "cell": cell["label"],
            "config": config,
            "config_sha256": cell["train_config_sha256"],
        },
        immutable=True,
    )
    _verify_file(TRAIN_PATH, TRAIN_SHA256)

    import torch

    torch.cuda.reset_peak_memory_stats("cuda")
    np.random.seed(cell["seed"])
    torch.manual_seed(cell["seed"])
    torch.cuda.manual_seed_all(cell["seed"])
    X = input_array(cell["dimension"])
    model = _new_model(config)
    _configure_runtime(
        model, output=output,
        floor=config["execution"]["minimum_train_upd_s"])
    started = time.monotonic()
    model.fit(
        X,
        low_memory=False,
        verbose=False,
        n_processes=6,
        random_state=cell["seed"],
        resample_negatives=False,
        precomputed_edges_path=GRAPH_PATH,
        use_wandb=False,
    )
    wall = time.monotonic() - started
    stats = dict(model._train_stats)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    exact = {
        "amp_dtype": "bfloat16",
        "schedule_version": "cosine-v3-positive-budget",
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": GRAPH_EDGES,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "amp_overflow_skips": 0,
    }
    mismatches = _pipeline_mismatches(stats, config)
    mismatches.update({
        key: {"expected": value, "observed": stats.get(key)}
        for key, value in exact.items()
        if stats.get(key) != value
    })
    verified = stats.get("verified_hashes", {})
    for key, value in {
        "graph_sha256": GRAPH_SHA256,
        "graph_manifest_sha256": config["graph"]["manifest_sha256"],
    }.items():
        if verified.get(key) != value:
            mismatches[f"verified_{key}"] = {
                "expected": value, "observed": verified.get(key)}
    rate = float(stats.get("updates_per_s") or 0.0)
    if rate < config["execution"]["minimum_train_upd_s"]:
        mismatches["updates_per_s"] = {
            "expected": f">={config['execution']['minimum_train_upd_s']}",
            "observed": rate,
        }
    if profiler.get("aborted") is not False:
        mismatches["performance_profile_aborted"] = {
            "expected": False,
            "observed": profiler.get("aborted"),
        }
    if int(profiler.get("n_windows") or 0) < 5:
        mismatches["performance_profile_windows"] = {
            "expected": ">=5",
            "observed": profiler.get("n_windows"),
        }
    for name in ("lr_used_first", "lr_used_last", "first_lr", "final_lr"):
        value = stats.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            mismatches[name] = {"expected": "finite positive", "observed": value}
    if mismatches:
        raise RuntimeError(
            f"Round 0027 {cell['label']} exact accounting failed: {mismatches}")
    model_path = os.path.join(output, "model.pt")
    _publish_model(model, model_path)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    receipt = {
        "schema": "round0027-train-receipt-v1",
        "round_id": "0027",
        "cell": cell["label"],
        "dimension": cell["dimension"],
        "seed": cell["seed"],
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": cell["train_config_sha256"],
        "model": expected_input_signature(model_path),
        "train_accounting": stats,
        "performance_profile": profiler,
        "actual_pipeline": {
            key: stats[f"pipeline_{key}"]
            for key in config["execution"]["expected_pipeline_stamp"]
        },
        "verified_inputs": verified,
        "train_wall_seconds": wall,
        "updates_per_s": rate,
        "memory": {
            "device_total_bytes": int(total_bytes),
            "post_train_free_bytes": int(free_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
        "retry_count": 0,
    }
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), _seal(receipt),
        immutable=True)
    del model, X
    torch.cuda.empty_cache()


def _authenticate_model(cell: dict[str, Any], train_output: str):
    train_path = os.path.join(train_output, "train-receipt.json")
    train = _read_sealed(train_path)
    model_signature = train.get("model")
    if (train.get("cell") != cell["label"] or
            train.get("production_config_sha256") !=
            cell["train_config_sha256"] or
            not isinstance(model_signature, dict) or
            expected_input_signature(
                model_signature.get("canonical_path", "")) != model_signature):
        raise RuntimeError(f"Round 0027 {cell['label']} train receipt changed")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device="cuda")
    expected = cell["train_config"]["model"]
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
    dtypes = sorted({str(parameter.dtype) for parameter in model.model.parameters()})
    if observed != expected or dtypes != ["torch.float32"]:
        raise RuntimeError(
            f"Round 0027 model architecture changed: {observed} {dtypes}")
    return model, train, expected_input_signature(train_path)


def run_transform(active: dict[str, Any], job: dict[str, Any]) -> None:
    cell = validate_job_cell(job)
    output = create_fresh_directory(
        job["outputs"][0], label=f"Round 0027 {cell['label']} transform output")
    model, train, train_signature = _authenticate_model(
        cell, job["train_output"])
    shared_receipt_path = os.path.join(
        job["shared_reference_output"], "receipt.json")
    shared = _read_sealed(shared_receipt_path)
    query_signature = shared.get("query_embeddings")
    if (not isinstance(query_signature, dict) or
            expected_input_signature(query_signature.get("canonical_path", ""))
            != query_signature):
        raise RuntimeError("Round 0027 shared query embedding changed")
    ids = np.load(job["query_ids_path"], allow_pickle=False)
    if not np.array_equal(ids, query_row_ids()):
        raise RuntimeError("Round 0027 transform query IDs changed")
    X = input_array(cell["dimension"])
    raw_queries = np.load(
        query_signature["canonical_path"], mmap_mode="r", allow_pickle=False)
    _, stamp = preprocessing_for_dimension(cell["dimension"])
    from basemap.data_loader import PrefixL2NormalizedArray

    Xq = PrefixL2NormalizedArray(
        raw_queries,
        source_dimension=SOURCE_DIMENSION,
        output_dimension=cell["dimension"],
        normalize=cell["dimension"] < SOURCE_DIMENSION,
        source_paths=[query_signature["canonical_path"]],
    )
    if Xq.execution_preprocessing_stamp != stamp:
        raise RuntimeError("Round 0027 transform preprocessing stamp changed")
    started = time.monotonic()
    Z = np.asarray(model.transform(X, batch_size=8192), dtype=np.float32)
    Zq = np.asarray(model.transform(Xq, batch_size=8192), dtype=np.float32)
    wall = time.monotonic() - started
    if (Z.shape != (ROWS, 2) or Zq.shape != (QUERY_ROWS, 2) or
            not np.isfinite(Z).all() or not np.isfinite(Zq).all()):
        raise RuntimeError(f"Round 0027 {cell['label']} transform is invalid")
    coordinates_path = os.path.join(output, "coordinates.npy")
    queries_path = os.path.join(output, "oos-query-coordinates.npy")
    atomic_save_new_npy(coordinates_path, Z, immutable=True)
    atomic_save_new_npy(queries_path, Zq, immutable=True)
    receipt = {
        "schema": "round0027-transform-receipt-v1",
        "round_id": "0027",
        "cell": cell["label"],
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": cell["train_config_sha256"],
        "train_receipt": train_signature,
        "model": train["model"],
        "input_preprocessing": stamp,
        "source": _verify_file(TRAIN_PATH, TRAIN_SHA256),
        "query_source": query_signature,
        "query_ids": expected_input_signature(job["query_ids_path"]),
        "coordinates": expected_input_signature(coordinates_path),
        "query_coordinates": expected_input_signature(queries_path),
        "wall_seconds": wall,
        "finite": True,
    }
    atomic_write_new_json(
        os.path.join(output, "transform-receipt.json"), _seal(receipt),
        immutable=True)


def run_score(active: dict[str, Any], job: dict[str, Any]) -> None:
    cell = validate_job_cell(job)
    output = create_fresh_directory(
        job["outputs"][0], label=f"Round 0027 {cell['label']} score output")
    train_path = os.path.join(job["train_output"], "train-receipt.json")
    train = _read_sealed(train_path)
    transform_path = os.path.join(
        job["transform_output"], "transform-receipt.json")
    transform = _read_sealed(transform_path)
    if (train.get("cell") != cell["label"] or
            transform.get("cell") != cell["label"] or
            train.get("production_config_sha256") !=
            cell["train_config_sha256"] or
            transform.get("production_config_sha256") !=
            cell["train_config_sha256"]):
        raise RuntimeError(f"Round 0027 {cell['label']} score input changed")

    X = input_array(768)
    Z = np.load(
        _verified_signature_path(
            transform.get("coordinates"), label="coordinate"),
        mmap_mode="r", allow_pickle=False)
    Zq = np.load(
        _verified_signature_path(
            transform.get("query_coordinates"),
            label="OOS query-coordinate"),
        mmap_mode="r", allow_pickle=False)
    shared_root = job["shared_reference_output"]
    shared_path = os.path.join(shared_root, "receipt.json")
    shared = _read_sealed(shared_path)
    from basemap.panel_v2 import (
        cross_knn,
        ffr_from_neighbors,
        load_hiD_reference,
        load_query_truth,
        recall_at_k_from_neighbors,
        score_panel,
    )

    reference = load_hiD_reference(
        shared["high_d_reference"]["canonical_path"],
        expected_key=shared["high_d_reference_key"])
    truth = load_query_truth(
        shared["query_truth"]["canonical_path"],
        expected_key=shared["query_truth_key"],
        expected_candidate_compute_backend="cuda",
    )
    centroids = {
        k: np.load(item["path"], mmap_mode="r", allow_pickle=False)
        for k, item in CENTROIDS.items()
    }
    cfg = _panel_config()
    panel = score_panel(
        X,
        Z,
        config=cfg,
        centroids_by_k=centroids,
        hiD_reference=reference,
        scale_admission=None,
        provenance={
            "round_id": "0027",
            "cell": cell["label"],
            "release_sha": active["manifest"]["release_sha"],
            "train_receipt": expected_input_signature(train_path),
            "transform_receipt": expected_input_signature(transform_path),
            "shared_reference_receipt": expected_input_signature(shared_path),
        },
    )
    kf = max(cfg.k_hit, int(np.ceil(cfg.frac * len(Z))))
    low = cross_knn(Zq, Z, kf, cfg, hi_dim=False)
    high = truth["neighbors"][:, :cfg.k_hit]
    projection = {
        "proj_ffr": round(ffr_from_neighbors(high, low, cfg.k_hit), 4),
        "proj_recall@k": round(
            recall_at_k_from_neighbors(high, low, cfg.k_hit), 5),
        "proj_n_queries": QUERY_ROWS,
        "proj_k_frac": kf,
        "truth": shared["query_truth"],
        "truth_payload_sha256": truth["payload_sha256"],
        "truth_exactness": truth["key_parts"]["policy"],
    }
    guards = panel.get("guards") or {}
    numerical = bool(
        guards.get("coords_finite") is True
        and guards.get("coords_collapsed") is False
        and guards.get("emb_finite") is True
        and guards.get("emb_zero_rows") == 0
        and np.isfinite(projection["proj_ffr"])
    )
    if not numerical:
        raise RuntimeError(
            f"Round 0027 {cell['label']} numerical guards failed: {guards}")
    report = {
        "schema": "round0027-cell-score-v1",
        "round_id": "0027",
        "cell": cell["label"],
        "dimension": cell["dimension"],
        "seed": cell["seed"],
        "release_sha": active["manifest"]["release_sha"],
        "production_config_sha256": cell["train_config_sha256"],
        "train_receipt": expected_input_signature(train_path),
        "transform_receipt": expected_input_signature(transform_path),
        "shared_reference_receipt": expected_input_signature(shared_path),
        "panel": panel,
        "projection": projection,
        "actual_pipeline": train["actual_pipeline"],
        "numerical_guards_passed": True,
    }
    atomic_write_new_json(
        os.path.join(output, "panel.json"), _seal(report), immutable=True)


def _fixed_axis_render(
    path: str, cell_outputs: dict[str, dict[str, str]]
) -> dict[str, Any]:
    rng = np.random.RandomState(20_260_727)
    sample_ids = np.sort(rng.choice(ROWS, 20_000, replace=False)).astype(
        np.int64)
    samples = {}
    for label in CELL_LABELS:
        transform = _read_sealed(os.path.join(
            cell_outputs[label]["transform"], "transform-receipt.json"))
        coordinate_path = _verified_signature_path(
            transform.get("coordinates"), label=f"{label} render coordinate")
        coords = np.load(
            coordinate_path, mmap_mode="r", allow_pickle=False)
        samples[label] = np.asarray(coords[sample_ids], dtype=np.float32)
    all_points = np.concatenate(list(samples.values()), axis=0)
    lo = np.quantile(all_points, 0.001, axis=0)
    hi = np.quantile(all_points, 0.999, axis=0)
    pad = np.maximum((hi - lo) * 0.03, 1e-6)
    xlim = [float(lo[0] - pad[0]), float(hi[0] + pad[0])]
    ylim = [float(lo[1] - pad[1]), float(hi[1] + pad[1])]

    def draw(destination: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 2, figsize=(10, 14))
        for axis, label in zip(axes.flat, CELL_LABELS):
            points = samples[label]
            axis.scatter(
                points[:, 0], points[:, 1], s=0.2, alpha=0.3,
                linewidths=0, rasterized=True)
            axis.set_title(label)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle("R0027 — shared sample and fixed axes")
        figure.tight_layout()
        figure.savefig(destination, format="png", dpi=180)
        plt.close(figure)

    atomic_build_new_file(path, draw, immutable=True)
    return {
        "image": expected_input_signature(path),
        "sample_rows": len(sample_ids),
        "sample_ids_sha256": ordered_array_sha256(sample_ids),
        "xlim": xlim,
        "ylim": ylim,
        "axis_policy": "shared 0.1%-99.9% union quantiles plus 3% padding",
    }


def run_decision(active: dict[str, Any], job: dict[str, Any]) -> None:
    output = create_fresh_directory(
        job["outputs"][0], label="Round 0027 decision output")
    cells: dict[str, dict[str, Any]] = {}
    for label in CELL_LABELS:
        paths = job["cell_outputs"][label]
        panel_path = os.path.join(paths["panel"], "panel.json")
        train_path = os.path.join(paths["train"], "train-receipt.json")
        panel = _read_sealed(panel_path)
        train = _read_sealed(train_path)
        dimension, seed = parse_cell(label)
        if (panel.get("cell") != label or train.get("cell") != label or
                panel.get("numerical_guards_passed") is not True):
            raise RuntimeError(f"Round 0027 decision input {label} changed")
        cells[label] = {
            "dimension": dimension,
            "seed": seed,
            "train": expected_input_signature(train_path),
            "panel": expected_input_signature(panel_path),
            "ffr": float(panel["panel"]["ffr"]),
            "purity_k256": float(panel["panel"]["purity"]["k256"]),
            "purity_k1024": float(panel["panel"]["purity"]["k1024"]),
            "density": float(panel["panel"]["density"]),
            "oos_proj_ffr": float(panel["projection"]["proj_ffr"]),
            "updates_per_s": float(train["updates_per_s"]),
            "peak_reserved_bytes": int(train["memory"]["peak_reserved_bytes"]),
            "pipeline": train["actual_pipeline"],
        }
    decision_summary = build_registered_decision(cells)
    render = _fixed_axis_render(
        os.path.join(output, "fixed-axis-six-cell.png"),
        job["cell_outputs"],
    )
    ceilings = {
        str(dimension): {
            "x_only_rows_at_31GB_fp16": x_only_row_ceiling(dimension),
            "bytes_per_input_row": dimension * 2,
            "note": (
                "X-residency arithmetic only; graph/CDF/model/headroom reduce "
                "the end-to-end pipeline ceiling"
            ),
        }
        for dimension in DIMENSIONS
    }
    body = {
        "schema": "round0027-mrl-input-decision-v1",
        "round_id": "0027",
        "release_sha": active["manifest"]["release_sha"],
        "cells": cells,
        "seed_means": decision_summary["seed_means"],
        "control_768_ffr_seed_spread_max_minus_min": decision_summary[
            "control_768_ffr_seed_spread_max_minus_min"],
        "registered_rule": {
            "choose_smallest_dimension": [256, 384],
            "oos_proj_ffr_ratio_min": 0.90,
            "transductive_ffr_tolerance": (
                "observed max-minus-min spread of the two 768d control seeds"),
            "all_numerical_and_execution_guards_required": True,
        },
        "qualification": decision_summary["qualification"],
        "decision": decision_summary["decision"],
        "adopted_input_dimension": decision_summary[
            "adopted_input_dimension"],
        "x_residency_ceiling_arithmetic": ceilings,
        "fixed_axis_render": render,
        "literal_prefix_proof": _read_sealed(os.path.join(
            job["shared_reference_output"], "literal-prefix-proof.json")),
    }
    atomic_write_new_json(
        os.path.join(output, "mrl-input-decision-v1.json"), _seal(body),
        immutable=True)
