"""Execute the conditional R0138 device-sampler/runtime bridge."""
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
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import (
    DIMENSION,
    InventoryFp16Array,
    ROWS,
    SEED,
    SUCCESSFUL_UPDATES,
    preprocessing_stamp,
    source_prefix_proof,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0134_functional_showdown import (
    CURRENT_R0104_SEED42,
    HISTORICAL_SEED42,
)
from basemap.round0138_sampler_bridge import (
    CAPABILITY,
    CELL_ORDER,
    CONTROL,
    HISTORICAL,
    PANEL_SCHEMA,
    PIPELINE,
    ROUND_ID,
    SAMPLER_CLASS,
    TRAIN_MINIMUM_UPDATES_PER_S,
    TREATMENT,
    Round0138Error,
    build_decision,
    train_config,
)
from experiments import round0104_nodes as r0104
from experiments.round0027_nodes import _panel_config
from experiments.round0119_nodes import SOURCE_ROWS
from experiments.round0134_nodes import (
    _load_reference,
    _load_shared_evaluation_inputs,
    _projection_metrics,
)


TRANSFORM_BATCH_ROWS = 8_192
PERFORMANCE_WARMUP_UPDATES = 200
PERFORMANCE_WINDOWS = 5


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0138Error(f"{label} bytes changed")
    return actual


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0138Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value


class DeviceInventoryFp16Array(InventoryFp16Array):
    """R0104 source with the generic device loader's identity hooks."""

    def __init__(self) -> None:
        super().__init__(0, ROWS)
        self.execution_preprocessing_stamp = preprocessing_stamp("fp16_control")
        self.loaded_shard_paths = [
            item["shard"]["canonical_path"] for item in self.segments
        ]


def _actual_pipeline(stats: Mapping[str, Any], expected: Mapping[str, Any]) -> dict:
    return {
        key: stats.get(f"pipeline_{key}")
        for key in expected
    } | {"path_reason": stats.get("pipeline_path_reason")}


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    graph_signature = _signature(job["graph"], label="R0104 graph")
    manifest_signature = _signature(
        job["device_graph_manifest"], label="R0138 device graph manifest"
    )
    parent_manifest_signature = _signature(
        job["parent_graph_manifest"], label="R0104 parent graph manifest"
    )
    with open(manifest_signature["canonical_path"], encoding="utf-8") as handle:
        graph_manifest = json.load(handle)
    if (
        not isinstance(graph_manifest, dict)
        or graph_manifest.get("schema") != "graph_manifest.v2"
        or graph_manifest.get("parent_manifest") != parent_manifest_signature
        or graph_manifest.get("graph_sha256") != graph_signature["sha256"]
        or graph_manifest.get("n_nodes") != ROWS
        or graph_manifest.get("n_edges") != int(job["graph_edges"])
        or graph_manifest.get("input_preprocessing")
        != preprocessing_stamp("fp16_control")
    ):
        raise Round0138Error("R0138 device graph adapter changed")
    proof = source_prefix_proof()
    if proof != job["source_prefix_proof"]:
        raise Round0138Error("R0138 source prefix proof changed")
    config, config_sha = train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        graph_edges=int(job["graph_edges"]),
    )
    if config != job["production_config"] or config_sha != job["production_config_sha256"]:
        raise Round0138Error("R0138 device-sampler production config changed")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0138 device-sampler train output"
    )
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        seal({
            "schema": "round0138-device-sampler-production-config-v1",
            "round_id": ROUND_ID,
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
    source = DeviceInventoryFp16Array()
    model = r0104._new_model(config)
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
        source,
        low_memory=False,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=graph_signature["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started
    accounting = dict(model._train_stats)
    expected_pipeline = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": accounting.get(f"pipeline_{key}")}
        for key, value in expected_pipeline.items()
        if accounting.get(f"pipeline_{key}") != value
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
        "n_pos_edges": int(job["graph_edges"]),
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    verified = accounting.get("verified_hashes") or {}
    if verified.get("graph_sha256") != graph_signature["sha256"]:
        mismatches["verified_graph_sha256"] = {
            "expected": graph_signature["sha256"],
            "observed": verified.get("graph_sha256"),
        }
    if verified.get("graph_manifest_sha256") != manifest_signature["sha256"]:
        mismatches["verified_graph_manifest_sha256"] = {
            "expected": manifest_signature["sha256"],
            "observed": verified.get("graph_manifest_sha256"),
        }
    if mismatches:
        raise Round0138Error(f"R0138 train accounting failed: {mismatches}")
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - PERFORMANCE_WARMUP_UPDATES) / model._bench_seconds
        if model._bench_seconds else 0.0
    )
    if profiler.get("aborted") is not False or rate < TRAIN_MINIMUM_UPDATES_PER_S:
        raise Round0138Error("R0138 device-sampler performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    receipt = seal({
        "schema": "round0138-device-sampler-train-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "causal_change": "host-to-device-sampler-runtime-only",
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "model": expected_input_signature(model_path),
        "graph": graph_signature,
        "graph_manifest": manifest_signature,
        "parent_graph_manifest": parent_manifest_signature,
        "source_prefix_proof": proof,
        "train_accounting": accounting,
        "exact_execution_receipt": _actual_pipeline(accounting, expected_pipeline),
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_seconds": wall,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "device_residency_and_sampler_proved": True,
        },
        "memory": {
            "device_total_bytes": int(total_bytes),
            "post_train_free_bytes": int(free_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
        },
        "retry_count": 0,
    })
    path = os.path.join(output, "train-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del model, source
    torch.cuda.empty_cache()
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _authenticate_model(job: Mapping[str, Any], *, device: str = "cuda"):
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train = _read_sealed(expected_input_signature(train_path), label="R0138 train")
    graph_signature = _signature(job["graph"], label="R0104 graph")
    manifest_signature = _signature(
        job["device_graph_manifest"], label="R0138 device graph manifest"
    )
    config, config_sha = train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        graph_edges=int(job["graph_edges"]),
    )
    if (
        train.get("round_id") != ROUND_ID
        or train.get("production_config_sha256") != config_sha
        or train.get("graph") != graph_signature
        or train.get("graph_manifest") != manifest_signature
        or (train.get("exact_execution_receipt") or {}).get("pipeline") != PIPELINE
        or (train.get("exact_execution_receipt") or {}).get("sampler_class")
        != SAMPLER_CLASS
    ):
        raise Round0138Error("R0138 train receipt/config changed")
    model_signature = _signature(train["model"], label="R0138 model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device=device)
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
        raise Round0138Error("R0138 model architecture changed")
    return model, train, expected_input_signature(train_path), config_sha


def _render(
    *, output: str, cells: Mapping[str, Mapping[str, Any]], labels: np.ndarray,
    sample: np.ndarray,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = create_fresh_directory(
        os.path.join(output, "renders"), label="R0138 sampler-bridge render"
    )
    sample_path = os.path.join(root, "sample-row-ids.npy")
    atomic_save_new_npy(sample_path, sample, immutable=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    titles = {
        HISTORICAL: "historical R0037 device s42",
        CONTROL: "current R0104 host s42",
        TREATMENT: "current device treatment s42",
    }
    color = np.asarray(labels[sample] % 20, dtype=np.int16)
    limits: dict[str, Any] = {}
    for axis, key in zip(axes, CELL_ORDER, strict=True):
        coords = np.load(
            cells[key]["coordinates"]["canonical_path"], mmap_mode="r",
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
        axis.set_title(titles[key], fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        limits[key] = {"quantile_low": low.tolist(), "quantile_high": high.tolist()}
    figure.tight_layout()
    path = os.path.join(root, "device-sampler-bridge.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    os.chmod(path, 0o444)
    receipt = seal({
        "schema": "round0138-device-sampler-render-v1",
        "round_id": ROUND_ID,
        "sample": expected_input_signature(sample_path),
        "same_ordered_rows": True,
        "color": "frozen R0037 k256 label modulo 20",
        "axes": "per-cell 0.1%-99.9% robust axes; diagnostic only",
        "limits": limits,
        "render": expected_input_signature(path),
    })
    manifest = os.path.join(root, "render-manifest.json")
    atomic_write_new_json(manifest, receipt, immutable=True)
    return {**receipt, "manifest": expected_input_signature(manifest)}


def run_panel(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0138 functional sampler panel"
    )
    started = time.monotonic()
    r0134 = _read_sealed(job["r0134_panel"], label="R0134 functional panel")
    source_cells = r0134.get("cells")
    if not isinstance(source_cells, Mapping):
        raise Round0138Error("R0134 functional cells are missing")
    source_signature, source, queries = _load_shared_evaluation_inputs(job)
    shared, shared_signature, reference, truth, centroids = _load_reference(job)
    model, train, train_signature, config_sha = _authenticate_model(job)
    coordinates = np.asarray(
        model.transform(source, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
    )
    query_coordinates = np.asarray(
        model.transform(queries, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
    )
    if (
        coordinates.shape != (SOURCE_ROWS, 2)
        or query_coordinates.shape != (20_000, 2)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(query_coordinates).all()
    ):
        raise Round0138Error("R0138 treatment transform is malformed")
    coords_path = os.path.join(output, "treatment-coordinates.npy")
    query_path = os.path.join(output, "treatment-query-coordinates.npy")
    atomic_save_new_npy(coords_path, coordinates, immutable=True)
    atomic_save_new_npy(query_path, query_coordinates, immutable=True)
    coords_signature = expected_input_signature(coords_path)
    query_signature = expected_input_signature(query_path)

    from basemap.panel_v2 import score_panel

    treatment_panel = score_panel(
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
            "coordinates": coords_signature,
            "shared_reference_receipt": shared_signature,
        },
    )
    projection = _projection_metrics(
        coordinates=coordinates, query_coordinates=query_coordinates, truth=truth
    )
    if (
        treatment_panel.get("guards", {}).get("coords_finite") is not True
        or treatment_panel.get("guards", {}).get("coords_collapsed") is not False
    ):
        raise Round0138Error("R0138 treatment panel guards failed")

    cells: dict[str, Any] = {}
    for key, source_key in (
        (HISTORICAL, HISTORICAL_SEED42),
        (CONTROL, CURRENT_R0104_SEED42),
    ):
        cell = source_cells.get(source_key)
        if not isinstance(cell, Mapping):
            raise Round0138Error(f"R0134 control cell missing: {source_key}")
        _signature(cell["coordinates"], label=f"{key} coordinates")
        _signature(cell["query_coordinates"], label=f"{key} query coordinates")
        cells[key] = dict(cell)
    cells[TREATMENT] = {
        "seed": SEED,
        "role": "device-sampler-runtime-treatment",
        "training": {
            "train": train_signature,
            "model": train["model"],
            "production_config_sha256": config_sha,
            "graph": train["graph"],
            "graph_manifest": train["graph_manifest"],
            "actual_pipeline": train["exact_execution_receipt"],
            "train_accounting": train["train_accounting"],
        },
        "coordinates": coords_signature,
        "query_coordinates": query_signature,
        "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
        "query_coordinates_ordered_sha256": ordered_array_sha256(query_coordinates),
        "panel": treatment_panel,
        "projection": projection,
    }
    sample = np.load(
        _signature(r0134["render"]["sample"], label="R0134 render sample")[
            "canonical_path"
        ], allow_pickle=False,
    )
    render = _render(
        output=output,
        cells=cells,
        labels=np.asarray(reference["labels"][256], dtype=np.int32),
        sample=np.asarray(sample, dtype=np.int64),
    )
    receipt = seal({
        "schema": PANEL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "r0134_panel": dict(job["r0134_panel"]),
        "same_functional_universe_and_truth": True,
        "treatment": {
            "only_changed_factor": (
                "R0104 host endpoint sampler/runtime to generic device-resident "
                "DeviceEdgeSampler runtime"
            ),
            "same_rows_representation_graph_model_optimizer_seed_and_updates": True,
            "same_positive_and_negative_distributions": True,
            "rng_implementation_and_draw_stream_differ": True,
        },
        "cells": cells,
        "render": render,
        "training_performed": True,
        "optimizer_updates": int(train["train_accounting"]["optimizer_steps_succeeded"]),
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "functional-sampler-bridge.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del model, coordinates, query_coordinates
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0138 sampler-bridge decision"
    )
    panel_path = os.path.join(
        str(job["panel_output"]), "functional-sampler-bridge.json"
    )
    panel = _read_sealed(expected_input_signature(panel_path), label="R0138 panel")
    if panel.get("schema") != PANEL_SCHEMA or panel.get("round_id") != ROUND_ID:
        raise Round0138Error("R0138 panel identity changed")
    decision = build_decision(panel["cells"])
    receipt = seal({
        **decision,
        "release_sha": active["manifest"]["release_sha"],
        "panel": expected_input_signature(panel_path),
        "capability": CAPABILITY,
    })
    path = os.path.join(output, "decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> Any:
    if job is None or (active.get("manifest") or {}).get("round_id") != ROUND_ID:
        raise Round0138Error("R0138 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "train_device_sampler":
        return run_train(active, job)
    if action == "score_device_sampler":
        return run_panel(active, job)
    if action == "decide_device_sampler":
        return run_decision(active, job)
    raise Round0138Error(f"unknown R0138 action: {action}")
