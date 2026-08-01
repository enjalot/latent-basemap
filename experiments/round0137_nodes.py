"""Execute the conditional high-recall graph-construction bridge."""
from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0134_functional_showdown import (
    CURRENT_R0104_SEED42,
    HISTORICAL_SEED42,
)
from basemap.round0137_graph_bridge import (
    CAPABILITY,
    CELL_ORDER,
    CONTROL,
    DECISION_SCHEMA,
    HISTORICAL,
    PANEL_SCHEMA,
    ROUND_ID,
    TREATMENT,
    Round0137Error,
    build_decision,
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


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0137Error(f"{label} bytes changed")
    return actual


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0137Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value


def run_build(active: dict[str, Any], job: dict[str, Any]) -> None:
    if job.get("forced_nprobe") != 256 or job.get("shared_arms") != ["fp16_control"]:
        raise Round0137Error("R0137 high-recall graph treatment changed")
    r0104.run_build_shared(active, job)


def run_train(active: dict[str, Any], job: dict[str, Any]) -> None:
    if (
        job.get("arm") != "fp16_control"
        or job.get("shared_round_id") != ROUND_ID
        or job.get("shared_arms") != ["fp16_control"]
    ):
        raise Round0137Error("R0137 train treatment changed")
    r0104.run_train(active, job)


def _render(
    *, output: str, cells: Mapping[str, Mapping[str, Any]], labels: np.ndarray,
    sample: np.ndarray,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = create_fresh_directory(
        os.path.join(output, "renders"), label="R0137 graph-bridge render"
    )
    sample_path = os.path.join(root, "sample-row-ids.npy")
    atomic_save_new_npy(sample_path, sample, immutable=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    titles = {
        HISTORICAL: "historical R0037 s42",
        CONTROL: "current R0104 nprobe32 s42",
        TREATMENT: "treatment nprobe256 s42",
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
    path = os.path.join(root, "high-recall-graph-bridge.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    os.chmod(path, 0o444)
    receipt = seal({
        "schema": "round0137-high-recall-graph-render-v1",
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
        str(job["outputs"][0]), label="R0137 functional bridge panel"
    )
    started = time.monotonic()
    r0134 = _read_sealed(job["r0134_panel"], label="R0134 functional panel")
    source_cells = r0134.get("cells")
    if not isinstance(source_cells, Mapping):
        raise Round0137Error("R0134 functional cells are missing")
    source_signature, source, queries = _load_shared_evaluation_inputs(job)
    shared, shared_signature, reference, truth, centroids = _load_reference(job)

    model, train, train_signature, graph, config_sha = r0104._authenticate_model(job)
    # R0104 graph manifests predate the sealed-JSON convention. Authenticate
    # the immutable bytes before inspecting the registered search treatment.
    graph_manifest_signature = _signature(
        graph["graph_manifest"], label="R0137 high-recall graph manifest"
    )
    with open(graph_manifest_signature["canonical_path"], encoding="utf-8") as handle:
        graph_manifest = json.load(handle)
    if not isinstance(graph_manifest, dict):
        raise Round0137Error("R0137 high-recall graph manifest is malformed")
    search = ((graph_manifest.get("graph_construction_truth") or {}).get("search") or {})
    if (
        search.get("selected_nprobe") != 256
        or (search.get("quality_cells") or {}).get("256", {}).get("passed") is not True
        or train.get("round_id") != ROUND_ID
        or (train.get("exact_execution_receipt") or {}).get("pipeline")
        != "host_weighted_jina_paired"
    ):
        raise Round0137Error("R0137 graph/train treatment did not execute")
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
        raise Round0137Error("R0137 treatment transform is malformed")
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
        raise Round0137Error("R0137 treatment panel guards failed")

    cells: dict[str, Any] = {}
    for key, source_key in (
        (HISTORICAL, HISTORICAL_SEED42),
        (CONTROL, CURRENT_R0104_SEED42),
    ):
        cell = source_cells.get(source_key)
        if not isinstance(cell, Mapping):
            raise Round0137Error(f"R0134 control cell missing: {source_key}")
        _signature(cell["coordinates"], label=f"{key} coordinates")
        _signature(cell["query_coordinates"], label=f"{key} query coordinates")
        cells[key] = dict(cell)
    cells[TREATMENT] = {
        "seed": 42,
        "role": "graph-treatment",
        "training": {
            "train": train_signature,
            "model": train["model"],
            "production_config_sha256": config_sha,
            "graph": graph["graph"],
            "graph_manifest": graph["graph_manifest"],
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
            "only_changed_factor": "IVF-Flat graph nprobe 32 to 256",
            "control_selected_nprobe": 32,
            "treatment_selected_nprobe": 256,
            "same_rows_representation_k_fuzzy_kernel_sampler_seed_and_updates": True,
        },
        "cells": cells,
        "render": render,
        "training_performed": True,
        "optimizer_updates": int(train["train_accounting"]["optimizer_steps_succeeded"]),
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "functional-graph-bridge.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del model, coordinates, query_coordinates
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def run_decision(active: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0137 graph-bridge decision"
    )
    panel_path = os.path.join(str(job["panel_output"]), "functional-graph-bridge.json")
    panel = _read_sealed(expected_input_signature(panel_path), label="R0137 panel")
    if panel.get("schema") != PANEL_SCHEMA or panel.get("round_id") != ROUND_ID:
        raise Round0137Error("R0137 panel identity changed")
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
        raise Round0137Error("R0137 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "build_high_recall_graph":
        return run_build(active, job)
    if action == "train_high_recall_graph":
        return run_train(active, job)
    if action == "score_high_recall_graph":
        return run_panel(active, job)
    if action == "decide_high_recall_graph":
        return run_decision(active, job)
    raise Round0137Error(f"unknown R0137 action: {action}")
