#!/usr/bin/env python3
"""Materialize, but never launch, the fixed-row R0140 bisection queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0027_program import (
    CENTROIDS,
    GRAPH_PATH,
    SOURCE_4M_PATH,
    TRAIN_PATH,
)
from basemap.round0037_program import (
    graph_manifest_for_dimension,
    train_config_for_cell,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0140_subsystem_bisection import (
    CAPABILITY,
    CURRENT_GRAPH_CURRENT_HOST,
    HISTORICAL_GRAPH_CURRENT_HOST,
    HISTORICAL_GRAPH_DEVICE_REPRO,
    NEW_CELLS,
    ROUND_ID,
    Round0140Error,
    build_decision,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _embedded_signatures,
    _frontmatter,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0140"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0140-*.md")
R0037_SHARED = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/"
    "shared-reference/receipt.json"
)
R0134_PANEL = (
    "/data/latent-basemap/runs/round-0134/queue-attempt-3-exact-views/artifacts/"
    "functional-showdown/functional-showdown.json"
)

REVIEW_CAPABILITIES = {
    "0037": "jina-mrl-seed42-screen-v1",
    "0104": "jina-full768-host-int8-training-validation-v1",
    "0115": "jina-fineweb-2m-prompt-map-contrast-v1",
    "0134": "jina-density-functional-showdown-v1",
    "0138": "jina-current-2m-device-sampler-bridge-v1",
}

GPU_HOURS_MINIMUM = 3.65
GPU_HOURS_EXPECTED = 4.25
GPU_HOURS_P90 = 5.25
GPU_HOURS_MAXIMUM = 6.75


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"R0140 requires exactly one issued round; found {len(candidates)}")
    if _frontmatter(candidates[0]).get("base_commit") != release_sha:
        raise RuntimeError("R0140 issued base_commit differs from release")
    return candidates[0], expected_input_signature(candidates[0])


def _write_historical_manifest(root: str) -> dict[str, Any]:
    manifest = graph_manifest_for_dimension(768)
    manifest["verified_by"] = "round0140-historical-graph-fixed-row-adapter-v1"
    truth = manifest["graph_construction_truth"]
    truth.pop("shared_across_registered_cells", None)
    truth.pop("shared_across_all_six_cells", None)
    truth.update({
        "shared_across_r0140_cells": [
            HISTORICAL_GRAPH_CURRENT_HOST,
            HISTORICAL_GRAPH_DEVICE_REPRO,
        ],
        "graph_bytes_changed_by_adapter": False,
    })
    path = os.path.join(root, "historical-graph-manifest.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return expected_input_signature(path)


def _smoke_cpu(*, r0134: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise model reload/transform, panel-shaped sealing and selector on CPU."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("R0140 smoke must run with CUDA hidden")
    started = time.monotonic()
    historical = r0134["cells"]["historical_r0037_seed42"]
    model_signature = historical["training"]["model"]
    if expected_input_signature(model_signature["canonical_path"]) != model_signature:
        raise RuntimeError("R0140 smoke historical model bytes changed")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device="cpu")
    source = np.load(TRAIN_PATH, mmap_mode="r", allow_pickle=False)
    probe = np.asarray(source[:32], dtype=np.float32)
    transformed = np.asarray(model.transform(probe, batch_size=16), dtype=np.float32)
    if transformed.shape != (32, 2) or not np.isfinite(transformed).all():
        raise RuntimeError("R0140 CPU smoke transform failed")
    prototype = {
        "panel": {
            "ffr": 0.57,
            "purity": {"k256": 1.01, "k1024": 0.95},
            "density": 0.2,
        },
        "projection": {"ffr": 0.53, "recall_at_10": 0.011},
    }
    decision = build_decision({key: prototype for key in NEW_CELLS})
    sealed = seal({
        "schema": "round0140-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "model": model_signature,
        "probe_rows": 32,
        "transform_finite": True,
        "decision_outcome": decision["outcome"],
    })
    validate_seal(sealed, label="R0140 CPU smoke")
    return {
        **sealed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wall_seconds": time.monotonic() - started,
        "scope": "post-fit model reload -> transform -> sealed panel shape -> selector",
    }


def prepare_round0140(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0140 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    review_inputs: list[dict[str, Any]] = []
    for round_id, capability in REVIEW_CAPABILITIES.items():
        review_inputs.extend(_accepted_review(round_id, capability))

    r0134_signature = expected_input_signature(R0134_PANEL)
    r0134 = _read_json(R0134_PANEL)
    validate_seal(r0134, label="R0134 functional panel")
    if (
        r0134.get("round_id") != "0134"
        or r0134.get("source") != expected_input_signature(TRAIN_PATH)
        or not isinstance(r0134.get("cells"), Mapping)
    ):
        raise RuntimeError("R0134 functional context changed")
    shared_signature = expected_input_signature(R0037_SHARED)
    shared = _read_json(R0037_SHARED)
    validate_seal(shared, label="R0037 shared reference")
    for key in ("high_d_reference", "query_truth", "query_embeddings"):
        if expected_input_signature(shared[key]["canonical_path"]) != shared[key]:
            raise RuntimeError(f"R0037 shared {key} changed")

    smoke = _smoke_cpu(r0134=r0134)
    queue_root = create_fresh_directory(queue_root, label="R0140 bisection queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    historical_manifest = _write_historical_manifest(preflight)
    historical_graph = expected_input_signature(GRAPH_PATH)
    historical_config, historical_config_sha = train_config_for_cell(
        "d768_s42",
        graph_manifest_path=historical_manifest["canonical_path"],
        graph_manifest_sha256=historical_manifest["sha256"],
    )
    smoke_path = os.path.join(preflight, "cpu-smoke.json")
    atomic_write_new_json(smoke_path, smoke, immutable=True)

    expected_inputs = _dedupe([
        round_signature,
        *review_inputs,
        r0134_signature,
        shared_signature,
        expected_input_signature(TRAIN_PATH),
        expected_input_signature(SOURCE_4M_PATH),
        historical_graph,
        historical_manifest,
        expected_input_signature(smoke_path),
        *[expected_input_signature(item["path"]) for item in CENTROIDS.values()],
        *[dict(shared[key]) for key in ("high_d_reference", "query_truth", "query_embeddings")],
        *_embedded_signatures(r0134),
    ])

    current_graph_output = os.path.join(artifacts, "current-graph-fixed-rows")
    train_outputs = {
        cell: os.path.join(artifacts, cell, "train") for cell in NEW_CELLS
    }
    panel_output = os.path.join(artifacts, "functional-panel")
    decision_output = os.path.join(artifacts, "decision")

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_path,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest.update({
        "schema": "round0140-fixed-row-subsystem-bisection-queue-v1",
        "repo_root": RELEASE_ROOT,
        "required_reviews": list(REVIEW_CAPABILITIES),
        "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "scientific_contract": {
            "design": "three-cell subsystem bisection on one exact R0037 row universe",
            "cells": list(NEW_CELLS),
            "row_universe_fixed_across_new_cells": True,
            "source": expected_input_signature(TRAIN_PATH),
            "historical_graph": historical_graph,
            "successful_updates_per_cell": 500_000,
            "seed": 42,
            "current_cross_round_controls_are_context_not_paired_cells": True,
            "density_diagnostic_only": True,
            "cpu_smoke": expected_input_signature(smoke_path),
            "pinned_r0037_release_not_executed": True,
            "historical_reproduction_scope": (
                "exact registered R0037 recipe on current release; recipe-level, "
                "not release-level reproduction"
            ),
        },
    })
    common_panel = {
        "source": expected_input_signature(TRAIN_PATH),
        "shared_reference_receipt": shared_signature,
        "high_d_reference": dict(shared["high_d_reference"]),
        "query_truth": dict(shared["query_truth"]),
        "query_embeddings": dict(shared["query_embeddings"]),
        "centroids": {
            str(k): expected_input_signature(value["path"])
            for k, value in CENTROIDS.items()
        },
        "r0134_panel": r0134_signature,
    }
    jobs: list[dict[str, Any]] = [{
        "id": "build_current_graph_fixed_rows",
        "action": "build_current_graph",
        "handler_module": "experiments.round0140_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [current_graph_output],
        "done_marker": os.path.join(artifacts, "build-current-graph.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 600.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": f"train_{CURRENT_GRAPH_CURRENT_HOST}",
        "action": "train_host",
        "cell": CURRENT_GRAPH_CURRENT_HOST,
        "graph_kind": "current-fixed-row",
        "graph_output": current_graph_output,
        "handler_module": "experiments.round0140_nodes",
        "handler_callable": "run_job",
        "deps": ["build_current_graph_fixed_rows"],
        "outputs": [train_outputs[CURRENT_GRAPH_CURRENT_HOST]],
        "done_marker": os.path.join(artifacts, "train-current-current.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 5_400.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": f"train_{HISTORICAL_GRAPH_CURRENT_HOST}",
        "action": "train_host",
        "cell": HISTORICAL_GRAPH_CURRENT_HOST,
        "graph_kind": "historical-byte-exact",
        "graph": historical_graph,
        "graph_manifest": historical_manifest,
        "handler_module": "experiments.round0140_nodes",
        "handler_callable": "run_job",
        "deps": [f"train_{CURRENT_GRAPH_CURRENT_HOST}"],
        "outputs": [train_outputs[HISTORICAL_GRAPH_CURRENT_HOST]],
        "done_marker": os.path.join(artifacts, "train-historical-current.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 5_400.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": "historical_device_canary",
        "action": "historical_device_canary",
        "cell": "d768_s42",
        "graph_manifest_path": historical_manifest["canonical_path"],
        "graph_manifest_sha256": historical_manifest["sha256"],
        "production_config": historical_config,
        "production_config_sha256": historical_config_sha,
        "minimum_updates_per_s": 75.0,
        "handler_module": "experiments.round0140_nodes",
        "handler_callable": "run_job",
        "deps": [f"train_{HISTORICAL_GRAPH_CURRENT_HOST}"],
        "outputs": [os.path.join(artifacts, "historical-device-canary")],
        "done_marker": os.path.join(artifacts, "historical-device-canary.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 300.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": f"train_{HISTORICAL_GRAPH_DEVICE_REPRO}",
        "action": "historical_device_train",
        "cell": "d768_s42",
        "graph_manifest_path": historical_manifest["canonical_path"],
        "graph_manifest_sha256": historical_manifest["sha256"],
        "production_config": historical_config,
        "production_config_sha256": historical_config_sha,
        "canary_output": os.path.join(artifacts, "historical-device-canary"),
        "handler_module": "experiments.round0140_nodes",
        "handler_callable": "run_job",
        "deps": ["historical_device_canary"],
        "outputs": [train_outputs[HISTORICAL_GRAPH_DEVICE_REPRO]],
        "done_marker": os.path.join(artifacts, "train-historical-device.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 5_400.0,
        "node_policy": {"gpu_required": True, "training_performed": True},
    }, {
        "id": "score_functional_bisection",
        "action": "functional_panel",
        "handler_module": "experiments.round0140_nodes",
        "handler_callable": "run_job",
        "deps": [f"train_{cell}" for cell in NEW_CELLS],
        "train_outputs": train_outputs,
        **common_panel,
        "outputs": [panel_output],
        "done_marker": os.path.join(artifacts, "functional-panel.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 300.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }, {
        "id": "decide_subsystem_bisection",
        "action": "decide",
        "handler_module": "experiments.round0140_nodes",
        "handler_callable": "run_job",
        "deps": ["score_functional_bisection"],
        "panel_output": panel_output,
        "outputs": [decision_output],
        "done_marker": os.path.join(artifacts, "decision.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    }]
    manifest["jobs"] = jobs
    manifest["p90_gpu_seconds"] = {
        "current_graph_build": 600.0,
        "three_trains": 16_200.0,
        "historical_device_canary": 300.0,
        "functional_panel": 300.0,
        "total": 17_400.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({"queue_manifest": prepare_round0140(
        release_sha=args.release_sha, queue_root=args.queue_root
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
