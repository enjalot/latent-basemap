#!/usr/bin/env python3
"""Prepare the slim queues for R0029 artifact qualification and R0031 Path B."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
    _materialized_chunk_inputs,
)
from experiments.round0029_program import (
    DUPLICATE_CAP,
    INPUT_PACK,
    R0029_BUILD_ROOT,
    R0029_V3,
    UNIFORM_GRAPH,
    artifact_path,
)


R0029_VALIDATION_ROOT = (
    "/data/latent-basemap/runs/round-0029/queue/artifacts/cpu-validation"
)
R0029_CANARY_ROOT = (
    "/data/latent-basemap/runs/round-0029/queue/artifacts/production-canary"
)
R0029_WORKDIR = "/data/latent-basemap/runs/round-0029/staging/weighted-graph-v2"
INDEX_3M = "/data/checkpoints/pumap/faiss_ivf_pq_3m.index"
R0032_V4_ROOT = (
    "/data/latent-basemap/runs/round-0032/queue/artifacts/v4-requalification"
)
R0032_CANARY_ROOT = (
    "/data/latent-basemap/runs/round-0032/queue/artifacts/production-canary"
)


def _round_file(round_id: str) -> str:
    return os.path.join(LAB_ROOT, f"round-{round_id}-2026-07-21.md")


def _publish(queue_root: str, manifest: dict[str, Any]) -> str:
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def prepare_round0029(release_sha: str) -> str:
    round_root = ensure_data_directory("/data/latent-basemap/runs/round-0029")
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="R0029 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    ensure_data_directory(os.path.join(round_root, "staging"))
    inputs = _dedupe([
        *_file_inputs([_round_file("0029"), INPUT_PACK, UNIFORM_GRAPH, DUPLICATE_CAP]),
        *_materialized_chunk_inputs(),
    ])
    manifest = _base_manifest(
        round_id="0029",
        release_sha=release_sha,
        round_file=_round_file("0029"),
        queue_root=queue_root,
        gpu_hours_cap=0.75,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0013", "0020"]
    manifest["capability_dependencies"] = [
        "30m-input-pack-v1", "global-exact-family-cap-v1"]
    manifest["capabilities_produced"] = ["30m-weighted-fuzzy-graph-v2"]
    manifest["scientific_contract"] = {
        "training_performed": False,
        "input_topology": expected_input_signature(UNIFORM_GRAPH),
        "duplicate_cap": expected_input_signature(DUPLICATE_CAP),
        "artifact_name": "edges_30m_k15_fuzzy-v2.npz",
        "sampler_canary": {
            "pipeline": "hybrid",
            "sampler_class": "HostStreamEdgeSampler",
            "positive_sampling": "weighted_with_replacement",
            "x_residency": "device_fp16",
            "multiplicity_policy": "exact_duplicate_cap_one",
        },
    }
    manifest["jobs"] = [
        {
            "id": "build_weighted_graph",
            "handler": "round0029_build",
            "deps": [],
            "done_marker": os.path.join(artifacts, "build_weighted_graph.done.json"),
            "outputs": [R0029_BUILD_ROOT],
            "workdir": R0029_WORKDIR,
            "expected_inputs": inputs,
            "p90_wall_s": 1_200.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            "id": "validate_weighted_graph",
            "handler": "round0029_validate",
            "deps": ["build_weighted_graph"],
            "done_marker": os.path.join(artifacts, "validate_weighted_graph.done.json"),
            "outputs": [R0029_VALIDATION_ROOT],
            "build_root": R0029_BUILD_ROOT,
            "expected_inputs": inputs,
            "p90_wall_s": 1_200.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            "id": "production_canary",
            "handler": "round0029_production_canary",
            "deps": ["validate_weighted_graph"],
            "done_marker": os.path.join(artifacts, "production_canary.done.json"),
            "outputs": [R0029_CANARY_ROOT],
            "build_root": R0029_BUILD_ROOT,
            "validation_root": R0029_VALIDATION_ROOT,
            "expected_inputs": inputs,
            "p90_wall_s": 900.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
    ]
    return _publish(queue_root, manifest)


def _first_three_embedding_inputs() -> list[dict[str, Any]]:
    with open(INPUT_PACK, encoding="utf-8") as handle:
        members = json.load(handle)["capability_payload"][
            "materialized_fp16"]["ordered_members"][:3]
    return [
        {
            "canonical_path": os.path.realpath(item["path"]),
            "kind": "file",
            "bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in members
    ]


def prepare_round0031(release_sha: str) -> str:
    round_root = ensure_data_directory("/data/latent-basemap/runs/round-0031")
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="R0031 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe([
        *_file_inputs([_round_file("0031"), INPUT_PACK, INDEX_3M]),
        *_first_three_embedding_inputs(),
    ])
    manifest = _base_manifest(
        round_id="0031",
        release_sha=release_sha,
        round_file=_round_file("0031"),
        queue_root=queue_root,
        gpu_hours_cap=0.75,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = []
    manifest["capability_dependencies"] = ["30m-input-pack-v1"]
    manifest["capabilities_produced"] = ["path-b-3m-candidate-coverage-v1"]
    manifest["scientific_contract"] = {
        "training_performed": False,
        "base_rows": 3_000_000,
        "sample_rows": 50_000,
        "k": 15,
        "nprobe": 64,
        "candidate_widths": [64, 128],
        "distance_compute_dtype": "float32",
        "self_exclusion": "explicit-query-row-id",
        "exact_duplicate_policy": "report-and-stratify-top-k-boundary-ties",
    }
    output = os.path.join(artifacts, "path-b")
    manifest["jobs"] = [{
        "id": "path_b_candidate_rerank",
        "handler": "round0031_path_b",
        "deps": [],
        "done_marker": os.path.join(artifacts, "path_b_candidate_rerank.done.json"),
        "outputs": [output],
        "expected_inputs": inputs,
        "p90_wall_s": 1_800.0,
        "node_policy": {"gpu_required": True, "training_performed": False},
    }]
    return _publish(queue_root, manifest)


def prepare_round0032(release_sha: str) -> str:
    round_root = ensure_data_directory("/data/latent-basemap/runs/round-0032")
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="R0032 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    graph = artifact_path(R0029_BUILD_ROOT)
    inputs = _dedupe([
        *_file_inputs([
            _round_file("0032"),
            os.path.join(LAB_ROOT, "review-0029-2026-07-21.md"),
            INPUT_PACK,
            UNIFORM_GRAPH,
            DUPLICATE_CAP,
            graph,
            graph + ".manifest.json",
            os.path.join(R0029_BUILD_ROOT, "build-receipt.json"),
            R0029_V3,
        ]),
        *_materialized_chunk_inputs(),
    ])
    manifest = _base_manifest(
        round_id="0032",
        release_sha=release_sha,
        round_file=_round_file("0032"),
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0013", "0020", "0029"]
    manifest["capability_dependencies"] = [
        "30m-input-pack-v1", "30m-duplicate-census-v1"]
    manifest["capabilities_produced"] = ["30m-weighted-fuzzy-graph-v2"]
    manifest["scientific_contract"] = {
        "training_performed": False,
        "graph": expected_input_signature(graph),
        "manifest": expected_input_signature(graph + ".manifest.json"),
        "r0029_v3": expected_input_signature(R0029_V3),
        "v4_compute_device": "cuda",
        "v4_sample_seed": 0,
        "v4_sample_nodes": 20,
        "v4_pairs_per_node": 3,
        "production_canary": {
            "pipeline": "hybrid",
            "sampler_class": "HostStreamEdgeSampler",
            "positive_sampling": "weighted_with_replacement",
            "x_residency": "device_fp16",
            "multiplicity_policy": "exact_duplicate_cap_one",
            "optimizer_updates": 0,
        },
    }
    manifest["jobs"] = [
        {
            "id": "v4_cuda_requalification",
            "handler": "round0032_v4",
            "deps": [],
            "done_marker": os.path.join(
                artifacts, "v4_cuda_requalification.done.json"),
            "outputs": [R0032_V4_ROOT],
            "build_root": R0029_BUILD_ROOT,
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            "id": "production_canary",
            "handler": "round0032_canary",
            "deps": ["v4_cuda_requalification"],
            "done_marker": os.path.join(artifacts, "production_canary.done.json"),
            "outputs": [R0032_CANARY_ROOT],
            "build_root": R0029_BUILD_ROOT,
            "v3_path": R0029_V3,
            "v4_root": R0032_V4_ROOT,
            "expected_inputs": inputs,
            "p90_wall_s": 900.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
    ]
    return _publish(queue_root, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_id", choices=("0029", "0031", "0032"))
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args()
    prepare = globals()[f"prepare_round{args.round_id}"]
    print(prepare(args.release_sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
