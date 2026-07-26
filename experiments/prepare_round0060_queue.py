#!/usr/bin/env python3
"""Prepare, but never launch, the matched balanced-30M GPU graph queue."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0049_program import INDEX_PATH
from basemap.round0053_program import (
    EXPECTED_RETAINED_ROWS,
    validate_control_substrate,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0054_nodes import _validate_quality
from experiments.round0059_nodes import (
    FAISS_WHEEL,
    RECEIPT_SCHEMA as R0059_RECEIPT_SCHEMA,
    _load_sealed_json,
)
from experiments.round0060_nodes import (
    MAX_PROJECTED_GRAPH_HOURS,
    NPROBE,
    RUNTIME_SPEC,
)


ROUND_ID = "0060"
ROUND_ROOT = "/data/latent-basemap/runs/round-0060"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0060"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0060-2026-07-26.md",
)
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
QUALITY_RECEIPT = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "candidate-quality-30m/balanced-30m-candidate-quality-v1.json"
)
GPU_QUALIFICATION_RECEIPT = (
    "/data/latent-basemap/runs/round-0059/queue/artifacts/"
    "gpu-ivfpq-qualification/gpu-ivfpq-qualification-v1.json"
)


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        lines = handle.readlines()
    statuses: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "status":
            statuses.append(value.strip().strip("\"'"))
    if statuses != ["issued"]:
        raise RuntimeError(
            f"R0060 requires one issued status; observed {statuses}"
        )


def prepare_round0060(
    *,
    release_sha: str,
    substrate_manifest_sha256: str,
    quality_receipt_sha256: str,
    gpu_qualification_receipt_sha256: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    substrate = validate_control_substrate(
        SUBSTRATE_MANIFEST,
        expected_sha256=substrate_manifest_sha256,
    )
    quality, quality_signature = _validate_quality(
        QUALITY_RECEIPT,
        expected_sha256=quality_receipt_sha256,
        nprobe=NPROBE,
    )
    gpu_qualification, gpu_qualification_signature = (
        _load_sealed_json(
            GPU_QUALIFICATION_RECEIPT,
            expected_sha256=gpu_qualification_receipt_sha256,
            schema=R0059_RECEIPT_SCHEMA,
        )
    )
    if gpu_qualification.get("validity_passed") is not True:
        raise RuntimeError("R0059 GPU qualification did not pass")
    runtime_signature = expected_input_signature(RUNTIME_SPEC)
    if runtime_signature["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0060 runtime specification changed")
    outputs = substrate["manifest"]["outputs"]
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0060 GPU queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    qualification_output = os.path.join(
        artifacts,
        "gpu-index-balanced-30m",
    )
    graph_output = os.path.join(
        artifacts,
        "native-graph-balanced-30m",
    )
    qualification_receipt = os.path.join(
        qualification_output,
        "gpu-index-qualification-v1.json",
    )
    filtered_index = os.path.join(
        qualification_output,
        "balanced-30m-retained.ivfpq",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SUBSTRATE_MANIFEST,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            QUALITY_RECEIPT,
            GPU_QUALIFICATION_RECEIPT,
            INDEX_PATH,
            RUNTIME_SPEC,
            FAISS_WHEEL,
            os.path.join(LAB_ROOT, "review-0053-2026-07-26.md"),
            os.path.join(LAB_ROOT, "review-0059-2026-07-26.md"),
        ]),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=2.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0060-balanced-30m-gpu-graph-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0053", "0059"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-int8-input-v1",
        "minilm-balanced-30m-candidate-quality-v1",
        "minilm-balanced-60m-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-30m-gpu-native-graph-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "rows": 30_000_000,
        "retained_rows": EXPECTED_RETAINED_ROWS,
        "k": 15,
        "substrate": substrate["signature"],
        "quality_validation": quality_signature,
        "gpu_search_qualification": gpu_qualification_signature,
        "quality_mean_recall_at_15_unambiguous": (
            quality["recall"]["mean_recall_at_15_unambiguous"]
        ),
        "nprobe": NPROBE,
        "search_width": 128,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "shard_rows": 100_000,
        "resumable_shards": True,
        "maximum_projected_graph_hours": MAX_PROJECTED_GRAPH_HOURS,
        "purpose": (
            "representation- and graph-policy-matched 30M control "
            "before the balanced 60M training rung"
        ),
        "no_training": True,
    }
    manifest["jobs"] = [
        {
            "id": "qualify_gpu_index_balanced_30m",
            "action": "qualify_gpu_index",
            "handler_module": "experiments.round0060_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [qualification_output],
            "done_marker": os.path.join(
                artifacts,
                "qualify_gpu_index_balanced_30m.done.json",
            ),
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "substrate_manifest": SUBSTRATE_MANIFEST,
            "substrate_manifest_sha256": substrate_manifest_sha256,
            "quality_receipt": QUALITY_RECEIPT,
            "quality_receipt_sha256": quality_receipt_sha256,
            "gpu_qualification_receipt": GPU_QUALIFICATION_RECEIPT,
            "gpu_qualification_receipt_sha256": (
                gpu_qualification_receipt_sha256
            ),
            "runtime_spec": RUNTIME_SPEC,
            "runtime_spec_sha256": runtime_spec_sha256,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            "id": "build_gpu_native_graph_balanced_30m",
            "action": "build_gpu_graph",
            "handler_module": "experiments.round0060_nodes",
            "handler_callable": "run_job",
            "deps": ["qualify_gpu_index_balanced_30m"],
            "outputs": [graph_output],
            "done_marker": os.path.join(
                artifacts,
                "build_gpu_native_graph_balanced_30m.done.json",
            ),
            "expected_inputs": inputs,
            "p90_wall_s": 5_400.0,
            "substrate_manifest": SUBSTRATE_MANIFEST,
            "substrate_manifest_sha256": substrate_manifest_sha256,
            "quality_receipt": QUALITY_RECEIPT,
            "quality_receipt_sha256": quality_receipt_sha256,
            "qualification_receipt": qualification_receipt,
            "filtered_index": filtered_index,
            "runtime_spec": RUNTIME_SPEC,
            "runtime_spec_sha256": runtime_spec_sha256,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
    ]
    manifest["p90_gpu_seconds"] = {
        "qualify_gpu_index_balanced_30m": 300.0,
        "build_gpu_native_graph_balanced_30m": 5_400.0,
        "total": 5_700.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--quality-receipt-sha256", required=True)
    parser.add_argument(
        "--gpu-qualification-receipt-sha256",
        required=True,
    )
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0060(
            release_sha=args.release_sha,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            quality_receipt_sha256=args.quality_receipt_sha256,
            gpu_qualification_receipt_sha256=(
                args.gpu_qualification_receipt_sha256
            ),
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
