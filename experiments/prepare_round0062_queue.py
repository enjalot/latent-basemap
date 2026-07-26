#!/usr/bin/env python3
"""Prepare, but never launch, the balanced-60M GPU graph queue."""
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
from basemap.round0049_program import (
    ROW_COUNT,
    validate_substrate_manifest,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0059_nodes import (
    FAISS_WHEEL,
    RECEIPT_SCHEMA as QUALIFICATION_SCHEMA,
    _load_sealed_json,
)
from experiments.round0062_nodes import (
    EXPECTED_RETAINED_ROWS,
    NPROBE,
    RUNTIME_SPEC,
    _validate_qualification,
)


ROUND_ID = "0062"
ROUND_ROOT = "/data/latent-basemap/runs/round-0062"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0062-2026-07-26.md",
)
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "balanced-60m-substrate/balanced-60m-substrate-v1.json"
)
GPU_QUALIFICATION_RECEIPT = (
    "/data/latent-basemap/runs/round-0059/queue/artifacts/"
    "gpu-ivfpq-qualification/gpu-ivfpq-qualification-v1.json"
)
FILTERED_INDEX = (
    "/data/latent-basemap/runs/round-0059/queue/artifacts/"
    "gpu-ivfpq-qualification/balanced-60m-retained.ivfpq"
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
            f"R0062 requires one issued status; observed {statuses}"
        )


def prepare_round0062(
    *,
    release_sha: str,
    substrate_manifest_sha256: str,
    gpu_qualification_receipt_sha256: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    substrate = validate_substrate_manifest(
        SUBSTRATE_MANIFEST,
        expected_sha256=substrate_manifest_sha256,
    )
    outputs = substrate["manifest"]["outputs"]
    qualification, qualification_signature = _load_sealed_json(
        GPU_QUALIFICATION_RECEIPT,
        expected_sha256=gpu_qualification_receipt_sha256,
        schema=QUALIFICATION_SCHEMA,
    )
    filtered_registered = _validate_qualification(
        qualification,
        substrate_signature=substrate["signature"],
        eligibility_signature=outputs["eligibility"],
    )
    filtered_signature = expected_input_signature(FILTERED_INDEX)
    if filtered_signature != filtered_registered:
        raise RuntimeError("R0059 filtered index changed")
    runtime_signature = expected_input_signature(RUNTIME_SPEC)
    if runtime_signature["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0062 runtime specification changed")

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0062 GPU queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    graph_output = os.path.join(
        artifacts,
        "native-graph-balanced-60m",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SUBSTRATE_MANIFEST,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            GPU_QUALIFICATION_RECEIPT,
            FILTERED_INDEX,
            RUNTIME_SPEC,
            FAISS_WHEEL,
            os.path.join(LAB_ROOT, "review-0049-2026-07-26.md"),
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
    manifest["schema"] = "round0062-balanced-60m-gpu-graph-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0049", "0059"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-60m-input-v1",
        "minilm-balanced-60m-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-60m-gpu-native-graph-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "purpose": (
            "materialize the frozen balanced-60M representative graph "
            "without the failed R0050 CPU path"
        ),
        "rows": ROW_COUNT,
        "retained_rows": EXPECTED_RETAINED_ROWS,
        "k": 15,
        "substrate": substrate["signature"],
        "gpu_search_qualification": qualification_signature,
        "filtered_index": filtered_signature,
        "nprobe": NPROBE,
        "search_width": 128,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "shard_rows": 100_000,
        "resumable_shards": True,
        "no_training": True,
    }
    manifest["jobs"] = [{
        "id": "build_gpu_native_graph_balanced_60m",
        "action": "build_gpu_graph",
        "handler_module": "experiments.round0062_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [graph_output],
        "done_marker": os.path.join(
            artifacts,
            "build_gpu_native_graph_balanced_60m.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 7_200.0,
        "substrate_manifest": SUBSTRATE_MANIFEST,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "gpu_qualification_receipt": GPU_QUALIFICATION_RECEIPT,
        "gpu_qualification_receipt_sha256": (
            gpu_qualification_receipt_sha256
        ),
        "filtered_index": FILTERED_INDEX,
        "runtime_spec": RUNTIME_SPEC,
        "runtime_spec_sha256": runtime_spec_sha256,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        "build_gpu_native_graph_balanced_60m": 7_200.0,
        "total": 7_200.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
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
        "queue_manifest": prepare_round0062(
            release_sha=args.release_sha,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
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
