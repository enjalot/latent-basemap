#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only native balanced-60M graph build."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0049_program import (
    INDEX_PATH,
    K,
    ROW_COUNT,
    validate_substrate_manifest,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0049_nodes import (
    MEAN_RECALL_FLOOR,
    QUALITY_RECEIPT_SCHEMA,
    SEARCH_WIDTH,
)


ROUND_ID = "0050"
ROUND_ROOT = "/data/latent-basemap/runs/round-0050"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0050"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0050-2026-07-26.md",
)
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "balanced-60m-substrate/balanced-60m-substrate-v1.json"
)
QUALITY_RECEIPT = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "candidate-quality-60m/balanced-60m-candidate-quality-v1.json"
)
R0047_QUALITY_RECEIPT = (
    "/data/latent-basemap/runs/round-0047/queue/artifacts/"
    "candidate-quality/candidate-quality-sweep-v1.json"
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
            f"R0050 requires one issued status; observed {statuses}"
        )


def _load_quality(
    path: str,
    *,
    expected_sha256: str,
    nprobe: int,
) -> tuple[dict, dict]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0049 candidate-quality receipt changed")
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value for key, value in receipt.items()
        if key != "identity_sha256"
    }
    if (
        receipt.get("schema") != QUALITY_RECEIPT_SCHEMA
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("validity_passed") is not True
        or int(
            receipt.get("candidate_generator", {}).get(
                "nprobe",
                -1,
            )
        )
        != nprobe
        or int(
            receipt.get("candidate_generator", {}).get(
                "search_width",
                -1,
            )
        )
        != SEARCH_WIDTH
        or receipt.get("candidate_generator", {}).get(
            "exact_rerank"
        )
        is not True
        or float(
            receipt.get("recall", {}).get(
                "mean_recall_at_15_unambiguous",
                -1,
            )
        )
        < MEAN_RECALL_FLOOR
    ):
        raise RuntimeError(
            "R0049 did not release this exact candidate policy"
        )
    return receipt, signature


def prepare_round0050(
    *,
    release_sha: str,
    substrate_manifest_sha256: str,
    quality_receipt_sha256: str,
    r0047_quality_receipt_sha256: str,
    nprobe: int,
    cpu_threads: int = 24,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if (
        nprobe <= 0
        or cpu_threads <= 0
        or cpu_threads > (os.cpu_count() or 1)
    ):
        raise ValueError("R0050 search resource geometry is invalid")
    substrate = validate_substrate_manifest(
        SUBSTRATE_MANIFEST,
        expected_sha256=substrate_manifest_sha256,
    )
    quality, quality_signature = _load_quality(
        QUALITY_RECEIPT,
        expected_sha256=quality_receipt_sha256,
        nprobe=nprobe,
    )
    r0047_signature = expected_input_signature(
        R0047_QUALITY_RECEIPT
    )
    if r0047_signature["sha256"] != r0047_quality_receipt_sha256:
        raise RuntimeError("R0047 candidate-quality receipt changed")
    index_signature = expected_input_signature(INDEX_PATH)
    outputs = substrate["manifest"]["outputs"]

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0050 CPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(artifacts, "native-graph-60m")
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SUBSTRATE_MANIFEST,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            QUALITY_RECEIPT,
            R0047_QUALITY_RECEIPT,
            INDEX_PATH,
            os.path.join(LAB_ROOT, "review-0047-2026-07-25.md"),
            os.path.join(LAB_ROOT, "review-0049-2026-07-26.md"),
        ]),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["schema"] = "round0050-balanced-60m-graph-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "cpu-research-parallel"
    manifest["required_reviews"] = ["0047", "0049"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-60m-input-v1",
        "minilm-balanced-60m-candidate-quality-v1",
        "path-b-balanced-3m-candidate-quality-v2",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-60m-native-graph-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "rows": ROW_COUNT,
        "k": K,
        "substrate": substrate["signature"],
        "quality_validation": quality_signature,
        "quality_mean_recall_at_15_unambiguous": (
            quality["recall"]["mean_recall_at_15_unambiguous"]
        ),
        "candidate_generator": {
            "index": index_signature,
            "r0047_quality_receipt": r0047_signature,
            "nprobe": nprobe,
            "search_width": SEARCH_WIDTH,
            "selected_neighbors": K,
            "native_representative_selector": True,
            "exact_rerank": True,
            "rerank_vector_source": (
                "balanced-subset int8-plus-fp16-scale exact cosine"
            ),
        },
        "shard_rows": 100_000,
        "resumable_shards": True,
        "cpu_threads": cpu_threads,
        "may_run_while_gpu_queue_is_active": True,
        "no_training": True,
    }
    manifest["jobs"] = [{
        "id": "build_native_representative_graph_60m",
        "action": "build_graph",
        "handler_module": "experiments.round0050_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "build_native_representative_graph_60m.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 39_600.0,
        "substrate_manifest": SUBSTRATE_MANIFEST,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "candidate_quality_receipt": R0047_QUALITY_RECEIPT,
        "candidate_quality_receipt_sha256": (
            r0047_quality_receipt_sha256
        ),
        "quality_validation_receipt": QUALITY_RECEIPT,
        "nprobe": nprobe,
        "cpu_threads": cpu_threads,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }]
    manifest["p90_cpu_seconds"] = {
        "build_native_representative_graph_60m": 39_600.0,
        "total": 39_600.0,
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
        "--r0047-quality-receipt-sha256",
        required=True,
    )
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0050(
            release_sha=args.release_sha,
            substrate_manifest_sha256=(
                args.substrate_manifest_sha256
            ),
            quality_receipt_sha256=args.quality_receipt_sha256,
            r0047_quality_receipt_sha256=(
                args.r0047_quality_receipt_sha256
            ),
            nprobe=args.nprobe,
            cpu_threads=args.cpu_threads,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
