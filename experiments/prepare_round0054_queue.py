#!/usr/bin/env python3
"""Prepare the CPU-only matched balanced-30M native graph build."""
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
from experiments.round0054_nodes import (
    QUALITY_RECEIPT_SCHEMA,
    _validate_quality,
)


ROUND_ID = "0054"
ROUND_ROOT = "/data/latent-basemap/runs/round-0054"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0054"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0054-2026-07-26.md",
)
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
QUALITY_RECEIPT = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "candidate-quality-30m/balanced-30m-candidate-quality-v1.json"
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
            f"R0054 requires one issued status; observed {statuses}"
        )


def prepare_round0054(
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
        raise ValueError("R0054 search resource geometry is invalid")
    substrate = validate_control_substrate(
        SUBSTRATE_MANIFEST,
        expected_sha256=substrate_manifest_sha256,
    )
    quality, quality_signature = _validate_quality(
        QUALITY_RECEIPT,
        expected_sha256=quality_receipt_sha256,
        nprobe=nprobe,
    )
    if quality.get("schema") != QUALITY_RECEIPT_SCHEMA:
        raise RuntimeError("R0053 quality schema changed")
    r0047_signature = expected_input_signature(
        R0047_QUALITY_RECEIPT
    )
    if (
        r0047_signature["sha256"]
        != r0047_quality_receipt_sha256
    ):
        raise RuntimeError("R0047 candidate quality changed")
    outputs = substrate["manifest"]["outputs"]
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0054 CPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(
        artifacts,
        "native-graph-balanced-30m",
    )
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
            os.path.join(LAB_ROOT, "review-0053-2026-07-26.md"),
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
    manifest["schema"] = "round0054-balanced-30m-graph-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "cpu-research-parallel"
    manifest["required_reviews"] = ["0047", "0053"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-int8-input-v1",
        "minilm-balanced-30m-candidate-quality-v1",
        "path-b-balanced-3m-candidate-quality-v2",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-30m-native-graph-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "rows": 30_000_000,
        "retained_rows": EXPECTED_RETAINED_ROWS,
        "k": 15,
        "substrate": substrate["signature"],
        "quality_validation": quality_signature,
        "quality_mean_recall_at_15_unambiguous": (
            quality["recall"]["mean_recall_at_15_unambiguous"]
        ),
        "nprobe": nprobe,
        "search_width": 128,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "shard_rows": 100_000,
        "resumable_shards": True,
        "cpu_threads": cpu_threads,
        "may_run_while_gpu_queue_is_active": True,
        "purpose": (
            "representation- and graph-policy-matched 30M control for "
            "the balanced 60M rung"
        ),
        "no_training": True,
    }
    manifest["jobs"] = [{
        "id": "build_native_representative_graph_balanced_30m",
        "action": "build_graph",
        "handler_module": "experiments.round0054_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "build_native_representative_graph_balanced_30m.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 21_600.0,
        "substrate_manifest": SUBSTRATE_MANIFEST,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "quality_validation_receipt": QUALITY_RECEIPT,
        "quality_validation_receipt_sha256": quality_receipt_sha256,
        "candidate_quality_receipt": R0047_QUALITY_RECEIPT,
        "candidate_quality_receipt_sha256": (
            r0047_quality_receipt_sha256
        ),
        "nprobe": nprobe,
        "cpu_threads": cpu_threads,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }]
    manifest["p90_cpu_seconds"] = {
        "build_native_representative_graph_balanced_30m": 21_600.0,
        "total": 21_600.0,
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
        "queue_manifest": prepare_round0054(
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
