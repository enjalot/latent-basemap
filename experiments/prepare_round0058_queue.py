#!/usr/bin/env python3
"""Prepare, but never launch, the balanced-60M nprobe calibration."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0049_program import INDEX_PATH, validate_substrate_manifest
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0058_nodes import NPROBES


ROUND_ID = "0058"
ROUND_ROOT = "/data/latent-basemap/runs/round-0058"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0058-2026-07-26.md",
)
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "balanced-60m-substrate/balanced-60m-substrate-v1.json"
)
BASELINE_QUALITY = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "candidate-quality-60m/balanced-60m-candidate-quality-v1.json"
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
            f"R0058 requires one issued status; observed {statuses}"
        )


def prepare_round0058(
    *,
    release_sha: str,
    substrate_manifest_sha256: str,
    baseline_quality_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    substrate = validate_substrate_manifest(
        SUBSTRATE_MANIFEST,
        expected_sha256=substrate_manifest_sha256,
    )
    outputs = substrate["manifest"]["outputs"]
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0058 queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(artifacts, "nprobe-sweep-60m")
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SUBSTRATE_MANIFEST,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            INDEX_PATH,
            BASELINE_QUALITY,
            os.path.join(LAB_ROOT, "review-0049-2026-07-26.md"),
        ]),
    ])
    by_path = {
        item["canonical_path"]: item
        for item in inputs
    }
    if (
        by_path[os.path.realpath(BASELINE_QUALITY)]["sha256"]
        != baseline_quality_sha256
    ):
        raise RuntimeError("R0049 quality receipt changed")

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.25,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0058-balanced-60m-nprobe-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0049"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-60m-input-v1",
        "minilm-balanced-60m-candidate-quality-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-60m-nprobe-calibration-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "purpose": (
            "find the smallest measured nprobe that preserves the frozen "
            "balanced-60M exact-reranked mean recall floor"
        ),
        "nprobe_grid": list(NPROBES),
        "search_width": 128,
        "selected_neighbors": 15,
        "sample_rows": 1_024,
        "sample_identity": "exact R0049 seed-49 sample",
        "mean_recall_at_15_unambiguous_floor": 0.90,
        "selection_rule": (
            "smallest registered passing nprobe; retain 64 if none lower passes"
        ),
        "no_training": True,
        "may_correct_r0050_resource_policy_only_after_review": True,
    }
    manifest["jobs"] = [{
        "id": "sweep_balanced_60m_nprobe",
        "action": "sweep_nprobe",
        "handler_module": "experiments.round0058_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "sweep_balanced_60m_nprobe.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 1_200.0,
        "substrate_manifest": SUBSTRATE_MANIFEST,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "baseline_quality_receipt": BASELINE_QUALITY,
        "baseline_quality_receipt_sha256": baseline_quality_sha256,
        "nprobes": list(NPROBES),
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        "sweep_balanced_60m_nprobe": 1_200.0,
        "total": 1_200.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--baseline-quality-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0058(
            release_sha=args.release_sha,
            substrate_manifest_sha256=(
                args.substrate_manifest_sha256
            ),
            baseline_quality_sha256=args.baseline_quality_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
