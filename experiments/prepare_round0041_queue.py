#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only Round 0041 queue."""
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
from basemap.round0041_program import (
    CENSUS_PATH,
    GRAPH_PATH,
    R0021_TRAIN_RECEIPT,
    R0030_TRAIN_RECEIPT,
    SELECTOR_PATH,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0041"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0041-2026-07-24.md")


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        lines = handle.readlines()
    statuses = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "status":
            statuses.append(value.strip().strip("\"'"))
    if statuses != ["issued"]:
        raise RuntimeError(
            f"Round 0041 requires one status: issued; observed {statuses}"
        )


def prepare_round0041(release_sha: str) -> str:
    _require_issued_round()
    queue_root = create_fresh_directory(
        os.path.join(ensure_data_directory(ROUND_ROOT), "queue"),
        label="Round 0041 queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "canonical-graph-30m")
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            GRAPH_PATH,
            CENSUS_PATH,
            SELECTOR_PATH,
            R0021_TRAIN_RECEIPT,
            R0030_TRAIN_RECEIPT,
        ])
    ])
    manifest = _base_manifest(
        round_id="0041",
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["required_reviews"] = ["0020", "0021", "0030", "0040"]
    manifest["capability_dependencies"] = [
        "30m-duplicate-census-v1",
        "duplicate-controlled-panel-v1",
    ]
    manifest["capabilities_produced"] = [
        "30m-canonical-source-major-graph-v1",
        "30m-sampler-semantics-audit-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "row_count": 30_000_000,
        "input_k": 15,
        "source_law": "uniform-positive-source",
        "destination_policy": (
            "map exact copies to R0020 representative; drop zero/self/repeated "
            "canonical destinations"
        ),
        "full_invalid_scan": "R0040 selector proves zero=0 and nonfinite=0",
        "historical_receipts": ["R0021", "R0030"],
        "no_training": True,
    }
    manifest["jobs"] = [{
        "id": "build_and_audit_30m",
        "handler": "round0041_build_and_audit",
        "handler_module": "experiments.round0041_nodes",
        "handler_callable": "run_build_and_audit",
        "deps": [],
        "done_marker": os.path.join(
            artifacts, "build_and_audit_30m.done.json"
        ),
        "outputs": [output],
        "expected_inputs": inputs,
        "p90_wall_s": 1_800.0,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }]
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0041(args.release_sha)
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
