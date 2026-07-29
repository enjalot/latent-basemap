#!/usr/bin/env python3
"""Materialize, but never launch, the corrected R0047 3M candidate sweep."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
)
from experiments.round0029_program import ordered_embedding_paths
from experiments.round0044_nodes import (
    CORRECTED_MEMBER_INDICES,
    CORRECTION_ROUND_ID,
    INDEX_PATH,
    R0031_MEASUREMENT,
)


ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0047-2026-07-25.md",
)
ROUND_ROOT = "/data/latent-basemap/runs/round-0047"


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        head = handle.read(4096)
    statuses = re.findall(r"(?m)^status:\s*([^\s]+)\s*$", head)
    if statuses != ["issued"]:
        raise RuntimeError(
            f"R0047 requires one issued status; observed {statuses}"
        )


def prepare_round0047(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0047 release SHA must be one full commit")
    _require_issued_round()
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0047 queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    paths = ordered_embedding_paths()
    balanced_members = [paths[index] for index in CORRECTED_MEMBER_INDICES]
    inputs = _dedupe(_file_inputs([
        ROUND_FILE,
        os.path.join(LAB_ROOT, "review-0013-2026-07-18-02.md"),
        os.path.join(LAB_ROOT, "result-0031-2026-07-21.md"),
        os.path.join(LAB_ROOT, "review-0031-2026-07-21.md"),
        os.path.join(LAB_ROOT, "result-0045-2026-07-25.md"),
        os.path.join(LAB_ROOT, "review-0045-2026-07-25.md"),
        R0031_MEASUREMENT,
        INDEX_PATH,
        *balanced_members,
    ]))
    manifest = _base_manifest(
        round_id=CORRECTION_ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0047-balanced-candidate-quality-queue-v1"
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0013"]
    manifest["capability_dependencies"] = ["30m-input-pack-v1"]
    manifest["capabilities_produced"] = [
        "3m-index-row-alignment-correction-v1",
        "path-b-balanced-3m-candidate-quality-v2",
    ]
    manifest["supersedes"] = ["0045"]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "compact_member_indices": list(CORRECTED_MEMBER_INDICES),
        "compact_row_layout": (
            "fineweb[0:1m]|redpajama[0:1m]|pile[0:1m]"
        ),
        "historical_layout": (
            "R0031/R0044/R0045 used member indices [0,1,2]"
        ),
        "historical_measurement_is_comparator_only": True,
        "recall_floor": 0.90,
        "training_performed": False,
    }
    output = os.path.join(artifacts, "candidate-quality")
    manifest["jobs"] = [{
        "id": "balanced_candidate_quality_sweep_3m",
        "action": "candidate_quality_sweep",
        "handler_module": "experiments.round0044_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "balanced_candidate_quality_sweep_3m.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 600.0,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        "balanced_candidate_quality_sweep_3m": 600.0,
        "total": 600.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0047(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
