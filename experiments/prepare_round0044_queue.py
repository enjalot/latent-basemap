#!/usr/bin/env python3
"""Materialize, but never launch, the Round 0044 candidate-quality queue."""
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
from experiments.round0044_nodes import (
    INDEX_PATH,
    R0031_MEASUREMENT,
    ROUND_ID,
)
from experiments.round0029_program import ordered_embedding_paths


ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0044-2026-07-25.md",
)
ROUND_ROOT = "/data/latent-basemap/runs/round-0044"


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        head = handle.read(4096)
    statuses = re.findall(r"(?m)^status:\s*([^\s]+)\s*$", head)
    if statuses != ["issued"]:
        raise RuntimeError(
            f"R0044 requires one issued status; observed {statuses}"
        )


def prepare_round0044(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0044 release SHA must be one full commit")
    _require_issued_round()
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0044 queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    inputs = _dedupe(_file_inputs([
        ROUND_FILE,
        os.path.join(LAB_ROOT, "review-0013-2026-07-18-02.md"),
        os.path.join(LAB_ROOT, "review-0031-2026-07-21.md"),
        R0031_MEASUREMENT,
        INDEX_PATH,
        *ordered_embedding_paths()[:3],
    ]))
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.75,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0044-candidate-quality-queue-v1"
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0013", "0031"]
    manifest["capability_dependencies"] = [
        "30m-input-pack-v1",
        "path-b-3m-candidate-coverage-v1",
    ]
    manifest["capabilities_produced"] = [
        "path-b-3m-candidate-quality-sweep-v1"
    ]
    manifest["training_performed"] = False
    output = os.path.join(artifacts, "candidate-quality")
    manifest["jobs"] = [{
        "id": "candidate_quality_sweep_3m",
        "action": "candidate_quality_sweep",
        "handler_module": "experiments.round0044_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "candidate_quality_sweep_3m.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 1_800.0,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        "candidate_quality_sweep_3m": 1_800.0,
        "total": 1_800.0,
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
        "queue_manifest": prepare_round0044(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
