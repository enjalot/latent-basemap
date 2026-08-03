#!/usr/bin/env python3
"""Prepare, but never launch, the light CPU-only R0182 synthesis queue."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0182_universality_packet import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0182"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0182-2026-08-03.md")
PROMPTED_PANEL = (
    "/data/latent-basemap/runs/round-0178/queue/artifacts/"
    "jina-prompted-universality-panel-v1/prompted-universality-panel.json"
)
RAW_PANEL = (
    "/data/latent-basemap/runs/round-0142/queue/artifacts/"
    "jina-diverse-universality-panel-v1/retention-table.json"
)
RAW_PREDICTORS = (
    "/data/latent-basemap/runs/round-0146/queue/artifacts/"
    "jina-diverse-projection-loss-predictors-v1/projection-loss-predictors.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0182 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def prepare_round0182(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0182 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = [
        *_accepted_review("0142", "jina-diverse-universality-panel-v1"),
        *_accepted_review("0146", "jina-diverse-projection-loss-predictors-v1"),
        *_accepted_review("0178", "jina-prompted-universality-panel-v1"),
    ]
    prompted = expected_input_signature(PROMPTED_PANEL)
    raw = expected_input_signature(RAW_PANEL)
    predictors = expected_input_signature(RAW_PREDICTORS)
    queue_root = create_fresh_directory(queue_root, label="R0182 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    expected_inputs = _dedupe([
        round_signature,
        *reviews,
        prompted,
        raw,
        predictors,
    ])
    job = {
        "id": "assemble_universality_readout",
        "action": "universality_packet",
        "handler_module": "experiments.round0182_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "prompted_panel": prompted,
        "raw_panel": raw,
        "raw_predictors": predictors,
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "assemble-universality-readout.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 60.0,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": False,
        },
    }
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0182-universality-readout-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-light",
        "required_reviews": ["0142", "0146", "0178"],
        "capability_dependencies": [
            "jina-diverse-universality-panel-v1",
            "jina-diverse-projection-loss-predictors-v1",
            "jina-prompted-universality-panel-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "what does the accepted prompted/raw OOD evidence say side by side?",
            "operation": "authenticated transcription and deterministic rendering only",
            "new_metric_computation": False,
            "quality_role": "diagnostic owner-facing packet",
            "map_registry_state_changed": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    path = prepare_round0182(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
