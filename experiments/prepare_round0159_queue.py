#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0159 seed-margin proposal."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0159_seed_margin_proposal import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0159"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0159-2026-08-02.md")
R0140_PANEL = (
    "/data/latent-basemap/runs/round-0140/queue-attempt-2/artifacts/"
    "functional-panel/functional-bisection.json"
)
R0149_PANEL = (
    "/data/latent-basemap/runs/round-0149/queue/artifacts/"
    "functional-drop-only-panel/functional-panel.json"
)
R0150_PANEL = (
    "/data/latent-basemap/runs/round-0150/queue-attempt-2/artifacts/"
    "paired-seed43-functional-panel/functional-panel.json"
)
R0153_DENSITY = (
    "/data/latent-basemap/runs/round-0153/queue/artifacts/"
    "jina-2m-track-a-density-forensics-v1/density-forensics.json"
)
R0154_EVIDENCE = (
    "/data/latent-basemap/runs/round-0154/queue/artifacts/"
    "jina-2m-raw-seed44-45-calibration-v1/seed-evidence.json"
)
R0158_EVIDENCE = (
    "/data/latent-basemap/runs/round-0158/queue/artifacts/"
    "jina-2m-drop-only-seed44-45-calibration-v1/seed-evidence.json"
)
STATIC_REVIEWS = (
    os.path.join(LAB_ROOT, "review-0140-2026-08-01-01.md"),
    os.path.join(LAB_ROOT, "review-0149-2026-08-02.md"),
    os.path.join(LAB_ROOT, "review-0150-2026-08-02.md"),
    os.path.join(LAB_ROOT, "review-0153-2026-08-02.md"),
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if frontmatter.get("status") != "issued":
        raise RuntimeError("R0159 round is not issued")
    if frontmatter.get("base_commit") != release_sha:
        raise RuntimeError("R0159 issued base_commit differs from release")
    return expected_input_signature(ROUND_FILE)


def _accepted_review(path: str, *, round_id: str) -> dict[str, Any]:
    frontmatter = _frontmatter(path)
    if frontmatter.get("status") != "accepted" or frontmatter.get("round_id") != round_id:
        raise RuntimeError(f"R0159 required Review {round_id} is not accepted")
    return expected_input_signature(path)


def _dynamic_review(round_id: str) -> dict[str, Any]:
    accepted = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md")))
        if _frontmatter(path).get("status") == "accepted"
    ]
    if len(accepted) != 1:
        raise RuntimeError(
            f"R0159 requires one accepted Review {round_id}; found {len(accepted)}"
        )
    return _accepted_review(accepted[0], round_id=round_id)


def prepare_round0159(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0159 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = [
        *[
            _accepted_review(path, round_id=round_id)
            for path, round_id in zip(
                STATIC_REVIEWS, ("0140", "0149", "0150", "0153"), strict=True
            )
        ],
        _dynamic_review("0154"),
        _dynamic_review("0158"),
    ]
    sources = {
        "r0140_panel": expected_input_signature(R0140_PANEL),
        "r0149_panel": expected_input_signature(R0149_PANEL),
        "r0150_panel": expected_input_signature(R0150_PANEL),
        "r0153_density": expected_input_signature(R0153_DENSITY),
        "r0154_evidence": expected_input_signature(R0154_EVIDENCE),
        "r0158_evidence": expected_input_signature(R0158_EVIDENCE),
    }
    expected_inputs = _dedupe([round_signature, *reviews, *sources.values()])
    queue_root = create_fresh_directory(queue_root, label="R0159 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "build_seed_margin_proposal",
        "action": "build_seed_margin_proposal",
        "handler_module": "experiments.round0159_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "seed-margin-proposal.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 60.0,
        "review_bindings": reviews,
        **sources,
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
        "schema": "round0159-seed-margin-proposal-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0140", "0149", "0150", "0153", "0154", "0158"],
        "capability_dependencies": [
            "jina-2m-track-a-density-forensics-v1",
            "jina-2m-raw-seed44-45-calibration-v1",
            "jina-2m-drop-only-seed44-45-calibration-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "what empirical seed variation should inform later functional and density margins?",
            "raw_seeds": [42, 43, 44, 45],
            "drop_only_seeds": [42, 43, 44, 45],
            "proposal_rule": "raw control mean minus two sample standard deviations",
            "paired_drop_minus_raw_reported": True,
            "small_n_warning_required": True,
            "owner_decision_required_for_adoption": True,
            "margin_or_floor_changed": False,
            "no_training": True,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0159(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

