#!/usr/bin/env python3
"""Prepare, but never launch, the review-gated R0205 registry queue."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0205_v0_registry import CANDIDATE_ID, CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0191_queue import _embedded_signatures
from experiments.prepare_round0195_queue import _review_lineage


ROUND_ROOT = "/data/latent-basemap/runs/round-0205"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0205-2026-08-06.md")
R0204_REVIEW = os.path.join(LAB_ROOT, "review-0204-2026-08-06.md")
R0204_BUNDLE = (
    "/data/latent-basemap/runs/round-0204/queue/artifacts/"
    "basemap-jina-v5-nano-en-2m-v0-release-bundle-v1/release-bundle.json"
)
R0115_SCORE = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
    "document/evaluation/score.json"
)
R0115_TRAIN = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
    "document/train/train-receipt.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0205 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _accepted_inputs() -> list[dict[str, Any]]:
    signatures = [
        *_review_lineage("0204", R0204_REVIEW),
        expected_input_signature(R0204_BUNDLE),
        expected_input_signature(R0115_SCORE),
        expected_input_signature(R0115_TRAIN),
    ]
    for path in (R0204_BUNDLE, R0115_SCORE, R0115_TRAIN):
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        _embedded_signatures(value, signatures)
    return _dedupe(signatures)


def prepare_round0205(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0205 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    expected_inputs = _dedupe([round_signature, *_accepted_inputs()])
    queue_root = create_fresh_directory(queue_root, label="R0205 registry queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "register_v0_locally",
        "action": "register_v0_locally",
        "handler_module": "experiments.round0205_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "local-registry.done.json"),
        "expected_inputs": expected_inputs,
        "r0204_bundle": R0204_BUNDLE,
        "r0115_score": R0115_SCORE,
        "r0115_train_receipt": R0115_TRAIN,
        "p90_wall_s": 300.0,
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
        "schema": "round0205-v0-local-registry-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0204"],
        "capability_dependencies": [
            "basemap-jina-v5-nano-en-2m-v0-release-bundle-v1"
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "candidate_id": CANDIDATE_ID,
            "canonical_seed": 42,
            "copy_exact_coordinates": True,
            "register_local_map": True,
            "mint_immutable_registry_snapshot": True,
            "publish_local_registry_site": True,
            "intended_use": "exploratory FineWeb-English only",
            "production_readiness_claim": False,
            "named_ood_failures_required": "7 of 11",
            "universal_ood_claim": False,
            "huggingface_upload": False,
            "external_publication": False,
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
    print(json.dumps({
        "queue_manifest": prepare_round0205(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
