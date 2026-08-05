#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0195 v0 proposal queue."""
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
from basemap.round0195_release_proposal import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0195"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0195-2026-08-05.md")
SOURCES = {
    "family": "/data/latent-basemap/runs/round-0160/queue/artifacts/jina-fineweb-2m-prompted-seed42-45-family-v1/prompted-seed-family.json",
    "gates": "/data/latent-basemap/runs/round-0161/queue/artifacts/jina-prompted-universe-quality-gates-v1/prompted-quality-gates.json",
    "universality": "/data/latent-basemap/runs/round-0182/queue/artifacts/jina-prompted-raw-universality-readout-v1/packet.json",
    "methods": "/data/latent-basemap/runs/round-0183/queue/artifacts/jina-heldout-projection-method-table-v1/table.json",
    "scale": "/data/latent-basemap/runs/round-0190/queue/artifacts/jina-composition-boundary-three-seed-synthesis-v1/three-seed-boundary-synthesis.json",
}
REVIEWS = {
    "0160": os.path.join(LAB_ROOT, "review-0160-2026-08-03.md"),
    "0161": os.path.join(LAB_ROOT, "review-0161-2026-08-03.md"),
    "0182": os.path.join(LAB_ROOT, "review-0182-2026-08-03-01.md"),
    "0183": os.path.join(LAB_ROOT, "review-0183-2026-08-03.md"),
    "0190": os.path.join(LAB_ROOT, "review-0190-2026-08-05.md"),
}


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0195 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _review_lineage(round_id: str, review_path: str) -> list[dict[str, Any]]:
    review = _frontmatter(review_path)
    if review.get("round_id") != round_id or review.get("status") != "accepted":
        raise RuntimeError(f"R{round_id} review is not accepted")
    round_path = os.path.join(LAB_ROOT, str(review.get("round") or ""))
    result_path = os.path.join(LAB_ROOT, str(review.get("result") or ""))
    round_signature = expected_input_signature(round_path)
    result_signature = expected_input_signature(result_path)
    if (
        review.get("round_sha256") != round_signature["sha256"]
        or review.get("result_sha256") != result_signature["sha256"]
    ):
        raise RuntimeError(f"R{round_id} review binding changed")
    queue_path = f"/data/latent-basemap/runs/round-{round_id}/queue/queue.json"
    terminal_path = f"/data/latent-basemap/runs/round-{round_id}/queue/runner-terminal.json"
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    queue_signature = expected_input_signature(queue_path)
    if (
        terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
    ):
        raise RuntimeError(f"R{round_id} terminal provenance changed")
    return [
        round_signature,
        result_signature,
        expected_input_signature(review_path),
        queue_signature,
        expected_input_signature(terminal_path),
    ]


def prepare_round0195(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0195 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = [
        signature
        for round_id, path in REVIEWS.items()
        for signature in _review_lineage(round_id, path)
    ]
    sources = {name: expected_input_signature(path) for name, path in SOURCES.items()}
    expected_inputs = _dedupe([round_signature, *lineage, *sources.values()])
    queue_root = create_fresh_directory(queue_root, label="R0195 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    review_signatures = [expected_input_signature(path) for path in REVIEWS.values()]
    job = {
        "id": "assemble_release_proposal",
        "action": "assemble_release_proposal",
        "handler_module": "experiments.round0195_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "release-proposal.done.json"),
        "expected_inputs": expected_inputs,
        **sources,
        "accepted_reviews": review_signatures,
        "p90_wall_s": 30.0,
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": False},
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
        "schema": "round0195-fineweb-2m-v0-release-proposal-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": list(REVIEWS),
        "capability_dependencies": [
            "jina-fineweb-2m-prompted-seed42-45-family-v1",
            "jina-prompted-universe-quality-gates-v1",
            "jina-prompted-raw-universality-readout-v1",
            "jina-heldout-projection-method-table-v1",
            "jina-composition-boundary-three-seed-synthesis-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "is the accepted FineWeb-English prompted 2M family ready for an owner registry decision?",
            "candidate_id": "basemap-jina-v5-nano-en-2m-v0",
            "canonical_seed": 42,
            "proposal_only": True,
            "registry_mutation": False,
            "production_or_publishing": False,
            "method_context_not_candidate_specific": True,
            "ood_caveats_required": True,
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
    print(json.dumps({"queue_manifest": prepare_round0195(release_sha=args.release_sha, queue_root=args.queue_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
