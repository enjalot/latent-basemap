#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0204 v0 bundle queue."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0204_v0_release_bundle import CANDIDATE_ID, CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0191_queue import _embedded_signatures
from experiments.prepare_round0195_queue import _review_lineage


ROUND_ROOT = "/data/latent-basemap/runs/round-0204"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0204-2026-08-06.md")
R0195_REVIEW = os.path.join(LAB_ROOT, "review-0195-2026-08-05.md")
R0195_PROPOSAL = (
    "/data/latent-basemap/runs/round-0195/queue/artifacts/"
    "jina-fineweb-2m-v0-release-proposal-v1/release-proposal.json"
)
R0182_PACKET = (
    "/data/latent-basemap/runs/round-0182/queue/artifacts/"
    "jina-prompted-raw-universality-readout-v1/packet.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0204 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _accepted_inputs() -> list[dict[str, Any]]:
    signatures = [
        *_review_lineage("0195", R0195_REVIEW),
        expected_input_signature(R0195_PROPOSAL),
        expected_input_signature(R0182_PACKET),
    ]
    for path, label in (
        (R0195_PROPOSAL, "accepted R0195 proposal"),
        (R0182_PACKET, "accepted R0182 packet"),
    ):
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        _embedded_signatures(value, signatures)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{label} changed")
    return _dedupe(signatures)


def prepare_round0204(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0204 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    expected_inputs = _dedupe([round_signature, *_accepted_inputs()])
    queue_root = create_fresh_directory(queue_root, label="R0204 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "assemble_v0_release_bundle",
        "action": "assemble_v0_release_bundle",
        "handler_module": "experiments.round0204_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "v0-release-bundle.done.json"),
        "expected_inputs": expected_inputs,
        "r0195_proposal": R0195_PROPOSAL,
        "r0182_packet": R0182_PACKET,
        "p90_wall_s": 30.0,
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
        "schema": "round0204-v0-release-bundle-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0195"],
        "capability_dependencies": ["jina-fineweb-2m-v0-release-proposal-v1"],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "candidate_id": CANDIDATE_ID,
            "canonical_seed": 42,
            "exact_coordinates_and_train_receipt_required": True,
            "all_four_seed_gate_table_required": True,
            "seed42_ood_named_failures_stated_plainly": "7 of 11",
            "seed43_ood_named_failures_stated_plainly": "6 of 11",
            "aumap_role": "historical method context; no method-winner claim",
            "draft_huggingface_model_card_required": True,
            "registry_mutation": False,
            "huggingface_upload": False,
            "production_or_publishing": False,
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
        "queue_manifest": prepare_round0204(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
