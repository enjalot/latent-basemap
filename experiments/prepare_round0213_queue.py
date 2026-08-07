#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0213 scaling-story synthesis."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0213_scaling_story_synthesis import (
    CAPABILITY,
    HIGH_DOSE,
    LOW_DOSE,
    OPERATING_RULE,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0191_queue import _embedded_signatures
from experiments.prepare_round0169_queue import _accepted_bundle


ROUND_ROOT = "/data/latent-basemap/runs/round-0213"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0213-2026-08-07.md")
R0190_SYNTHESIS = (
    "/data/latent-basemap/runs/round-0190/queue/artifacts/"
    "jina-composition-boundary-three-seed-synthesis-v1/"
    "three-seed-boundary-synthesis.json"
)
R0207_FACTORIAL = (
    "/data/latent-basemap/runs/round-0207/queue/artifacts/"
    "jina-width-by-n-factorial-capacity-economics-v1/width-factorial.json"
)
#: Every round the campaign asked this artifact to bind.
BOUND_ROUNDS = ("0187", "0189", "0190", "0191", "0202", "0203", "0207")


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0213 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def prepare_round0213(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0213 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    dependencies: list[dict[str, Any]] = []
    for round_id in BOUND_ROUNDS:
        dependencies.extend(_accepted_bundle(round_id))
    primaries = [
        expected_input_signature(R0190_SYNTHESIS),
        expected_input_signature(R0207_FACTORIAL),
    ]
    embedded: list[dict[str, Any]] = []
    for path in (R0190_SYNTHESIS, R0207_FACTORIAL):
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{path} changed")
        _embedded_signatures(value, embedded)
    expected_inputs = _dedupe([
        round_signature, *dependencies, *primaries, *embedded
    ])

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0213 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    job = {
        "id": "synthesise_scaling_story",
        "action": "synthesise_scaling_story",
        "handler_module": "experiments.round0213_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, "scaling-story.done.json"),
        "expected_inputs": expected_inputs,
        "r0190_synthesis": primaries[0],
        "r0207_factorial": primaries[1],
        "embedded_sources": _dedupe(embedded),
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
        "schema": "round0213-scaling-story-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": list(BOUND_ROUNDS),
        "capability_dependencies": [
            "jina-width-by-n-factorial-capacity-economics-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "registered_metric": "pile_ffr",
            "bound_rounds": list(BOUND_ROUNDS),
            "high_dose": HIGH_DOSE,
            "low_dose": LOW_DOSE,
            "operating_rule": OPERATING_RULE,
            "width_axis_measured_only_at_low_dose": True,
            "capacity_absorbs_dose_claim_supported": False,
            "training_performed": False,
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
        "queue_manifest": prepare_round0213(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
