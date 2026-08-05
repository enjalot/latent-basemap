#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0190 boundary synthesis."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0190_three_seed_boundary_synthesis import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0190"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0190-2026-08-05.md")

DECISION_PATHS = {
    "0187": (
        "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
        "nested-ladder-synthesis/ladder-decision.json"
    ),
    "0188": (
        "/data/latent-basemap/runs/round-0188/queue/artifacts/"
        "seed43-boundary-synthesis/boundary-decision.json"
    ),
    "0189": (
        "/data/latent-basemap/runs/round-0189/queue/artifacts/"
        "seed44-boundary-synthesis/boundary-decision.json"
    ),
}
EVALUATION_PATHS = {
    "seed42_quarter": (
        "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
        "quarter-common-core-evaluation/common-core-evaluation.json"
    ),
    "seed42_half": (
        "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
        "half-common-core-evaluation/common-core-evaluation.json"
    ),
    "seed42_full": (
        "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/"
        "full-common-core-evaluation/common-core-evaluation.json"
    ),
    "seed43_half": (
        "/data/latent-basemap/runs/round-0188/queue/artifacts/"
        "half-seed43-common-core-evaluation/common-core-evaluation.json"
    ),
    "seed43_full": (
        "/data/latent-basemap/runs/round-0188/queue/artifacts/"
        "full-seed43-common-core-evaluation/common-core-evaluation.json"
    ),
    "seed44_half": (
        "/data/latent-basemap/runs/round-0189/queue/artifacts/"
        "half-seed44-common-core-evaluation/common-core-evaluation.json"
    ),
    "seed44_full": (
        "/data/latent-basemap/runs/round-0189/queue/artifacts/"
        "full-seed44-common-core-evaluation/common-core-evaluation.json"
    ),
}
R0160_FAMILY = (
    "/data/latent-basemap/runs/round-0160/queue/artifacts/"
    "jina-fineweb-2m-prompted-seed42-45-family-v1/prompted-seed-family.json"
)
R0161_GATES = (
    "/data/latent-basemap/runs/round-0161/queue/artifacts/"
    "jina-prompted-universe-quality-gates-v1/prompted-quality-gates.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if frontmatter.get("status") != "issued" or not descendant:
        raise RuntimeError("R0190 round is not issued for this descendant release")
    return expected_input_signature(ROUND_FILE)


def _document(prefix: str, round_id: str, *, status: str) -> dict[str, Any]:
    matches = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"{prefix}-{round_id}-*.md")))
        if _frontmatter(path).get("status") == status
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"R0190 requires one {status} {prefix} for R{round_id}; found {len(matches)}"
        )
    return expected_input_signature(matches[0])


def prepare_round0190(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0190 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews = {
        round_id: _document("review", round_id, status="accepted")
        for round_id in ("0187", "0188", "0189")
    }
    lineage = [
        round_signature,
        *reviews.values(),
        *(
            _document("result", round_id, status="complete")
            for round_id in ("0187", "0188", "0189")
        ),
        _document("review", "0160", status="accepted"),
        _document("review", "0161", status="accepted"),
        expected_input_signature(R0160_FAMILY),
        expected_input_signature(R0161_GATES),
        *(expected_input_signature(path) for path in DECISION_PATHS.values()),
        *(expected_input_signature(path) for path in EVALUATION_PATHS.values()),
    ]
    expected_inputs = _dedupe(lineage)
    queue_root = create_fresh_directory(queue_root, label="R0190 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "synthesize_three_seed_boundary",
        "action": "synthesize_three_seed_boundary",
        "handler_module": "experiments.round0190_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "three-seed-synthesis.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 30.0,
        "review_signatures": reviews,
        "decision_paths": DECISION_PATHS,
        "evaluation_paths": EVALUATION_PATHS,
        "r0160_family_path": R0160_FAMILY,
        "r0161_gate_path": R0161_GATES,
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
        "schema": "round0190-three-seed-boundary-synthesis-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0160", "0161", "0187", "0188", "0189"],
        "capability_dependencies": [
            "jina-document-english-composition-controlled-nested-ladder-v1",
            "jina-document-english-composition-controlled-half-full-seed43-replay-v1",
            "jina-document-english-composition-controlled-half-full-seed44-replay-v1",
            "jina-prompted-universe-quality-gates-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": (
                "what aggregate decision follows from the reviewed seed-42/43/44 "
                "half-to-full Pile FFR boundary evidence?"
            ),
            "retention_floor": 0.97,
            "decision_rule": "at least two of three seeds below 0.97",
            "positive_outcome": "confirmed-2-of-3-seed-sensitive",
            "positive_branch": "activate exactly one R0191 h4096 full-rung sibling",
            "absolute_fineweb_gate_comparisons": "descriptive-only noncommensurate",
            "composition_shift": "mixed quarter seed42 versus FineWeb 2M seed42",
            "no_training": True,
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
        "queue_manifest": prepare_round0190(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
