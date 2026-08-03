#!/usr/bin/env python3
"""Prepare, but never launch, the R0183 conditional CPU synthesis queue."""
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
from basemap.round0183_baseline_table import CAPABILITY, NUMAP_CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import (
    _accepted_review,
    _frontmatter,
    _frontmatter_list,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0183"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0183-2026-08-03.md")
AUMAP_SYNTHESIS = (
    "/data/latent-basemap/runs/round-0175/queue/artifacts/"
    "jina-aumap-oos-baseline-v1/synthesis.json"
)
R0181_RESULT = os.path.join(LAB_ROOT, "result-0181-2026-08-03.md")
R0181_REVIEW = os.path.join(LAB_ROOT, "review-0181-2026-08-03.md")
NUMAP_SYNTHESIS = (
    "/data/latent-basemap/runs/round-0181/queue/artifacts/"
    f"{NUMAP_CAPABILITY}/synthesis.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0183 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _r0181_branch() -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    if not os.path.isfile(R0181_RESULT) or not os.path.isfile(R0181_REVIEW):
        raise RuntimeError("R0183 waits for terminal R0181 result and review")
    result = _frontmatter(R0181_RESULT)
    review = _frontmatter(R0181_REVIEW)
    if (
        result.get("round_id") != "0181"
        or result.get("status") not in {"complete", "failed", "blocked"}
        or review.get("round_id") != "0181"
        or review.get("status") != "accepted"
    ):
        raise RuntimeError("R0181 is not an accepted terminal result")
    releases = _frontmatter_list(review, "releases")
    capability_marker = f"capability:{NUMAP_CAPABILITY}"
    evidence = [
        expected_input_signature(R0181_RESULT),
        expected_input_signature(R0181_REVIEW),
    ]
    if capability_marker in releases:
        synthesis = expected_input_signature(NUMAP_SYNTHESIS)
        return "measured", synthesis, [*evidence, synthesis]
    if NUMAP_CAPABILITY in _frontmatter_list(result, "capabilities_produced"):
        raise RuntimeError("R0181 result claims an unreviewed NUMAP capability")
    return "terminal-retry-failed", None, evidence


def prepare_round0183(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0183 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    r0175_review = _accepted_review("0175", "jina-aumap-oos-baseline-v1")
    branch, numap, r0181_evidence = _r0181_branch()
    aumap = expected_input_signature(AUMAP_SYNTHESIS)
    queue_root = create_fresh_directory(queue_root, label="R0183 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    expected_inputs = _dedupe([
        round_signature,
        *r0175_review,
        *r0181_evidence,
        aumap,
    ])
    job = {
        "id": "assemble_heldout_projection_method_table",
        "action": "baseline_table",
        "handler_module": "experiments.round0183_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "aumap_synthesis": aumap,
        "numap_synthesis": numap,
        "numap_terminal_status": branch,
        "r0181_result": expected_input_signature(R0181_RESULT),
        "r0181_review": expected_input_signature(R0181_REVIEW),
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "assemble-heldout-projection-table.done.json"),
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
        "schema": "round0183-heldout-projection-table-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-light",
        "required_reviews": ["0175", "0181"],
        "capability_dependencies": ["jina-aumap-oos-baseline-v1"],
        "conditional_numap_branch": branch,
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "operation": "authenticated transcription and deterministic rendering only",
            "scales": ["200k", "500k", "2m"],
            "corrected_parametric_500k": "explicitly not measured",
            "numap_branch": branch,
            "method_winner_selector": False,
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
    path = prepare_round0183(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
