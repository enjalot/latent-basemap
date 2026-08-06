#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0193 mixed-gate registration."""
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
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0192_quarter_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0193_mixed_gate_registration import CAPABILITY, FORMULA, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0193"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0193-2026-08-06.md")
R0192_QUEUE = "/data/latent-basemap/runs/round-0192/queue/queue.json"
R0192_TERMINAL = "/data/latent-basemap/runs/round-0192/queue/runner-terminal.json"
R0192_FAMILY = (
    "/data/latent-basemap/runs/round-0192/queue/artifacts/"
    f"{FAMILY_CAPABILITY}/quarter-seed-family.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0193 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _accepted_r0192() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reviews = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, "review-0192-*.md")))
        if _frontmatter(path).get("status") == "accepted"
    ]
    results = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, "result-0192-*.md")))
        if _frontmatter(path).get("status") == "complete"
    ]
    if len(reviews) != 1 or len(results) != 1:
        raise RuntimeError("R0193 requires one accepted Review/Result 0192")
    review = _frontmatter(reviews[0])
    result_signature = expected_input_signature(results[0])
    round_signature = expected_input_signature(
        os.path.join(LAB_ROOT, str(review.get("round") or ""))
    )
    if (
        review.get("result_sha256") != result_signature["sha256"]
        or review.get("round_sha256") != round_signature["sha256"]
        or f"capability:{FAMILY_CAPABILITY}"
        not in _frontmatter_list(review, "releases")
        or _frontmatter(results[0]).get("release_commit")
        != review.get("verified_release_commit")
    ):
        raise RuntimeError("R0192 review binding changed")
    queue_signature = expected_input_signature(R0192_QUEUE)
    terminal_signature = expected_input_signature(R0192_TERMINAL)
    with open(R0192_TERMINAL, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        terminal.get("round_id") != "0192"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
    ):
        raise RuntimeError("R0192 terminal premise changed")
    family = expected_input_signature(R0192_FAMILY)
    return family, [
        round_signature,
        result_signature,
        expected_input_signature(reviews[0]),
        queue_signature,
        terminal_signature,
    ]


def prepare_round0193(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0193 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    family, lineage = _accepted_r0192()
    expected_inputs = _dedupe([round_signature, *lineage, family])
    queue_root = create_fresh_directory(queue_root, label="R0193 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "register_mixed_quality_gates",
        "action": "register_mixed_quality_gates",
        "handler_module": "experiments.round0193_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "mixed-quality-gates.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 30.0,
        "family_evidence": family["canonical_path"],
        "accepted_review": expected_input_signature(lineage[2]["canonical_path"]),
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
        "schema": "round0193-mixed-quality-gate-registration-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0192"],
        "capability_dependencies": [FAMILY_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": (
                "what binding mixed-English quarter quality floors follow from "
                "the reviewed three-seed family?"
            ),
            "formula": FORMULA,
            "sample_standard_deviation_ddof": 1,
            "n": 3,
            "higher_is_better": True,
            "formula_preregistered_in_campaign_and_r0192": True,
            "r0161_prompted_fineweb_floors_unchanged": True,
            "raw_floors_unchanged": True,
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
        "queue_manifest": prepare_round0193(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
