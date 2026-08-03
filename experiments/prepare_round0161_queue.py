#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0161 prompted gate registration."""
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
from basemap.round0160_prompted_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0161_prompted_gate_registration import CAPABILITY, FORMULA, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0161"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0161-2026-08-03.md")


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if frontmatter.get("status") != "issued" or frontmatter.get("base_commit") != release_sha:
        raise RuntimeError("R0161 round is not issued for this release")
    return expected_input_signature(ROUND_FILE)


def _accepted_review() -> dict[str, Any]:
    accepted = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, "review-0160-*.md")))
        if _frontmatter(path).get("status") == "accepted"
    ]
    if len(accepted) != 1:
        raise RuntimeError(f"R0161 requires one accepted Review 0160; found {len(accepted)}")
    return expected_input_signature(accepted[0])


def _family_evidence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = [
        path
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, "result-0160-*.md")))
        if _frontmatter(path).get("status") == "complete"
    ]
    if len(results) != 1:
        raise RuntimeError(f"R0161 requires one complete Result 0160; found {len(results)}")
    result = _frontmatter(results[0])
    queue_path = str(result.get("queue_manifest") or "")
    if queue_path.startswith("gsv:"):
        queue_path = queue_path[4:]
    queue_signature = expected_input_signature(queue_path)
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    terminal_path = os.path.join(os.path.dirname(queue_path), "runner-terminal.json")
    terminal_signature = expected_input_signature(terminal_path)
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    outputs = [
        str(output)
        for job in queue.get("jobs", [])
        for output in (job.get("outputs") or [])
        if os.path.basename(str(output)) == FAMILY_CAPABILITY
    ]
    if (
        queue.get("round_id") != "0160"
        or FAMILY_CAPABILITY not in (queue.get("capabilities_produced") or [])
        or terminal.get("verdict") != "succeeded"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or len(outputs) != 1
    ):
        raise RuntimeError("R0161 accepted R0160 execution lineage changed")
    evidence = expected_input_signature(os.path.join(outputs[0], "prompted-seed-family.json"))
    return evidence, [expected_input_signature(results[0]), queue_signature, terminal_signature]


def prepare_round0161(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0161 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    review = _accepted_review()
    family, lineage = _family_evidence()
    expected_inputs = _dedupe([round_signature, review, *lineage, family])
    queue_root = create_fresh_directory(queue_root, label="R0161 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "register_prompted_quality_gates",
        "action": "register_prompted_quality_gates",
        "handler_module": "experiments.round0161_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "prompted-quality-gates.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 30.0,
        "family_evidence": family["canonical_path"],
        "accepted_review": review,
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
        "schema": "round0161-prompted-quality-gate-registration-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0160"],
        "capability_dependencies": [FAMILY_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "what binding quality floors follow from the four-seed prompted family?",
            "formula": FORMULA,
            "sample_standard_deviation_ddof": 1,
            "n": 4,
            "higher_is_better": True,
            "registered_before_r0160_outcomes": True,
            "raw_floor_changed": False,
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
    print(json.dumps({"queue_manifest": prepare_round0161(release_sha=args.release_sha, queue_root=args.queue_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
