#!/usr/bin/env python3
"""Prepare, but never launch, authority-corrected Track F1 Round 0199."""
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
from basemap.round0199_grease_batch_stable import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0181_queue import _package_files
from experiments.prepare_round0196_queue import (
    CHECKPOINT,
    QUERIES,
    REFERENCE_SCRIPT,
    REVIEWED_CHECKPOINT_SIGNATURE,
    REVIEWED_QUERY_SIGNATURE,
    _accepted_r0181,
    _reviewed_large_file,
)
from experiments.round0179_nodes import TOOLCHAIN_PYTHON, TOOLCHAIN_ROOT
from experiments.round0199_nodes import __file__ as NODE_SCRIPT


ROUND_ROOT = "/data/latent-basemap/runs/round-0199"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0199-2026-08-05.md")


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
        or frontmatter.get("execution_authority") != "autonomous-cpu"
        or frontmatter.get("supersedes") != ["0196"]
    ):
        raise RuntimeError("R0199 is not the issued authority-corrected R0196 replacement")
    return expected_input_signature(ROUND_FILE)


def prepare_round0199(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0199 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    r0181_lineage = _accepted_r0181()
    checkpoint = _reviewed_large_file(REVIEWED_CHECKPOINT_SIGNATURE)
    queries = _reviewed_large_file(REVIEWED_QUERY_SIGNATURE)
    reference_script = expected_input_signature(REFERENCE_SCRIPT)
    node_script = expected_input_signature(os.path.realpath(NODE_SCRIPT))
    toolchain_python = {
        "invocation_path": TOOLCHAIN_PYTHON,
        "resolved_interpreter": expected_input_signature(
            os.path.realpath(TOOLCHAIN_PYTHON)
        ),
    }
    package_files = [expected_input_signature(path) for path in _package_files()]
    expected_inputs = _dedupe([
        round_signature,
        *r0181_lineage,
        checkpoint,
        queries,
        reference_script,
        node_script,
        toolchain_python["resolved_interpreter"],
        expected_input_signature(os.path.join(TOOLCHAIN_ROOT, "pyvenv.cfg")),
        *package_files,
    ])
    queue_root = create_fresh_directory(queue_root, label="R0199 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "diagnose_grease_batch_stability",
        "action": "diagnose_grease_batch_stability",
        "handler_module": "experiments.round0199_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "grease-diagnosis.done.json"),
        "expected_inputs": expected_inputs,
        "checkpoint": checkpoint,
        "queries": queries,
        "reference_script": reference_script,
        "toolchain_python": toolchain_python,
        "accepted_r0181_review": r0181_lineage[2],
        "p90_wall_s": 7_200.0,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
            "cpu_heavy": True,
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
        "schema": "round0199-grease-batch-stability-diagnosis-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0181"],
        "capability_dependencies": [],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "can a minimal fixed-geometry pure-network inference patch close the exact R0181 reload guard?",
            "scientific_contract_reused_from": "0196",
            "supersedes_round": "0196",
            "correction": "queue execution authority now matches the issued autonomous-cpu contract",
            "source_checkpoint_round": "0181",
            "reload_tolerance": 1.0e-4,
            "fixed_chunk_rows": 256,
            "cpu_wall_seconds_maximum": 7_200.0,
            "f2_gpu_hours_maximum_if_positive": 0.5,
            "f3_terminal_negative_if_not_positive": True,
            "additional_debug_or_f4_authorized": False,
            "numap_revival_authorized": False,
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
        "queue_manifest": prepare_round0199(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
