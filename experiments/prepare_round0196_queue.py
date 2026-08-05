#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0196 GrEASE diagnosis."""
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
from basemap.round0196_grease_batch_stable import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0181_queue import _package_files
from experiments.round0179_nodes import TOOLCHAIN_PYTHON, TOOLCHAIN_ROOT
from experiments.round0196_nodes import __file__ as NODE_SCRIPT


ROUND_ROOT = "/data/latent-basemap/runs/round-0196"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0196-2026-08-05.md")
R0181_ROUND = os.path.join(LAB_ROOT, "round-0181-2026-08-03.md")
R0181_RESULT = os.path.join(LAB_ROOT, "result-0181-2026-08-03.md")
R0181_REVIEW = os.path.join(LAB_ROOT, "review-0181-2026-08-03.md")
R0181_QUEUE = "/data/latent-basemap/runs/round-0181/queue/queue.json"
R0181_TERMINAL = "/data/latent-basemap/runs/round-0181/queue/runner-terminal.json"
CHECKPOINT = (
    "/data/latent-basemap/runs/round-0181/queue/artifacts/"
    "numap-fixed-normalization-200k/reference/numap-model.dill"
)
QUERIES = (
    "/data/latent-basemap/runs/round-0181/queue/artifacts/"
    "numap-fixed-normalization-200k/held-query-embeddings.npy"
)
REFERENCE_SCRIPT = os.path.join(
    os.path.dirname(__file__), "round0196_grease_batch_stable_reference.py"
)
REVIEWED_CHECKPOINT_SIGNATURE = {
    "kind": "file",
    "canonical_path": CHECKPOINT,
    "bytes": 1_237_132_280,
    "sha256": "b3e0488f2cf72b9fd93f8343e2babb8d7360bb9d19c71cb5f72af9648efb7490",
}
REVIEWED_QUERY_SIGNATURE = {
    "kind": "file",
    "canonical_path": QUERIES,
    "bytes": 61_440_128,
    "sha256": "d246c3ec10cb2ab9e7a9812b9a1bd8f2fcb690eea7c5349a51e9d74e2aab02ad",
}


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0196 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _accepted_r0181() -> list[dict[str, Any]]:
    review = _frontmatter(R0181_REVIEW)
    result = _frontmatter(R0181_RESULT)
    if (
        review.get("round_id") != "0181"
        or review.get("status") != "accepted"
        or result.get("round_id") != "0181"
        or result.get("status") != "failed"
    ):
        raise RuntimeError("R0196 requires the accepted terminal R0181 failure")
    result_signature = expected_input_signature(R0181_RESULT)
    round_signature = expected_input_signature(R0181_ROUND)
    if (
        review.get("result_sha256") != result_signature["sha256"]
        or review.get("round_sha256") != round_signature["sha256"]
    ):
        raise RuntimeError("R0181 review bindings changed")
    with open(R0181_TERMINAL, encoding="utf-8") as handle:
        terminal = json.load(handle)
    queue_signature = expected_input_signature(R0181_QUEUE)
    if (
        terminal.get("round_id") != "0181"
        or terminal.get("verdict") != "failed"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
    ):
        raise RuntimeError("R0181 terminal failure provenance changed")
    return [
        round_signature,
        result_signature,
        expected_input_signature(R0181_REVIEW),
        queue_signature,
        expected_input_signature(R0181_TERMINAL),
    ]


def _reviewed_large_file(signature: dict[str, Any]) -> dict[str, Any]:
    stat = os.stat(signature["canonical_path"])
    if stat.st_size != signature["bytes"]:
        raise RuntimeError(f"reviewed artifact size changed: {signature['canonical_path']}")
    return dict(signature)


def prepare_round0196(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0196 release SHA must be one full commit")
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
    queue_root = create_fresh_directory(queue_root, label="R0196 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "diagnose_grease_batch_stability",
        "action": "diagnose_grease_batch_stability",
        "handler_module": "experiments.round0196_nodes",
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
        execution_authority="owner-campaign-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0196-grease-batch-stability-diagnosis-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0181"],
        "capability_dependencies": [],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "can a minimal fixed-geometry pure-network inference patch close the exact R0181 reload guard?",
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
        "queue_manifest": prepare_round0196(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
