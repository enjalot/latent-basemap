#!/usr/bin/env python3
"""Prepare, but never launch, the R0157 prompted-density recovery queue."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0157_prompted_density import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0157"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0157-2026-08-02.md")
ISSUED_BASE_COMMIT = "a9814820eb2b578d4ef5854db402b1df902b1e59"
_MECHANICAL_CORRECTION_FILES = {
    "experiments/prepare_round0157_queue.py",
    "experiments/round0157_nodes.py",
    "tests/test_round0157_prompted_density.py",
}
ASSEMBLY = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/"
    "compact-arrays/assembly-manifest.json"
)
DOCUMENT_COMPACT = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/"
    "compact-arrays/document-compact.f16"
)
REVIEW_FILES = {
    42: os.path.join(LAB_ROOT, "review-0115-2026-07-30.md"),
    43: os.path.join(LAB_ROOT, "review-0117-2026-07-31.md"),
}
CELLS = {
    42: {
        "coordinates": (
            "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
            "document/evaluation/coordinates.npy"
        ),
        "score": (
            "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
            "document/evaluation/score.json"
        ),
    },
    43: {
        "coordinates": (
            "/data/latent-basemap/runs/round-0117/queue/artifacts/"
            "document/evaluation/coordinates.npy"
        ),
        "score": (
            "/data/latent-basemap/runs/round-0117/queue/artifacts/"
            "document/evaluation/score.json"
        ),
    },
}


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _accepted_review(path: str) -> dict[str, Any]:
    if _frontmatter(path).get("status") != "accepted":
        raise RuntimeError(f"R0157 required review is not accepted: {path}")
    return expected_input_signature(path)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if frontmatter.get("status") != "issued":
        raise RuntimeError("R0157 round is not issued")
    if frontmatter.get("base_commit") != ISSUED_BASE_COMMIT:
        raise RuntimeError("R0157 issued base_commit changed")
    if release_sha != ISSUED_BASE_COMMIT:
        ancestor = subprocess.run(
            [
                "git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
                ISSUED_BASE_COMMIT, release_sha,
            ],
            check=False,
            timeout=10,
        )
        changed = subprocess.run(
            [
                "git", "-C", RELEASE_ROOT, "diff", "--name-only",
                f"{ISSUED_BASE_COMMIT}..{release_sha}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        if ancestor.returncode != 0 or not set(changed) <= _MECHANICAL_CORRECTION_FILES:
            raise RuntimeError("R0157 release exceeds the setup correction")
    return expected_input_signature(ROUND_FILE)


def prepare_round0157(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0157 release SHA must be one full commit")
    round_file = _issued_round(release_sha)
    reviews = [_accepted_review(REVIEW_FILES[seed]) for seed in sorted(REVIEW_FILES)]
    assembly_signature = expected_input_signature(ASSEMBLY)
    assembly = _read_json(ASSEMBLY)
    document_signature = expected_input_signature(DOCUMENT_COMPACT)
    if assembly.get("outputs", {}).get("document") != document_signature:
        raise RuntimeError("R0157 document compact binding changed")
    cells = []
    cell_inputs = []
    for seed, paths in sorted(CELLS.items()):
        coordinates = expected_input_signature(paths["coordinates"])
        score = expected_input_signature(paths["score"])
        cells.append({"seed": seed, "coordinates": coordinates, "score": score})
        cell_inputs.extend((coordinates, score))

    expected_inputs = _dedupe([
        round_file,
        *reviews,
        assembly_signature,
        document_signature,
        *cell_inputs,
    ])
    queue_root = create_fresh_directory(queue_root, label="R0157 GPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "recover_native_prompted_density_v2",
        "action": "recover_native_prompted_density_v2",
        "handler_module": "experiments.round0157_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "prompted-density-v2.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 900.0,
        "assembly": assembly_signature,
        "document_compact": document_signature,
        "accepted_reviews": reviews,
        "cells": cells,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "model_produced": False,
        },
    }
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0157-native-prompted-density-v2-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0115", "0117"],
        "ordering_dependencies": [],
        "capability_dependencies": [
            "jina-fineweb-2m-prompt-map-contrast-v1",
            "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "what calibrated-style density-v2 do the accepted native prompted seed-42/43 maps achieve?",
            "population": "R0113 1,993,761 shared prompt-union representatives",
            "anchors": 10_000,
            "anchor_seed": 123,
            "k_density": 15,
            "high_and_low_search": "exact panel-v2 self-excluded mean radius",
            "raw_universe_floor_is_context_only": True,
            "quality_gate": None,
            "prompt_noninferiority_verdict_changed": False,
            "floor_changed": False,
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
    print(json.dumps({
        "queue_manifest": prepare_round0157(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
