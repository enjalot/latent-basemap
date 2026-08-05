#!/usr/bin/env python3
"""Prepare, but never launch, corrected CPU-only R0198 Track D."""
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
from basemap.round0198_pile_loss_localization import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments import prepare_round0194_queue as prior


ROUND_ROOT = "/data/latent-basemap/runs/round-0198"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0198-2026-08-05.md")
REVIEW_CAPABILITIES = {
    "0187": "jina-document-english-composition-controlled-nested-ladder-v1",
    "0188": "jina-document-english-composition-controlled-half-full-seed43-replay-v1",
    "0189": "jina-document-english-composition-controlled-half-full-seed44-replay-v1",
}


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0198 is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _review_lineage(round_id: str) -> list[dict[str, Any]]:
    review_path = prior.REVIEWS[round_id]
    review = _frontmatter(review_path)
    round_path = os.path.join(LAB_ROOT, review.get("round") or "")
    result_path = os.path.join(LAB_ROOT, review.get("result") or "")
    round_signature = expected_input_signature(round_path)
    result_signature = expected_input_signature(result_path)
    if (
        review.get("round_id") != round_id
        or review.get("status") != "accepted"
        or review.get("round_sha256") != round_signature["sha256"]
        or review.get("result_sha256") != result_signature["sha256"]
        or f"capability:{REVIEW_CAPABILITIES[round_id]}"
        not in _frontmatter_list(review, "releases")
        or _frontmatter(result_path).get("release_commit")
        != review.get("verified_release_commit")
    ):
        raise RuntimeError(f"R{round_id} accepted capability binding changed")
    queue_path = prior.QUEUE_PATHS[round_id]
    queue_signature = expected_input_signature(queue_path)
    terminal_path = os.path.join(os.path.dirname(queue_path), "runner-terminal.json")
    terminal_signature = expected_input_signature(terminal_path)
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
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
        terminal_signature,
    ]


def prepare_round0198(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0198 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = [
        signature
        for round_id in ("0187", "0188", "0189")
        for signature in _review_lineage(round_id)
    ]
    population, source, graph, reference = prior._source_inputs()
    cells = {
        seed: {
            rung: {
                key: expected_input_signature(path)
                for key, path in spec.items()
            }
            for rung, spec in rungs.items()
        }
        for seed, rungs in prior.CELLS.items()
    }
    cell_inputs = [
        signature
        for rungs in cells.values()
        for spec in rungs.values()
        for signature in spec.values()
    ]
    accepted_reviews = [
        expected_input_signature(prior.REVIEWS[round_id])
        for round_id in ("0187", "0188", "0189")
    ]
    expected_inputs = _dedupe([
        round_signature,
        *lineage,
        population,
        source,
        graph,
        reference,
        *cell_inputs,
    ])
    queue_root = create_fresh_directory(queue_root, label="R0198 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "localize_pile_boundary_loss",
        "action": "localize_pile_boundary_loss",
        "handler_module": "experiments.round0198_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, "pile-loss-localization.done.json"),
        "expected_inputs": expected_inputs,
        "population": population,
        "source": source,
        "graph_manifest": graph,
        "reference": reference,
        "cells": cells,
        "accepted_reviews": accepted_reviews,
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
        "schema": "round0198-pile-boundary-loss-localization-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0187", "0188", "0189"],
        "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "p90_gpu_seconds": {"total": 0.0},
        "scientific_contract": {
            "question": "is the three-seed half-to-full Pile FFR loss diffuse or localized in frozen high-D clusters/geometry?",
            "supersedes_round": "0194",
            "correction_scope": "capability identity and deterministic boundary membership only; scientific estimand unchanged",
            "seeds": [42, 43, 44],
            "anchors": 4_000,
            "pile_rows": 566_340,
            "ffr_k_hit": 10,
            "ffr_k_fraction": 567,
            "cluster_labels": [256, 1024],
            "loss_group": "mean full-minus-half per-anchor FFR below zero",
            "retaining_group": "mean full-minus-half per-anchor FFR at least zero",
            "boundary_tie_policy": "fail if any k567/k568 fp32 squared-L2 gap is zero",
            "diffuse_heuristic": "k256 top-decile loss-mass share <=0.35 and losing-cluster coverage >=0.75",
            "concentrated_heuristic": "k256 top-decile loss-mass share >=0.50",
            "hypothesis_generating": True,
            "quality_gate": False,
            "causal_claim": False,
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
        "queue_manifest": prepare_round0198(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
