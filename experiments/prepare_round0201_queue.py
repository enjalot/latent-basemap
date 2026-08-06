#!/usr/bin/env python3
"""Prepare, but never launch, R0201's CPU-only Track-D correction."""
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
from basemap.round0201_pile_loss_localization import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments import prepare_round0198_queue as prior


ROUND_ROOT = "/data/latent-basemap/runs/round-0201"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0201-2026-08-06.md")
R0198_RESULT = os.path.join(LAB_ROOT, "result-0198-2026-08-06.md")
R0198_QUEUE = "/data/latent-basemap/runs/round-0198/queue/queue.json"
R0198_TERMINAL = os.path.join(os.path.dirname(R0198_QUEUE), "runner-terminal.json")
R0198_FAILED = os.path.join(
    os.path.dirname(R0198_QUEUE), "artifacts", "pile-loss-localization.failed.json"
)


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
        or frontmatter.get("supersedes") != '["0198"]'
    ):
        raise RuntimeError("R0201 is not issued for this exact release/supersession")
    return expected_input_signature(ROUND_FILE)


def _failed_r0198_lineage() -> list[dict[str, Any]]:
    result = _frontmatter(R0198_RESULT)
    queue_signature = expected_input_signature(R0198_QUEUE)
    terminal_signature = expected_input_signature(R0198_TERMINAL)
    failed_signature = expected_input_signature(R0198_FAILED)
    with open(R0198_TERMINAL, encoding="utf-8") as handle:
        terminal = json.load(handle)
    with open(R0198_FAILED, encoding="utf-8") as handle:
        failed = json.load(handle)
    if (
        result.get("round_id") != "0198"
        or result.get("status") != "failed"
        or result.get("outcome") != "ambiguous-fp32-boundary-tie"
        or result.get("release_commit") != prior._frontmatter(
            os.path.join(LAB_ROOT, "round-0198-2026-08-05.md")
        ).get("base_commit")
        or result.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("round_id") != "0198"
        or terminal.get("verdict") != "failed"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("boundary_problems") != []
        or failed.get("node") != "localize_pile_boundary_loss"
        or failed.get("returncode") != 1
        or failed.get("queue_manifest_sha256") != queue_signature["sha256"]
    ):
        raise RuntimeError("R0198 failed tie-guard lineage changed")
    return [
        expected_input_signature(
            os.path.join(LAB_ROOT, "round-0198-2026-08-05.md")
        ),
        expected_input_signature(R0198_RESULT),
        queue_signature,
        terminal_signature,
        failed_signature,
    ]


def prepare_round0201(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0201 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    upstream_lineage = [
        signature
        for round_id in ("0187", "0188", "0189")
        for signature in prior._review_lineage(round_id)
    ]
    failed_lineage = _failed_r0198_lineage()
    population, source, graph, reference = prior.prior._source_inputs()
    cells = {
        seed: {
            rung: {
                key: expected_input_signature(path)
                for key, path in spec.items()
            }
            for rung, spec in rungs.items()
        }
        for seed, rungs in prior.prior.CELLS.items()
    }
    cell_inputs = [
        signature
        for rungs in cells.values()
        for spec in rungs.values()
        for signature in spec.values()
    ]
    accepted_reviews = [
        expected_input_signature(prior.prior.REVIEWS[round_id])
        for round_id in ("0187", "0188", "0189")
    ]
    expected_inputs = _dedupe([
        round_signature,
        *upstream_lineage,
        *failed_lineage,
        population,
        source,
        graph,
        reference,
        *cell_inputs,
    ])
    queue_root = create_fresh_directory(queue_root, label="R0201 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "localize_pile_boundary_loss_float64",
        "action": "localize_pile_boundary_loss",
        "handler_module": "experiments.round0201_nodes",
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
        "failed_r0198_lineage": failed_lineage,
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
        "schema": "round0201-pile-boundary-loss-localization-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0187", "0188", "0189"],
        "capability_dependencies": list(prior.REVIEW_CAPABILITIES.values()),
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "p90_gpu_seconds": {"total": 0.0},
        "scientific_contract": {
            "question": (
                "is the three-seed half-to-full Pile FFR loss diffuse or "
                "localized in frozen high-D clusters/geometry?"
            ),
            "supersedes_round": "0198",
            "correction_scope": (
                "float64 squared-L2 candidate rerank only; population, six "
                "cells, anchors, metric, groups, and selector unchanged"
            ),
            "expected_fp32_boundary_ties": {
                "seed42_half": 0,
                "seed42_full": 0,
                "seed43_half": 0,
                "seed43_full": 0,
                "seed44_half": 0,
                "seed44_full": 1,
            },
            "float64_boundary_rule": (
                "rerank exact cKDTree candidates in float64 over exact float32 "
                "coordinates; canonical row ID secondary; require positive "
                "k567/k568 gap in all six cells"
            ),
            "seeds": [42, 43, 44],
            "anchors": 4_000,
            "pile_rows": 566_340,
            "ffr_k_hit": 10,
            "ffr_k_fraction": 567,
            "cluster_labels": [256, 1024],
            "loss_group": "mean full-minus-half per-anchor FFR below zero",
            "retaining_group": (
                "mean full-minus-half per-anchor FFR at least zero"
            ),
            "diffuse_heuristic": (
                "k256 top-decile loss-mass share <=0.35 and "
                "losing-cluster coverage >=0.75"
            ),
            "concentrated_heuristic": (
                "k256 top-decile loss-mass share >=0.50"
            ),
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
        "queue_manifest": prepare_round0201(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
