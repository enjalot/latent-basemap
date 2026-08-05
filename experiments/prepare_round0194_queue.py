#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-only R0194 Pile localization queue."""
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
from basemap.round0113_prompt_contrast import read_sealed
from basemap.round0194_pile_loss_localization import CAPABILITY, ROUND_ID
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0194"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0194-2026-08-05.md")
POPULATION = "/data/latent-basemap/runs/round-0187/queue/artifacts/nested-populations/quarter/population.json"
GRAPH_MANIFEST = "/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/quarter-graph/graph-manifest.json"
QUEUE_PATHS = {
    "0187": "/data/latent-basemap/runs/round-0187/queue-correction-1/queue.json",
    "0188": "/data/latent-basemap/runs/round-0188/queue/queue.json",
    "0189": "/data/latent-basemap/runs/round-0189/queue/queue.json",
}
REVIEWS = {
    round_id: os.path.join(LAB_ROOT, f"review-{round_id}-2026-08-{'04' if round_id == '0187' else '05'}.md")
    for round_id in ("0187", "0188", "0189")
}
CELLS = {
    "42": {
        rung: {
            "evaluation": f"/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/{rung}-common-core-evaluation/common-core-evaluation.json",
            "coordinates": f"/data/latent-basemap/runs/round-0187/queue-correction-1/artifacts/{rung}-common-core-evaluation/common-quarter-coordinates.npy",
        }
        for rung in ("half", "full")
    },
    "43": {
        rung: {
            "evaluation": f"/data/latent-basemap/runs/round-0188/queue/artifacts/{rung}-seed43-common-core-evaluation/common-core-evaluation.json",
            "coordinates": f"/data/latent-basemap/runs/round-0188/queue/artifacts/{rung}-seed43-common-core-evaluation/common-quarter-coordinates.npy",
        }
        for rung in ("half", "full")
    },
    "44": {
        rung: {
            "evaluation": f"/data/latent-basemap/runs/round-0189/queue/artifacts/{rung}-seed44-common-core-evaluation/common-core-evaluation.json",
            "coordinates": f"/data/latent-basemap/runs/round-0189/queue/artifacts/{rung}-seed44-common-core-evaluation/common-quarter-coordinates.npy",
        }
        for rung in ("half", "full")
    },
}


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0194 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _review_lineage(round_id: str) -> list[dict[str, Any]]:
    review_path = REVIEWS[round_id]
    review = _frontmatter(review_path)
    if review.get("round_id") != round_id or review.get("status") != "accepted":
        raise RuntimeError(f"R{round_id} review is not accepted")
    round_signature = expected_input_signature(os.path.join(LAB_ROOT, review["round"]))
    result_signature = expected_input_signature(os.path.join(LAB_ROOT, review["result"]))
    if (
        review.get("round_sha256") != round_signature["sha256"]
        or review.get("result_sha256") != result_signature["sha256"]
    ):
        raise RuntimeError(f"R{round_id} review binding changed")
    queue_signature = expected_input_signature(QUEUE_PATHS[round_id])
    terminal_path = os.path.join(os.path.dirname(QUEUE_PATHS[round_id]), "runner-terminal.json")
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
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
        expected_input_signature(terminal_path),
    ]


def _source_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    population_signature = expected_input_signature(POPULATION)
    population = read_sealed(POPULATION, label="accepted R0187 quarter population")
    if (
        population.get("retained_rows") != 1_988_104
        or population.get("corpus_compact_ranges", {}).get("pile")
        != [1_421_764, 1_988_104]
    ):
        raise RuntimeError("R0187 quarter population contract changed")
    source = dict(population["document_compact"])
    graph_signature = expected_input_signature(GRAPH_MANIFEST)
    graph = read_sealed(GRAPH_MANIFEST, label="accepted R0187 quarter graph")
    pile = (graph.get("comparison_references") or {}).get("pile") or {}
    if (
        pile.get("rows") != 566_340
        or pile.get("compact_range") != [1_421_764, 1_988_104]
    ):
        raise RuntimeError("R0187 Pile common reference changed")
    reference = dict(pile["high_d_reference"])
    return population_signature, source, graph_signature, reference


def prepare_round0194(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0194 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    lineage = [
        signature
        for round_id in ("0187", "0188", "0189")
        for signature in _review_lineage(round_id)
    ]
    population, source, graph, reference = _source_inputs()
    cells = {
        seed: {
            rung: {
                key: expected_input_signature(path)
                for key, path in spec.items()
            }
            for rung, spec in rungs.items()
        }
        for seed, rungs in CELLS.items()
    }
    cell_inputs = [
        signature
        for rungs in cells.values()
        for spec in rungs.values()
        for signature in spec.values()
    ]
    accepted_reviews = [expected_input_signature(path) for path in REVIEWS.values()]
    expected_inputs = _dedupe([
        round_signature,
        *lineage,
        population,
        source,
        graph,
        reference,
        *cell_inputs,
    ])
    queue_root = create_fresh_directory(queue_root, label="R0194 CPU queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, CAPABILITY)
    job = {
        "id": "localize_pile_boundary_loss",
        "action": "localize_pile_boundary_loss",
        "handler_module": "experiments.round0194_nodes",
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
        "node_policy": {"gpu_required": False, "training_performed": False, "cpu_heavy": True},
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
        "schema": "round0194-pile-boundary-loss-localization-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0187", "0188", "0189"],
        "capability_dependencies": [
            "jina-document-english-composition-controlled-nested-ladder-v1",
            "jina-document-english-boundary-seed43-replay-v1",
            "jina-document-english-boundary-seed44-replay-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "scientific_contract": {
            "question": "is the three-seed half-to-full Pile FFR loss diffuse or localized in frozen high-D clusters/geometry?",
            "seeds": [42, 43, 44],
            "anchors": 4_000,
            "pile_rows": 566_340,
            "ffr_k_hit": 10,
            "ffr_k_fraction": 567,
            "cluster_labels": [256, 1024],
            "loss_group": "mean full-minus-half per-anchor FFR below zero",
            "retaining_group": "mean full-minus-half per-anchor FFR at least zero",
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
    print(json.dumps({"queue_manifest": prepare_round0194(release_sha=args.release_sha, queue_root=args.queue_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
