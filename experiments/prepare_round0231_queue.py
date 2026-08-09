#!/usr/bin/env python3
"""Prepare, but never launch, the R0231 robust-floor queue.

One CPU node. It registers the mixed-MiniLM 2M floors at `n = 13` on a **robust
scale estimator**, gates FFR and both purity fidelities (purity two-sidedly on the
unfolded log-ratio, lower and upper), carries `density_v2` descriptive-only,
reports every prior floor family side by side against the thirteen exact cells and
the twelve held-out cells with attainability beside every count, and assesses the
R0161/R0193 exposure read-only.

`gpu_required: False`, `gpu_hours_cap: 0.0`, `CUDA_VISIBLE_DEVICES=""` in the child
environment. Nothing here trains, scores a map, or reads a GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0231_robust_floors import (
    CANDIDATE_CLUSTER_COUNTS,
    CANDIDATE_SEEDS,
    CHOSEN_ESTIMATOR,
    CHOSEN_ESTIMATOR_JUSTIFICATION,
    CUVS_FAMILY_SEEDS,
    DENSITY_V2_DEFECT,
    DESCRIPTIVE_METRICS,
    EXACT_FAMILY_SEEDS,
    FAMILY_DEFINITIONS,
    FLOOR_ESTIMATORS,
    GATED_METRICS,
    GATE_CAPABILITY,
    METRICS,
    ONE_SIDED_DERIVATION,
    PURITY_METRICS,
    ROBUST_ESTIMATORS,
    ROUND_ID,
    SD_DDOF,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    TWO_SIDED_DERIVATION,
    VARIANCE_ESTIMATORS,
    identity_bound,
)
from experiments.round0231_nodes import GATE_ACTION, PRECEDENTS, SOURCES
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0231"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0231-2026-08-09.md")

#: Pure arithmetic over 13 + 12 cells and six JSON reads. Measured in seconds;
#: allowed twenty minutes.
GATE_P90_WALL_S = 1_200.0
GPU_HOURS_CAP = 0.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit, release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0231 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0231 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))):
            frontmatter = _frontmatter(path)
            reviews.append({
                "file": os.path.basename(path),
                "status": frontmatter.get("status"),
                "sha256": expected_input_signature(path)["sha256"],
            })
        accepted = [item for item in reviews if item["status"] == "accepted"]
        state[round_id] = {
            "reviews_present": reviews,
            "accepted_reviews": len(accepted),
        }
        if not accepted:
            contingent.append(round_id)
    return {
        "required_reviews": list(required),
        "by_round": state,
        "rounds_without_an_accepted_review": contingent,
        "gate_release_contingent_on": contingent,
        "note": (
            "Review is post-hoc: it blocks the downstream claim, not the launch. "
            "The n = 13 floors are registered here and released only once every "
            "listed round carries an accepted review."
        ),
    }


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0231 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0231_robust_floors.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0231-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": (
            "the five floor estimators and their measured self-loosening, the "
            "outlier-invariance of the three robust candidates, the derived "
            "tolerance factors, the two-sided unfolded log-ratio bands, the "
            "identity bound at every n in play and the explicit witness that a "
            "defining cell can fail every registered family, and the "
            "attainability arithmetic behind every pass count"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0231 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0231(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0231 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))
    if CHOSEN_ESTIMATOR not in ROBUST_ESTIMATORS:
        raise RuntimeError("R0231 must register a robust estimator")

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0231 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)

    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        *[expected_input_signature(path) for path in SOURCES.values()],
        *[expected_input_signature(path) for path in PRECEDENTS.values()],
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    node = "register_robust_floors_n13"
    jobs = [
        {
            "id": node,
            "action": GATE_ACTION,
            "handler_module": "experiments.round0231_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": GATE_P90_WALL_S,
            "capability": GATE_CAPABILITY,
            "upstream_review_state": review_state,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
    ]

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    queue.update({
        "schema": "round0231-robust-floors-n13-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-analysis",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-seed-family-panel-n13-v1",
            "minilm-mixed-2m-quality-gates-n8-v1",
            "minilm-mixed-2m-tolerance-gates-n8-v1",
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
            "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1",
        ],
        "capabilities_produced": [GATE_CAPABILITY],
        "training_performed": False,
        "p90_gpu_seconds": {node: 0.0, "total": 0.0},
        "jobs": jobs,
        "scientific_contract": {
            "question": (
                "on a robust scale estimator and at an n where a defining cell can "
                "actually fail, what floors does the mixed-MiniLM 2M family "
                "register, and which cells clear them?"
            ),
            "population": (
                "the 13 exact-graph cells (seeds 42-54) that define the floors, "
                "plus 12 held-out cells that define nothing: R0223's 3 cuVS-igd48 "
                "cells and R0228's 9 cluster-spill candidate cells"
            ),
            "cells_defining_the_floors": len(EXACT_FAMILY_SEEDS),
            "held_out_cells": (
                len(CUVS_FAMILY_SEEDS)
                + len(CANDIDATE_CLUSTER_COUNTS) * len(CANDIDATE_SEEDS)
            ),
            "metrics": list(METRICS),
            "gated_metrics": list(GATED_METRICS),
            "descriptive_metrics": list(DESCRIPTIVE_METRICS),
            "density_v2_defect": DENSITY_V2_DEFECT,
            "purity_metrics_gated_two_sidedly_unfolded": list(PURITY_METRICS),
            "candidate_estimators": sorted(FLOOR_ESTIMATORS),
            "robust_estimators": list(ROBUST_ESTIMATORS),
            "variance_estimators": list(VARIANCE_ESTIMATORS),
            "chosen_estimator": CHOSEN_ESTIMATOR,
            "chosen_estimator_justification": CHOSEN_ESTIMATOR_JUSTIFICATION,
            "floor_families": FAMILY_DEFINITIONS,
            "identity_bound_at_n13": identity_bound(len(EXACT_FAMILY_SEEDS)),
            "identity_bound_note": (
                "max|x - xbar|/s <= (n-1)/sqrt(n) = 3.3282 at n = 13, above "
                "mean-2s's 2.0, the one-sided 95/95 factor 2.6705 and Howe's "
                "two-sided 3.1008 — so a defining cell CAN fail every variance "
                "family reported here. For the robust families the identity is "
                "INAPPLICABLE (it bounds deviation in sample-sd units) and the "
                "round exhibits an explicit witness family in which a defining "
                "cell falls below its own robust floor rather than inferring it."
            ),
            "self_loosening_is_measured": (
                "for every metric and every candidate estimator, by injecting a "
                "worse worst-cell at 1, 2 and 3 sample-sd below the observed "
                "minimum and re-fitting. A floor that moves 0.0 is the point of "
                "the exercise; a floor that drops is self-loosening and is "
                "reported as such."
            ),
            "strictness_is_measured": (
                "3*MAD_n / s and the effective sigma multiplier are published per "
                "metric so a reader can see whether the robust floor is stricter "
                "or laxer than mean - 2s on THIS family. If a robust floor fails "
                "a cell that mean - 2s passed, that is reported, not softened."
            ),
            "tolerance_content": TOLERANCE_CONTENT,
            "tolerance_confidence": TOLERANCE_CONFIDENCE,
            "sample_standard_deviation_ddof": SD_DDOF,
            "one_sided_derivation": ONE_SIDED_DERIVATION,
            "two_sided_derivation": TWO_SIDED_DERIVATION,
            "prior_families_reported_side_by_side": [
                "r0219_n4_mean_minus_2sd",
                "r0222_n8_mean_minus_2sd",
                "r0225_n8_tolerance_95_95",
            ],
            "precedents_read_only": sorted(PRECEDENTS),
            "training_performed": False,
            "evaluation_performed": False,
            "gpu_used": False,
            "gate_registerable_here": True,
            "gate_status": "registered-and-contingent-pending-review",
            "upstream_review_state": review_state,
            "production_or_publishing": False,
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
        "queue_manifest": prepare_round0231(
            release_sha=args.release_sha, queue_root=args.queue_root
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
