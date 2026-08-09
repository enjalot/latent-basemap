#!/usr/bin/env python3
"""Prepare, but never launch, the R0234 calibrated-robust-floor queue.

One CPU node. It calibrates every candidate floor multiplier on the Gaussian
null at `n = 13` before opening a sealed cell, applies the pre-registered
selection rule (coverage, invariance, attainability; then power, materiality,
breakdown point), registers what survives, scores all `25` cells against every
candidate, enumerates every published verdict the registration would change, and
retains rather than supersedes any released criterion it would un-fail.

`gpu_required: False`, `gpu_hours_cap: 0.0`, `CUDA_VISIBLE_DEVICES=""` in the
child environment. Nothing here trains, scores a map, or reads a GPU.
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
from basemap import round0234_calibration as calibration
from basemap.round0234_calibrated_floors import (
    CANDIDATE_CLUSTER_COUNTS,
    CANDIDATE_ORDER,
    CANDIDATE_SEEDS,
    CANDIDATES,
    COVERAGE_TARGET,
    COVERAGE_TOLERANCE,
    CUVS_FAMILY_SEEDS,
    DENSITY_V2_DEFECT,
    DESCRIPTIVE_METRICS,
    EXACT_FAMILY_SEEDS,
    EXTERNAL_CALIBRATION_TARGETS,
    GATED_METRICS,
    GATE_CAPABILITY,
    METRICS,
    N_EXACT,
    N_HELD_OUT,
    POWER_MATERIALITY,
    POWER_SELECTION_ALTERNATIVE,
    PURITY_METRICS,
    REQUIRED_INVARIANCE_DEPTH,
    ROUND_ID,
    SELECTION_RULE,
    identity_bound,
)
from experiments.round0234_nodes import (
    GATE_ACTION,
    POWER_LADDER_SIZES,
    PRECEDENTS,
    RELEASED_FLOOR_FAMILIES,
    SOURCES,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0234"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0234-2026-08-09.md")

#: Four million Gaussian families at `n = 13` for six estimators, a two-sided
#: bisection over each, a 32-rung power ladder and eight JSON reads. Measured in
#: minutes; allowed twenty.
GATE_P90_WALL_S = 1_200.0
GPU_HOURS_CAP = 0.0


def _issued_round(release_sha: str) -> tuple[dict[str, Any], list[str]]:
    frontmatter = _frontmatter(ROUND_FILE)
    base_commit = str(frontmatter.get("base_commit") or "")
    descendant = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor", base_commit,
         release_sha],
        check=False,
        timeout=10,
    ).returncode == 0
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or not descendant
    ):
        raise RuntimeError("R0234 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0234 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _upstream_review_state(required: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    contingent: list[str] = []
    for round_id in required:
        reviews = []
        for path in sorted(
            glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
        ):
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
            "The calibrated n = 13 floors are registered here and released only "
            "once every listed round carries an accepted review."
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
        raise RuntimeError("R0234 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0234_calibrated_floors.py",
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
        "schema": "round0234-release-cpu-smoke-v1",
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
            "every scale estimator against hand-computed order statistics; the "
            "Monte-Carlo calibrator against the noncentral-t and Howe closed "
            "forms; the measurement that 3 under-delivers for MAD_n at n = 13; "
            "the derived attainability bounds, including the 1-trimmed family "
            "losing attainability at its own calibrated multiplier; the "
            "positive-scale witness and the degeneracy of R0231's; the injection "
            "ladder separating variance from robust scales in both tail "
            "directions; two-sided band scoring with direction; and the "
            "verdict-change enumerator that makes a reversal impossible to miss"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0234 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0234(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0234 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)
    review_state = _upstream_review_state(list(required_reviews))

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0234 queue")
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
    node = "register_calibrated_robust_floors_n13"
    jobs = [
        {
            "id": node,
            "action": GATE_ACTION,
            "handler_module": "experiments.round0234_nodes",
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
        "schema": "round0234-calibrated-robust-floors-n13-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-analysis",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-seed-family-panel-n13-v1",
            "minilm-mixed-2m-quality-gates-v1",
            "minilm-mixed-2m-quality-gates-n8-v1",
            "minilm-mixed-2m-tolerance-gates-n8-v1",
            "minilm-mixed-2m-robust-floors-n13-v1",
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
            "minilm-mixed-2m-cluster-spill-graph-map-comparison-v1",
        ],
        "capabilities_produced": [GATE_CAPABILITY],
        "supersedes": [],
        "training_performed": False,
        "p90_gpu_seconds": {node: 0.0, "total": 0.0},
        "jobs": jobs,
        "scientific_contract": {
            "question": (
                "is there a floor family that is exactly outlier-invariant AND "
                "delivers its nominal 95% confidence at n = 13, and if so which "
                "one, at what multiplier, and which cells does it fail?"
            ),
            "population": (
                "the 13 exact-graph cells (seeds 42-54) that define the floors, "
                "plus 12 held-out cells that define nothing: R0223's 3 cuVS-igd48 "
                "cells and R0228's 9 cluster-spill candidate cells"
            ),
            "cells_defining_the_floors": N_EXACT,
            "held_out_cells": N_HELD_OUT,
            "cells_scored_in_total": N_EXACT + N_HELD_OUT,
            "metrics": list(METRICS),
            "gated_metrics": list(GATED_METRICS),
            "descriptive_metrics": list(DESCRIPTIVE_METRICS),
            "density_v2_defect": DENSITY_V2_DEFECT,
            "purity_metrics_gated_two_sidedly_unfolded": list(PURITY_METRICS),
            "candidate_estimators": list(CANDIDATE_ORDER),
            "candidate_properties": {
                name: {
                    "centre": item["centre"],
                    "scale": item["scale_name"],
                    "asymptotic_breakdown_point": item["breakdown_point"],
                    "gaussian_efficiency": item["gaussian_efficiency"],
                    "robust": item["robust"],
                }
                for name, item in CANDIDATES.items()
            },
            "selection_rule": SELECTION_RULE,
            "coverage_target": COVERAGE_TARGET,
            "coverage_tolerance": COVERAGE_TOLERANCE,
            "required_invariance_depth": REQUIRED_INVARIANCE_DEPTH,
            "power_selection_alternative_sigma": POWER_SELECTION_ALTERNATIVE,
            "power_materiality": POWER_MATERIALITY,
            "calibration": {
                "families_simulated_at_n13": calibration.CALIBRATION_FAMILIES,
                "seed": calibration.CALIBRATION_SEED,
                "one_sided_definition": (
                    "P(centre - k*scale <= mu - z_0.95*sigma) = 0.95, so the "
                    "calibrated multiplier is the 0.95-quantile of "
                    "(centre + z_0.95)/scale over standard-normal families"
                ),
                "two_sided_definition": (
                    "P(Phi(upper) - Phi(lower) >= 0.95) = 0.95 -- the standard "
                    "content definition Howe's factor approximates, NOT the "
                    "stricter requirement that the interval bracket mu +/- 1.96 "
                    "sigma"
                ),
                "external_reference_targets": [
                    dict(item) for item in EXTERNAL_CALIBRATION_TARGETS
                ],
                "calibration_reads_no_cell_of_this_program": True,
            },
            "attainability": (
                "derived per estimator rather than asserted. A mean - k*s family "
                "obeys max|x - xbar|/s <= (n-1)/sqrt(n) = "
                f"{identity_bound(N_EXACT)!r} at n = 13. A 1-trimmed family obeys "
                "the same identity on its KEPT subsample, so above "
                f"{identity_bound(N_EXACT - 2)!r} only the trimmed cells can fail. "
                "A scale built from an order statistic of rank r out of m terms, "
                "at most a of which involve any single cell, is bounded whenever "
                "m - a >= r, so max (centre - x_i)/scale is unbounded and EVERY "
                "defining cell can fail at ANY multiplier. Each claim is carried "
                "by a witness family with a strictly POSITIVE scale estimate; "
                "R0231's [1.0 x 12, 0.0] vector is reported alongside and labelled "
                "degenerate."
            ),
            "invariance": (
                "Review 0225-01's injection, deepened: the d worst cells are each "
                "made 1, 2 and 3 sample-sd worse, for d = 1..6, on every gated "
                "metric series and on both unfolded purity log-ratio series, and "
                "for a two-sided band on the bound facing the contaminated tail. "
                "Depth 1 is the hard bar; the full ladder is reported."
            ),
            "no_supersession": (
                "R0234 supersedes nothing. If the registered floor passes a cell "
                "that a RELEASED floor failed, the reversal is named beside the "
                "floor and the released criterion is RETAINED as a second "
                "criterion on that metric. Released families watched for this: "
                f"{list(RELEASED_FLOOR_FAMILIES)}."
            ),
            "power_parity_ladder_sizes": list(POWER_LADDER_SIZES),
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
        "queue_manifest": prepare_round0234(
            release_sha=args.release_sha, queue_root=args.queue_root
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
