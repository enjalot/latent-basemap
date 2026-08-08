#!/usr/bin/env python3
"""Prepare, but never launch, the R0225 tolerance-gate queue.

One CPU node. It re-registers the mixed-MiniLM 2M gate from sealed artifacts
under a 95/95 tolerance interval, gates the purity metrics two-sidedly on the
unfolded log-ratio scale, measures the self-loosening defect, scores all eleven
cells, reproduces `density_v2` for all eight exact-graph cells, and assesses the
R0161/R0193 exposure read-only.

`gpu_required: False`. Nothing here trains, scores a new map, or reads a GPU.
"""
from __future__ import annotations

import argparse
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
from basemap.round0225_tolerance_gate import (
    CUVS_FAMILY_SEEDS,
    EXACT_FAMILY_SEEDS,
    FAMILY_DEFINITIONS,
    GATE_CAPABILITY,
    GATE_METRICS,
    ONE_SIDED_DERIVATION,
    PURITY_METRICS,
    ROUND_ID,
    SD_DDOF,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    TWO_SIDED_DERIVATION,
)
from experiments.round0225_nodes import (
    GATE_ACTION,
    HIGH_D_REFERENCE,
    PRECEDENTS,
    R0222_GATE,
    R0223_COMPARISON,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0225"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0225-2026-08-08.md")

#: Eight cKDTree queries over 2,000,000 two-dimensional points plus the
#: arithmetic. Measured at well under a minute; allowed twenty.
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
        raise RuntimeError("R0225 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0225 round must declare its required reviews")
    return expected_input_signature(ROUND_FILE), reviews


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0225 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0225_tolerance_gate.py",
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
        "schema": "round0225-release-cpu-smoke-v1",
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
            "the derived one-sided and two-sided tolerance factors and their "
            "simulated coverage, the unfolded two-sided log-ratio band, the "
            "self-loosening measurement including the finding that the "
            "tolerance floor loosens MORE, the eleven-cell scoring, and "
            "agreement with R0222's and R0223's independently sealed floors"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0225 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0225(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0225 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0225 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)

    # Every sealed input this round reads, bound by size and hash at prepare
    # time. All are small enough to rehash cheaply except the reference npz
    # (80 MB), which is also cheap.
    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(R0222_GATE),
        expected_input_signature(R0223_COMPARISON),
        expected_input_signature(HIGH_D_REFERENCE),
        *[expected_input_signature(path) for path in PRECEDENTS.values()],
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    node = "reregister_tolerance_gate"
    jobs = [
        {
            "id": node,
            "action": GATE_ACTION,
            "handler_module": "experiments.round0225_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [os.path.join(artifacts, GATE_CAPABILITY)],
            "done_marker": os.path.join(artifacts, f"{node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": GATE_P90_WALL_S,
            "capability": GATE_CAPABILITY,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
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
        "schema": "round0225-tolerance-gate-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-analysis",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-quality-gates-n8-v1",
            "minilm-mixed-2m-cuvs-graph-map-comparison-v1",
        ],
        "capabilities_produced": [GATE_CAPABILITY],
        "training_performed": False,
        "p90_gpu_seconds": {node: 0.0, "total": 0.0},
        "jobs": jobs,
        "scientific_contract": {
            "question": (
                "re-registered on a method that is calibrated and on the right "
                "scale, which cells does the mixed-MiniLM 2M gate actually pass, "
                "and how much of R0222's 7/8 was the estimator rather than the "
                "maps?"
            ),
            "population": (
                "the 8 exact-graph cells (seeds 42-49) that define the floors, "
                "plus R0223's 3 cuVS-igd48 cells that do not"
            ),
            "cells_scored": len(EXACT_FAMILY_SEEDS) + len(CUVS_FAMILY_SEEDS),
            "cells_defining_the_floors": len(EXACT_FAMILY_SEEDS),
            "metrics": list(GATE_METRICS),
            "purity_metrics_gated_two_sidedly": list(PURITY_METRICS),
            "floor_families": FAMILY_DEFINITIONS,
            "tolerance_content": TOLERANCE_CONTENT,
            "tolerance_confidence": TOLERANCE_CONFIDENCE,
            "sample_standard_deviation_ddof": SD_DDOF,
            "one_sided_derivation": ONE_SIDED_DERIVATION,
            "two_sided_derivation": TWO_SIDED_DERIVATION,
            "factor_is_derived_not_copied": (
                "one_sided_tolerance_factor(8) is computed from the noncentral t "
                "and cross-checked against review-0222-01's published 3.187; a "
                "disagreement beyond 1e-3 aborts the node"
            ),
            "self_loosening_is_measured": (
                "for every metric and both one-sided families, by injecting a "
                "worse worst-cell and re-fitting. The round reports that the "
                "tolerance floor is ALSO self-loosening, and moves further than "
                "mean-2sigma does, because k > 2. The tolerance interval fixes "
                "calibration, not self-loosening; the structural fix is "
                "held-out calibration and is NOT claimed here."
            ),
            "density_v2_positive_control": (
                "seeds 42-45, reproduced by review-0218-01, must reproduce here "
                "before seeds 46-49 are reported. Review-0222-01's harness "
                "failed exactly this control."
            ),
            "precedents_read_only": sorted(PRECEDENTS),
            "training_performed": False,
            "evaluation_performed": False,
            "gpu_used": False,
            "gate_registerable_here": True,
            "gate_status": "registered-and-contingent-pending-review",
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
        "queue_manifest": prepare_round0225(
            release_sha=args.release_sha, queue_root=args.queue_root
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
