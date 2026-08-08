#!/usr/bin/env python3
"""Prepare, but never launch, the R0226 graph-builder qualification queue.

Two GPU nodes: the ascending ladders for both candidates, then recall against
R0220's sealed exact k15 truth plus the registered 100M device verdict and the
per-rung Phase 2 recommendation.

Nothing here trains, registers a gate, or seals a graph for downstream use.
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
from basemap.round0226_graph_builders import (
    A_CLUSTER_CAPACITY_ROWS,
    A_CLUSTER_TARGET_ROWS,
    A_SCRATCH_BUDGET_BYTES,
    A_SPILL,
    B_NLIST,
    B_NPROBE,
    B_SHARD_ROWS,
    BUILD_TIMEOUT_S,
    CANDIDATES,
    DEVICE_BUDGET_INSTRUMENT,
    DEVICE_INSTRUMENT_QUANTUM_BYTES,
    DEVICE_TOTAL_BYTES,
    DIMENSION,
    FLATNESS_TOLERANCE,
    GPU_HOURS_CAP,
    GRAPH_K,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    INSTRUMENTS,
    INSTRUMENT_APPLICABILITY,
    INSTRUMENT_NOTE,
    LADDER_ROWS,
    METRIC_EQUIVALENCE,
    PHASE2_RUNGS,
    PROJECTION_ROWS,
    QUALIFICATION_CAPABILITY,
    RECALL_ROWS,
    ROUND_ID,
    SENSITIVITY_ARGUMENT,
    SUBSTRATE_16M_PATH,
    SUBSTRATE_2M_PATH,
    TRUTH_COS_PATH,
    TRUTH_IDS_PATH,
    TRUTH_RECEIPT_PATH,
    WATCHDOG_POLL_S,
    ladder_settings,
)
from experiments.round0226_nodes import (
    EVALUATE_ACTION,
    QUALIFY_ACTION,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0226"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0226-2026-08-08.md")

#: Eight ladder cells, each a fresh process. Candidate B's search cost grows
#: quadratically in N at fixed nprobe/nlist, so the top rungs dominate.
QUALIFY_P90_WALL_S = 7_200.0
#: Two 2M graphs scored against sealed truth, cosines recomputed on the device.
EVALUATE_P90_WALL_S = 900.0


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
        raise RuntimeError("R0226 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0226 round must declare its required reviews")
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
        raise RuntimeError("R0226 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0226_builders.py",
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
        timeout=180,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0226-release-cpu-smoke-v1",
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
            "the exact top-k merge including FAISS -1 sentinel exclusion, self "
            "exclusion, duplicate collapse and deterministic tie-breaks; the "
            "predictive guard and its refusal path; the ascending stop-on-failure "
            "ladder; the flatness rule; the registered 100M device verdict in "
            "both its flat and non-flat branches; the wall fit and its labelled "
            "projection; and the per-rung recommendation including the "
            "no-candidate-qualifies branch"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0226 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0226(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0226 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0226 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(TRUTH_RECEIPT_PATH),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    qualification_output = os.path.join(artifacts, QUALIFICATION_CAPABILITY)
    qualification_manifest = os.path.join(
        qualification_output, "graph-builder-qualification.json"
    )
    builds_root = os.path.join(qualification_output, "builds")
    recall_output = os.path.join(artifacts, "graph-builder-recall-and-verdict")
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")

    qualify_node = "qualify_graph_builders"
    evaluate_node = "evaluate_recall_and_verdict"
    jobs = [
        {
            "id": qualify_node,
            "action": QUALIFY_ACTION,
            "handler_module": "experiments.round0226_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [qualification_output],
            "done_marker": os.path.join(artifacts, f"{qualify_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": QUALIFY_P90_WALL_S,
            "capability": QUALIFICATION_CAPABILITY,
            "cuvs_cache_root": cache_root,
            "scratch_root": scratch_root,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": evaluate_node,
            "action": EVALUATE_ACTION,
            "handler_module": "experiments.round0226_nodes",
            "handler_callable": "run_job",
            "deps": [qualify_node],
            "outputs": [recall_output],
            "done_marker": os.path.join(artifacts, f"{evaluate_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": EVALUATE_P90_WALL_S,
            "qualification_manifest": qualification_manifest,
            "builds_root": builds_root,
            "node_policy": {
                "gpu_required": True,
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
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0226-graph-builder-qualification-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-benchmark",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-substrate-and-exact-k15-graph-v1",
            "minilm-mixed-16m-benchmark-substrate-v1",
        ],
        "capabilities_produced": [QUALIFICATION_CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            qualify_node: QUALIFY_P90_WALL_S,
            evaluate_node: EVALUATE_P90_WALL_S,
            "total": QUALIFY_P90_WALL_S + EVALUATE_P90_WALL_S,
        },
        "scientific_contract": {
            "question": (
                "what builds the 100,000,000-row k15 graph within 31.37 GiB of "
                "device memory on this box?"
            ),
            "qualification_not_adoption": (
                "this round qualifies candidate builders and recommends one per "
                "Phase 2 rung; it trains no map, registers no gate and seals no "
                "graph for downstream consumption"
            ),
            "candidates": list(CANDIDATES),
            "ladder_rows": list(LADDER_ROWS),
            "cells": len(ladder_settings()),
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrates": {
                "2000000": SUBSTRATE_2M_PATH,
                "prefixes_above_2m": SUBSTRATE_16M_PATH,
            },
            "recall_rows": RECALL_ROWS,
            "recall_truth_ids": TRUTH_IDS_PATH,
            "recall_truth_cosines": TRUTH_COS_PATH,
            "recall_mean_floor": RECALL_MEAN_FLOOR,
            "recall_p10_floor": RECALL_P10_FLOOR,
            "metric_equivalence": METRIC_EQUIVALENCE,
            "candidate_a_spill": A_SPILL,
            "candidate_a_cluster_target_rows": A_CLUSTER_TARGET_ROWS,
            "candidate_a_cluster_capacity_rows": A_CLUSTER_CAPACITY_ROWS,
            "candidate_a_scratch_budget_bytes": A_SCRATCH_BUDGET_BYTES,
            "candidate_b_shard_rows": B_SHARD_ROWS,
            "candidate_b_nlist": B_NLIST,
            "candidate_b_nprobe": B_NPROBE,
            "instruments": list(INSTRUMENTS),
            "instrument_applicability": dict(INSTRUMENT_APPLICABILITY),
            "instrument_note": INSTRUMENT_NOTE,
            "device_budget_instrument": DEVICE_BUDGET_INSTRUMENT,
            "device_instrument_quantum_bytes": DEVICE_INSTRUMENT_QUANTUM_BYTES,
            "sensitivity_argument": SENSITIVITY_ARGUMENT,
            "flatness_tolerance": FLATNESS_TOLERANCE,
            "device_total_bytes": DEVICE_TOTAL_BYTES,
            "projection_rows": PROJECTION_ROWS,
            "phase2_rungs": list(PHASE2_RUNGS),
            "degree_zero_tripwire": (
                "R0215's tripwire: a candidate that emits any edgeless row is "
                "disqualified regardless of speed"
            ),
            "guard_device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
            "guard_host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
            "guard_swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
            "guard_budget_note": GUARD_BUDGET_NOTE,
            "watchdog_poll_s": WATCHDOG_POLL_S,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "refusal_is_data": (
                "a cell whose predicted footprint exceeds a budget is recorded "
                "as refused_a_priori with its prediction and never launched"
            ),
            "never_sigkill_a_cuda_context": (
                "aborts are SIGTERM with a 180 s grace; SIGKILL is a recorded "
                "last resort and its use is an execution-check failure"
            ),
            "projection_discipline": (
                "every 100M number carries its basis. A flat measured device "
                "series is published as a plateau with no extrapolation; a "
                "non-flat one carries its fitted range and extrapolation "
                "factor. No projection is divided by another projection."
            ),
            "training_performed": False,
            "evaluation_performed": True,
            "gate_registerable_here": False,
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
        "queue_manifest": prepare_round0226(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
