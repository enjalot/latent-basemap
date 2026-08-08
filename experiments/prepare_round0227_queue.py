#!/usr/bin/env python3
"""Prepare, but never launch, the R0227 low-cluster-count configuration queue.

Three GPU nodes: the reachability map against `c`, the ascending build ladder,
then recall against exact truth (sealed at 2M, computed here at 16M), loss
concentration, memory-law verification and the per-rung configuration.

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
from basemap.round0227_low_c_contract import (
    BUILD_TIMEOUT_S,
    CANDIDATE,
    CLUSTER_CAPACITY_ROWS,
    C_MIN,
    DATA_COLD_READ_BYTES_PER_S,
    DATA_READ_NOTE,
    DENSITY_DECILES,
    DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
    DEVICE_LAW_INTERCEPT_BYTES,
    DEVICE_LAW_NOTE,
    DEVICE_TOTAL_BYTES,
    DIMENSION,
    GPU_HOURS_CAP,
    GRAPH_K,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    LARGE_RECALL_ROWS,
    LARGE_RECALL_SEED,
    LARGE_RECALL_SEED_ROWS,
    LOW_C_CAPABILITY,
    PHASE2_RUNGS,
    PROJECTION_ROWS,
    R0226_A_2M_GRAPH_IDS,
    REACHABILITY_CLUSTERS,
    REACHABILITY_QUERY_SEED,
    REACHABILITY_ROWS,
    REACHABILITY_TIE_QUERY_ROWS,
    RMM_LAW_BYTES_PER_MAX_CLUSTER_ROW,
    ROUND_ID,
    SCRATCH_BUDGET_BYTES,
    SPILL,
    SUBSTRATE_16M_PATH,
    SUBSTRATE_2M_PATH,
    TRUTH_COS_PATH,
    TRUTH_IDS_PATH,
    TRUTH_RECEIPT_PATH,
    WATCHDOG_POLL_S,
    cluster_settings,
)
from experiments.round0227_nodes import (
    EVALUATE_ACTION,
    LADDER_ACTION,
    LADDER_INSTRUMENTS,
    REACHABILITY_ACTION,
    RECALL_MEAN_FLOOR,
    RECALL_P10_FLOOR,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0227"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0227-2026-08-08.md")

#: Nine k-means fits and nine similarity scans over 2,000,000 rows.
REACHABILITY_P90_WALL_S = 1_800.0
#: Nine builds, the largest with a ~9M-row cluster.
LADDER_P90_WALL_S = 3_600.0
#: Two brute-force exact-truth passes over 16,000,000 rows plus six scored graphs.
EVALUATE_P90_WALL_S = 1_800.0


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
        raise RuntimeError("R0227 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0227 round must declare its required reviews")
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
        raise RuntimeError("R0227 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0227_low_c.py",
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
        "schema": "round0227-release-cpu-smoke-v1",
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
            "the cluster-count law and its ascending-max-cluster ordering; the "
            "predictive guard in both its allow and refuse branches; the "
            "cluster-into-group packing against the scratch budget; the memory "
            "law and its agreement statistic; the smallest-feasible-c "
            "calculation; the density-decile, autocorrelation, concentration "
            "and edge-precision statistics against constructed graphs with "
            "known answers; the thread-lifecycle guard over every "
            "threading.Thread subclass this round can reach; and the "
            "phase-by-phase projection with its labelled bases"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0227 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0227(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0227 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0227 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
        expected_input_signature(TRUTH_RECEIPT_PATH),
        expected_input_signature(R0226_A_2M_GRAPH_IDS),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    reachability_output = os.path.join(artifacts, "low-c-reachability")
    reachability_manifest = os.path.join(
        reachability_output, "reachability-vs-cluster-count.json"
    )
    ladder_output = os.path.join(artifacts, LOW_C_CAPABILITY)
    ladder_manifest = os.path.join(ladder_output, "low-c-build-ladder.json")
    builds_root = os.path.join(ladder_output, "builds")
    evaluate_output = os.path.join(artifacts, "low-c-recall-and-recommendation")
    cache_root = os.path.join(ROUND_ROOT, "child-cache")
    scratch_root = os.path.join(ROUND_ROOT, "spill-scratch")

    reachability_node = "map_reachability_vs_c"
    ladder_node = "build_low_c_ladder"
    evaluate_node = "evaluate_low_c"
    jobs = [
        {
            "id": reachability_node,
            "action": REACHABILITY_ACTION,
            "handler_module": "experiments.round0227_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [reachability_output],
            "done_marker": os.path.join(artifacts, f"{reachability_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": REACHABILITY_P90_WALL_S,
            "cuvs_cache_root": cache_root,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
                "cpu_heavy": False,
            },
        },
        {
            "id": ladder_node,
            "action": LADDER_ACTION,
            "handler_module": "experiments.round0227_nodes",
            "handler_callable": "run_job",
            "deps": [reachability_node],
            "outputs": [ladder_output],
            "done_marker": os.path.join(artifacts, f"{ladder_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": LADDER_P90_WALL_S,
            "capability": LOW_C_CAPABILITY,
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
            "handler_module": "experiments.round0227_nodes",
            "handler_callable": "run_job",
            "deps": [ladder_node],
            "outputs": [evaluate_output],
            "done_marker": os.path.join(artifacts, f"{evaluate_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": EVALUATE_P90_WALL_S,
            "ladder_manifest": ladder_manifest,
            "reachability_manifest": reachability_manifest,
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
        "schema": "round0227-low-cluster-count-configuration-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-benchmark",
        "required_reviews": list(required_reviews),
        "capability_dependencies": [
            "minilm-mixed-2m-substrate-and-exact-k15-graph-v1",
            "minilm-mixed-16m-benchmark-substrate-v1",
            "minilm-100m-graph-builder-qualification-v1",
        ],
        "capabilities_produced": [LOW_C_CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            reachability_node: REACHABILITY_P90_WALL_S,
            ladder_node: LADDER_P90_WALL_S,
            evaluate_node: EVALUATE_P90_WALL_S,
            "total": (
                REACHABILITY_P90_WALL_S + LADDER_P90_WALL_S + EVALUATE_P90_WALL_S
            ),
        },
        "scientific_contract": {
            "question": (
                "candidate A's recall ceiling is set by its cluster count and "
                "~25 GiB of the card goes unused at every rung. Does spending "
                "that headroom on fewer, larger clusters buy the reachability "
                "back, and does it fix the CONCENTRATION of the loss and not "
                "merely its mean?"
            ),
            "configuration_study_not_adoption": (
                "this round varies R0226's cluster count under the same builder "
                "and the same nn-descent setting; it trains no map, registers "
                "no gate and seals no graph for downstream consumption"
            ),
            "candidate": CANDIDATE,
            "spill": SPILL,
            "cluster_floor": C_MIN,
            "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
            "reachability_cluster_counts": list(REACHABILITY_CLUSTERS),
            "reachability_rows": REACHABILITY_ROWS,
            "reachability_tie_query_rows": REACHABILITY_TIE_QUERY_ROWS,
            "reachability_query_seed": REACHABILITY_QUERY_SEED,
            "build_cells": [dict(item) for item in cluster_settings()],
            "ladder_axis": (
                "predicted max_cluster_rows ascending — the axis the memory law "
                "charges. N ascends within each fixed c. The ladder stops at "
                "its first refusal, abort, timeout or failure."
            ),
            "dimension": DIMENSION,
            "k": GRAPH_K,
            "substrates": {
                "2000000": SUBSTRATE_2M_PATH,
                "prefixes_above_2m": SUBSTRATE_16M_PATH,
            },
            "recall_truth_ids": TRUTH_IDS_PATH,
            "recall_truth_cosines": TRUTH_COS_PATH,
            "large_recall_rows": LARGE_RECALL_ROWS,
            "large_recall_seed_rows": LARGE_RECALL_SEED_ROWS,
            "large_recall_seed": LARGE_RECALL_SEED,
            "large_recall_note": (
                "exact truth at 16,000,000 rows is computed inside the "
                "evaluation node by brute force over the whole population for a "
                "seeded query set and its exact neighbours; review-0226-01 "
                "named this the highest-value follow-up in the program"
            ),
            "recall_mean_floor": RECALL_MEAN_FLOOR,
            "recall_p10_floor": RECALL_P10_FLOOR,
            "density_deciles": DENSITY_DECILES,
            "concentration_note": (
                "review-0226-01 measured A's loss as monotone in local density "
                "(0.9130 sparsest tenth against 0.9957 densest) and spatially "
                "autocorrelated at r = 0.6216 against a null near zero. A "
                "configuration whose mean recall rises while its loss stays "
                "concentrated in the sparse tail has not been fixed, because "
                "R0215 showed sparse regions are where the v1 map broke. Both "
                "statistics are measured for every emitted graph, including "
                "R0226's own c=8 graph re-scored with this round's code."
            ),
            "device_law": {
                "intercept_bytes": DEVICE_LAW_INTERCEPT_BYTES,
                "bytes_per_max_cluster_row": DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
                "rmm_bytes_per_max_cluster_row": RMM_LAW_BYTES_PER_MAX_CLUSTER_ROW,
                "note": DEVICE_LAW_NOTE,
            },
            "instruments": list(LADDER_INSTRUMENTS),
            "device_total_bytes": DEVICE_TOTAL_BYTES,
            "projection_rows": PROJECTION_ROWS,
            "phase2_rungs": list(PHASE2_RUNGS),
            "scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
            "spill_io_read_bytes_per_s": DATA_COLD_READ_BYTES_PER_S,
            "spill_io_note": DATA_READ_NOTE,
            "degree_zero_tripwire": (
                "R0215's tripwire on every emitted graph, plus its structural "
                "analogue in the reachability map: rows with zero reachable "
                "true neighbours, reported at every c"
            ),
            "guard_device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
            "guard_host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
            "guard_swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
            "guard_budget_note": GUARD_BUDGET_NOTE,
            "watchdog_poll_s": WATCHDOG_POLL_S,
            "build_timeout_s": BUILD_TIMEOUT_S,
            "refusal_is_data": (
                "a cell whose predicted footprint exceeds a budget is refused "
                "before launch; a cell whose REALISED largest cluster exceeds "
                "the registered capacity refuses itself after assignment and "
                "before any per-cluster build. Both are recorded with their "
                "predictions and never launched further."
            ),
            "never_sigkill_a_cuda_context": (
                "aborts are SIGTERM with a 180 s grace; SIGKILL is a recorded "
                "last resort and its use is an execution-check failure"
            ),
            "projection_discipline": (
                "the 100M wall is built phase by phase from measured "
                "coefficients — a per-cluster nn-descent cost curve in cluster "
                "rows, linear cosine and merge rates per spilled row, and an "
                "explicit spill-I/O term from the realised group packing and a "
                "measured /data throughput. Every term carries its fitted range "
                "and extrapolation factor. No projection is divided by another "
                "projection."
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
        "queue_manifest": prepare_round0227(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
