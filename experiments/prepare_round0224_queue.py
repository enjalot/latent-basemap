#!/usr/bin/env python3
"""Prepare, but never launch, the R0224 cuVS memory-scaling queue.

Two nodes: a CPU assembly of the 16,000,000-row benchmark substrate, then the
GPU sweep. The assembly node is declared `gpu_required: False` because it is
disk- and CPU-bound and must not be charged against the GPU cap; the runner's
accounting honours that.

Nothing here trains, evaluates a map, or registers a gate.
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
from basemap.round0224_cuvs_memory import (
    BENCHMARK_COMPOSITION_SCALE,
    BENCHMARK_NOTE,
    BENCHMARK_ROWS,
    BENCHMARK_SELECTION_SEED,
    BENCHMARK_SHUFFLE_SEED,
    CONTROL_INSTRUMENT,
    DIMENSION,
    GPU_HOURS_CAP,
    HOST_RSS_LIMIT_GIB,
    INSTRUMENTS,
    PROJECTION_DISCIPLINE,
    PROJECTION_ROWS,
    PROJECTION_SUBSTRATE_BYTES,
    REGISTERED_DEVICE_TOTAL_BYTES,
    ROUND_ID,
    SENSITIVITY_RULE,
    SUBSTRATE_CAPABILITY,
    SWEEP_CAPABILITY,
    SWEEP_GRAPH_DEGREE,
    SWEEP_INTERMEDIATE_DEGREES,
    SWEEP_MAX_ITERATIONS,
    SWEEP_METRIC,
    SWEEP_ROWS,
    residency_probe_settings,
    sweep_settings,
)
from experiments.round0224_nodes import ASSEMBLE_ACTION, SWEEP_ACTION
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list


ROUND_ROOT = "/data/latent-basemap/runs/round-0224"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0224-2026-08-08.md")

#: The 16M assembly reads ~16,000,000 scattered rows across ~426 shards. R0216
#: took ~4 minutes for 2M; this is allowed an hour and is not GPU-charged.
ASSEMBLE_P90_WALL_S = 5_400.0
#: 12 sweep cells plus 2 residency probes, each a fresh RAPIDS process that
#: loads up to 24.6 GB from disk before building.
SWEEP_P90_WALL_S = 7_200.0


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
        raise RuntimeError("R0224 round is not issued for this release")
    reviews = _frontmatter_list(frontmatter, "required_reviews")
    if not reviews:
        raise RuntimeError("R0224 round must declare its required reviews")
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
        raise RuntimeError("R0224 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0224_cuvs_memory.py",
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
        "schema": "round0224-release-cpu-smoke-v1",
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
            "sweep matrix construction, instrument-sensitivity rule, linear and "
            "power-law fits, budget verdicts, projection labelling, the "
            "no-sensitivity abort path, and prefix-composition validation"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0224 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0224(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0224 release SHA must be one full commit")
    round_signature, required_reviews = _issued_round(release_sha)

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0224 queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    expected_inputs = _dedupe([
        round_signature,
        expected_input_signature(smoke_path),
    ])

    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    substrate_output = os.path.join(artifacts, SUBSTRATE_CAPABILITY)
    substrate_manifest = os.path.join(substrate_output, "benchmark-substrate.json")
    cache_root = os.path.join(ROUND_ROOT, "cuvs-cache")

    assemble_node = "assemble_benchmark_substrate"
    sweep_node = "sweep_cuvs_memory"
    jobs = [
        {
            "id": assemble_node,
            "action": ASSEMBLE_ACTION,
            "handler_module": "experiments.round0224_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [substrate_output],
            "done_marker": os.path.join(artifacts, f"{assemble_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": ASSEMBLE_P90_WALL_S,
            "capability": SUBSTRATE_CAPABILITY,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
                "cpu_heavy": True,
            },
        },
        {
            "id": sweep_node,
            "action": SWEEP_ACTION,
            "handler_module": "experiments.round0224_nodes",
            "handler_callable": "run_job",
            "deps": [assemble_node],
            "outputs": [os.path.join(artifacts, SWEEP_CAPABILITY)],
            "done_marker": os.path.join(artifacts, f"{sweep_node}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": SWEEP_P90_WALL_S,
            "capability": SWEEP_CAPABILITY,
            "benchmark_substrate_manifest_signature": {
                "kind": "file",
                "canonical_path": substrate_manifest,
            },
            "cuvs_cache_root": cache_root,
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
        "schema": "round0224-cuvs-memory-scaling-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-benchmark",
        "required_reviews": list(required_reviews),
        "capability_dependencies": ["cuvs-k15-graph-builder-qualification-v1"],
        "capabilities_produced": [SUBSTRATE_CAPABILITY, SWEEP_CAPABILITY],
        "training_performed": False,
        "jobs": jobs,
        "p90_gpu_seconds": {
            assemble_node: 0.0,
            sweep_node: SWEEP_P90_WALL_S,
            "total": SWEEP_P90_WALL_S,
        },
        "scientific_contract": {
            "question": (
                "which cuVS nn-descent intermediate_graph_degree settings fit a "
                "100,000,000-row build on this box, measured with an instrument "
                "that can actually see the allocation?"
            ),
            "population": (
                f"{BENCHMARK_ROWS}-row mixed MiniLM benchmark substrate, R0216's "
                f"selection law with composition scaled x{BENCHMARK_COMPOSITION_SCALE}"
            ),
            "benchmark_only": True,
            "benchmark_note": BENCHMARK_NOTE,
            "selection_seed": BENCHMARK_SELECTION_SEED,
            "row_order_seed": BENCHMARK_SHUFFLE_SEED,
            "dimension": DIMENSION,
            "matrix_rows": list(SWEEP_ROWS),
            "matrix_intermediate_degrees": list(SWEEP_INTERMEDIATE_DEGREES),
            "graph_degree_held_fixed": SWEEP_GRAPH_DEGREE,
            "max_iterations_held_fixed": SWEEP_MAX_ITERATIONS,
            "metric": SWEEP_METRIC,
            "cells": len(sweep_settings()),
            "residency_probe_cells": len(residency_probe_settings()),
            "instruments": list(INSTRUMENTS),
            "control_instrument": CONTROL_INSTRUMENT,
            "sensitivity_rule": SENSITIVITY_RULE,
            "projection_discipline": PROJECTION_DISCIPLINE,
            "projection_rows": PROJECTION_ROWS,
            "projection_substrate_bytes": PROJECTION_SUBSTRATE_BYTES,
            "registered_device_total_bytes": REGISTERED_DEVICE_TOTAL_BYTES,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "training_performed": False,
            "evaluation_performed": False,
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
        "queue_manifest": prepare_round0224(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
