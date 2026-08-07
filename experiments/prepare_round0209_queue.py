#!/usr/bin/env python3
"""Prepare, but never launch, the R0209 prompted-diverse U12 graph queue."""
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
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY as STAGING_CAPABILITY,
    MANIFEST_SCHEMA as STAGING_SCHEMA,
)
from basemap.round0209_prompted_diverse_graph import (
    CAPABILITY,
    DIMENSION,
    GRAPH_EXECUTION,
    GRAPH_K,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE,
    GRAPH_NPROBE_GRID,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_SHARD_ROWS,
    GRAPH_VECTOR_STORAGE,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    ROWS,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0169_queue import (
    STAGING_MANIFEST,
    _accepted_bundle,
    _read_sealed,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0209"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0209-2026-08-07.md")
R0168_REVIEW = os.path.join(LAB_ROOT, "review-0168-2026-08-03-01.md")
GPU_HOURS_CAP = 3.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0209 round is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0209 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0209_prompted_diverse_graph.py",
        "tests/test_round0169_prompted_diverse.py",
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
        timeout=120,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0209-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": "diverse graph law rebinding and node dispatch",
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0209 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0209(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0209 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    dependencies = [
        *_accepted_bundle("0132"),
        *_accepted_bundle("0168", review_path=R0168_REVIEW),
    ]
    staging_signature = expected_input_signature(STAGING_MANIFEST)
    staging = _read_sealed(staging_signature, label="accepted R0168 staging")
    if (
        staging.get("schema") != STAGING_SCHEMA
        or staging.get("round_id") != "0168"
        or staging.get("capability") != STAGING_CAPABILITY
        or int(staging.get("rows", -1)) != ROWS
        or int(staging.get("dimension", -1)) != DIMENSION
        or staging.get("embedding_convention") != "Document: "
        or (staging.get("population") or {}).get("polish_held_out") is not True
    ):
        raise RuntimeError("R0209 accepted staging contract changed")
    staging_inputs = [
        staging_signature,
        dict(staging["host_fp16"]),
        dict(staging["population"]["mapping"]),
        dict(staging["duplicate_control"]["arrays"]),
    ]

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(queue_root, label="R0209 GPU queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(smoke_path, _release_cpu_smoke(release_sha), immutable=True)
    expected_inputs = _dedupe([
        round_signature,
        *dependencies,
        *staging_inputs,
        expected_input_signature(smoke_path),
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    graph_output = os.path.join(artifacts, "fuzzy-k50-graph-and-reference")
    job = {
        "id": "build_graph_and_reference",
        "action": "build_graph_and_reference",
        "handler_module": "experiments.round0209_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [graph_output],
        "done_marker": os.path.join(artifacts, "graph-reference.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 9_000.0,
        "staging_manifest": staging_signature,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
            "cpu_heavy": True,
        },
    }
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
        "schema": "round0209-prompted-diverse-u12-graph-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-graph",
        "required_reviews": ["0132", "0168"],
        "capability_dependencies": [STAGING_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "jobs": [job],
        "p90_gpu_seconds": {"build_graph_and_reference": 9_000.0, "total": 9_000.0},
        "scientific_contract": {
            "population": "exact accepted R0168 12,474,331-row prompted U12 matrix",
            "graph_k": GRAPH_K,
            "graph_nlist": GRAPH_NLIST,
            "graph_nprobe": GRAPH_NPROBE,
            "qualification_nprobe_grid": list(GRAPH_NPROBE_GRID),
            "mean_recall_at_49_floor": GRAPH_MEAN_RECALL_FLOOR,
            "p10_recall_at_49_floor": GRAPH_P10_RECALL_FLOOR,
            "graph_vector_storage": GRAPH_VECTOR_STORAGE,
            "graph_execution": GRAPH_EXECUTION,
            "shard_rows_maximum": GRAPH_SHARD_ROWS,
            "host_rss_limit_gib": HOST_RSS_LIMIT_GIB,
            "training_performed": False,
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
        "graph_execution": GRAPH_EXECUTION,
        "queue_manifest": prepare_round0209(
            release_sha=args.release_sha, queue_root=args.queue_root
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
