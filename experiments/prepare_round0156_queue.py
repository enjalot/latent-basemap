#!/usr/bin/env python3
"""Prepare, but never launch, the R0156 12.5M scale-native treatment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import time
from typing import Any, Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.round0156_scale_rescue import (
    CAPABILITY,
    DECISION_SCHEMA,
    DENSITY_FLOOR,
    FUNCTIONAL_SCHEMA,
    GRAPH_DEGREE,
    GRAPH_K,
    GRAPH_PART_SCHEMA,
    GRAPH_SCHEMA,
    GRAPH_SHARD_SCHEMA,
    INDEX_SCHEMA,
    NATIVE_SCHEMA,
    N_NEIGHBORS,
    OOD_RETENTION,
    OOD_SCHEMA,
    PARENT_CAPABILITY,
    PARENT_ROUND_ID,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    SEED,
    SUBSET_SCHEMA,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    UPDATE_RULE,
)
from experiments import prepare_round0152_queue as base


ROUND_ROOT = "/data/latent-basemap/runs/round-0156"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(base.LAB_ROOT, "round-0156-*.md")
PARENT_OUTPUT = os.path.join(
    "/data/latent-basemap/runs/round-0155/queue/artifacts", PARENT_CAPABILITY
)
PARENT_CENSUS = os.path.join(PARENT_OUTPUT, "census.json")
PARENT_MAPPING = os.path.join(PARENT_OUTPUT, "compact-to-global.i64.npy")
PARENT_GROUP_IDS = os.path.join(PARENT_OUTPUT, "compact-group-ids.u8.npy")

GPU_HOURS_MINIMUM = 2.10
GPU_HOURS_EXPECTED = 2.55
GPU_HOURS_P90 = 3.45
GPU_HOURS_MAXIMUM = 5.0
P90_GPU_SECONDS = {
    "build_search_index": 240.0,
    "qualify_fixed_search": 300.0,
    "graph_part": 700.0,
    "train_map": 8_000.0,
    "transform_map": 300.0,
    "score_matched_native": 600.0,
    "score_matched_ood": 430.0,
    "score_functional_density": 450.0,
}

REVIEW_CAPABILITIES = {
    "0087": "jina-diverse-25m-inventory-v1",
    "0103": "jina-diverse-25m-full768-int8-substrate-v1",
    "0105": "jina-diverse-25m-full768-search-qualified-v1",
    "0106": "jina-diverse-25m-full768-fuzzy-graph-v1",
    "0107": "jina-diverse-25m-full768-trained-map-seed42-v1",
    "0108": "jina-diverse-25m-map-registry-v1",
    "0119": "jina-density-failure-localization-v1",
    "0132": "jina-diverse-12p5m-25m-scale-policy-geometry-v1",
    "0140": "jina-2m-subsystem-bisection-v1",
    "0155": PARENT_CAPABILITY,
}


def _pytest_receipt(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0156 run checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0156_scale_rescue.py",
        "tests/test_round0152_scale_rescue.py",
        "tests/test_round0155_scale_census.py",
        "tests/test_round0151_scale_census.py",
        "tests/test_round0132_scale_bridge.py::test_cpu_train_seal_reload_transform_panel_smoke",
        "tests/test_round0132_scale_bridge.py::test_actual_r0132_train_contract_seals_reloads_and_scores_on_cpu",
        "tests/test_round0107_training.py",
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
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
    receipt = base.seal({
        "schema": "round0156-release-pytest-and-cpu-path-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0156 release smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


@contextmanager
def _configured() -> Iterator[None]:
    values = {
        "ROUND_ROOT": ROUND_ROOT,
        "RELEASE_ROOT": RELEASE_ROOT,
        "ROUND_FILE_GLOB": ROUND_FILE_GLOB,
        "HANDLER_MODULE": "experiments.round0156_nodes",
        "QUEUE_SCHEMA": "round0156-historical-prefix-rescue-queue-v1",
        "PARENT_ROUND_ID": PARENT_ROUND_ID,
        "R0151_OUTPUT": PARENT_OUTPUT,
        "R0151_CENSUS": PARENT_CENSUS,
        "R0151_MAPPING": PARENT_MAPPING,
        "R0151_GROUP_IDS": PARENT_GROUP_IDS,
        "R0151_CAPABILITY": PARENT_CAPABILITY,
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "RETAINED_ROWS": RETAINED_ROWS,
        "GRAPH_K": GRAPH_K,
        "N_NEIGHBORS": N_NEIGHBORS,
        "SEED": SEED,
        "SUBSET_SCHEMA": SUBSET_SCHEMA,
        "INDEX_SCHEMA": INDEX_SCHEMA,
        "QUALIFICATION_SCHEMA": QUALIFICATION_SCHEMA,
        "GRAPH_SHARD_SCHEMA": GRAPH_SHARD_SCHEMA,
        "GRAPH_PART_SCHEMA": GRAPH_PART_SCHEMA,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "TRAIN_CONFIG_SCHEMA": TRAIN_CONFIG_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "TRAIN_RECEIPT_SCHEMA": TRAIN_RECEIPT_SCHEMA,
        "NATIVE_SCHEMA": NATIVE_SCHEMA,
        "OOD_SCHEMA": OOD_SCHEMA,
        "FUNCTIONAL_SCHEMA": FUNCTIONAL_SCHEMA,
        "DECISION_SCHEMA": DECISION_SCHEMA,
        "PIPELINE": PIPELINE,
        "PIPELINE_SCHEMA": PIPELINE_SCHEMA,
        "POSITIVE_DESTINATION_POLICY": POSITIVE_DESTINATION_POLICY,
        "UPDATE_RULE": UPDATE_RULE,
        "GRAPH_DEGREE": GRAPH_DEGREE,
        "DENSITY_FLOOR": DENSITY_FLOOR,
        "OOD_RETENTION": OOD_RETENTION,
        "GPU_HOURS_MINIMUM": GPU_HOURS_MINIMUM,
        "GPU_HOURS_EXPECTED": GPU_HOURS_EXPECTED,
        "GPU_HOURS_P90": GPU_HOURS_P90,
        "GPU_HOURS_MAXIMUM": GPU_HOURS_MAXIMUM,
        "P90_GPU_SECONDS": P90_GPU_SECONDS,
        "REVIEW_CAPABILITIES": REVIEW_CAPABILITIES,
        "SCIENTIFIC_QUESTION": (
            "does the R0153 density-restoring historical-prefix population "
            "transfer under the scale-native 12.5M recipe?"
        ),
        "SCIENTIFIC_ESTIMAND": (
            "historical-prefix population under the same k15 scale graph and "
            "coverage-aligned dose as R0132; not a unique duplicate cause"
        ),
        "POSITIVE_BRANCH": (
            "release a reviewed shippable-v1 candidate for a separate immutable "
            "registry-promotion decision"
        ),
    }
    values["_pytest_receipt"] = _pytest_receipt
    snapshots = {
        name: (hasattr(base, name), getattr(base, name, None)) for name in values
    }
    try:
        for name, value in values.items():
            setattr(base, name, value)
        yield
    finally:
        for name, (present, value) in snapshots.items():
            if present:
                setattr(base, name, value)
            else:
                delattr(base, name)


def prepare_round0156(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    with _configured():
        return base.prepare_round0152(release_sha=release_sha, queue_root=queue_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0156(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
