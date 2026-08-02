#!/usr/bin/env python3
"""Prepare, but never launch, the activated R0155 CPU census."""
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
from basemap.round0105_search import ELIGIBILITY_PATH
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0132_scale_bridge import SCALE_POLICY_CAPABILITY
from basemap.round0153_density_forensics import CAPABILITY as R0153_CAPABILITY
from basemap.round0155_scale_census import (
    CAPABILITY,
    EXPECTED_DROPPED_ROWS,
    EXPECTED_GROUP_IDS_ORDERED_SHA256,
    EXPECTED_MAPPING_ORDERED_SHA256,
    EXPECTED_RETAINED_ROWS,
    EXPECTED_U12_OVERLAP,
    FULL_RAW_ROWS,
    RAW_PREFIX_TARGET,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter


ROUND_ROOT = "/data/latent-basemap/runs/round-0155"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0155-2026-08-02.md")
INVENTORY = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-inventory-v1.json"
)
U12_ROOT = "/data/latent-basemap/runs/round-0132/queue/artifacts/half-subset"
U12_MANIFEST = os.path.join(U12_ROOT, "subset-manifest.json")
U12_MAPPING = os.path.join(U12_ROOT, "compact-to-global.i64.npy")
R0153_EVIDENCE = (
    "/data/latent-basemap/runs/round-0153/queue/artifacts/"
    f"{R0153_CAPABILITY}/density-forensics.json"
)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _read_sealed(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    value = _read_json(path)
    validate_seal(value, label=label)
    return value, signature


def _issued_round(release_sha: str) -> dict[str, Any]:
    if _frontmatter(ROUND_FILE).get("status") != "issued":
        raise RuntimeError("R0155 round is not issued")
    if _frontmatter(ROUND_FILE).get("base_commit") != release_sha:
        raise RuntimeError("R0155 issued base_commit differs from release")
    return expected_input_signature(ROUND_FILE)


def _accepted_activation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviews = [
        *_accepted_review("0087", "jina-diverse-25m-inventory-v1"),
        *_accepted_review("0132", SCALE_POLICY_CAPABILITY),
        *_accepted_review("0153", R0153_CAPABILITY),
    ]
    evidence, evidence_signature = _read_sealed(
        R0153_EVIDENCE, label="accepted R0153 density evidence"
    )
    decision = evidence.get("decision")
    if (
        evidence.get("round_id") != "0153"
        or evidence.get("capability") != R0153_CAPABILITY
        or not isinstance(decision, dict)
        or decision.get("outcome") != "density-restores-with-row-universe"
        or decision.get("track_f_activated") is not True
        or decision.get("floor_changed") is not False
    ):
        raise RuntimeError("R0155 positive R0153 activation is absent")
    return reviews, evidence_signature


def _pytest_receipt(*, release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0155 CPU checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0155_scale_census.py",
        "tests/test_round0151_scale_census.py",
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
    receipt = seal({
        "schema": "round0155-release-pytest-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0155 release pytest failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def prepare_round0155(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0155 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    reviews, activation_signature = _accepted_activation()
    inventory, inventory_signature = _read_sealed(
        INVENTORY, label="accepted R0087 inventory"
    )
    u12, u12_signature = _read_sealed(
        U12_MANIFEST, label="accepted R0132 U12"
    )
    eligibility_signature = expected_input_signature(ELIGIBILITY_PATH)
    u12_mapping_signature = expected_input_signature(U12_MAPPING)
    if (
        inventory.get("duplicate_control", {}).get("eligibility")
        != eligibility_signature
        or u12.get("mapping") != u12_mapping_signature
    ):
        raise RuntimeError("R0155 parent selection bindings changed")

    queue_root = create_fresh_directory(
        queue_root, label="R0155 historical-prefix census queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    pytest_path = os.path.join(preflight, "release-pytest.json")
    atomic_write_new_json(
        pytest_path, _pytest_receipt(release_sha=release_sha), immutable=True
    )
    pytest_signature = expected_input_signature(pytest_path)
    inputs = _dedupe([
        round_signature,
        *reviews,
        activation_signature,
        inventory_signature,
        eligibility_signature,
        u12_signature,
        u12_mapping_signature,
        pytest_signature,
    ])
    output = os.path.join(artifacts, CAPABILITY)
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
        "schema": "round0155-historical-prefix-census-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "cpu-research-parallel",
        "required_reviews": ["0087", "0132", "0153"],
        "capability_dependencies": [
            "jina-diverse-25m-inventory-v1",
            SCALE_POLICY_CAPABILITY,
            R0153_CAPABILITY,
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": False,
        "p90_gpu_seconds": {"total": 0.0},
        "scientific_contract": {
            "question": "what exact 12.5M historical-prefix/drop-only population is activated by R0153 density restoration?",
            "raw_population_rows": FULL_RAW_ROWS,
            "raw_prefix_target": RAW_PREFIX_TARGET,
            "expected_retained_rows": EXPECTED_RETAINED_ROWS,
            "expected_dropped_rows": EXPECTED_DROPPED_ROWS,
            "expected_u12_overlap": EXPECTED_U12_OVERLAP,
            "expected_mapping_ordered_sha256": EXPECTED_MAPPING_ORDERED_SHA256,
            "expected_group_ids_ordered_sha256": EXPECTED_GROUP_IDS_ORDERED_SHA256,
            "allocation": "integer-largest-remainder across the 22 raw R0087 groups",
            "within_group": "raw global-row prefix then R0087 exclusions without replacement",
            "must_differ_from_r0132_u12": True,
            "no_graph": True,
            "no_training": True,
            "no_map_outcomes": True,
            "release_pytest": pytest_signature,
        },
        "jobs": [{
            "id": "build_historical_prefix_census",
            "action": "build_historical_prefix_census",
            "r0153_evidence": activation_signature,
            "inventory": inventory_signature,
            "eligibility": eligibility_signature,
            "u12_manifest": u12_signature,
            "u12_mapping": u12_mapping_signature,
            "handler_module": "experiments.round0155_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [output],
            "done_marker": os.path.join(
                artifacts, "build-historical-prefix-census.done.json"
            ),
            "expected_inputs": inputs,
            "p90_wall_s": 120.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
        }],
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0155(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
