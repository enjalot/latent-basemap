#!/usr/bin/env python3
"""Prepare the content-bound R0034 graph/canary/train core queue.

The canonical graph is deliberately a pre-issuance CPU artifact.  Its exact
retained-positive-source count determines the immutable successful-update
horizon, so a single immutable queue cannot both create that count and claim
to have registered it beforehand.

This builder covers the expensive core (canary + one train).  Transform and
registered panels remain downstream finalization work and are not represented
as completed by this queue.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0034_pipeline import load_canonical_graph
from basemap.round0034_program import (
    INT8_PATH,
    SCALES_PATH,
    train_config_from_capabilities,
)
from experiments.build_round0034_graph import load_released_eligibility


RUN_ROOT = "/home/enjalot/code/latent-basemap-run"
LAB_ROOT = "/home/enjalot/code/latent-labs/basemap-100m"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0034-2026-07-22.md")


def _assert_issued_round(path: str) -> None:
    text = open(path, encoding="utf-8").read(4096)
    match = re.search(r"(?m)^status:\s*([^\s]+)\s*$", text)
    if not match or match.group(1) != "issued":
        raise RuntimeError("R0034 remains draft; refuse to materialize a runnable queue")


def _dedupe(signatures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for signature in signatures:
        path = signature["canonical_path"]
        if path not in seen:
            seen.add(path)
            output.append(signature)
    return output


def prepare(
    *,
    release_sha: str,
    eligibility_path: str,
    eligibility_sha256: str,
    canonical_graph_manifest: str,
    canonical_graph_manifest_sha256: str,
    queue_root: str = "/data/latent-basemap/runs/round-0034/queue-core",
) -> str:
    _assert_issued_round(ROUND_FILE)
    _eligibility = load_released_eligibility(
        eligibility_path, eligibility_sha256, 150_000_000
    )
    graph = load_canonical_graph(
        canonical_graph_manifest,
        expected_sha256=canonical_graph_manifest_sha256,
        expected_eligibility_sha256=eligibility_sha256,
        row_count=150_000_000,
    )
    config, config_sha256 = train_config_from_capabilities(
        graph["manifest"],
        canonical_graph_manifest_path=graph["signature"]["canonical_path"],
        canonical_graph_manifest_sha256=graph["signature"]["sha256"],
        eligibility_sha256=eligibility_sha256,
    )
    queue_root = create_fresh_directory(queue_root, label="R0034 core queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    canary_output = os.path.join(artifacts, "two-endpoint-canary")
    train_output = os.path.join(artifacts, "train-150m")

    evidence_files = [
        ROUND_FILE,
        os.path.join(LAB_ROOT, "review-0019-2026-07-19.md"),
        os.path.join(LAB_ROOT, "review-0025-2026-07-20.md"),
        os.path.join(LAB_ROOT, "review-0033-2026-07-22.md"),
        eligibility_path,
        canonical_graph_manifest,
        graph["manifest"]["outputs"]["targets"]["canonical_path"],
        graph["manifest"]["outputs"]["degrees"]["canonical_path"],
        INT8_PATH,
        SCALES_PATH,
    ]
    inputs = _dedupe([expected_input_signature(path) for path in evidence_files])

    common = {
        "handler_module": "experiments.run_round0034_node",
        "handler_callable": "run_job",
        "expected_inputs": inputs,
        "eligibility_path": eligibility_path,
        "eligibility_sha256": eligibility_sha256,
        "canonical_graph_manifest": canonical_graph_manifest,
        "canonical_graph_manifest_sha256": canonical_graph_manifest_sha256,
        "train_config_sha256": config_sha256,
        "successful_positive_lr_updates": config["optimizer"][
            "successful_positive_lr_updates"
        ],
        "batch_size": config["optimizer"]["batch_size"],
    }
    jobs = [
        {
            **common,
            "id": "two_endpoint_canary",
            "action": "canary",
            "deps": [],
            "outputs": [canary_output],
            "done_marker": os.path.join(artifacts, "two_endpoint_canary.done.json"),
            "p90_wall_s": 1800.0,
            "canary_warmup_steps": 20,
            "canary_measured_steps": 100,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            **common,
            "id": "train_seed42_150m",
            "action": "train",
            "deps": ["two_endpoint_canary"],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, "train_seed42_150m.done.json"),
            "p90_wall_s": 43_200.0,
            "canary_output": canary_output,
            "node_policy": {"gpu_required": True, "training_performed": True},
        },
    ]
    manifest = {
        "schema_version": 1,
        "schema": "round0034-core-training-queue-v1",
        "program": "basemap-100m-round-0034",
        "round_id": "0034",
        "round_sha256": sha256_file(ROUND_FILE),
        "release_sha": release_sha,
        "execution_authority": "autonomous-gpu",
        "gpu_hours_cap": 12.0,
        "queue_class": "gpu-research",
        "deadline_utc": (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=16)
        ).isoformat(timespec="seconds"),
        "repo_root": RUN_ROOT,
        "lease_path": "/data/latent-basemap/.gpu_lease",
        "scope": {
            "included": ["two-endpoint production canary", "one 150M training run"],
            "downstream_not_claimed": [
                "150M transform", "same-domain panel", "OOD card", "fixed render"
            ],
        },
        "coverage_alignment": config["execution"]["coverage_alignment"],
        "production_config": config,
        "production_config_sha256": config_sha256,
        "child_environment": {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPYCACHEPREFIX": os.path.join(queue_root, "cache", "pycache"),
        },
        "jobs": jobs,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--eligibility-sha256", required=True)
    parser.add_argument("--canonical-graph-manifest", required=True)
    parser.add_argument("--canonical-graph-manifest-sha256", required=True)
    parser.add_argument(
        "--queue-root", default="/data/latent-basemap/runs/round-0034/queue-core"
    )
    args = parser.parse_args(argv)
    print(prepare(
        release_sha=args.release_sha,
        eligibility_path=args.eligibility,
        eligibility_sha256=args.eligibility_sha256,
        canonical_graph_manifest=args.canonical_graph_manifest,
        canonical_graph_manifest_sha256=args.canonical_graph_manifest_sha256,
        queue_root=args.queue_root,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
