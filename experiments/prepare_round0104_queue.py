#!/usr/bin/env python3
"""Prepare the self-contained paired fp16/int8 Round 0104 queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0104_training import (
    ARMS,
    DIMENSION,
    GRAPH_K,
    NONINFERIORITY_RATIO,
    QUERY_ROWS,
    QUERY_START,
    ROUND_ID,
    ROWS,
    SEED,
    SUBSTRATE_MANIFEST,
    SUCCESSFUL_UPDATES,
    source_prefix_proof,
    source_segments,
    validate_substrate_manifest,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0104"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0104-*.md")

REVIEW_DEFAULTS = {
    "0037": (
        os.path.join(LAB_ROOT, "review-0037-2026-07-23.md"),
        "8192d5478c63c1e961283c398370619144bfa97828aabcecbbd56ed7fbdb39a1",
        ("capability:jina-mrl-seed42-screen-v1", "d768_s42"),
    ),
    "0038": (
        os.path.join(LAB_ROOT, "review-0038-2026-07-24.md"),
        "fdafdb50286526e6a8a491f4a281c0a95967dc8e4238d8fd270d52b28798cc78",
        ("capability:jina-mrl-two-seed-decision-v1", "reject-384d"),
    ),
    "0103": (
        os.path.join(LAB_ROOT, "review-0103-2026-07-29.md"),
        "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51",
        ("capability:jina-diverse-25m-full768-int8-substrate-v1",),
    ),
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0104 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str, *, expected_sha256: str, required_text: tuple[str, ...]
) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError(f"{path} does not release reviewed evidence")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if any(value not in text for value in required_text):
        raise RuntimeError(f"{path} lacks the required capability evidence")
    return signature


def prepare_round0104(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0104 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            path, expected_sha256=sha, required_text=required
        )
        for round_id, (path, sha, required) in REVIEW_DEFAULTS.items()
    }
    substrate = validate_substrate_manifest(verify_payloads=False)
    proof = source_prefix_proof()
    query_segments = source_segments(QUERY_START, QUERY_START + QUERY_ROWS)

    queue_root = create_fresh_directory(
        queue_root, label="R0104 self-contained paired queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe(
        [
            expected_input_signature(round_file),
            *reviews.values(),
            substrate["signature"],
            substrate["payloads"]["int8"],
            substrate["payloads"]["scales"],
            substrate["payloads"]["labels"],
            substrate["payloads"]["reconstruction_sample"],
            *[dict(item["shard"]) for item in proof["segments"]],
            *[dict(item["shard"]) for item in query_segments],
        ]
    )

    shared_output = os.path.join(artifacts, "shared")
    outputs: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        outputs[arm] = {
            "train": os.path.join(artifacts, arm, "train"),
            "transform": os.path.join(artifacts, arm, "transform"),
            "score": os.path.join(artifacts, arm, "score"),
        }
    decision_output = os.path.join(artifacts, "decision")

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=5.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0104-self-contained-paired-queue-v2"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["required_reviews"] = ["0037", "0038", "0103"]
    manifest["capability_dependencies"] = [
        "jina-mrl-seed42-screen-v1",
        "jina-mrl-two-seed-decision-v1",
        "jina-diverse-25m-full768-int8-substrate-v1",
    ]
    manifest["capabilities_produced"] = [
        "jina-full768-host-int8-training-validation-v1"
    ]
    manifest["training_performed"] = True
    manifest["scientific_contract"] = {
        "design": "self-contained paired fp16-control versus int8-treatment",
        "rows": ROWS,
        "dimension": DIMENSION,
        "seed": SEED,
        "graph_k": GRAPH_K,
        "graph_built_once_from_fp16_and_shared": True,
        "sampler_and_endpoint_schedule_shared": True,
        "successful_positive_lr_updates_per_arm": SUCCESSFUL_UPDATES,
        "query_rows": [QUERY_START, QUERY_START + QUERY_ROWS],
        "query_disjoint_from_training": True,
        "noninferiority_ratio": NONINFERIORITY_RATIO,
        "purity_not_registered": (
            "the gate slice is one FineWeb corpus and has no meaningful "
            "corpus-label purity contrast"
        ),
        "cross_round_row_equivalence_claimed": False,
        "full_run_retry_count": 0,
    }
    jobs: list[dict[str, Any]] = [
        {
            "id": "build_shared_graph_and_reference",
            "action": "build_shared",
            "handler_module": "experiments.round0104_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [shared_output],
            "done_marker": os.path.join(artifacts, "build_shared.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 3_600.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        }
    ]
    previous = "build_shared_graph_and_reference"
    for arm in ARMS:
        train_id = f"train_{arm}"
        transform_id = f"transform_{arm}"
        score_id = f"score_{arm}"
        jobs.extend(
            [
                {
                    "id": train_id,
                    "action": "train",
                    "arm": arm,
                    "handler_module": "experiments.round0104_nodes",
                    "handler_callable": "run_job",
                    "deps": [previous],
                    "shared_output": shared_output,
                    "outputs": [outputs[arm]["train"]],
                    "done_marker": os.path.join(artifacts, f"{train_id}.done.json"),
                    "expected_inputs": expected_inputs,
                    "p90_wall_s": 6_600.0,
                    "node_policy": {
                        "gpu_required": True,
                        "training_performed": True,
                    },
                },
                {
                    "id": transform_id,
                    "action": "transform",
                    "arm": arm,
                    "handler_module": "experiments.round0104_nodes",
                    "handler_callable": "run_job",
                    "deps": [train_id],
                    "shared_output": shared_output,
                    "train_output": outputs[arm]["train"],
                    "outputs": [outputs[arm]["transform"]],
                    "done_marker": os.path.join(
                        artifacts, f"{transform_id}.done.json"
                    ),
                    "expected_inputs": expected_inputs,
                    "p90_wall_s": 600.0,
                    "node_policy": {
                        "gpu_required": True,
                        "training_performed": False,
                    },
                },
                {
                    "id": score_id,
                    "action": "score",
                    "arm": arm,
                    "handler_module": "experiments.round0104_nodes",
                    "handler_callable": "run_job",
                    "deps": [transform_id],
                    "shared_output": shared_output,
                    "train_output": outputs[arm]["train"],
                    "transform_output": outputs[arm]["transform"],
                    "outputs": [outputs[arm]["score"]],
                    "done_marker": os.path.join(artifacts, f"{score_id}.done.json"),
                    "expected_inputs": expected_inputs,
                    "p90_wall_s": 1_200.0,
                    "node_policy": {
                        "gpu_required": True,
                        "training_performed": False,
                    },
                },
            ]
        )
        previous = score_id
    jobs.append(
        {
            "id": "decide_paired_noninferiority",
            "action": "decide",
            "handler_module": "experiments.round0104_nodes",
            "handler_callable": "run_job",
            "deps": ["score_fp16_control", "score_int8_treatment"],
            "score_outputs": {
                arm: outputs[arm]["score"] for arm in ARMS
            },
            "outputs": [decision_output],
            "done_marker": os.path.join(artifacts, "decision.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 60.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        }
    )
    manifest["jobs"] = jobs
    manifest["p90_gpu_seconds"] = {
        "shared_graph_and_reference": 3_600.0,
        "fp16_control_train_transform_score": 8_400.0,
        "int8_treatment_train_transform_score": 8_400.0,
        "total": 20_400.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "queue_manifest": prepare_round0104(
                    release_sha=args.release_sha, queue_root=args.queue_root
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
