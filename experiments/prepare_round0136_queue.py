#!/usr/bin/env python3
"""Materialize, but never launch, conditional R0136 density-v3."""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0108_evaluation import validate_seal
from basemap.round0136_density_v3 import (
    ATLAS_CAPABILITY,
    DENSITY_CAPABILITY,
    REPLAY_CELLS,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe


ROUND_ROOT = "/data/latent-basemap/runs/round-0136"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0136-*.md")

R0108_CALIBRATION = (
    "/data/latent-basemap/runs/round-0108/queue-attempt-3/artifacts/"
    "jina-density-calibration/jina-density-calibration.json"
)
R0119_QUEUE = "/data/latent-basemap/runs/round-0119/queue/queue.json"
R0119_RECEIPT = (
    "/data/latent-basemap/runs/round-0119/queue/artifacts/"
    "density-localization-panel/density-localization-panel.json"
)
R0119_ARRAYS = (
    "/data/latent-basemap/runs/round-0119/queue/artifacts/"
    "density-localization-panel/density-localization-arrays.npz"
)
R0122_RECEIPT = (
    "/data/latent-basemap/runs/round-0122/queue/artifacts/"
    "density-provenance-bridge/density-bridge-panel.json"
)
R0122_ARRAYS = (
    "/data/latent-basemap/runs/round-0122/queue/artifacts/"
    "density-provenance-bridge/density-bridge-arrays.npz"
)
R0118_RECEIPT = (
    "/data/latent-basemap/runs/round-0118/queue/artifacts/"
    "matched-calibration-density-seed44/matched-density.json"
)
R0118_ARRAYS = (
    "/data/latent-basemap/runs/round-0118/queue/artifacts/"
    "matched-calibration-density-seed44/matched-density-arrays.npz"
)
R0110_DECISION = (
    "/data/latent-basemap/runs/round-0110/queue/artifacts/"
    "two-seed-decision/two-seed-decision.json"
)
R0118_DECISION = (
    "/data/latent-basemap/runs/round-0118/queue/artifacts/"
    "three-seed-decision/three-seed-decision.json"
)
R0134_DECISION = (
    "/data/latent-basemap/runs/round-0134/queue/artifacts/decision/decision.json"
)

REVIEW_CAPABILITIES = {
    "0107": "jina-diverse-25m-full768-trained-map-seed42-v1",
    "0109": "jina-diverse-25m-full768-trained-map-seed43-v1",
    "0111": "jina-diverse-25m-full768-trained-map-seed44-v1",
    "0118": "jina-diverse-25m-seed44-map-registry-v1",
    "0119": "jina-density-failure-localization-v1",
    "0122": "jina-density-provenance-representation-bridge-v1",
    "0134": "jina-density-functional-showdown-v1",
}

GPU_HOURS_MINIMUM = 0.01
GPU_HOURS_EXPECTED = 0.04
GPU_HOURS_P90 = 0.10
GPU_HOURS_MAXIMUM = 0.20


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"frontmatter missing: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"frontmatter unterminated: {path}")
    output: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            output[key.strip()] = value.strip().strip("\"'")
    return output


def _frontmatter_list(frontmatter: Mapping[str, str], key: str) -> list[str]:
    value = json.loads(frontmatter.get(key) or "[]")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(f"frontmatter {key} is malformed")
    return value


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"R0136 requires exactly one issued round; found {len(candidates)}")
    if _frontmatter(candidates[0]).get("base_commit") != release_sha:
        raise RuntimeError("R0136 issued base_commit differs from release")
    return candidates[0], expected_input_signature(candidates[0])


def _accepted_review(round_id: str, capability: str) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for review_path in sorted(glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))):
        frontmatter = _frontmatter(review_path)
        if (
            frontmatter.get("round_id") != round_id
            or frontmatter.get("status") != "accepted"
            or f"capability:{capability}" not in _frontmatter_list(frontmatter, "releases")
        ):
            continue
        result_path = os.path.join(LAB_ROOT, frontmatter.get("result") or "")
        round_path = os.path.join(LAB_ROOT, frontmatter.get("round") or "")
        result = expected_input_signature(result_path)
        issued = expected_input_signature(round_path)
        if (
            result["sha256"] != frontmatter.get("result_sha256")
            or issued["sha256"] != frontmatter.get("round_sha256")
            or _frontmatter(result_path).get("release_commit")
            != frontmatter.get("verified_release_commit")
        ):
            raise RuntimeError(f"Review {round_id} result/round binding changed")
        accepted.append(
            {
                "round": issued,
                "result": result,
                "review": expected_input_signature(review_path),
            }
        )
    if len(accepted) != 1:
        raise RuntimeError(f"R0136 requires one accepted Review {round_id}; found {len(accepted)}")
    value = accepted[0]
    return [value["round"], value["result"], value["review"]]


def _require_positive_r0134() -> dict[str, Any]:
    signature = expected_input_signature(R0134_DECISION)
    decision = _read_json(R0134_DECISION)
    validate_seal(decision, label="R0134 functional showdown decision")
    if (
        decision.get("round_id") != "0134"
        or decision.get("outcome") != "current-recipe-functionally-noninferior"
        or decision.get("density_v3_calibration_authorized") is not True
    ):
        raise RuntimeError("R0134 does not activate density-v3 calibration")
    return signature


def _embedded_signatures(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if set(("canonical_path", "kind", "bytes", "sha256")) <= set(value):
            signature = expected_input_signature(str(value["canonical_path"]))
            if signature != dict(value):
                raise RuntimeError(f"embedded artifact changed: {value['canonical_path']}")
            output.append(signature)
        else:
            for child in value.values():
                output.extend(_embedded_signatures(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_embedded_signatures(child))
    return output


def _model_bundles() -> list[dict[str, Any]]:
    queue = _read_json(R0119_QUEUE)
    score_jobs = [job for job in queue["jobs"] if job.get("action") == "score_density_localization"]
    if len(score_jobs) != 1:
        raise RuntimeError("R0119 score job changed")
    source = {
        int(spec["seed"]): copy.deepcopy(spec)
        for spec in score_jobs[0]["model_bundles"]
        if spec.get("group") == "current_25m"
    }
    if set(source) != {42, 43}:
        raise RuntimeError("R0119 25M model pair changed")
    source[42]["key"] = REPLAY_CELLS[0]
    source[43]["key"] = REPLAY_CELLS[1]

    seed44 = copy.deepcopy(source[43])
    seed44.update(
        {
            "key": REPLAY_CELLS[2],
            "seed": 44,
            "round_id": "0111",
            "config_receipt_round_id": "0111",
            "config_receipt_schema": "round0111-production-config-v1",
            "config_schema": "round0111-diverse-jina-train-config-v1",
            "train_schema": "round0111-diverse-jina-train-receipt-v1",
            "reviewed_capability": "capability:jina-diverse-25m-full768-trained-map-seed44-v1",
            "accepted_result": expected_input_signature(
                os.path.join(LAB_ROOT, "result-0111-2026-07-31.md")
            ),
            "accepted_review": expected_input_signature(
                os.path.join(LAB_ROOT, "review-0111-2026-07-31.md")
            ),
            "model": expected_input_signature(
                "/data/latent-basemap/runs/round-0111/queue/artifacts/"
                "train-diverse-jina-25m-seed44/model.pt"
            ),
            "production_config": expected_input_signature(
                "/data/latent-basemap/runs/round-0111/queue/artifacts/"
                "train-diverse-jina-25m-seed44/production-config.json"
            ),
            "train_receipt": expected_input_signature(
                "/data/latent-basemap/runs/round-0111/queue/artifacts/"
                "train-diverse-jina-25m-seed44/train-receipt.json"
            ),
        }
    )
    return [source[42], source[43], seed44]


def _calibration_inputs() -> dict[str, dict[str, Any]]:
    return {
        "r0122": {
            "receipt": expected_input_signature(R0122_RECEIPT),
            "arrays": expected_input_signature(R0122_ARRAYS),
        },
        "r0119": {
            "receipt": expected_input_signature(R0119_RECEIPT),
            "arrays": expected_input_signature(R0119_ARRAYS),
        },
        "r0118": {
            "receipt": expected_input_signature(R0118_RECEIPT),
            "arrays": expected_input_signature(R0118_ARRAYS),
        },
    }


def prepare_round0136(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0136 release SHA must be one full commit")
    round_path, round_signature = _issued_round(release_sha)
    reviews = _dedupe(
        [
            signature
            for round_id, capability in REVIEW_CAPABILITIES.items()
            for signature in _accepted_review(round_id, capability)
        ]
    )
    r0134_decision = _require_positive_r0134()
    calibration_sources = _calibration_inputs()
    model_bundles = _model_bundles()
    calibration = _read_json(R0108_CALIBRATION)
    validate_seal(calibration, label="R0108 density calibration")
    replay_inputs = _dedupe(
        [
            expected_input_signature(R0108_CALIBRATION),
            *_embedded_signatures(calibration),
            expected_input_signature(R0119_QUEUE),
            *_embedded_signatures(model_bundles),
        ]
    )
    r0110_decision = expected_input_signature(R0110_DECISION)
    r0118_decision = expected_input_signature(R0118_DECISION)
    common = _dedupe(
        [
            round_signature,
            *reviews,
            r0134_decision,
            *[value for source in calibration_sources.values() for value in source.values()],
            *replay_inputs,
            r0110_decision,
            r0118_decision,
        ]
    )

    queue_root = create_fresh_directory(queue_root, label="R0136 density-v3 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    calibration_output = os.path.join(artifacts, "density-v3-calibration")
    replay_output = os.path.join(artifacts, "three-seed-density-replay")
    decision_output = os.path.join(artifacts, "density-v3-decision")
    jobs = [
        {
            "id": "calibrate_density_v3",
            "action": "calibrate_density_v3",
            "handler_module": "experiments.round0136_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [calibration_output],
            "done_marker": os.path.join(artifacts, "calibrate_density_v3.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 90.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
            "calibration_sources": calibration_sources,
        },
        {
            "id": "replay_three_seed_density_v3",
            "action": "replay_three_seed_density_v3",
            "handler_module": "experiments.round0136_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [replay_output],
            "done_marker": os.path.join(
                artifacts, "replay_three_seed_density_v3.done.json"
            ),
            "expected_inputs": common,
            "p90_wall_s": 360.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "r0108_calibration": expected_input_signature(R0108_CALIBRATION),
            "model_bundles": model_bundles,
        },
        {
            "id": "decide_density_v3",
            "action": "decide_density_v3",
            "handler_module": "experiments.round0136_nodes",
            "handler_callable": "run_job",
            "deps": ["calibrate_density_v3", "replay_three_seed_density_v3"],
            "outputs": [decision_output],
            "done_marker": os.path.join(artifacts, "decide_density_v3.done.json"),
            "expected_inputs": common,
            "p90_wall_s": 30.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
            "calibration_output": calibration_output,
            "replay_output": replay_output,
            "r0110_decision": r0110_decision,
            "r0118_decision": r0118_decision,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_path,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update(
        {
            "schema": "round0136-density-v3-queue-v1",
            "repo_root": RELEASE_ROOT,
            "queue_class": "gpu-research",
            "required_reviews": list(REVIEW_CAPABILITIES),
            "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
            "capabilities_produced": [DENSITY_CAPABILITY, ATLAS_CAPABILITY],
            "training_performed": False,
            "gpu_hours": {
                "minimum": GPU_HOURS_MINIMUM,
                "expected": GPU_HOURS_EXPECTED,
                "p90": GPU_HOURS_P90,
                "maximum": GPU_HOURS_MAXIMUM,
            },
            "jobs": jobs,
        }
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args()
    print(prepare_round0136(release_sha=args.release_sha, queue_root=args.queue_root))


if __name__ == "__main__":
    main()
