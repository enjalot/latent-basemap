#!/usr/bin/env python3
"""Prepare the frozen-protocol evaluation of the seed-43 diverse-Jina map."""
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
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ID = "0110"
ROUND_ROOT = "/data/latent-basemap/runs/round-0110"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0110-*.md")
R0108_QUEUE = "/data/latent-basemap/runs/round-0108/queue/queue.json"
R0108_TERMINAL = (
    "/data/latent-basemap/runs/round-0108/queue/runner-terminal.json"
)
R0108_SELECTION = (
    "/data/latent-basemap/runs/round-0108/queue/inputs/"
    "registered-selections.npz"
)
R0108_CALIBRATION_OUTPUT = (
    "/data/latent-basemap/runs/round-0108/queue/artifacts/"
    "jina-density-calibration"
)
R0108_TRANSFORM_OUTPUT = (
    "/data/latent-basemap/runs/round-0108/queue/artifacts/coordinates"
)
R0108_CORE_OUTPUT = (
    "/data/latent-basemap/runs/round-0108/queue/artifacts/core-geometry"
)
R0108_OOD_OUTPUT = "/data/latent-basemap/runs/round-0108/queue/artifacts/ood"
R0108_DECISION_OUTPUT = (
    "/data/latent-basemap/runs/round-0108/queue/artifacts/decision"
)
R0109_TRAIN_OUTPUT = (
    "/data/latent-basemap/runs/round-0109/queue/artifacts/"
    "train-diverse-jina-25m-seed43"
)


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"unterminated frontmatter: {path}")
    values: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0110 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_accepted_review(
    path: str,
    *,
    expected_sha256: str,
    round_id: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"Review {round_id} bytes changed")
    frontmatter = _frontmatter(path)
    if (
        frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
    ):
        raise RuntimeError(f"Review {round_id} is not accepted")
    return signature


def _require_clean_terminal(path: str, *, round_id: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("completed_jobs") != terminal.get("required_jobs")
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("queue_manifest_unchanged") is not True
    ):
        raise RuntimeError(f"R{round_id} terminal is not a clean success")
    return signature


def _jobs_by_action(queue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = queue.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("R0108 queue jobs are missing")
    result = {str(job.get("action")): job for job in jobs}
    required = {
        "transform_retained_map",
        "score_core_geometry",
        "score_ood",
    }
    if not required.issubset(result):
        raise RuntimeError("R0108 scorer jobs are incomplete")
    return result


def _inherited_inputs(
    job: dict[str, Any],
    *,
    round_file: str,
    reviews: list[dict[str, Any]],
    r0108_queue: dict[str, Any],
    r0108_terminal: dict[str, Any],
    r0109_train: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _dedupe([
        expected_input_signature(round_file),
        *reviews,
        r0108_queue,
        r0108_terminal,
        *job["expected_inputs"],
        *r0109_train,
    ])


def prepare_round0110(
    *,
    release_sha: str,
    r0108_review_path: str,
    r0108_review_sha256: str,
    r0109_review_path: str,
    r0109_review_sha256: str,
    r0108_queue_path: str = R0108_QUEUE,
    r0109_train_output: str = R0109_TRAIN_OUTPUT,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0110 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = [
        _require_accepted_review(
            r0108_review_path,
            expected_sha256=r0108_review_sha256,
            round_id="0108",
        ),
        _require_accepted_review(
            r0109_review_path,
            expected_sha256=r0109_review_sha256,
            round_id="0109",
        ),
    ]
    r0108_root = os.path.dirname(os.path.realpath(r0108_queue_path))
    r0108_terminal_path = os.path.join(r0108_root, "runner-terminal.json")
    r0108_selection = os.path.join(
        r0108_root, "inputs", "registered-selections.npz"
    )
    r0108_artifacts = os.path.join(r0108_root, "artifacts")
    r0108_calibration_output = os.path.join(
        r0108_artifacts, "jina-density-calibration"
    )
    r0108_core_output = os.path.join(r0108_artifacts, "core-geometry")
    r0108_ood_output = os.path.join(r0108_artifacts, "ood")
    r0108_decision_output = os.path.join(r0108_artifacts, "decision")
    r0108_terminal = _require_clean_terminal(
        r0108_terminal_path, round_id="0108"
    )
    r0108_queue_signature = expected_input_signature(r0108_queue_path)
    with open(r0108_queue_path, encoding="utf-8") as handle:
        r0108_queue = json.load(handle)
    if (
        r0108_queue.get("schema")
        != "round0108-diverse-jina-evaluation-queue-v1"
        or r0108_queue.get("round_id") != "0108"
    ):
        raise RuntimeError("R0108 queue identity changed")
    source_jobs = _jobs_by_action(r0108_queue)
    r0109_train = [
        expected_input_signature(
            os.path.join(r0109_train_output, name)
        )
        for name in (
            "train-receipt.json",
            "production-config.json",
            "model.pt",
        )
    ]
    fixed_r0108_outputs = [
        expected_input_signature(r0108_selection),
        expected_input_signature(
            os.path.join(
                r0108_calibration_output,
                "jina-density-calibration.json",
            )
        ),
        expected_input_signature(
            os.path.join(r0108_core_output, "core-geometry.json")
        ),
        expected_input_signature(
            os.path.join(r0108_ood_output, "ood-evaluation.json")
        ),
        expected_input_signature(
            os.path.join(r0108_decision_output, "atlas-decision.json")
        ),
    ]

    queue_root = create_fresh_directory(
        queue_root, label="R0110 seed-43 frozen evaluation queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    transform_output = os.path.join(artifacts, "coordinates-seed43")
    core_output = os.path.join(artifacts, "core-geometry-seed43")
    ood_output = os.path.join(artifacts, "ood-seed43")
    decision_output = os.path.join(artifacts, "two-seed-decision")

    transform_source = source_jobs["transform_retained_map"]
    core_source = source_jobs["score_core_geometry"]
    ood_source = source_jobs["score_ood"]
    jobs = [
        {
            **{
                key: value
                for key, value in transform_source.items()
                if key not in {
                    "id",
                    "action",
                    "handler_module",
                    "handler_callable",
                    "deps",
                    "outputs",
                    "done_marker",
                    "expected_inputs",
                    "p90_wall_s",
                }
            },
            "id": "transform_seed43",
            "action": "transform_seed43",
            "handler_module": "experiments.round0110_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [transform_output],
            "done_marker": os.path.join(
                artifacts, "transform_seed43.done.json"
            ),
            "expected_inputs": _inherited_inputs(
                transform_source,
                round_file=round_file,
                reviews=reviews,
                r0108_queue=r0108_queue_signature,
                r0108_terminal=r0108_terminal,
                r0109_train=r0109_train,
            ),
            "p90_wall_s": 1_800.0,
            "train_output": r0109_train_output,
            "release_sha": release_sha,
        },
        {
            **{
                key: value
                for key, value in core_source.items()
                if key not in {
                    "id",
                    "action",
                    "handler_module",
                    "handler_callable",
                    "deps",
                    "outputs",
                    "done_marker",
                    "expected_inputs",
                    "p90_wall_s",
                }
            },
            "id": "score_seed43_core",
            "action": "score_seed43_core",
            "handler_module": "experiments.round0110_nodes",
            "handler_callable": "run_job",
            "deps": ["transform_seed43"],
            "outputs": [core_output],
            "done_marker": os.path.join(
                artifacts, "score_seed43_core.done.json"
            ),
            "expected_inputs": _dedupe([
                *_inherited_inputs(
                    core_source,
                    round_file=round_file,
                    reviews=reviews,
                    r0108_queue=r0108_queue_signature,
                    r0108_terminal=r0108_terminal,
                    r0109_train=r0109_train,
                ),
                *fixed_r0108_outputs,
            ]),
            "p90_wall_s": 3_600.0,
            "train_output": r0109_train_output,
            "transform_output": transform_output,
            "calibration_output": r0108_calibration_output,
            "selection": r0108_selection,
            "release_sha": release_sha,
        },
        {
            **{
                key: value
                for key, value in ood_source.items()
                if key not in {
                    "id",
                    "action",
                    "handler_module",
                    "handler_callable",
                    "deps",
                    "outputs",
                    "done_marker",
                    "expected_inputs",
                    "p90_wall_s",
                }
            },
            "id": "score_seed43_ood",
            "action": "score_seed43_ood",
            "handler_module": "experiments.round0110_nodes",
            "handler_callable": "run_job",
            "deps": ["transform_seed43"],
            "outputs": [ood_output],
            "done_marker": os.path.join(
                artifacts, "score_seed43_ood.done.json"
            ),
            "expected_inputs": _dedupe([
                *_inherited_inputs(
                    ood_source,
                    round_file=round_file,
                    reviews=reviews,
                    r0108_queue=r0108_queue_signature,
                    r0108_terminal=r0108_terminal,
                    r0109_train=r0109_train,
                ),
                *fixed_r0108_outputs,
            ]),
            "p90_wall_s": 7_200.0,
            "train_output": r0109_train_output,
            "transform_output": transform_output,
            "selection": r0108_selection,
            "release_sha": release_sha,
        },
        {
            "id": "decide_seed_stability",
            "action": "decide_seed_stability",
            "handler_module": "experiments.round0110_nodes",
            "handler_callable": "run_job",
            "deps": ["score_seed43_core", "score_seed43_ood"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_seed_stability.done.json"
            ),
            "expected_inputs": _dedupe([
                expected_input_signature(round_file),
                *reviews,
                r0108_queue_signature,
                r0108_terminal,
                *fixed_r0108_outputs,
            ]),
            "p90_wall_s": 300.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
            "seed42_decision": os.path.join(
                r0108_decision_output, "atlas-decision.json"
            ),
            "seed42_core": os.path.join(
                r0108_core_output, "core-geometry.json"
            ),
            "seed42_ood": os.path.join(
                r0108_ood_output, "ood-evaluation.json"
            ),
            "core_output": core_output,
            "ood_output": ood_output,
            "release_sha": release_sha,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=4.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0110-diverse-jina-seed43-evaluation-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = ["0108", "0109"]
    queue["capability_dependencies"] = [
        "jina-diverse-25m-full768-trained-map-seed43-v1",
        "jina-diverse-25m-map-registry-v1",
    ]
    queue["capabilities_produced"] = [
        "jina-diverse-25m-two-seed-quality-v1"
    ]
    queue["training_performed"] = False
    queue["scientific_contract"] = {
        "role": (
            "seed-43 replay of R0108's frozen seed-42 evaluation; "
            "no outcome-conditioned selector or threshold"
        ),
        "seed42_training_round": "0107",
        "seed43_training_round": "0109",
        "seed42": 42,
        "seed43": 43,
        "selection": expected_input_signature(r0108_selection),
        "density_calibration": expected_input_signature(
            os.path.join(
                r0108_calibration_output,
                "jina-density-calibration.json",
            )
        ),
        "absolute_core_gate": "identical to R0108",
        "absolute_headline_polish_ood_gate": "identical to R0108",
        "two_seed_release_rule": (
            "both seeds independently pass the same frozen core and Polish "
            "OOD gates"
        ),
        "cross_seed_metric_deltas": "diagnostic-only",
        "projection_ffr": "diagnostic-only",
        "thresholds_tunable_after_seed42_or_seed43": False,
        "map_decision": True,
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {
        "transform_seed43": 1_800.0,
        "score_seed43_core": 3_600.0,
        "score_seed43_ood": 7_200.0,
        "total": 12_600.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0108-review", required=True)
    parser.add_argument("--r0108-review-sha256", required=True)
    parser.add_argument("--r0109-review", required=True)
    parser.add_argument("--r0109-review-sha256", required=True)
    parser.add_argument("--r0108-queue", default=R0108_QUEUE)
    parser.add_argument("--r0109-train-output", default=R0109_TRAIN_OUTPUT)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0110(
        release_sha=args.release_sha,
        r0108_review_path=args.r0108_review,
        r0108_review_sha256=args.r0108_review_sha256,
        r0109_review_path=args.r0109_review,
        r0109_review_sha256=args.r0109_review_sha256,
        r0108_queue_path=args.r0108_queue,
        r0109_train_output=args.r0109_train_output,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
