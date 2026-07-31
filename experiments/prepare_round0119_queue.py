#!/usr/bin/env python3
"""Prepare the no-training R0119 matched-density localization queue."""
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
from basemap.round0108_evaluation import validate_seal
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.round0119_nodes import CELL_ORDER, MATCHED_SCHEMA


ROUND_ID = "0119"
REQUIRED_REVIEWS = ("0037", "0038", "0110", "0115", "0117")
ROUND_ROOT = "/data/latent-basemap/runs/round-0119"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0119-*.md")
R0037_TRAIN = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/d768_s42/train"
)
R0038_TRAIN = (
    "/data/latent-basemap/runs/round-0038/queue/artifacts/d768_s43/train"
)
R0115_RAW_TRAIN = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/"
    "artifacts/raw/train"
)
R0117_RAW_TRAIN = (
    "/data/latent-basemap/runs/round-0117/queue/artifacts/raw/train"
)
R0107_TRAIN = (
    "/data/latent-basemap/runs/round-0107/queue/artifacts/"
    "train-diverse-jina-25m"
)
R0109_TRAIN = (
    "/data/latent-basemap/runs/round-0109/queue/artifacts/"
    "train-diverse-jina-25m-seed43"
)
R0110_QUEUE = "/data/latent-basemap/runs/round-0110/queue/queue.json"
R0110_TERMINAL = (
    "/data/latent-basemap/runs/round-0110/queue/runner-terminal.json"
)
R0110_MATCHED = (
    "/data/latent-basemap/runs/round-0110/queue/artifacts/"
    "matched-calibration-density/matched-density.json"
)
R0115_QUEUE = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/queue.json"
)
R0115_TERMINAL = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/"
    "runner-terminal.json"
)
R0117_QUEUE = "/data/latent-basemap/runs/round-0117/queue/queue.json"
R0117_TERMINAL = (
    "/data/latent-basemap/runs/round-0117/queue/runner-terminal.json"
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


def _issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0119 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_review(
    path: str,
    expected_sha256: str,
    *,
    round_id: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    frontmatter = _frontmatter(path)
    if (
        signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
    ):
        raise RuntimeError(f"R{round_id} review is not exact and accepted")
    return signature


def _clean_terminal(path: str, *, round_id: str) -> dict[str, Any]:
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


def _bundle_signature(
    *,
    key: str,
    group: str,
    root: str,
    round_id: str,
    seed: int,
    train_schema: str,
    config_receipt_schema: str,
    config_receipt_round_id: str | None,
    config_schema: str,
    training_population: str,
    training_graph: str,
    training_dose: str,
    arm: str | None = None,
    legacy_integer_key_json_roundtrip: bool = False,
) -> dict[str, Any]:
    bundle = {
        "key": key,
        "group": group,
        "round_id": round_id,
        "seed": seed,
        "train_schema": train_schema,
        "config_receipt_schema": config_receipt_schema,
        "config_receipt_round_id": config_receipt_round_id,
        "config_schema": config_schema,
        "training_population": training_population,
        "training_graph": training_graph,
        "training_dose": training_dose,
        "train_receipt": expected_input_signature(
            os.path.join(root, "train-receipt.json")
        ),
        "production_config": expected_input_signature(
            os.path.join(root, "production-config.json")
        ),
        "model": expected_input_signature(os.path.join(root, "model.pt")),
        "legacy_integer_key_json_roundtrip": (
            legacy_integer_key_json_roundtrip
        ),
    }
    if arm is not None:
        bundle["arm"] = arm
    return bundle


def _model_bundles() -> list[dict[str, Any]]:
    return [
        _bundle_signature(
            key="historical_2m_seed42",
            group="historical_2m",
            root=R0037_TRAIN,
            round_id="0037",
            seed=42,
            train_schema="round0037-train-receipt-v1",
            config_receipt_schema="round0037-production-config-receipt-v1",
            config_receipt_round_id=None,
            config_schema="round0037-d768_s42-production-config-v1",
            training_population="R0037 jina-en-2M-nested exact 2M rows",
            training_graph="R0037 fuzzy k50 graph",
            training_dose="500000 successful positive-LR updates",
            legacy_integer_key_json_roundtrip=True,
        ),
        _bundle_signature(
            key="historical_2m_seed43",
            group="historical_2m",
            root=R0038_TRAIN,
            round_id="0038",
            seed=43,
            train_schema="round0038-train-receipt-v1",
            config_receipt_schema="round0038-production-config-receipt-v1",
            config_receipt_round_id=None,
            config_schema="round0038-d768_s43-production-config-v1",
            training_population="R0037 jina-en-2M-nested exact 2M rows",
            training_graph="R0037 fuzzy k50 graph",
            training_dose="500000 successful positive-LR updates",
            legacy_integer_key_json_roundtrip=True,
        ),
        _bundle_signature(
            key="current_2m_seed42",
            group="current_2m",
            root=R0115_RAW_TRAIN,
            round_id="0115",
            seed=42,
            train_schema="round0113-train-receipt-v1",
            config_receipt_schema="round0113-production-config-v1",
            config_receipt_round_id="0115",
            config_schema="round0113-prompt-arm-train-config-v1",
            training_population=(
                "R0113 raw prompt-family-union representatives, 1993761 rows"
            ),
            training_graph="accepted R0115 raw fuzzy k50 graph",
            training_dose="500000 successful positive-LR updates",
            arm="raw",
        ),
        _bundle_signature(
            key="current_2m_seed43",
            group="current_2m",
            root=R0117_RAW_TRAIN,
            round_id="0117",
            seed=43,
            train_schema="round0113-train-receipt-v1",
            config_receipt_schema="round0113-production-config-v1",
            config_receipt_round_id="0117",
            config_schema="round0113-prompt-arm-train-config-v1",
            training_population=(
                "R0113 raw prompt-family-union representatives, 1993761 rows"
            ),
            training_graph="accepted R0115 raw fuzzy k50 graph reused",
            training_dose="500000 successful positive-LR updates",
            arm="raw",
        ),
        _bundle_signature(
            key="current_25m_seed42",
            group="current_25m",
            root=R0107_TRAIN,
            round_id="0107",
            seed=42,
            train_schema="round0107-diverse-jina-train-receipt-v1",
            config_receipt_schema="round0107-production-config-v1",
            config_receipt_round_id="0107",
            config_schema="round0107-diverse-jina-train-config-v1",
            training_population=(
                "R0106 diverse Jina exact-family representatives, 24948663 rows"
            ),
            training_graph="R0106 canonical diverse fuzzy k50 graph",
            training_dose="1459722 successful positive-LR updates",
        ),
        _bundle_signature(
            key="current_25m_seed43",
            group="current_25m",
            root=R0109_TRAIN,
            round_id="0109",
            seed=43,
            train_schema="round0109-diverse-jina-train-receipt-v1",
            config_receipt_schema="round0109-production-config-v1",
            config_receipt_round_id="0109",
            config_schema="round0109-diverse-jina-train-config-v1",
            training_population=(
                "R0106 diverse Jina exact-family representatives, 24948663 rows"
            ),
            training_graph="R0106 canonical diverse fuzzy k50 graph",
            training_dose="1459722 successful positive-LR updates",
        ),
    ]


def prepare_round0119(
    *,
    release_sha: str,
    reviews: Mapping[str, tuple[str, str]],
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0119 release SHA must be one full commit")
    round_file = _issued_round()
    if set(reviews) != set(REQUIRED_REVIEWS):
        raise RuntimeError("R0119 review set is incomplete")
    review_signatures = [
        _accepted_review(path, sha256, round_id=round_id)
        for round_id in REQUIRED_REVIEWS
        for path, sha256 in [reviews[round_id]]
    ]
    terminals = [
        _clean_terminal(R0110_TERMINAL, round_id="0110"),
        _clean_terminal(R0115_TERMINAL, round_id="0115"),
        _clean_terminal(R0117_TERMINAL, round_id="0117"),
    ]
    queue_signatures = [
        expected_input_signature(path)
        for path in (R0110_QUEUE, R0115_QUEUE, R0117_QUEUE)
    ]
    with open(R0110_MATCHED, encoding="utf-8") as handle:
        matched = json.load(handle)
    validate_seal(matched, label="R0110 matched-density receipt")
    if matched.get("schema") != MATCHED_SCHEMA:
        raise RuntimeError("R0110 matched-density schema changed")
    matched_signature = expected_input_signature(R0110_MATCHED)
    calibration_signature = dict(matched["calibration"])
    model_bundles = _model_bundles()
    if [bundle["key"] for bundle in model_bundles] != list(CELL_ORDER):
        raise RuntimeError("R0119 model-cell order changed")

    common_inputs = _dedupe([
        expected_input_signature(round_file),
        *review_signatures,
        *terminals,
        *queue_signatures,
        matched_signature,
        dict(matched["arrays"]),
        calibration_signature,
        dict(matched["census_receipt"]),
        dict(matched["source"]),
        dict(matched["representative_reference"]),
        *[
            dict(bundle[field])
            for bundle in model_bundles
            for field in ("train_receipt", "production_config", "model")
        ],
    ])
    queue_root = create_fresh_directory(
        queue_root, label="R0119 density localization queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    score_output = os.path.join(artifacts, "density-localization-panel")
    decision_output = os.path.join(
        artifacts, "density-localization-decision"
    )
    jobs = [
        {
            "id": "score_density_localization",
            "action": "score_density_localization",
            "handler_module": "experiments.round0119_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [score_output],
            "done_marker": os.path.join(
                artifacts, "score_density_localization.done.json"
            ),
            "expected_inputs": common_inputs,
            "p90_wall_s": 180.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "r0110_matched_receipt": matched_signature,
            "r0108_calibration": calibration_signature,
            "model_bundles": model_bundles,
        },
        {
            "id": "decide_density_localization",
            "action": "decide_density_localization",
            "handler_module": "experiments.round0119_nodes",
            "handler_callable": "run_job",
            "deps": ["score_density_localization"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_density_localization.done.json"
            ),
            "expected_inputs": _dedupe([
                expected_input_signature(round_file),
                *review_signatures,
                *terminals,
                *queue_signatures,
                matched_signature,
            ]),
            "p90_wall_s": 30.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
            "score_output": score_output,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.25,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0119-jina-density-localization-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": list(REQUIRED_REVIEWS),
        "capability_dependencies": [
            "jina-fineweb-2m-prompt-map-contrast-v1",
            "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
        ],
        "capabilities_produced": [
            "jina-density-failure-localization-v1",
        ],
        "training_performed": False,
        "scientific_contract": {
            "universe": (
                "exact reviewed R0110/R0040 1996279-row FineWeb "
                "representative universe"
            ),
            "anchors_and_high_d_radii": "exact reviewed R0110 10000",
            "family_filter": "exact family size <16",
            "density_floor": (
                "unchanged R0108/R0110 registered floor "
                "0.17589389755990817"
            ),
            "cells": list(CELL_ORDER),
            "historical_control_requirement": (
                "both R0037/R0038 controls reproduce frozen R0108 arrays "
                "within fixed 1e-6 absolute/relative tolerance and clear "
                "the unchanged floor"
            ),
            "localization_rule": (
                "if controls reproduce, current 2M pair clears, and current "
                "25M pair does not both clear, localize only to the bundled "
                "2M-to-25M population/graph/dose/execution transition"
            ),
            "scale_specific_rejection_rule": (
                "if either current 2M seed fails, reject a scale-specific "
                "explanation"
            ),
            "single_cause_localization": False,
            "matched_cell_can_rescue_native_quality": False,
            "production_or_prompt_transfer": False,
            "map_decision": False,
        },
        "p90_gpu_seconds": {
            "score_density_localization": 180.0,
            "total": 180.0,
        },
        "estimate_basis": {
            "measured_r0110_two_model_matched_density_wall_s": (
                13.835361725185066
            ),
            "six_model_linear_projection_s": 41.5060851755552,
            "expected_gpu_seconds": 45.0,
            "p90_multiplier_over_linear_projection": (
                180.0 / 41.5060851755552
            ),
            "hard_cap_gpu_seconds": 900.0,
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    for round_id in REQUIRED_REVIEWS:
        parser.add_argument(f"--r{round_id}-review", required=True)
        parser.add_argument(f"--r{round_id}-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    reviews = {
        round_id: (
            getattr(args, f"r{round_id}_review"),
            getattr(args, f"r{round_id}_review_sha256"),
        )
        for round_id in REQUIRED_REVIEWS
    }
    path = prepare_round0119(
        release_sha=args.release_sha,
        reviews=reviews,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
