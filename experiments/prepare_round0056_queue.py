#!/usr/bin/env python3
"""Prepare the review-selected seed-43 negative-BCE confirmation queue."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0014_program import accepted_reference_records
from basemap.round0014_transform import build_transform_template
from basemap.round0034_pipeline import load_canonical_graph
from basemap.round0048_program import (
    ELIGIBILITY_SHA256,
    REFERENCE_RECEIPT,
    ROW_COUNT,
    SELECTOR_PATH,
)
from basemap.round0056_program import (
    BASELINE_COORDINATES,
    BASELINE_TRAIN_RECEIPT,
    R0051_COMPARISON,
    ROUND_ID,
    selected_arm,
    train_config_from_graph,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.prepare_round0042_queue import _coordinate_inputs
from experiments.round0046_nodes import _read_sealed


ROUND_ROOT = "/data/latent-basemap/runs/round-0056"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0056-2026-07-26.md",
)
DEFAULT_GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0041/attempt-01/queue/artifacts/"
    "canonical-graph-30m/canonical-graph-v1.json"
)
DEFAULT_GRAPH_MANIFEST_SHA256 = (
    "656389789cd68196442272e2596195850c5a70b56f3acb2b7b8271a9f0b6662c"
)


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        lines = handle.readlines()
    statuses: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "status":
            statuses.append(value.strip().strip("\"'"))
    if statuses != ["issued"]:
        raise RuntimeError(
            f"R0056 requires one issued status; observed {statuses}"
        )


def prepare_round0056(
    *,
    release_sha: str,
    r0051_comparison_sha256: str,
    canonical_graph_manifest: str = DEFAULT_GRAPH_MANIFEST,
    canonical_graph_manifest_sha256: str = (
        DEFAULT_GRAPH_MANIFEST_SHA256
    ),
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    comparison_signature = expected_input_signature(R0051_COMPARISON)
    if comparison_signature["sha256"] != r0051_comparison_sha256:
        raise RuntimeError("R0051 comparison bytes changed")
    comparison = _read_sealed(
        R0051_COMPARISON,
        label="R0051 negative-BCE selection",
    )
    arm = selected_arm(comparison)
    graph = load_canonical_graph(
        canonical_graph_manifest,
        expected_sha256=canonical_graph_manifest_sha256,
        expected_eligibility_sha256=ELIGIBILITY_SHA256,
        row_count=ROW_COUNT,
    )
    config, config_sha256 = train_config_from_graph(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        arm=arm,
    )
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0056 queue",
    )
    inputs_root = ensure_data_directory(
        os.path.join(queue_root, "inputs")
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    template_path = os.path.join(
        inputs_root,
        "round0056-transform-spec-template.json",
    )
    atomic_write_new_json(
        template_path,
        build_transform_template(
            release_root=RELEASE_ROOT,
            release_sha=release_sha,
            train_output_relative_path="artifacts/train/model.pt",
            production_config=config,
            production_config_sha256=config_sha256,
        ),
        immutable=True,
    )
    with open(REFERENCE_RECEIPT, encoding="utf-8") as handle:
        reference_receipt = json.load(handle)
    inputs = _dedupe([
        *accepted_reference_records(full_hash=False),
        *_file_inputs([
            REFERENCE_RECEIPT,
            reference_receipt["reference"]["canonical_path"],
            reference_receipt["query_truth"]["canonical_path"],
            SELECTOR_PATH,
            canonical_graph_manifest,
            graph["manifest"]["outputs"]["targets"]["canonical_path"],
            graph["manifest"]["outputs"]["degrees"]["canonical_path"],
            ROUND_FILE,
            os.path.join(LAB_ROOT, "review-0040-2026-07-24.md"),
            os.path.join(LAB_ROOT, "review-0041-2026-07-24.md"),
            os.path.join(LAB_ROOT, "review-0048-2026-07-26.md"),
            os.path.join(LAB_ROOT, "review-0051-2026-07-26.md"),
            BASELINE_TRAIN_RECEIPT,
            os.path.join(
                os.path.dirname(BASELINE_TRAIN_RECEIPT),
                "production-config.json",
            ),
            R0051_COMPARISON,
            template_path,
        ]),
        *_coordinate_inputs(BASELINE_COORDINATES),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=2.25,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0056-negative-bce-seed43-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0040", "0041", "0048", "0051"]
    manifest["capability_dependencies"] = [
        "duplicate-controlled-panel-v1",
        "30m-canonical-source-major-graph-v1",
        "30m-seed43-paired-source-exposure-replication-v1",
        "30m-normalized-negative-bce-calibration-v1",
    ]
    manifest["capabilities_produced"] = [
        "30m-negative-bce-two-seed-confirmation-v1",
    ]
    manifest["training_performed"] = True
    manifest["production_config"] = config
    manifest["production_config_sha256"] = config_sha256
    manifest["scientific_contract"] = {
        "selection_source": comparison_signature,
        "selected_arm": arm,
        "negative_bce_multiplier": (
            config["optimizer"]["negative_bce_multiplier"]
        ),
        "baseline_round": "0048",
        "baseline_arm": "edge_uniform",
        "seed": 43,
        "only_intended_change": (
            "negative BCE contribution relative to positive BCE"
        ),
        "external_ood_adoption_gate_run": False,
        "selected_treatment_remains_candidate_only": True,
        "no_scale_claim": True,
    }
    train_output = os.path.join(artifacts, "train")
    transform_output = os.path.join(artifacts, "coordinates")
    panel_output = os.path.join(artifacts, "matched-panel")
    common = {
        "handler_module": "experiments.round0056_nodes",
        "handler_callable": "run_job",
        "expected_inputs": inputs,
        "canonical_graph_manifest": canonical_graph_manifest,
        "canonical_graph_manifest_sha256": (
            canonical_graph_manifest_sha256
        ),
        "r0051_comparison_sha256": r0051_comparison_sha256,
        "selected_arm": arm,
        "train_config_sha256": config_sha256,
    }
    manifest["jobs"] = [
        {
            **common,
            "id": "train_seed43_selected_negative_bce",
            "action": "train",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(
                artifacts,
                "train_seed43_selected_negative_bce.done.json",
            ),
            "p90_wall_s": 5_100.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
            },
        },
        {
            **common,
            "id": "transform_seed43_selected_negative_bce",
            "action": "transform",
            "deps": ["train_seed43_selected_negative_bce"],
            "outputs": [transform_output],
            "done_marker": os.path.join(
                artifacts,
                "transform_seed43_selected_negative_bce.done.json",
            ),
            "p90_wall_s": 300.0,
            "train_output": train_output,
            "transform_spec_template": template_path,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            **common,
            "id": "matched_seed43_negative_bce_panel",
            "action": "matched_panel",
            "deps": ["transform_seed43_selected_negative_bce"],
            "outputs": [panel_output],
            "done_marker": os.path.join(
                artifacts,
                "matched_seed43_negative_bce_panel.done.json",
            ),
            "p90_wall_s": 500.0,
            "train_output": train_output,
            "transform_output": transform_output,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
    ]
    manifest["p90_gpu_seconds"] = {
        "train_seed43_selected_negative_bce": 5_100.0,
        "transform_seed43_selected_negative_bce": 300.0,
        "matched_seed43_negative_bce_panel": 500.0,
        "total": 5_900.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0051-comparison-sha256", required=True)
    parser.add_argument(
        "--canonical-graph-manifest",
        default=DEFAULT_GRAPH_MANIFEST,
    )
    parser.add_argument(
        "--canonical-graph-manifest-sha256",
        default=DEFAULT_GRAPH_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0056(
            release_sha=args.release_sha,
            r0051_comparison_sha256=args.r0051_comparison_sha256,
            canonical_graph_manifest=args.canonical_graph_manifest,
            canonical_graph_manifest_sha256=(
                args.canonical_graph_manifest_sha256
            ),
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
