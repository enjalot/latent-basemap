#!/usr/bin/env python3
"""Prepare, but never launch, the Round 0042 matched training queue."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

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
from basemap.round0042_program import (
    ELIGIBILITY_SHA256,
    R0021_COORDINATES,
    REFERENCE_RECEIPT,
    ROUND_ID,
    ROW_COUNT,
    SELECTOR_PATH,
    train_config_from_graph,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    RUN_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_FILE = os.path.join(
    LAB_ROOT, "round-0042-2026-07-24.md"
)
ROUND_ROOT = "/data/latent-basemap/runs/round-0042"
DEFAULT_GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0041/attempt-01/queue/artifacts/"
    "canonical-graph-30m/canonical-graph-v1.json"
)
DEFAULT_GRAPH_MANIFEST_SHA256 = (
    "656389789cd68196442272e2596195850c5a70b56f3acb2b7b8271a9f0b6662c"
)


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        head = handle.read(4096)
    statuses = [
        line.partition(":")[2].strip().strip("\"'")
        for line in head.splitlines()[1:]
        if line.partition(":")[0].strip() == "status"
    ]
    if statuses != ["issued"]:
        raise RuntimeError(
            f"R0042 requires one issued status; observed {statuses}"
        )


def _coordinate_inputs(root: str) -> list[dict[str, Any]]:
    receipt_path = os.path.join(root, "actual-transform.json")
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    ordered = receipt["stream_capability"]["capability_payload"][
        "ordered_chunks"
    ]
    paths = [
        receipt_path,
        os.path.join(root, "heldout-query-coordinates.npy"),
    ]
    paths.extend(
        os.path.join(
            root,
            f"chunk-{int(item['chunk_index']):05d}",
            "coordinates.npy",
        )
        for item in ordered
    )
    return _file_inputs(paths)


def prepare_round0042(
    *,
    release_sha: str,
    canonical_graph_manifest: str = DEFAULT_GRAPH_MANIFEST,
    canonical_graph_manifest_sha256: str = (
        DEFAULT_GRAPH_MANIFEST_SHA256
    ),
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
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
    )
    queue_root = create_fresh_directory(
        queue_root, label="Round 0042 queue"
    )
    inputs_root = ensure_data_directory(
        os.path.join(queue_root, "inputs")
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    template = build_transform_template(
        release_root=RUN_ROOT,
        release_sha=release_sha,
        train_output_relative_path="artifacts/train/model.pt",
        production_config=config,
        production_config_sha256=config_sha256,
    )
    template_path = os.path.join(
        inputs_root, "round0042-transform-spec-template.json"
    )
    atomic_write_new_json(template_path, template, immutable=True)

    with open(REFERENCE_RECEIPT, encoding="utf-8") as handle:
        reference_receipt = json.load(handle)
    reference_paths = [
        REFERENCE_RECEIPT,
        reference_receipt["reference"]["canonical_path"],
        reference_receipt["query_truth"]["canonical_path"],
        SELECTOR_PATH,
    ]
    graph_paths = [
        canonical_graph_manifest,
        graph["manifest"]["outputs"]["targets"]["canonical_path"],
        graph["manifest"]["outputs"]["degrees"]["canonical_path"],
    ]
    protocol_paths = [
        ROUND_FILE,
        os.path.join(LAB_ROOT, "review-0021-2026-07-20.md"),
        os.path.join(LAB_ROOT, "review-0040-2026-07-24.md"),
        os.path.join(LAB_ROOT, "review-0041-2026-07-24.md"),
        (
            "/data/latent-basemap/runs/round-0021/queue/artifacts/"
            "train/train-receipt.json"
        ),
        (
            "/data/latent-basemap/runs/round-0021/queue/artifacts/"
            "train/production-config.json"
        ),
        (
            "/data/latent-basemap/runs/round-0021/queue/artifacts/"
            "panel/panel.json"
        ),
        template_path,
    ]
    inputs = _dedupe([
        *accepted_reference_records(full_hash=False),
        *_file_inputs(reference_paths),
        *_file_inputs(graph_paths),
        *_file_inputs(protocol_paths),
        *_coordinate_inputs(R0021_COORDINATES),
    ])

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=2.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0042-matched-training-queue-v1"
    manifest["required_reviews"] = ["0021", "0040", "0041"]
    manifest["capability_dependencies"] = [
        "30m-input-pack-v1",
        "duplicate-controlled-panel-v1",
        "30m-canonical-source-major-graph-v1",
        "30m-sampler-semantics-audit-v1",
    ]
    manifest["capabilities_produced"] = [
        "30m-canonical-destination-map-v1",
        "30m-canonical-destination-isolation-v1",
    ]
    manifest["training_performed"] = True
    manifest["production_config"] = config
    manifest["production_config_sha256"] = config_sha256
    manifest["scientific_contract"] = {
        "control": "R0021 raw-destination source-normalized fixed-k map",
        "treatment": (
            "R0041 canonical destinations, uniform positive source then "
            "uniform valid destination"
        ),
        "matched": [
            "30M feature universe",
            "source-major IVF-PQ k15 topology",
            "seed42",
            "model h2048",
            "optimizer",
            "bf16",
            "500k successful updates",
        ],
        "mechanical_source_universe_delta": (
            "139 post-canonicalization zero-degree rows; 4.67e-6 of control"
        ),
        "primary_evaluation": "R0040 exact-vector representatives",
        "no_scale_claim": True,
    }
    common = {
        "handler_module": "experiments.round0042_nodes",
        "handler_callable": "run_job",
        "expected_inputs": inputs,
        "canonical_graph_manifest": canonical_graph_manifest,
        "canonical_graph_manifest_sha256": (
            canonical_graph_manifest_sha256
        ),
        "train_config_sha256": config_sha256,
    }
    train_output = os.path.join(artifacts, "train")
    transform_output = os.path.join(artifacts, "coordinates")
    panel_output = os.path.join(artifacts, "matched-panel")
    manifest["jobs"] = [
        {
            **common,
            "id": "train_seed42_canonical_30m",
            "action": "train",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(
                artifacts, "train_seed42_canonical_30m.done.json"
            ),
            "p90_wall_s": 5_100.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
            },
        },
        {
            **common,
            "id": "transform_canonical_30m",
            "action": "transform",
            "deps": ["train_seed42_canonical_30m"],
            "outputs": [transform_output],
            "done_marker": os.path.join(
                artifacts, "transform_canonical_30m.done.json"
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
            "id": "matched_representative_panel",
            "action": "matched_panel",
            "deps": ["transform_canonical_30m"],
            "outputs": [panel_output],
            "done_marker": os.path.join(
                artifacts, "matched_representative_panel.done.json"
            ),
            "p90_wall_s": 600.0,
            "train_output": train_output,
            "transform_output": transform_output,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
    ]
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
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
        "queue_manifest": prepare_round0042(
            release_sha=args.release_sha,
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
