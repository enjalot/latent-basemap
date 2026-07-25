#!/usr/bin/env python3
"""Prepare, but never launch, the Round 0048 paired seed-43 queue."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0014_program import accepted_reference_records
from basemap.round0014_transform import build_transform_template
from basemap.round0034_pipeline import load_canonical_graph
from basemap.round0048_program import (
    ARMS,
    CENTROIDS_K1024_PATH,
    CENTROIDS_K256_PATH,
    ELIGIBILITY_SHA256,
    QUERIES_PATH,
    QUERY_PROVENANCE_PATH,
    REFERENCE_RECEIPT,
    ROUND_ID,
    ROW_COUNT,
    SELECTOR_PATH,
    train_configs_from_graph,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    RUN_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0048-2026-07-25.md",
)
ROUND_ROOT = "/data/latent-basemap/runs/round-0048"
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
            f"R0048 requires one issued status; observed {statuses}"
        )


def prepare_round0048(
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
    configs = train_configs_from_graph(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
    )
    config_hashes = {
        arm: value[1]
        for arm, value in configs.items()
    }
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0048 queue",
    )
    inputs_root = ensure_data_directory(
        os.path.join(queue_root, "inputs")
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    templates: dict[str, str] = {}
    for arm in ARMS:
        config, config_sha256 = configs[arm]
        template = build_transform_template(
            release_root=RUN_ROOT,
            release_sha=release_sha,
            train_output_relative_path=(
                f"artifacts/{arm}/train/model.pt"
            ),
            production_config=config,
            production_config_sha256=config_sha256,
        )
        path = os.path.join(
            inputs_root,
            f"round0048-{arm}-transform-spec-template.json",
        )
        atomic_write_new_json(path, template, immutable=True)
        templates[arm] = path

    with open(REFERENCE_RECEIPT, encoding="utf-8") as handle:
        reference_receipt = json.load(handle)
    reference_paths = [
        REFERENCE_RECEIPT,
        reference_receipt["reference"]["canonical_path"],
        reference_receipt["query_truth"]["canonical_path"],
        SELECTOR_PATH,
        QUERIES_PATH,
        QUERY_PROVENANCE_PATH,
        CENTROIDS_K256_PATH,
        CENTROIDS_K1024_PATH,
    ]
    graph_paths = [
        canonical_graph_manifest,
        graph["manifest"]["outputs"]["targets"]["canonical_path"],
        graph["manifest"]["outputs"]["degrees"]["canonical_path"],
    ]
    protocol_paths = [
        ROUND_FILE,
        os.path.join(LAB_ROOT, "review-0040-2026-07-24.md"),
        os.path.join(LAB_ROOT, "review-0041-2026-07-24.md"),
        os.path.join(LAB_ROOT, "review-0042-2026-07-25.md"),
        *templates.values(),
    ]
    inputs = _dedupe([
        *accepted_reference_records(full_hash=False),
        *_file_inputs(reference_paths),
        *_file_inputs(graph_paths),
        *_file_inputs(protocol_paths),
    ])

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=4.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0048-paired-source-exposure-queue-v1"
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0040", "0041", "0042"]
    manifest["capability_dependencies"] = [
        "duplicate-controlled-panel-v1",
        "30m-canonical-source-major-graph-v1",
        "30m-canonical-destination-map-v1",
        "30m-canonical-destination-isolation-v1",
    ]
    manifest["capabilities_produced"] = [
        "30m-seed43-paired-source-exposure-replication-v1",
    ]
    manifest["training_performed"] = True
    manifest["production_configs"] = {
        arm: configs[arm][0]
        for arm in ARMS
    }
    manifest["production_config_sha256_by_arm"] = config_hashes
    manifest["scientific_contract"] = {
        "seed": 43,
        "arms": list(ARMS),
        "only_intended_change": "positive-source exposure law",
        "source_uniform": (
            "uniform retained positive source then uniform valid "
            "canonical destination"
        ),
        "edge_uniform": (
            "uniform valid canonical edge with degree-proportional source "
            "exposure"
        ),
        "shared": [
            "30M exact fp16 feature universe",
            "R0041 canonical targets and degrees",
            "R0020 duplicate-controlled row universe",
            "canonical destination policy",
            "uniform retained nonself negatives",
            "seed43",
            "h2048 model",
            "500k successful updates",
            "bf16",
            "R0040 representative and held-out panel",
        ],
        "no_scale_claim": True,
    }
    common = {
        "handler_module": "experiments.round0048_nodes",
        "handler_callable": "run_job",
        "expected_inputs": inputs,
        "canonical_graph_manifest": canonical_graph_manifest,
        "canonical_graph_manifest_sha256": (
            canonical_graph_manifest_sha256
        ),
        "train_config_sha256_by_arm": config_hashes,
    }
    train_outputs = {
        arm: os.path.join(artifacts, arm, "train")
        for arm in ARMS
    }
    transform_outputs = {
        arm: os.path.join(artifacts, arm, "coordinates")
        for arm in ARMS
    }
    train_ids = {
        arm: f"train_seed43_{arm}_30m"
        for arm in ARMS
    }
    transform_ids = {
        arm: f"transform_seed43_{arm}_30m"
        for arm in ARMS
    }
    jobs: list[dict] = []
    for arm in ARMS:
        jobs.append({
            **common,
            "id": train_ids[arm],
            "action": "train",
            "arm": arm,
            "deps": [],
            "outputs": [train_outputs[arm]],
            "done_marker": os.path.join(
                artifacts,
                f"{train_ids[arm]}.done.json",
            ),
            "p90_wall_s": 5_100.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
            },
        })
    for arm in ARMS:
        jobs.append({
            **common,
            "id": transform_ids[arm],
            "action": "transform",
            "arm": arm,
            "deps": list(train_ids.values()),
            "outputs": [transform_outputs[arm]],
            "done_marker": os.path.join(
                artifacts,
                f"{transform_ids[arm]}.done.json",
            ),
            "p90_wall_s": 300.0,
            "train_output": train_outputs[arm],
            "transform_spec_template": templates[arm],
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        })
    panel_output = os.path.join(artifacts, "matched-panel")
    jobs.append({
        **common,
        "id": "matched_seed43_source_exposure_panel",
        "action": "matched_panel",
        "deps": list(transform_ids.values()),
        "outputs": [panel_output],
        "done_marker": os.path.join(
            artifacts,
            "matched_seed43_source_exposure_panel.done.json",
        ),
        "p90_wall_s": 700.0,
        **{
            f"{arm}_train_output": train_outputs[arm]
            for arm in ARMS
        },
        **{
            f"{arm}_transform_output": transform_outputs[arm]
            for arm in ARMS
        },
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    })
    manifest["jobs"] = jobs
    manifest["p90_gpu_seconds"] = {
        **{train_ids[arm]: 5_100.0 for arm in ARMS},
        **{transform_ids[arm]: 300.0 for arm in ARMS},
        "matched_seed43_source_exposure_panel": 700.0,
        "total": 11_500.0,
    }
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
        "queue_manifest": prepare_round0048(
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
