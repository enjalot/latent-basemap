#!/usr/bin/env python3
"""Prepare, but never launch, the balanced-60M R0052 train queue."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0034_pipeline import load_canonical_graph
from basemap.round0049_program import (
    ROW_COUNT,
    validate_substrate_manifest,
)
from basemap.round0052_program import (
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0052"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0052"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0052-2026-07-26.md",
)
SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "balanced-60m-substrate/balanced-60m-substrate-v1.json"
)
GRAPH_MANIFEST = (
    "/data/latent-basemap/runs/round-0050/queue/artifacts/"
    "native-graph-60m/canonical-graph-v1.json"
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
            f"R0052 requires one issued status; observed {statuses}"
        )


def prepare_round0052(
    *,
    release_sha: str,
    substrate_manifest_sha256: str,
    canonical_graph_manifest_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    substrate = validate_substrate_manifest(
        SUBSTRATE_MANIFEST,
        expected_sha256=substrate_manifest_sha256,
    )
    outputs = substrate["manifest"]["outputs"]
    load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    graph = load_canonical_graph(
        GRAPH_MANIFEST,
        expected_sha256=canonical_graph_manifest_sha256,
        expected_eligibility_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    config, config_sha256 = train_config_from_capabilities(
        graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        substrate_manifest=substrate["manifest"],
        substrate_manifest_path=substrate["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate["signature"]["sha256"],
    )
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0052 queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(artifacts, "train-balanced-60m")
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            os.path.join(LAB_ROOT, "review-0046-2026-07-25.md"),
            os.path.join(LAB_ROOT, "review-0049-2026-07-26.md"),
            os.path.join(LAB_ROOT, "review-0050-2026-07-26.md"),
            SUBSTRATE_MANIFEST,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            GRAPH_MANIFEST,
            graph["manifest"]["outputs"]["targets"]["canonical_path"],
            graph["manifest"]["outputs"]["degrees"]["canonical_path"],
        ]),
    ])

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=3.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0052-balanced-60m-train-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0046", "0049", "0050"]
    manifest["capability_dependencies"] = [
        "30m-canonical-source-exposure-isolation-v1",
        "minilm-balanced-60m-input-v1",
        "minilm-balanced-60m-native-graph-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-60m-trained-model-seed42-v1",
    ]
    manifest["training_performed"] = True
    manifest["production_config"] = config
    manifest["production_config_sha256"] = config_sha256
    manifest["scientific_contract"] = {
        "question": (
            "does the frozen MiniLM recipe execute stably at the first "
            "principled balanced scale rung?"
        ),
        "rows": ROW_COUNT,
        "retained_scientific_rows": (
            config["graph"]["positive_source_rows"]
        ),
        "successful_updates": SUCCESSFUL_UPDATES,
        "coverage_alignment": config["execution"]["coverage_alignment"],
        "graph": graph["signature"],
        "substrate": substrate["signature"],
        "source_edge_uniform_equivalence": (
            config["graph"]["source_edge_uniform_equivalence"]
        ),
        "known_representation_and_residency_confound": True,
        "interpretation_limit": (
            "R0052 establishes feasibility and a 60M map candidate; it "
            "does not isolate a pure training scale law against fp16 R0046. "
            "A matched balanced-30M int8/native-graph control is required."
        ),
        "runtime_safety": {
            "standalone_canary": False,
            "reason": (
                "the reviewed R0034 host-int8 implementation already ran "
                "2.45M updates; the full run has bounded live performance "
                "windows and exact post-run accounting"
            ),
            "minimum_updates_per_second": (
                config["execution"]["minimum_train_upd_s"]
            ),
            "consecutive_subfloor_windows_before_abort": (
                config["execution"]["performance_subfloor_patience"]
            ),
        },
        "training_wall_only": True,
        "geometry_claim_requires_successor_evaluation": True,
        "pure_scale_claim": False,
    }
    manifest["jobs"] = [{
        "id": "train_seed42_balanced_60m",
        "action": "train",
        "handler_module": "experiments.round0052_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "train_seed42_balanced_60m.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 10_500.0,
        "substrate_manifest": SUBSTRATE_MANIFEST,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "canonical_graph_manifest": GRAPH_MANIFEST,
        "canonical_graph_manifest_sha256": (
            canonical_graph_manifest_sha256
        ),
        "train_config_sha256": config_sha256,
        "successful_updates": SUCCESSFUL_UPDATES,
        "batch_size": config["optimizer"]["batch_size"],
        "node_policy": {
            "gpu_required": True,
            "training_performed": True,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        "train_seed42_balanced_60m": 10_500.0,
        "total": 10_500.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--substrate-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--canonical-graph-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0052(
            release_sha=args.release_sha,
            substrate_manifest_sha256=(
                args.substrate_manifest_sha256
            ),
            canonical_graph_manifest_sha256=(
                args.canonical_graph_manifest_sha256
            ),
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
