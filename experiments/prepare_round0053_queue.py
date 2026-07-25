#!/usr/bin/env python3
"""Prepare the matched balanced-30M int8 substrate and quality queue."""
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
from basemap.round0049_program import (
    INDEX_PATH,
    validate_substrate_manifest,
)
from basemap.round0053_program import (
    EXPECTED_RETAINED_ROWS,
    ROUND_ID,
    SOURCE_SUBSTRATE_MANIFEST,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0049_nodes import SEARCH_WIDTH


ROUND_ROOT = "/data/latent-basemap/runs/round-0053"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0053"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0053-2026-07-26.md",
)
R0047_QUALITY = (
    "/data/latent-basemap/runs/round-0047/queue/artifacts/"
    "candidate-quality/candidate-quality-sweep-v1.json"
)
R0049_QUALITY = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "candidate-quality-60m/balanced-60m-candidate-quality-v1.json"
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
            f"R0053 requires one issued status; observed {statuses}"
        )


def prepare_round0053(
    *,
    release_sha: str,
    source_substrate_manifest_sha256: str,
    r0047_quality_sha256: str,
    r0049_quality_sha256: str,
    nprobe: int,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if nprobe <= 0:
        raise ValueError("R0053 nprobe must be positive")
    source = validate_substrate_manifest(
        SOURCE_SUBSTRATE_MANIFEST,
        expected_sha256=source_substrate_manifest_sha256,
    )
    source_outputs = source["manifest"]["outputs"]
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0053 queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    substrate_output = os.path.join(
        artifacts,
        "balanced-30m-int8-substrate",
    )
    substrate_manifest = os.path.join(
        substrate_output,
        "balanced-30m-int8-substrate-v1.json",
    )
    quality_output = os.path.join(
        artifacts,
        "candidate-quality-30m",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SOURCE_SUBSTRATE_MANIFEST,
            source_outputs["int8"]["canonical_path"],
            source_outputs["scales"]["canonical_path"],
            source_outputs["eligibility"]["canonical_path"],
            INDEX_PATH,
            R0047_QUALITY,
            R0049_QUALITY,
            os.path.join(LAB_ROOT, "review-0047-2026-07-25.md"),
            os.path.join(LAB_ROOT, "review-0049-2026-07-26.md"),
        ]),
    ])
    by_path = {
        item["canonical_path"]: item
        for item in inputs
    }
    for path, expected in (
        (R0047_QUALITY, r0047_quality_sha256),
        (R0049_QUALITY, r0049_quality_sha256),
    ):
        if by_path[os.path.realpath(path)]["sha256"] != expected:
            raise RuntimeError(
                f"reviewed quality receipt changed: {path}"
            )

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0053-balanced-30m-control-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "mixed-research"
    manifest["required_reviews"] = ["0047", "0049"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-60m-input-v1",
        "minilm-balanced-60m-candidate-quality-v1",
        "path-b-balanced-3m-candidate-quality-v2",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-30m-int8-input-v1",
        "minilm-balanced-30m-candidate-quality-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "purpose": (
            "construct the missing representation-matched 30M control "
            "needed to interpret the 60M rung as a scale experiment"
        ),
        "rows": 30_000_000,
        "retained_rows": EXPECTED_RETAINED_ROWS,
        "source": source["signature"],
        "first_rows_per_corpus": 10_000_000,
        "exact_families_recomputed_after_subset_restriction": True,
        "candidate_generator": {
            "same_index_and_policy_as_r0049": True,
            "nprobe": nprobe,
            "search_width": SEARCH_WIDTH,
            "selected_neighbors": 15,
            "exact_rerank": True,
        },
        "quality_validation": {
            "sample_rows": 1_024,
            "exact_truth": (
                "streamed fp32 cosine over native retained 30M int8 universe"
            ),
            "mean_recall_at_15_unambiguous_floor": 0.90,
        },
        "no_training": True,
        "no_scale_claim": True,
    }
    common = {
        "handler_module": "experiments.round0053_nodes",
        "handler_callable": "run_job",
        "expected_inputs": inputs,
    }
    manifest["jobs"] = [
        {
            **common,
            "id": "build_balanced_30m_int8_substrate",
            "action": "build_substrate",
            "deps": [],
            "outputs": [substrate_output],
            "done_marker": os.path.join(
                artifacts,
                "build_balanced_30m_int8_substrate.done.json",
            ),
            "p90_wall_s": 1_200.0,
            "source_substrate_manifest_sha256": (
                source_substrate_manifest_sha256
            ),
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        },
        {
            **common,
            "id": "validate_native_candidate_quality_30m",
            "action": "validate_candidate_quality",
            "deps": ["build_balanced_30m_int8_substrate"],
            "outputs": [quality_output],
            "done_marker": os.path.join(
                artifacts,
                "validate_native_candidate_quality_30m.done.json",
            ),
            "p90_wall_s": 1_200.0,
            "substrate_manifest": substrate_manifest,
            "nprobe": nprobe,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
    ]
    manifest["p90_cpu_seconds"] = {
        "build_balanced_30m_int8_substrate": 1_200.0,
        "total": 1_200.0,
    }
    manifest["p90_gpu_seconds"] = {
        "validate_native_candidate_quality_30m": 1_200.0,
        "total": 1_200.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--source-substrate-manifest-sha256",
        required=True,
    )
    parser.add_argument("--r0047-quality-sha256", required=True)
    parser.add_argument("--r0049-quality-sha256", required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0053(
            release_sha=args.release_sha,
            source_substrate_manifest_sha256=(
                args.source_substrate_manifest_sha256
            ),
            r0047_quality_sha256=args.r0047_quality_sha256,
            r0049_quality_sha256=args.r0049_quality_sha256,
            nprobe=args.nprobe,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
