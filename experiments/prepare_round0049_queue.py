#!/usr/bin/env python3
"""Prepare, but never launch, the CPU-parallel balanced-60M graph queue."""
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
    CORPUS_INTERVALS,
    INDEX_PATH,
    ROUND_ID,
    ROW_COUNT,
    SOURCE_ELIGIBILITY_PATH,
    SOURCE_INT8_PATH,
    SOURCE_SCALES_PATH,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0049_nodes import SEARCH_WIDTH


ROUND_ROOT = "/data/latent-basemap/runs/round-0049"
RELEASE_ROOT = (
    "/home/enjalot/code/latent-basemap-worktrees/round-0049"
)
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0049-2026-07-25.md",
)
DEFAULT_CANDIDATE_QUALITY_RECEIPT = (
    "/data/latent-basemap/runs/round-0047/queue/artifacts/"
    "candidate-quality/candidate-quality-sweep-v1.json"
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
            f"R0049 requires one issued status; observed {statuses}"
        )


def prepare_round0049(
    *,
    release_sha: str,
    nprobe: int,
    candidate_quality_receipt: str,
    candidate_quality_receipt_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if nprobe <= 0:
        raise ValueError("nprobe must be positive")
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0049 CPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    substrate_output = os.path.join(
        artifacts,
        "balanced-60m-substrate",
    )
    substrate_manifest = os.path.join(
        substrate_output,
        "balanced-60m-substrate-v1.json",
    )
    quality_output = os.path.join(
        artifacts,
        "candidate-quality-60m",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            SOURCE_INT8_PATH,
            SOURCE_SCALES_PATH,
            SOURCE_ELIGIBILITY_PATH,
            INDEX_PATH,
            candidate_quality_receipt,
            os.path.join(LAB_ROOT, "review-0033-2026-07-22.md"),
            os.path.join(LAB_ROOT, "review-0047-2026-07-25.md"),
        ]),
    ])
    observed_quality = next(
        item for item in inputs
        if item["canonical_path"]
        == os.path.realpath(candidate_quality_receipt)
    )
    if observed_quality["sha256"] != candidate_quality_receipt_sha256:
        raise RuntimeError("R0047 candidate-quality receipt hash changed")

    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.75,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0049-balanced-60m-graph-queue-v1"
    # This long CPU build has its own immutable worktree so the detached GPU
    # checkout can advance to R0048 while graph shards continue.
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "mixed-research-parallel"
    manifest["required_reviews"] = ["0033", "0047"]
    manifest["capability_dependencies"] = [
        "minilm-int8-shards-v1",
        "minilm-150m-row-eligibility-v1",
        "path-b-balanced-3m-candidate-quality-v2",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-60m-input-v1",
        "minilm-balanced-60m-candidate-quality-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "row_count": ROW_COUNT,
        "corpus_intervals_in_150m_namespace": [
            list(value) for value in CORPUS_INTERVALS
        ],
        "geometry_unit": "one exact nonzero encoded vector in the subset",
        "graph": {
            "kind": "native directed representative-only k15",
            "index": "accepted aligned 150M IVF-PQ",
            "nprobe": nprobe,
            "search_width": SEARCH_WIDTH,
            "selected_neighbors": 15,
            "exact_rerank": True,
            "rerank_vector_source": (
                "balanced-subset int8-plus-fp16-scale exact cosine"
            ),
            "quality_basis": observed_quality,
            "matched_60m_validation": {
                "sample_rows": 1_024,
                "exact_truth": (
                    "streamed fp32 cosine over native retained 60M universe"
                ),
                "mean_recall_at_15_floor": 0.90,
                "p10_reported": True,
                "successor_graph_build_runs_only_after_pass": True,
            },
        },
        "parallelism": {
            "quality_validation_gpu_required": True,
            "successor_graph_build_gpu_required": False,
            "successor_graph_phase_may_run_while_gpu_research_queue_is_active": True,
        },
        "no_training": True,
        "graph_build": "separate CPU-only successor so this queue releases its GPU lease",
        "no_scale_quality_claim": True,
    }
    common = {
        "handler_module": "experiments.round0049_nodes",
        "handler_callable": "run_job",
        "expected_inputs": inputs,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }
    manifest["jobs"] = [
        {
            **common,
            "id": "build_balanced_60m_substrate",
            "action": "build_substrate",
            "deps": [],
            "outputs": [substrate_output],
            "done_marker": os.path.join(
                artifacts,
                "build_balanced_60m_substrate.done.json",
            ),
            "p90_wall_s": 1_800.0,
        },
        {
            **common,
            "id": "validate_native_candidate_quality_60m",
            "action": "validate_candidate_quality",
            "deps": ["build_balanced_60m_substrate"],
            "outputs": [quality_output],
            "done_marker": os.path.join(
                artifacts,
                "validate_native_candidate_quality_60m.done.json",
            ),
            "p90_wall_s": 1_800.0,
            "substrate_manifest": substrate_manifest,
            "nprobe": nprobe,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
    ]
    manifest["p90_cpu_seconds"] = {
        "build_balanced_60m_substrate": 1_800.0,
        "total": 1_800.0,
    }
    manifest["p90_gpu_seconds"] = {
        "validate_native_candidate_quality_60m": 1_800.0,
        "total": 1_800.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument(
        "--candidate-quality-receipt",
        default=DEFAULT_CANDIDATE_QUALITY_RECEIPT,
    )
    parser.add_argument(
        "--candidate-quality-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0049(
            release_sha=args.release_sha,
            nprobe=args.nprobe,
            candidate_quality_receipt=args.candidate_quality_receipt,
            candidate_quality_receipt_sha256=(
                args.candidate_quality_receipt_sha256
            ),
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
