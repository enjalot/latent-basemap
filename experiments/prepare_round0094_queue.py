#!/usr/bin/env python3
"""Prepare the reviewed sharded-search qualification queue for Round 0094."""
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
from basemap.round0086_program import validate_substrate
from basemap.round0093_policy import load_decision as load_r0093_decision
from basemap.round0094_sharded_search import (
    MAX_MEDIAN_SECONDS_PER_QUERY,
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    ROUND_ID,
    SHARD_SPECS,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0094"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0094-*.md")


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
            f"R0094 requires one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    required_text: tuple[str, ...],
) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError("R0093 review does not release evidence")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0093 review bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if any(value not in text for value in required_text):
        raise RuntimeError("R0093 review does not bind requested evidence")
    return signature


def prepare_round0094(
    *,
    release_sha: str,
    r0093_review_path: str,
    r0093_review_sha256: str,
    r0093_qualification_path: str,
    r0093_qualification_sha256: str,
    r0093_decision_path: str,
    r0093_decision_sha256: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    filtered_index_path: str,
    filtered_index_sha256: str,
    filter_receipt_path: str,
    filter_receipt_sha256: str,
    runtime_spec_path: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    round_file = _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0094 release SHA must be one full commit")
    review = _require_review(
        r0093_review_path,
        expected_sha256=r0093_review_sha256,
        required_text=(
            "capability:minilm-graph-recall-operational-floor-0p84-v1",
            (
                "capability:minilm-balanced-150m-gpu-ivfpq-"
                "search-qualified-low-recall-v1"
            ),
            r0093_qualification_sha256,
            r0093_decision_sha256,
        ),
    )
    decision = load_r0093_decision(
        r0093_decision_path,
        expected_sha256=r0093_decision_sha256,
    )
    qualification = expected_input_signature(r0093_qualification_path)
    substrate = validate_substrate(
        substrate_manifest_path,
        expected_sha256=substrate_manifest_sha256,
    )
    filtered = expected_input_signature(filtered_index_path)
    filter_receipt = expected_input_signature(filter_receipt_path)
    runtime = expected_input_signature(runtime_spec_path)
    if (
        qualification["sha256"] != r0093_qualification_sha256
        or filtered["sha256"] != filtered_index_sha256
        or filter_receipt["sha256"] != filter_receipt_sha256
        or runtime["sha256"] != runtime_spec_sha256
        or decision["receipt"].get("qualification") != qualification
        or decision["receipt"].get("substrate") != substrate["signature"]
        or decision["receipt"].get("filtered_index") != filtered
        or decision["receipt"].get("filter_receipt") != filter_receipt
    ):
        raise RuntimeError("R0094 R0093/staging evidence changed")

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0094 sharded-search queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    split_output = os.path.join(artifacts, "corpus-index-shards")
    qualification_output = os.path.join(
        artifacts,
        "sharded-search-qualification",
    )
    inputs = _dedupe([
        expected_input_signature(round_file),
        review,
        qualification,
        decision["signature"],
        substrate["signature"],
        filtered,
        filter_receipt,
        runtime,
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=1.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0094-sharded-search-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0093"]
    manifest["capability_dependencies"] = [
        "minilm-graph-recall-operational-floor-0p84-v1",
        "minilm-balanced-150m-gpu-ivfpq-search-qualified-low-recall-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-150m-sharded-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        "review_0093": review,
        "r0093_qualification": qualification,
        "r0093_decision": decision["signature"],
        "substrate": substrate["signature"],
        "filtered_index": filtered,
        "filter_receipt": filter_receipt,
    }
    manifest["scientific_contract"] = {
        "rows": 150_000_000,
        "retained_candidates": 147_221_757,
        "candidate_shards": {
            key: dict(value) for key, value in SHARD_SPECS.items()
        },
        "registered_mean_recall_floor": MEAN_RECALL_FLOOR,
        "maximum_median_seconds_per_query": (
            MAX_MEDIAN_SECONDS_PER_QUERY
        ),
        "policy_grid": [
            {
                "nprobe_per_shard": nprobe,
                "width_per_shard": width,
                "total_shortlist_width": width * len(SHARD_SPECS),
            }
            for nprobe, width in POLICY_GRID
        ],
        "same_sample_and_exact_truth_as_r0093": True,
        "global_exact_rerank_after_shard_union": True,
        "no_index_retraining": True,
        "no_graph": True,
        "no_training": True,
        "no_scale_decision": True,
    }
    split_id = "split_retained_index_by_corpus"
    qualify_id = "qualify_sharded_150m_search"
    common = {
        "substrate_manifest": substrate_manifest_path,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "filtered_index": filtered_index_path,
        "filtered_index_sha256": filtered_index_sha256,
        "r0093_decision": r0093_decision_path,
        "r0093_decision_sha256": r0093_decision_sha256,
    }
    manifest["jobs"] = [
        {
            "id": split_id,
            "action": "split_corpus_indices",
            "handler_module": "experiments.round0094_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [split_output],
            "done_marker": os.path.join(
                artifacts, f"{split_id}.done.json"
            ),
            "expected_inputs": inputs,
            "p90_wall_s": 600.0,
            **common,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        },
        {
            "id": qualify_id,
            "action": "qualify_sharded_search",
            "handler_module": "experiments.round0094_nodes",
            "handler_callable": "run_job",
            "deps": [split_id],
            "outputs": [qualification_output],
            "done_marker": os.path.join(
                artifacts, f"{qualify_id}.done.json"
            ),
            "expected_inputs": inputs,
            "p90_wall_s": 3_600.0,
            **common,
            "split_root": split_output,
            "runtime_spec": runtime_spec_path,
            "runtime_spec_sha256": runtime_spec_sha256,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
    ]
    manifest["p90_gpu_seconds"] = {
        split_id: 0.0,
        qualify_id: 3_600.0,
        "total": 3_600.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0093-review", required=True)
    parser.add_argument("--r0093-review-sha256", required=True)
    parser.add_argument("--r0093-qualification", required=True)
    parser.add_argument("--r0093-qualification-sha256", required=True)
    parser.add_argument("--r0093-decision", required=True)
    parser.add_argument("--r0093-decision-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--filtered-index", required=True)
    parser.add_argument("--filtered-index-sha256", required=True)
    parser.add_argument("--filter-receipt", required=True)
    parser.add_argument("--filter-receipt-sha256", required=True)
    parser.add_argument("--runtime-spec", required=True)
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0094(
            release_sha=args.release_sha,
            r0093_review_path=args.r0093_review,
            r0093_review_sha256=args.r0093_review_sha256,
            r0093_qualification_path=args.r0093_qualification,
            r0093_qualification_sha256=args.r0093_qualification_sha256,
            r0093_decision_path=args.r0093_decision,
            r0093_decision_sha256=args.r0093_decision_sha256,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            filtered_index_path=args.filtered_index,
            filtered_index_sha256=args.filtered_index_sha256,
            filter_receipt_path=args.filter_receipt,
            filter_receipt_sha256=args.filter_receipt_sha256,
            runtime_spec_path=args.runtime_spec,
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
