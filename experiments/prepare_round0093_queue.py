#!/usr/bin/env python3
"""Prepare the reviewed lower-recall 150M policy qualification queue."""
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
from basemap.round0093_policy import (
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    ROUND_ID,
    validate_r0083_sensitivity,
    validate_r0084_stability,
    validate_r0086_qualification,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0093"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0093-*.md")


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
            "R0093 requires exactly one issued round document; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    required_text: tuple[str, ...],
) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError(f"{path} does not release reviewed evidence")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if any(value not in text for value in required_text):
        raise RuntimeError(f"{path} does not bind the requested capability")
    return signature


def _bind_declared_path(
    path: str,
    *,
    expected_sha256: str,
    declared: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if (
        canonical != declared.get("canonical_path")
        or expected_sha256 != declared.get("sha256")
    ):
        raise RuntimeError(f"{label} differs from R0086 qualification")
    return dict(declared)


def prepare_round0093(
    *,
    release_sha: str,
    r0083_review_path: str,
    r0083_review_sha256: str,
    r0083_sensitivity_path: str,
    r0083_sensitivity_sha256: str,
    r0084_review_path: str,
    r0084_review_sha256: str,
    r0084_seed_contrast_path: str,
    r0084_seed_contrast_sha256: str,
    r0086_review_path: str,
    r0086_review_sha256: str,
    r0086_qualification_path: str,
    r0086_qualification_sha256: str,
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
        raise ValueError("R0093 release SHA must be one full commit")
    r0083 = validate_r0083_sensitivity(
        r0083_sensitivity_path,
        expected_sha256=r0083_sensitivity_sha256,
    )
    r0084 = validate_r0084_stability(
        r0084_seed_contrast_path,
        expected_sha256=r0084_seed_contrast_sha256,
    )
    r0086 = validate_r0086_qualification(
        r0086_qualification_path,
        expected_sha256=r0086_qualification_sha256,
    )
    r0086_receipt = r0086["receipt"]
    substrate = _bind_declared_path(
        substrate_manifest_path,
        expected_sha256=substrate_manifest_sha256,
        declared=dict(r0086_receipt["substrate"]),
        label="R0086 substrate",
    )
    filtered = _bind_declared_path(
        filtered_index_path,
        expected_sha256=filtered_index_sha256,
        declared=dict(r0086_receipt["filtered_index"]),
        label="R0086 filtered index",
    )
    filter_receipt = expected_input_signature(filter_receipt_path)
    runtime = expected_input_signature(runtime_spec_path)
    if (
        filter_receipt["sha256"] != filter_receipt_sha256
        or runtime["sha256"] != runtime_spec_sha256
    ):
        raise RuntimeError("R0093 R0086/runtime evidence bytes changed")
    reviews = {
        "0083": _require_review(
            r0083_review_path,
            expected_sha256=r0083_review_sha256,
            required_text=(
                "capability:minilm-30m-graph-recall-sensitivity-v1",
                r0083_sensitivity_sha256,
            ),
        ),
        "0084": _require_review(
            r0084_review_path,
            expected_sha256=r0084_review_sha256,
            required_text=(
                "capability:minilm-balanced-90m-seed43-sensitivity-v1",
                r0084_seed_contrast_sha256,
            ),
        ),
        "0086": _require_review(
            r0086_review_path,
            expected_sha256=r0086_review_sha256,
            required_text=(
                "capability:minilm-balanced-150m-int8-input-v1",
                "capability:minilm-balanced-150m-gpu-ivfpq-search-qualified-v1",
                r0086_qualification_sha256,
            ),
        ),
    }
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0093 lower-recall policy queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(
        artifacts,
        "lower-recall-policy-qualification-150m",
    )
    inputs = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        r0083["signature"],
        r0084["signature"],
        r0086["signature"],
        substrate,
        filtered,
        filter_receipt,
        runtime,
    ])
    node_id = "qualify_lower_recall_150m_policy"
    p90_seconds = 1_800.0
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0093-lower-recall-150m-policy-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0083", "0084", "0086"]
    manifest["capability_dependencies"] = [
        "minilm-30m-graph-recall-sensitivity-v1",
        "minilm-balanced-90m-seed43-sensitivity-v1",
        "minilm-balanced-150m-int8-input-v1",
        "minilm-balanced-150m-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-graph-recall-operational-floor-0p84-v1",
        "minilm-balanced-150m-gpu-ivfpq-search-qualified-low-recall-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "r0083_sensitivity": r0083["signature"],
        "r0084_seed_contrast": r0084["signature"],
        "r0086_qualification": r0086["signature"],
        "substrate": substrate,
        "filtered_index": filtered,
        "filter_receipt": filter_receipt,
    }
    manifest["scientific_contract"] = {
        "tier": "150m",
        "rows": 150_000_000,
        "registered_mean_recall_floor": MEAN_RECALL_FLOOR,
        "policy_grid": [
            {"nprobe": nprobe, "shortlist_width": width}
            for nprobe, width in POLICY_GRID
        ],
        "r0083_direct_treatment_recall": float(
            r0083["supporting_cell"][
                "candidate_recall_at_15_unambiguous"
            ]
        ),
        "r0084_descriptive_stability_screen": {
            "matched_absolute_deltas": r0084[
                "matched_absolute_deltas"
            ],
            "margins": r0084["margins"],
            "not_variance_or_error_bar": True,
        },
        "quality_selector": (
            "mean unambiguous exact-reranked recall@15 at least 0.84"
        ),
        "performance_selector": (
            "lowest median three-repeat 10000-query search-plus-rerank "
            "wall among passing cells"
        ),
        "sample_rows": 4_096,
        "sample_seed": 86,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "full_150m_map_evaluation_still_required": True,
        "no_graph": True,
        "no_training": True,
    }
    manifest["jobs"] = [{
        "id": node_id,
        "action": "qualify_lower_recall_150m_policy",
        "handler_module": "experiments.round0093_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": p90_seconds,
        "r0083_sensitivity": r0083_sensitivity_path,
        "r0083_sensitivity_sha256": r0083_sensitivity_sha256,
        "r0084_seed_contrast": r0084_seed_contrast_path,
        "r0084_seed_contrast_sha256": r0084_seed_contrast_sha256,
        "r0086_qualification": r0086_qualification_path,
        "r0086_qualification_sha256": r0086_qualification_sha256,
        "substrate_manifest": substrate_manifest_path,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "filtered_index": filtered_index_path,
        "filtered_index_sha256": filtered_index_sha256,
        "filter_receipt": filter_receipt_path,
        "filter_receipt_sha256": filter_receipt_sha256,
        "runtime_spec": runtime_spec_path,
        "runtime_spec_sha256": runtime_spec_sha256,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {node_id: p90_seconds, "total": p90_seconds}
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    for round_id in ("0083", "0084", "0086"):
        parser.add_argument(f"--r{round_id}-review", required=True)
        parser.add_argument(f"--r{round_id}-review-sha256", required=True)
    parser.add_argument("--r0083-sensitivity", required=True)
    parser.add_argument("--r0083-sensitivity-sha256", required=True)
    parser.add_argument("--r0084-seed-contrast", required=True)
    parser.add_argument("--r0084-seed-contrast-sha256", required=True)
    parser.add_argument("--r0086-qualification", required=True)
    parser.add_argument("--r0086-qualification-sha256", required=True)
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
        "queue_manifest": prepare_round0093(
            release_sha=args.release_sha,
            r0083_review_path=args.r0083_review,
            r0083_review_sha256=args.r0083_review_sha256,
            r0083_sensitivity_path=args.r0083_sensitivity,
            r0083_sensitivity_sha256=args.r0083_sensitivity_sha256,
            r0084_review_path=args.r0084_review,
            r0084_review_sha256=args.r0084_review_sha256,
            r0084_seed_contrast_path=args.r0084_seed_contrast,
            r0084_seed_contrast_sha256=args.r0084_seed_contrast_sha256,
            r0086_review_path=args.r0086_review,
            r0086_review_sha256=args.r0086_review_sha256,
            r0086_qualification_path=args.r0086_qualification,
            r0086_qualification_sha256=args.r0086_qualification_sha256,
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
