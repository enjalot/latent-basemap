#!/usr/bin/env python3
"""Prepare the corrected retained-row search audit for Round 0095."""
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
from basemap.round0094_sharded_search import load_split_receipt
from basemap.round0095_unbiased_audit import (
    MONOLITHIC_POLICIES,
    ROUND_ID,
    SAMPLE_ROWS,
    SAMPLE_SEED,
    SAMPLE_SHA256,
    SHARDED_POLICIES,
    load_r0094_negative,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0095"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0095-*.md")


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4_096)
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
            f"R0095 requires one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    required_text: tuple[str, ...],
) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial", "rejected"}:
        raise RuntimeError("R0094 review is not terminal")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0094 review bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if any(value not in text for value in required_text):
        raise RuntimeError("R0094 review does not release corrected audit")
    return signature


def prepare_round0095(
    *,
    release_sha: str,
    r0094_review_path: str,
    r0094_review_sha256: str,
    r0094_qualification_path: str,
    r0094_qualification_sha256: str,
    r0094_split_receipt_path: str,
    r0094_split_receipt_sha256: str,
    r0093_qualification_path: str,
    r0093_qualification_sha256: str,
    r0093_decision_path: str,
    r0093_decision_sha256: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    filtered_index_path: str,
    filtered_index_sha256: str,
    runtime_spec_path: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    round_file = _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0095 release SHA must be one full commit")
    review = _require_review(
        r0094_review_path,
        expected_sha256=r0094_review_sha256,
        required_text=(
            "0095",
            r0094_qualification_sha256,
            "sample",
        ),
    )
    substrate = validate_substrate(
        substrate_manifest_path,
        expected_sha256=substrate_manifest_sha256,
    )
    filtered = expected_input_signature(filtered_index_path)
    if filtered["sha256"] != filtered_index_sha256:
        raise RuntimeError("R0095 filtered index changed")
    r0093_qualification = expected_input_signature(
        r0093_qualification_path
    )
    r0093_decision = load_r0093_decision(
        r0093_decision_path,
        expected_sha256=r0093_decision_sha256,
    )
    if (
        r0093_qualification["sha256"] != r0093_qualification_sha256
        or r0093_decision["receipt"].get("qualification")
        != r0093_qualification
        or r0093_decision["receipt"].get("substrate")
        != substrate["signature"]
        or r0093_decision["receipt"].get("filtered_index") != filtered
    ):
        raise RuntimeError("R0095 R0093 evidence changed")
    r0094 = load_r0094_negative(
        r0094_qualification_path,
        expected_sha256=r0094_qualification_sha256,
    )
    split_signature = expected_input_signature(r0094_split_receipt_path)
    if split_signature["sha256"] != r0094_split_receipt_sha256:
        raise RuntimeError("R0094 split receipt bytes changed")
    split = load_split_receipt(
        r0094_split_receipt_path,
        expected_source=filtered,
        expected_release_sha=str(r0094["receipt"]["release_sha"]),
    )
    if (
        split["signature"] != split_signature
        or r0094["receipt"].get("split_receipt") != split_signature
        or r0094["receipt"].get("source_index") != filtered
    ):
        raise RuntimeError("R0094 negative/split lineage changed")
    runtime = expected_input_signature(runtime_spec_path)
    if runtime["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0095 runtime spec changed")

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0095 unbiased search audit queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "unbiased-search-audit")
    shard_inputs = [
        dict(value["index"])
        for value in split["receipt"]["shards"].values()
    ]
    inputs = _dedupe([
        expected_input_signature(round_file),
        review,
        r0094["signature"],
        split_signature,
        r0093_qualification,
        r0093_decision["signature"],
        substrate["signature"],
        filtered,
        runtime,
        *shard_inputs,
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0095-unbiased-search-audit-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0094"]
    manifest["capability_dependencies"] = []
    manifest["capabilities_produced"] = [
        "minilm-balanced-150m-unbiased-search-audit-v1",
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "sample_rows": SAMPLE_ROWS,
        "sample_seed": SAMPLE_SEED,
        "sample_sha256": SAMPLE_SHA256,
        "sample_semantics": (
            "uniform without replacement over all retained rows; "
            "random subset before final sort"
        ),
        "monolithic_replays": [
            {"name": name, "nprobe": nprobe, "shortlist_width": width}
            for name, nprobe, width in MONOLITHIC_POLICIES
        ],
        "sharded_replays": [
            {
                "name": name,
                "nprobe_per_shard": nprobe,
                "width_per_shard": width,
            }
            for name, nprobe, width in SHARDED_POLICIES
        ],
        "no_training": True,
        "no_graph": True,
        "no_scale_decision": True,
    }
    job_id = "audit_unbiased_150m_search"
    manifest["jobs"] = [{
        "id": job_id,
        "action": job_id,
        "handler_module": "experiments.round0095_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": 1_800.0,
        "substrate_manifest": substrate_manifest_path,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "filtered_index": filtered_index_path,
        "filtered_index_sha256": filtered_index_sha256,
        "r0093_qualification": r0093_qualification_path,
        "r0093_qualification_sha256": r0093_qualification_sha256,
        "r0093_decision": r0093_decision_path,
        "r0093_decision_sha256": r0093_decision_sha256,
        "r0094_qualification": r0094_qualification_path,
        "r0094_qualification_sha256": r0094_qualification_sha256,
        "r0094_split_receipt": r0094_split_receipt_path,
        "r0094_split_receipt_sha256": r0094_split_receipt_sha256,
        "runtime_spec": runtime_spec_path,
        "runtime_spec_sha256": runtime_spec_sha256,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        job_id: 1_800.0,
        "total": 1_800.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0094-review", required=True)
    parser.add_argument("--r0094-review-sha256", required=True)
    parser.add_argument("--r0094-qualification", required=True)
    parser.add_argument("--r0094-qualification-sha256", required=True)
    parser.add_argument("--r0094-split-receipt", required=True)
    parser.add_argument("--r0094-split-receipt-sha256", required=True)
    parser.add_argument("--r0093-qualification", required=True)
    parser.add_argument("--r0093-qualification-sha256", required=True)
    parser.add_argument("--r0093-decision", required=True)
    parser.add_argument("--r0093-decision-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--filtered-index", required=True)
    parser.add_argument("--filtered-index-sha256", required=True)
    parser.add_argument("--runtime-spec", required=True)
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0095(
            release_sha=args.release_sha,
            r0094_review_path=args.r0094_review,
            r0094_review_sha256=args.r0094_review_sha256,
            r0094_qualification_path=args.r0094_qualification,
            r0094_qualification_sha256=(
                args.r0094_qualification_sha256
            ),
            r0094_split_receipt_path=args.r0094_split_receipt,
            r0094_split_receipt_sha256=(
                args.r0094_split_receipt_sha256
            ),
            r0093_qualification_path=args.r0093_qualification,
            r0093_qualification_sha256=(
                args.r0093_qualification_sha256
            ),
            r0093_decision_path=args.r0093_decision,
            r0093_decision_sha256=args.r0093_decision_sha256,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            filtered_index_path=args.filtered_index,
            filtered_index_sha256=args.filtered_index_sha256,
            runtime_spec_path=args.runtime_spec,
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
