#!/usr/bin/env python3
"""Prepare the selected 45M-or-120M GPU-native graph queue."""
from __future__ import annotations

import argparse
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
from basemap.round0065_substrates import (
    SUBSETS,
    validate_scale_substrate,
)
from basemap.round0066_quality import load_scale_decision
from basemap.round0067_graph import (
    ROUND_ID,
    load_gpu_qualification,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0059_nodes import FAISS_WHEEL
from experiments.round0067_nodes import RUNTIME_SPEC


ROUND_ROOT = "/data/latent-basemap/runs/round-0067"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0067-2026-07-26.md",
)


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


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
        raise RuntimeError(f"{path} does not bind the supplied capability")
    return signature


def _require_issued_round() -> None:
    if _frontmatter_status(ROUND_FILE) != "issued":
        raise RuntimeError("R0067 remains draft; refuse queue materialization")


def prepare_round0067(
    *,
    release_sha: str,
    scale_comparison_path: str,
    scale_comparison_sha256: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    gpu_qualification_path: str,
    gpu_qualification_sha256: str,
    filtered_index_path: str,
    r0064_review_path: str,
    r0064_review_sha256: str,
    r0065_review_path: str,
    r0065_review_sha256: str,
    r0066_review_path: str,
    r0066_review_sha256: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0067 release SHA must be one full commit")
    decision = load_scale_decision(
        scale_comparison_path,
        expected_sha256=scale_comparison_sha256,
    )
    tier = decision["tier"]
    spec = SUBSETS[tier]
    substrate = validate_scale_substrate(
        substrate_manifest_path,
        tier=tier,
        expected_sha256=substrate_manifest_sha256,
    )
    outputs = substrate["manifest"]["outputs"]
    qualification = load_gpu_qualification(
        gpu_qualification_path,
        expected_sha256=gpu_qualification_sha256,
        tier=tier,
        substrate_signature=substrate["signature"],
        eligibility_signature=outputs["eligibility"],
    )
    filtered = expected_input_signature(filtered_index_path)
    if (
        filtered
        != qualification["receipt"]["candidate_universe"]["filtered_index"]
    ):
        raise RuntimeError("R0067 filtered-index bytes changed")
    review64 = _require_review(
        r0064_review_path,
        expected_sha256=r0064_review_sha256,
        required_text=(scale_comparison_sha256, tier),
    )
    review65 = _require_review(
        r0065_review_path,
        expected_sha256=r0065_review_sha256,
        required_text=(substrate_manifest_sha256, f"balanced-{tier}"),
    )
    review66 = _require_review(
        r0066_review_path,
        expected_sha256=r0066_review_sha256,
        required_text=(
            gpu_qualification_sha256,
            filtered["sha256"],
            tier,
        ),
    )
    runtime = expected_input_signature(RUNTIME_SPEC)
    if runtime["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0067 runtime specification changed")
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0067 GPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(
        artifacts,
        f"native-graph-balanced-{tier}",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            scale_comparison_path,
            substrate_manifest_path,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            gpu_qualification_path,
            filtered_index_path,
            r0064_review_path,
            r0065_review_path,
            r0066_review_path,
            RUNTIME_SPEC,
            FAISS_WHEEL,
        ]),
    ])
    cap = 2.0 if tier == "45m" else 5.0
    p90 = 7_200.0 if tier == "45m" else 18_000.0
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=cap,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0067-selected-next-rung-graph-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0064", "0065", "0066"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-60m-scale-geometry-v1",
        f"minilm-balanced-{tier}-int8-input-v1",
        f"minilm-balanced-{tier}-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["capabilities_produced"] = [
        f"minilm-balanced-{tier}-gpu-native-graph-v1",
    ]
    manifest["training_performed"] = False
    manifest["late_bound_selection"] = {
        "tier": tier,
        "scale_comparison": decision["signature"],
        "substrate": substrate["signature"],
        "gpu_qualification": qualification["signature"],
        "filtered_index": filtered,
        "reviews": {
            "0064": review64,
            "0065": review65,
            "0066": review66,
        },
    }
    nprobe = int(qualification["receipt"]["selected_nprobe"])
    manifest["scientific_contract"] = {
        "tier_selected_only_by_r0064": tier,
        "rows": spec["row_count"],
        "retained_rows": spec["eligibility_summary"][
            "retained_row_count"
        ],
        "nprobe_selected_only_by_r0066": nprobe,
        "search_width": 128,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "shard_rows": 100_000,
        "resumable_shards": True,
        "fixed_degree": 15,
        "no_training": True,
        "no_scale_decision": True,
    }
    manifest["jobs"] = [{
        "id": f"build_gpu_native_graph_balanced_{tier}",
        "action": "build_selected_gpu_graph",
        "handler_module": "experiments.round0067_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            f"build_gpu_native_graph_balanced_{tier}.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": p90,
        "tier": tier,
        "scale_comparison": scale_comparison_path,
        "scale_comparison_sha256": scale_comparison_sha256,
        "substrate_manifest": substrate_manifest_path,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "gpu_qualification_receipt": gpu_qualification_path,
        "gpu_qualification_receipt_sha256": gpu_qualification_sha256,
        "filtered_index": filtered_index_path,
        "runtime_spec": RUNTIME_SPEC,
        "runtime_spec_sha256": runtime_spec_sha256,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        f"build_gpu_native_graph_balanced_{tier}": p90,
        "total": p90,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--scale-comparison", required=True)
    parser.add_argument("--scale-comparison-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--gpu-qualification", required=True)
    parser.add_argument("--gpu-qualification-sha256", required=True)
    parser.add_argument("--filtered-index", required=True)
    for round_id in ("0064", "0065", "0066"):
        parser.add_argument(f"--r{round_id}-review", required=True)
        parser.add_argument(f"--r{round_id}-review-sha256", required=True)
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0067(
            release_sha=args.release_sha,
            scale_comparison_path=args.scale_comparison,
            scale_comparison_sha256=args.scale_comparison_sha256,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            gpu_qualification_path=args.gpu_qualification,
            gpu_qualification_sha256=args.gpu_qualification_sha256,
            filtered_index_path=args.filtered_index,
            r0064_review_path=args.r0064_review,
            r0064_review_sha256=args.r0064_review_sha256,
            r0065_review_path=args.r0065_review,
            r0065_review_sha256=args.r0065_review_sha256,
            r0066_review_path=args.r0066_review,
            r0066_review_sha256=args.r0066_review_sha256,
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
