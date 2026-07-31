#!/usr/bin/env python3
"""Prepare the R0064-selected next-rung GPU quality qualification."""
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
from basemap.round0049_program import INDEX_PATH
from basemap.round0065_substrates import validate_scale_substrate
from basemap.round0066_quality import (
    NPROBE_GRID,
    ROUND_ID,
    load_scale_decision,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0059_nodes import FAISS_WHEEL
from experiments.round0066_nodes import RUNTIME_SPEC


ROUND_ROOT = "/data/latent-basemap/runs/round-0066"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0066-2026-07-26.md",
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
        raise RuntimeError("R0066 remains draft; refuse queue materialization")


def prepare_round0066(
    *,
    release_sha: str,
    scale_comparison_path: str,
    scale_comparison_sha256: str,
    r0064_review_path: str,
    r0064_review_sha256: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    r0065_review_path: str,
    r0065_review_sha256: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0066 release SHA must be one full commit")
    decision = load_scale_decision(
        scale_comparison_path,
        expected_sha256=scale_comparison_sha256,
    )
    tier = decision["tier"]
    substrate = validate_scale_substrate(
        substrate_manifest_path,
        tier=tier,
        expected_sha256=substrate_manifest_sha256,
    )
    r0064_review = _require_review(
        r0064_review_path,
        expected_sha256=r0064_review_sha256,
        required_text=(scale_comparison_sha256, tier),
    )
    r0065_review = _require_review(
        r0065_review_path,
        expected_sha256=r0065_review_sha256,
        required_text=(substrate_manifest_sha256, f"balanced-{tier}"),
    )
    runtime = expected_input_signature(RUNTIME_SPEC)
    if runtime["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0066 runtime specification changed")
    outputs = substrate["manifest"]["outputs"]
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0066 GPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(
        artifacts,
        f"gpu-ivfpq-qualification-{tier}",
    )
    review_0059 = os.path.join(
        LAB_ROOT,
        "review-0059-2026-07-26.md",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            scale_comparison_path,
            r0064_review_path,
            substrate_manifest_path,
            r0065_review_path,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            INDEX_PATH,
            RUNTIME_SPEC,
            FAISS_WHEEL,
            review_0059,
        ]),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=1.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0066-next-rung-gpu-quality-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0059", "0064", "0065"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-60m-scale-geometry-v1",
        f"minilm-balanced-{tier}-int8-input-v1",
        "minilm-balanced-60m-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["capabilities_produced"] = [
        f"minilm-balanced-{tier}-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["training_performed"] = False
    manifest["late_bound_selection"] = {
        "tier": tier,
        "scale_comparison": decision["signature"],
        "r0064_review": r0064_review,
        "substrate": substrate["signature"],
        "r0065_review": r0065_review,
    }
    manifest["scientific_contract"] = {
        "tier_selected_only_by_r0064": tier,
        "nprobe_grid": list(NPROBE_GRID),
        "selector": (
            "smallest nprobe with mean unambiguous exact-reranked "
            "recall@15 at least 0.90"
        ),
        "sample_rows": 1_024,
        "sample_seed": 66,
        "search_width": 128,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "gpu_engine": "faiss-classic-GpuIndexIVFPQ",
        "physically_filtered_representative_index": True,
        "graph_projection_is_informational": True,
        "no_graph": True,
        "no_training": True,
        "no_scale_decision": True,
    }
    manifest["jobs"] = [{
        "id": f"qualify_balanced_{tier}_gpu_ivfpq",
        "action": "qualify_next_rung_gpu_ivfpq",
        "handler_module": "experiments.round0066_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            f"qualify_balanced_{tier}_gpu_ivfpq.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 2_400.0,
        "tier": tier,
        "scale_comparison": scale_comparison_path,
        "scale_comparison_sha256": scale_comparison_sha256,
        "substrate_manifest": substrate_manifest_path,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "runtime_spec": RUNTIME_SPEC,
        "runtime_spec_sha256": runtime_spec_sha256,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        f"qualify_balanced_{tier}_gpu_ivfpq": 2_400.0,
        "total": 2_400.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--scale-comparison", required=True)
    parser.add_argument("--scale-comparison-sha256", required=True)
    parser.add_argument("--r0064-review", required=True)
    parser.add_argument("--r0064-review-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--r0065-review", required=True)
    parser.add_argument("--r0065-review-sha256", required=True)
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0066(
            release_sha=args.release_sha,
            scale_comparison_path=args.scale_comparison,
            scale_comparison_sha256=args.scale_comparison_sha256,
            r0064_review_path=args.r0064_review,
            r0064_review_sha256=args.r0064_review_sha256,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            r0065_review_path=args.r0065_review,
            r0065_review_sha256=args.r0065_review_sha256,
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
