#!/usr/bin/env python3
"""Prepare the fixed balanced-120M GPU candidate-search qualification."""
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
from basemap.round0065_substrates import (
    subset_spec,
    validate_scale_substrate,
)
from basemap.round0077_quality import (
    NPROBE_GRID,
    ROUND_ID,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0059_nodes import FAISS_WHEEL
from experiments.round0077_nodes import (
    QUALITY_SAMPLE_ROWS,
    QUALITY_SEED,
    RUNTIME_SPEC,
)


TIER = "120m"
SPEC = subset_spec(TIER)
ROUND_ROOT = "/data/latent-basemap/runs/round-0077"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0077-2026-07-27.md",
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
        raise RuntimeError("R0077 remains draft; refuse queue materialization")


def prepare_round0077(
    *,
    release_sha: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    r0065_review_path: str,
    r0065_review_sha256: str,
    r0072_review_path: str,
    r0072_review_sha256: str,
    r0076_review_path: str,
    r0076_review_sha256: str,
    runtime_spec_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0077 release SHA must be one full commit")
    substrate = validate_scale_substrate(
        substrate_manifest_path,
        tier=TIER,
        expected_sha256=substrate_manifest_sha256,
    )
    outputs = substrate["manifest"]["outputs"]
    review65 = _require_review(
        r0065_review_path,
        expected_sha256=r0065_review_sha256,
        required_text=(
            "capability:minilm-balanced-120m-int8-input-v1",
            substrate_manifest_sha256,
        ),
    )
    review72 = _require_review(
        r0072_review_path,
        expected_sha256=r0072_review_sha256,
        required_text=(
            "capability:minilm-balanced-90m-gpu-ivfpq-search-qualified-v1",
            "a18a840e2f0116a5ffac55a00b1df6f32d48a6226c35f4df12cae12c22679607",
        ),
    )
    review76 = _require_review(
        r0076_review_path,
        expected_sha256=r0076_review_sha256,
        required_text=(
            "capability:minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
            "6b0559b75198b736251c62c7524609c23554f16fbbeec566d240a3b6a6ac235f",
        ),
    )
    runtime = expected_input_signature(RUNTIME_SPEC)
    if runtime["sha256"] != runtime_spec_sha256:
        raise RuntimeError("R0077 runtime specification changed")
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0077 GPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(
        artifacts,
        "gpu-ivfpq-qualification-120m",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            substrate_manifest_path,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            INDEX_PATH,
            RUNTIME_SPEC,
            FAISS_WHEEL,
            r0065_review_path,
            r0072_review_path,
            r0076_review_path,
        ]),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = (
        "round0077-balanced-120m-gpu-quality-queue-v1"
    )
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0065", "0072", "0076"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-120m-int8-input-v1",
        "minilm-balanced-90m-gpu-ivfpq-search-qualified-v1",
        "minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-120m-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        "review_0065": review65,
        "review_0072": review72,
        "review_0076": review76,
        "substrate": substrate["signature"],
    }
    manifest["scientific_contract"] = {
        "fixed_tier": TIER,
        "rows": SPEC["row_count"],
        "retained_rows": SPEC["eligibility_summary"][
            "retained_row_count"
        ],
        "balanced_intervals": [
            list(value) for value in SPEC["intervals"]
        ],
        "nprobe_grid": list(NPROBE_GRID),
        "selector": (
            "smallest nprobe with mean unambiguous exact-reranked "
            "recall@15 at least 0.90"
        ),
        "sample_rows": QUALITY_SAMPLE_ROWS,
        "sample_seed": QUALITY_SEED,
        "search_width": 128,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "gpu_engine": "faiss-classic-GpuIndexIVFPQ",
        "physically_filtered_representative_index": True,
        "rung_advance_bound_to_reviewed_90m_geometry": True,
        "graph_projection_is_informational": True,
        "no_graph": True,
        "no_training": True,
        "no_120m_quality_claim": True,
    }
    manifest["jobs"] = [{
        "id": "qualify_balanced_120m_gpu_ivfpq",
        "action": "qualify_balanced_120m_gpu_ivfpq",
        "handler_module": "experiments.round0077_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "qualify_balanced_120m_gpu_ivfpq.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 900.0,
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
        "qualify_balanced_120m_gpu_ivfpq": 900.0,
        "total": 900.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    for round_id in ("0065", "0072", "0076"):
        parser.add_argument(f"--r{round_id}-review", required=True)
        parser.add_argument(
            f"--r{round_id}-review-sha256",
            required=True,
        )
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0077(
            release_sha=args.release_sha,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            r0065_review_path=args.r0065_review,
            r0065_review_sha256=args.r0065_review_sha256,
            r0072_review_path=args.r0072_review,
            r0072_review_sha256=args.r0072_review_sha256,
            r0076_review_path=args.r0076_review,
            r0076_review_sha256=args.r0076_review_sha256,
            runtime_spec_sha256=args.runtime_spec_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
