#!/usr/bin/env python3
"""Prepare, but never launch, the duplicate-anchor leverage queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0053_program import validate_control_substrate
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ID = "0074"
ROUND_ROOT = "/data/latent-basemap/runs/round-0074"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0074-2026-07-27.md")
R0019_ARTIFACTS = "/data/latent-basemap/runs/round-0019/queue/artifacts"
R0019_REFERENCE = os.path.join(
    R0019_ARTIFACTS,
    "high-d-reference",
    "reference.npz",
)
R0019_REFERENCE_RECEIPT = os.path.join(
    R0019_ARTIFACTS,
    "high-d-reference",
    "reference-receipt.json",
)
R0019_COORDINATES = os.path.join(R0019_ARTIFACTS, "coordinates")
R0019_PANEL = os.path.join(R0019_ARTIFACTS, "panel", "panel.json")
SUBSTRATE_30 = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
R0070_ARTIFACTS = "/data/latent-basemap/runs/round-0070/queue/artifacts"
R0070_MODERN_COORDINATES = os.path.join(
    R0070_ARTIFACTS,
    "coordinates-r0061-on-original-fp16-30m",
)
R0070_FACTORIAL = os.path.join(
    R0070_ARTIFACTS,
    "density-factorial",
    "density-factorial.json",
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
        raise RuntimeError("R0074 remains draft; refuse queue materialization")


def prepare_round0074(
    *,
    release_sha: str,
    substrate_30_sha256: str,
    legacy_model_sha256: str,
    modern_model_sha256: str,
    legacy_reference_sha256: str,
    r0070_factorial_sha256: str,
    review_0019_path: str,
    review_0019_sha256: str,
    review_0053_path: str,
    review_0053_sha256: str,
    review_0070_path: str,
    review_0070_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0074 release SHA must be one full commit")
    substrate = validate_control_substrate(
        SUBSTRATE_30,
        expected_sha256=substrate_30_sha256,
    )
    eligibility = substrate["manifest"]["outputs"]["eligibility"]
    reference = expected_input_signature(R0019_REFERENCE)
    factorial = expected_input_signature(R0070_FACTORIAL)
    if reference["sha256"] != legacy_reference_sha256:
        raise RuntimeError("R0019 reference bytes changed")
    if factorial["sha256"] != r0070_factorial_sha256:
        raise RuntimeError("R0070 factorial bytes changed")

    reviews = {
        "0019": _require_review(
            review_0019_path,
            expected_sha256=review_0019_sha256,
            required_text=(
                "capability:30m-minilm-map-seed42-duplicate-cap",
                legacy_model_sha256,
            ),
        ),
        "0053": _require_review(
            review_0053_path,
            expected_sha256=review_0053_sha256,
            required_text=(
                "capability:minilm-balanced-30m-int8-input-v1",
                substrate_30_sha256,
            ),
        ),
        "0070": _require_review(
            review_0070_path,
            expected_sha256=review_0070_sha256,
            required_text=(
                "capability:minilm-30m-density-model-universe-factorial-v1",
                r0070_factorial_sha256,
            ),
        ),
    }

    import numpy as np

    with np.load(R0019_REFERENCE, allow_pickle=False) as archive:
        anchor_rows = np.asarray(archive["anchor_ids"], dtype=np.int64)
        high_radius = np.asarray(archive["r_hd"], dtype=np.float64)
    if anchor_rows.shape != (10_000,) or high_radius.shape != (10_000,):
        raise RuntimeError("R0019 reference geometry changed")

    legacy_coordinate_files = sorted(
        glob.glob(
            os.path.join(
                R0019_COORDINATES,
                "chunk-*",
                "coordinates.npy",
            )
        )
    )
    modern_coordinate_files = sorted(
        glob.glob(
            os.path.join(
                R0070_MODERN_COORDINATES,
                "chunk-*",
                "coordinates.npy",
            )
        )
    )
    if len(legacy_coordinate_files) != 30 or len(modern_coordinate_files) != 6:
        raise RuntimeError("coordinate stream membership changed")

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0074 duplicate-anchor queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "duplicate-anchor-leverage")
    inputs = _dedupe(_file_inputs([
        ROUND_FILE,
        review_0019_path,
        review_0053_path,
        review_0070_path,
        SUBSTRATE_30,
        eligibility["canonical_path"],
        R0019_REFERENCE,
        R0019_REFERENCE_RECEIPT,
        R0019_PANEL,
        os.path.join(R0019_COORDINATES, "actual-transform.json"),
        *legacy_coordinate_files,
        os.path.join(R0070_MODERN_COORDINATES, "actual-transform.json"),
        *modern_coordinate_files,
        R0070_FACTORIAL,
    ]))
    job = {
        "id": "duplicate_anchor_leverage",
        "action": "anchor_leverage",
        "handler_module": "experiments.round0074_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "duplicate_anchor_leverage.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 120.0,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
        "legacy_reference_path": R0019_REFERENCE,
        "legacy_reference_sha256": reference["sha256"],
        "legacy_anchor_rows_sha256": ordered_array_sha256(anchor_rows),
        "legacy_high_radius_sha256": ordered_array_sha256(high_radius),
        "eligibility_path": eligibility["canonical_path"],
        "eligibility_sha256": eligibility["sha256"],
        "legacy_coordinates": R0019_COORDINATES,
        "modern_coordinates": R0070_MODERN_COORDINATES,
        "legacy_model_sha256": legacy_model_sha256,
        "modern_model_sha256": modern_model_sha256,
        "legacy_panel_path": R0019_PANEL,
        "r0070_factorial_path": R0070_FACTORIAL,
    }
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.10,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0074-duplicate-anchor-leverage-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0019", "0053", "0070"]
    manifest["capability_dependencies"] = [
        "30m-minilm-map-seed42-duplicate-cap",
        "minilm-balanced-30m-int8-input-v1",
        "minilm-30m-density-model-universe-factorial-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-30m-density-anchor-leverage-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "balanced_substrate": expected_input_signature(SUBSTRATE_30),
        "eligibility": expected_input_signature(eligibility["canonical_path"]),
        "legacy_reference": reference,
        "r0070_factorial": factorial,
    }
    manifest["scientific_contract"] = {
        "design": (
            "hold both model and original-fp16 all-row candidate universe "
            "fixed while comparing the R0019 all-row anchor population to the "
            "reviewed R0070 representative anchor population; attribute the "
            "R0019 Pearson covariance to exact-family size"
        ),
        "primary_metric": "correlation of log exact high-/low-D mean-k15 radii",
        "registered_bands": {
            "material_correlation_delta": 0.20,
            "dominant_covariance_numerator_fraction": 0.50,
        },
        "must_exactly_replay_r0019_density": 0.7767,
        "expected_anchor_family_facts": {
            "anchors_in_exact_families": 124,
            "anchors_in_families_ge_16": 20,
            "unique_canonical_family_rows": 9_983,
            "maximum_family_size": 30_088,
            "zero_high_d_radii": 20,
        },
        "calibrates_density_threshold": False,
        "separates_graph_from_sampler": False,
        "authorizes_larger_training_rung": False,
    }
    manifest["jobs"] = [job]
    manifest["p90_gpu_seconds"] = {
        "duplicate_anchor_leverage": 120.0,
        "total": 120.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-30-sha256", required=True)
    parser.add_argument("--legacy-model-sha256", required=True)
    parser.add_argument("--modern-model-sha256", required=True)
    parser.add_argument("--legacy-reference-sha256", required=True)
    parser.add_argument("--r0070-factorial-sha256", required=True)
    for round_id in ("0019", "0053", "0070"):
        parser.add_argument(f"--review-{round_id}", required=True)
        parser.add_argument(f"--review-{round_id}-sha256", required=True)
    args = parser.parse_args(argv)
    path = prepare_round0074(
        release_sha=args.release_sha,
        substrate_30_sha256=args.substrate_30_sha256,
        legacy_model_sha256=args.legacy_model_sha256,
        modern_model_sha256=args.modern_model_sha256,
        legacy_reference_sha256=args.legacy_reference_sha256,
        r0070_factorial_sha256=args.r0070_factorial_sha256,
        review_0019_path=args.review_0019,
        review_0019_sha256=args.review_0019_sha256,
        review_0053_path=args.review_0053,
        review_0053_sha256=args.review_0053_sha256,
        review_0070_path=args.review_0070,
        review_0070_sha256=args.review_0070_sha256,
    )
    print(json.dumps({"queue_manifest": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
