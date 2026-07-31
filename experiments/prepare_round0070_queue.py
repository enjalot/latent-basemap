#!/usr/bin/env python3
"""Prepare, but never launch, the matched 30M density-factorial queue."""
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
from basemap.round0064_evaluation import validate_train_bundle
from experiments.prepare_round0020_0022_queues import (
    INPUT_PACK_MANIFEST,
    LAB_ROOT,
    _base_manifest,
    _coordinate_inputs,
    _dedupe,
    _file_inputs,
    _materialized_chunk_inputs,
)


ROUND_ID = "0070"
ROUND_ROOT = "/data/latent-basemap/runs/round-0070"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0070-2026-07-27.md")
SUBSTRATE_30 = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
R0019_ARTIFACTS = "/data/latent-basemap/runs/round-0019/queue/artifacts"
R0019_MODEL = os.path.join(R0019_ARTIFACTS, "train", "model.pt")
R0019_RECEIPT = os.path.join(
    R0019_ARTIFACTS,
    "train",
    "train-receipt.json",
)
R0019_COORDINATES = os.path.join(R0019_ARTIFACTS, "coordinates")
R0019_PANEL = os.path.join(R0019_ARTIFACTS, "panel", "panel.json")
R0061_ARTIFACTS = "/data/latent-basemap/runs/round-0061/queue/artifacts"
R0061_MODEL = os.path.join(
    R0061_ARTIFACTS,
    "train-balanced-30m-int8",
    "model.pt",
)
R0061_RECEIPT = os.path.join(
    R0061_ARTIFACTS,
    "train-balanced-30m-int8",
    "train-receipt.json",
)
R0064_ARTIFACTS = "/data/latent-basemap/runs/round-0064/queue/artifacts"
R0064_MODERN_COORDINATES = os.path.join(
    R0064_ARTIFACTS,
    "coordinates-r0061-30m",
)
R0064_MODERN_PANEL = os.path.join(
    R0064_ARTIFACTS,
    "panel-r0061-30m",
    "panel.json",
)
R0064_REFERENCE = os.path.join(
    R0064_ARTIFACTS,
    "high-d-reference-30m",
)
R0064_SCALE_COMPARISON = os.path.join(
    R0064_ARTIFACTS,
    "scale-comparison",
    "scale-comparison.json",
)
COMMON_ANCHORS = os.path.join(
    R0064_REFERENCE,
    "anchor-substrate-rows.npy",
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
        raise RuntimeError("R0070 remains draft; refuse queue materialization")


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    p90_wall_s: float,
    inputs: list[dict[str, Any]],
    gpu: bool,
    **extra: Any,
) -> dict[str, Any]:
    return {
        **extra,
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0070_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(
            os.path.dirname(output),
            f"{node_id}.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": float(p90_wall_s),
        "node_policy": {
            "gpu_required": gpu,
            "training_performed": False,
        },
    }


def prepare_round0070(
    *,
    release_sha: str,
    substrate_30_sha256: str,
    legacy_model_sha256: str,
    legacy_receipt_sha256: str,
    balanced_model_sha256: str,
    balanced_receipt_sha256: str,
    review_0019_path: str,
    review_0019_sha256: str,
    review_0053_path: str,
    review_0053_sha256: str,
    review_0061_path: str,
    review_0061_sha256: str,
    review_0064_path: str,
    review_0064_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0070 release SHA must be one full commit")
    substrate = validate_control_substrate(
        SUBSTRATE_30,
        expected_sha256=substrate_30_sha256,
    )
    legacy_model = expected_input_signature(R0019_MODEL)
    legacy_receipt = expected_input_signature(R0019_RECEIPT)
    if (
        legacy_model["sha256"] != legacy_model_sha256
        or legacy_receipt["sha256"] != legacy_receipt_sha256
    ):
        raise RuntimeError("R0019 model bundle bytes changed")
    balanced = validate_train_bundle(
        label="r0061-30m",
        model_path=R0061_MODEL,
        model_sha256=balanced_model_sha256,
        train_receipt_path=R0061_RECEIPT,
        train_receipt_sha256=balanced_receipt_sha256,
    )
    scale_comparison = expected_input_signature(R0064_SCALE_COMPARISON)
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
        "0061": _require_review(
            review_0061_path,
            expected_sha256=review_0061_sha256,
            required_text=(
                "capability:minilm-balanced-30m-int8-trained-model-seed42-v1",
                balanced_model_sha256,
                balanced_receipt_sha256,
            ),
        ),
        "0064": _require_review(
            review_0064_path,
            expected_sha256=review_0064_sha256,
            required_text=(
                "capability:minilm-balanced-30m-60m-scale-geometry-v1",
                scale_comparison["sha256"],
            ),
        ),
    }

    outputs = substrate["manifest"]["outputs"]
    common_anchors = expected_input_signature(COMMON_ANCHORS)
    import numpy as np

    common_anchor_rows = np.load(
        COMMON_ANCHORS,
        mmap_mode="r",
        allow_pickle=False,
    )
    if common_anchor_rows.shape != (10_000,):
        raise RuntimeError("R0064 common anchor geometry changed")
    common_anchor_sha256 = ordered_array_sha256(common_anchor_rows)
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0070 density-factorial queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "modern_original": os.path.join(
            artifacts,
            "coordinates-r0061-on-original-fp16-30m",
        ),
        "original_reference": os.path.join(
            artifacts,
            "original-high-d-reference",
        ),
        "factorial": os.path.join(
            artifacts,
            "density-factorial",
        ),
    }
    modern_coordinate_files = sorted(
        glob.glob(
            os.path.join(
                R0064_MODERN_COORDINATES,
                "chunk-*",
                "coordinates.npy",
            )
        )
    )
    core = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            review_0019_path,
            review_0053_path,
            review_0061_path,
            review_0064_path,
            INPUT_PACK_MANIFEST,
            SUBSTRATE_30,
            R0019_MODEL,
            R0019_RECEIPT,
            R0019_PANEL,
            R0061_MODEL,
            R0061_RECEIPT,
            R0064_SCALE_COMPARISON,
            R0064_MODERN_PANEL,
            os.path.join(
                R0064_MODERN_COORDINATES,
                "actual-transform.json",
            ),
            *modern_coordinate_files,
            os.path.join(R0064_REFERENCE, "reference.npz"),
            os.path.join(R0064_REFERENCE, "reference-receipt.json"),
            COMMON_ANCHORS,
        ]),
        *_coordinate_inputs(),
        *_materialized_chunk_inputs(),
    ])
    balanced_inputs = _dedupe([
        *core,
        *_file_inputs([
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
        ]),
    ])
    legacy_fields = {
        "legacy_model_path": R0019_MODEL,
        "legacy_model_sha256": legacy_model_sha256,
        "legacy_receipt_path": R0019_RECEIPT,
        "legacy_receipt_sha256": legacy_receipt_sha256,
    }
    balanced_fields = {
        "balanced_model_path": balanced["model"]["canonical_path"],
        "balanced_model_sha256": balanced["model"]["sha256"],
        "balanced_receipt_path": balanced["train_receipt"]["canonical_path"],
        "balanced_receipt_sha256": balanced["train_receipt"]["sha256"],
    }
    substrate_fields = {
        "int8_path": outputs["int8"]["canonical_path"],
        "int8_sha256": outputs["int8"]["sha256"],
        "scales_path": outputs["scales"]["canonical_path"],
        "scales_sha256": outputs["scales"]["sha256"],
        "eligibility_path": outputs["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs["eligibility"]["sha256"],
    }
    jobs = [
        _job(
            node_id="transform_modern_on_original_fp16",
            action="modern_transform",
            deps=[],
            output=paths["modern_original"],
            p90_wall_s=180,
            inputs=core,
            gpu=True,
            map_key="r0061-on-original-fp16-30m",
            **balanced_fields,
        ),
        _job(
            node_id="original_high_d_reference",
            action="original_reference",
            deps=[],
            output=paths["original_reference"],
            p90_wall_s=1_500,
            inputs=balanced_inputs,
            gpu=True,
            common_anchor_rows_path=COMMON_ANCHORS,
            common_anchor_rows_sha256=common_anchor_sha256,
            **substrate_fields,
        ),
        _job(
            node_id="matched_density_factorial",
            action="density_factorial",
            deps=[
                "transform_modern_on_original_fp16",
                "original_high_d_reference",
            ],
            output=paths["factorial"],
            p90_wall_s=900,
            inputs=balanced_inputs,
            gpu=True,
            common_anchor_rows_path=COMMON_ANCHORS,
            common_anchor_rows_sha256=common_anchor_sha256,
            fp16_reference_archive=os.path.join(
                paths["original_reference"],
                "fp16-high-d-radii.npz",
            ),
            int8_reference_path=os.path.join(
                R0064_REFERENCE,
                "reference.npz",
            ),
            legacy_original_coordinates=R0019_COORDINATES,
            modern_original_coordinates=paths["modern_original"],
            modern_int8_coordinates=R0064_MODERN_COORDINATES,
            legacy_original_panel_path=R0019_PANEL,
            modern_int8_panel_path=R0064_MODERN_PANEL,
            **legacy_fields,
            **balanced_fields,
            **substrate_fields,
        ),
    ]
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=1.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0070-density-factorial-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0019", "0053", "0061", "0064"]
    manifest["capability_dependencies"] = [
        "30m-minilm-map-seed42-duplicate-cap",
        "minilm-balanced-30m-int8-input-v1",
        "minilm-balanced-30m-int8-trained-model-seed42-v1",
        "minilm-balanced-30m-60m-scale-geometry-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-30m-density-model-universe-factorial-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "r0064_scale_comparison": scale_comparison,
        "legacy_model": legacy_model,
        "legacy_train_receipt": legacy_receipt,
        "modern_model": balanced["model"],
        "modern_train_receipt": balanced["train_receipt"],
        "balanced_substrate": expected_input_signature(SUBSTRATE_30),
        "common_anchor_rows": common_anchors,
        "common_anchor_rows_sha256": common_anchor_sha256,
    }
    manifest["scientific_contract"] = {
        "design": (
            "cross R0019 and R0061 models with original all-row and exact-"
            "representative 30M universes at fixed fp16 precision on "
            "identical global anchors; report int8 as a separate bridge"
        ),
        "primary_metric": "correlation of log exact high-/low-D mean-k15 radii",
        "registered_bands": {
            "near_equivalence": 0.05,
            "material_main_effect": 0.20,
            "interaction": 0.10,
        },
        "calibrates_density_threshold": False,
        "separates_graph_from_sampler": False,
        "authorizes_larger_training_rung": False,
    }
    manifest["jobs"] = jobs
    gpu_seconds = sum(float(job["p90_wall_s"]) for job in jobs)
    manifest["p90_gpu_seconds"] = {
        **{job["id"]: job["p90_wall_s"] for job in jobs},
        "total": gpu_seconds,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-30-sha256", required=True)
    parser.add_argument("--legacy-model-sha256", required=True)
    parser.add_argument("--legacy-receipt-sha256", required=True)
    parser.add_argument("--balanced-model-sha256", required=True)
    parser.add_argument("--balanced-receipt-sha256", required=True)
    for round_id in ("0019", "0053", "0061", "0064"):
        parser.add_argument(f"--review-{round_id}", required=True)
        parser.add_argument(f"--review-{round_id}-sha256", required=True)
    args = parser.parse_args(argv)
    path = prepare_round0070(
        release_sha=args.release_sha,
        substrate_30_sha256=args.substrate_30_sha256,
        legacy_model_sha256=args.legacy_model_sha256,
        legacy_receipt_sha256=args.legacy_receipt_sha256,
        balanced_model_sha256=args.balanced_model_sha256,
        balanced_receipt_sha256=args.balanced_receipt_sha256,
        review_0019_path=args.review_0019,
        review_0019_sha256=args.review_0019_sha256,
        review_0053_path=args.review_0053,
        review_0053_sha256=args.review_0053_sha256,
        review_0061_path=args.review_0061,
        review_0061_sha256=args.review_0061_sha256,
        review_0064_path=args.review_0064,
        review_0064_sha256=args.review_0064_sha256,
    )
    print(json.dumps({"queue_manifest": path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
