#!/usr/bin/env python3
"""Prepare the matched/full balanced-120M scale-evaluation queue."""
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
from basemap.round0064_evaluation import validate_seal, validate_train_bundle
from basemap.round0065_substrates import validate_scale_substrate
from basemap.round0071_substrate import validate_substrate
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
    _hf_snapshot_file_inputs,
    _materialized_chunk_inputs,
)
from experiments.prepare_round0036_queue import _static_ood_paths
from experiments.run_round0036_node import (
    CENTROIDS,
    MINILM_QUERIES,
    MINILM_QUERY_PROVENANCE,
)


ROUND_ID = "0080"
ROUND_ROOT = "/data/latent-basemap/runs/round-0080"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0080-2026-07-27.md")
SUBSTRATE_90 = (
    "/data/latent-basemap/runs/round-0071/queue/artifacts/"
    "balanced-90m-int8-substrate/balanced-90m-substrate-v1.json"
)
R0076_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0076/queue/artifacts"
)
CONTROL_TRANSFORM = os.path.join(
    R0076_ARTIFACTS,
    "coordinates-r0075-90m",
)
CONTROL_REFERENCE = os.path.join(
    R0076_ARTIFACTS,
    "high-d-reference-90m",
)
CONTROL_PANEL = os.path.join(
    R0076_ARTIFACTS,
    "panel-r0075-90m",
    "panel.json",
)
MATCHED_SAMPLE = os.path.join(
    R0076_ARTIFACTS,
    "semantic-renders",
    "full-90m-sample-rows.npy",
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
        raise RuntimeError("R0080 remains draft; refuse queue materialization")


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
        "handler_module": "experiments.round0080_nodes",
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


def prepare_round0080(
    *,
    release_sha: str,
    substrate_90_sha256: str,
    substrate_120_path: str,
    substrate_120_sha256: str,
    scale_geometry_path: str,
    scale_geometry_sha256: str,
    model_120_path: str,
    model_120_sha256: str,
    receipt_120_path: str,
    receipt_120_sha256: str,
    r0065_review_path: str,
    r0065_review_sha256: str,
    r0076_review_path: str,
    r0076_review_sha256: str,
    r0079_review_path: str,
    r0079_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0080 release SHA must be one full commit")
    substrate90 = validate_substrate(
        SUBSTRATE_90,
        expected_sha256=substrate_90_sha256,
    )
    substrate120 = validate_scale_substrate(
        substrate_120_path,
        tier="120m",
        expected_sha256=substrate_120_sha256,
    )
    bundle120 = validate_train_bundle(
        label="r0079-120m",
        model_path=model_120_path,
        model_sha256=model_120_sha256,
        train_receipt_path=receipt_120_path,
        train_receipt_sha256=receipt_120_sha256,
    )
    with open(scale_geometry_path, encoding="utf-8") as handle:
        scale_geometry = json.load(handle)
    validate_seal(scale_geometry, label="R0080 R0076 scale geometry")
    scale_signature = expected_input_signature(scale_geometry_path)
    if (
        scale_signature["sha256"] != scale_geometry_sha256
        or scale_geometry.get("schema")
        != "round0076-scale-geometry-comparison-v1"
        or scale_geometry.get("decision", {}).get(
            "90m_supported_as_deliberate_ladder_rung"
        )
        is not True
    ):
        raise RuntimeError("R0076 scale geometry changed")
    control_panel = expected_input_signature(CONTROL_PANEL)
    with open(CONTROL_PANEL, encoding="utf-8") as handle:
        control_value = json.load(handle)
    validate_seal(control_value, label="R0080 R0076 full-90M panel")
    if (
        control_value.get("schema") != "round0076-registered-panel-v1"
        or control_value.get("map_key") != "r0075-90m-on-90m"
    ):
        raise RuntimeError("R0076 full-90M control panel changed")
    reviews = {
        "0065": _require_review(
            r0065_review_path,
            expected_sha256=r0065_review_sha256,
            required_text=(
                "capability:minilm-balanced-120m-int8-input-v1",
                substrate_120_sha256,
            ),
        ),
        "0076": _require_review(
            r0076_review_path,
            expected_sha256=r0076_review_sha256,
            required_text=(
                "capability:minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
                scale_geometry_sha256,
            ),
        ),
        "0079": _require_review(
            r0079_review_path,
            expected_sha256=r0079_review_sha256,
            required_text=(model_120_sha256, receipt_120_sha256),
        ),
    }

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0080 scale evaluation queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "transform_matched": os.path.join(
            artifacts, "coordinates-r0079-on-90m"
        ),
        "transform_full": os.path.join(
            artifacts, "coordinates-r0079-120m"
        ),
        "reference_full": os.path.join(
            artifacts, "high-d-reference-120m"
        ),
        "panel_matched": os.path.join(
            artifacts, "panel-r0079-on-90m"
        ),
        "panel_full": os.path.join(artifacts, "panel-r0079-120m"),
        "comparison": os.path.join(artifacts, "scale-comparison"),
        "ood": os.path.join(artifacts, "ood-r0079-120m"),
        "renders": os.path.join(artifacts, "semantic-renders"),
        "registry": os.path.join(artifacts, "registry"),
    }
    outputs90 = substrate90["manifest"]["outputs"]
    outputs120 = substrate120["manifest"]["outputs"]
    control_coordinate_files = sorted(glob.glob(os.path.join(
        CONTROL_TRANSFORM,
        "chunk-*",
        "coordinates.npy",
    )))
    control_coordinate_inputs = _file_inputs(control_coordinate_files)
    core = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            r0065_review_path,
            r0076_review_path,
            r0079_review_path,
            SUBSTRATE_90,
            substrate_120_path,
            scale_geometry_path,
            bundle120["model"]["canonical_path"],
            bundle120["train_receipt"]["canonical_path"],
            MINILM_QUERIES,
            MINILM_QUERY_PROVENANCE,
            *CENTROIDS.values(),
            CONTROL_PANEL,
            MATCHED_SAMPLE,
            os.path.join(CONTROL_REFERENCE, "reference.npz"),
            os.path.join(CONTROL_REFERENCE, "reference-receipt.json"),
            os.path.join(CONTROL_REFERENCE, "recall50-truth.npy"),
            os.path.join(CONTROL_REFERENCE, "anchor-substrate-rows.npy"),
            os.path.join(CONTROL_TRANSFORM, "actual-transform.json"),
        ]),
        *control_coordinate_inputs,
    ])
    inputs90 = _dedupe([
        *core,
        *_file_inputs([
            outputs90["int8"]["canonical_path"],
            outputs90["scales"]["canonical_path"],
            outputs90["eligibility"]["canonical_path"],
        ]),
    ])
    inputs120 = _dedupe([
        *core,
        *_file_inputs([
            outputs120["int8"]["canonical_path"],
            outputs120["scales"]["canonical_path"],
            outputs120["eligibility"]["canonical_path"],
        ]),
    ])
    ood_inputs = _dedupe([
        *core,
        *(expected_input_signature(path) for path in _static_ood_paths()),
        *_materialized_chunk_inputs(),
        *_hf_snapshot_file_inputs(),
    ])
    model120 = {
        "model_label": "r0079-120m",
        "model_path": bundle120["model"]["canonical_path"],
        "model_sha256": bundle120["model"]["sha256"],
        "train_receipt_path": bundle120["train_receipt"]["canonical_path"],
        "train_receipt_sha256": bundle120["train_receipt"]["sha256"],
    }
    data90 = {
        "substrate_label": "balanced-90m",
        "row_count": 90_000_000,
        "rows_per_corpus": 30_000_000,
        "row_order": "first 30M rows of each FineWeb/RedPajama/Pile block",
        "int8_path": outputs90["int8"]["canonical_path"],
        "int8_sha256": outputs90["int8"]["sha256"],
        "scales_path": outputs90["scales"]["canonical_path"],
        "scales_sha256": outputs90["scales"]["sha256"],
        "eligibility_path": outputs90["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs90["eligibility"]["sha256"],
    }
    data120 = {
        "substrate_label": "balanced-120m",
        "row_count": 120_000_000,
        "rows_per_corpus": 40_000_000,
        "row_order": "first 40M rows of each FineWeb/RedPajama/Pile block",
        "int8_path": outputs120["int8"]["canonical_path"],
        "int8_sha256": outputs120["int8"]["sha256"],
        "scales_path": outputs120["scales"]["canonical_path"],
        "scales_sha256": outputs120["scales"]["sha256"],
        "eligibility_path": outputs120["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs120["eligibility"]["sha256"],
    }
    jobs = [
        _job(
            node_id="transform_r0079_on_90m",
            action="transform",
            deps=[],
            output=paths["transform_matched"],
            p90_wall_s=600,
            inputs=inputs90,
            gpu=True,
            map_key="r0079-120m-on-90m",
            **model120,
            **data90,
        ),
        _job(
            node_id="transform_r0079_120m",
            action="transform",
            deps=[],
            output=paths["transform_full"],
            p90_wall_s=800,
            inputs=inputs120,
            gpu=True,
            map_key="r0079-120m-on-120m",
            **model120,
            **data120,
        ),
        _job(
            node_id="high_d_reference_120m",
            action="high_d_reference",
            deps=[],
            output=paths["reference_full"],
            p90_wall_s=1_600,
            inputs=inputs120,
            gpu=True,
            reference_schema="round0080-high-d-reference-v1",
            **data120,
        ),
        _job(
            node_id="panel_r0079_on_90m",
            action="panel",
            deps=["transform_r0079_on_90m"],
            output=paths["panel_matched"],
            p90_wall_s=2_400,
            inputs=inputs90,
            gpu=True,
            map_key="r0079-120m-on-90m",
            transform_output=paths["transform_matched"],
            reference_output=CONTROL_REFERENCE,
            panel_schema="round0080-registered-panel-v1",
            **model120,
            **data90,
        ),
        _job(
            node_id="panel_r0079_120m",
            action="panel",
            deps=["transform_r0079_120m", "high_d_reference_120m"],
            output=paths["panel_full"],
            p90_wall_s=3_200,
            inputs=inputs120,
            gpu=True,
            map_key="r0079-120m-on-120m",
            transform_output=paths["transform_full"],
            reference_output=paths["reference_full"],
            panel_schema="round0080-registered-panel-v1",
            **model120,
            **data120,
        ),
        _job(
            node_id="scale_comparison",
            action="comparison",
            deps=["panel_r0079_on_90m", "panel_r0079_120m"],
            output=paths["comparison"],
            p90_wall_s=60,
            inputs=core,
            gpu=False,
            control_panel=CONTROL_PANEL,
            matched_panel=os.path.join(paths["panel_matched"], "panel.json"),
            full_panel=os.path.join(paths["panel_full"], "panel.json"),
        ),
        _job(
            node_id="ood_r0079_120m",
            action="ood",
            deps=["transform_r0079_120m"],
            output=paths["ood"],
            p90_wall_s=300,
            inputs=ood_inputs,
            gpu=True,
            map_key="r0079-120m-on-120m",
            transform_output=paths["transform_full"],
            ood_schema="round0080-ood-bundle-v1",
            **model120,
        ),
        _job(
            node_id="matched_renders",
            action="renders",
            deps=["transform_r0079_on_90m", "transform_r0079_120m"],
            output=paths["renders"],
            p90_wall_s=180,
            inputs=core,
            gpu=False,
            matched_sample_rows=MATCHED_SAMPLE,
            matched_maps=[
                {
                    "map_key": "r0075-90m-on-90m",
                    "transform_output": CONTROL_TRANSFORM,
                },
                {
                    "map_key": "r0079-120m-on-90m",
                    "transform_output": paths["transform_matched"],
                },
            ],
            full_transform=paths["transform_full"],
            matched_eligibility_path=outputs90["eligibility"][
                "canonical_path"
            ],
            matched_eligibility_sha256=outputs90["eligibility"]["sha256"],
            full_eligibility_path=outputs120["eligibility"][
                "canonical_path"
            ],
            full_eligibility_sha256=outputs120["eligibility"]["sha256"],
        ),
        _job(
            node_id="registry_publication",
            action="registry",
            deps=[
                "panel_r0079_120m",
                "ood_r0079_120m",
                "matched_renders",
            ],
            output=paths["registry"],
            p90_wall_s=180,
            inputs=core,
            gpu=False,
        ),
    ]
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=3.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0080-scale-evaluation-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0065", "0076", "0079"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-120m-int8-input-v1",
        "minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
        "minilm-balanced-120m-trained-model-seed42-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-90m-120m-scale-geometry-v1",
        "minilm-balanced-120m-map-registry-v1",
    ]
    manifest["training_performed"] = False
    manifest["late_bound_model"] = {
        "model": bundle120["model"],
        "train_receipt": bundle120["train_receipt"],
        "review": reviews["0079"],
    }
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "substrate_90m": substrate90["signature"],
        "substrate_120m": substrate120["signature"],
        "scale_geometry_90m": scale_signature,
        "control_panel_90m": control_panel,
    }
    manifest["scientific_contract"] = {
        "primary_comparison": (
            "90M and 120M models on the exact same R0071 representative "
            "rows, high-D reference, and anchors"
        ),
        "120m_noninferiority_control": "90m",
        "matched_noninferiority_margins": {
            "ffr": 0.02,
            "density": 0.05,
            "purity_k256": 0.05,
            "purity_k1024": 0.05,
            "projection_ffr": 0.02,
        },
        "density": {
            "anchors": "representative-only",
            "candidate_universe": "representative-only",
            "legacy_absolute_floor_is_decision_gating": False,
            "new_threshold_calibrated": False,
        },
        "full_120m_non_density_checks_required": True,
        "ood": (
            "Dadabase, TREC-COVID, code, science, Latin; map-card evidence, "
            "non-gating"
        ),
    }
    manifest["jobs"] = jobs
    gpu_jobs = [
        job for job in jobs if job["node_policy"]["gpu_required"]
    ]
    manifest["p90_gpu_seconds"] = {
        **{job["id"]: job["p90_wall_s"] for job in gpu_jobs},
        "total": sum(job["p90_wall_s"] for job in gpu_jobs),
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-90-sha256", required=True)
    parser.add_argument("--substrate-120", required=True)
    parser.add_argument("--substrate-120-sha256", required=True)
    parser.add_argument("--scale-geometry", required=True)
    parser.add_argument("--scale-geometry-sha256", required=True)
    parser.add_argument("--model-120", required=True)
    parser.add_argument("--model-120-sha256", required=True)
    parser.add_argument("--receipt-120", required=True)
    parser.add_argument("--receipt-120-sha256", required=True)
    for round_id in ("0065", "0076", "0079"):
        parser.add_argument(f"--review-{round_id}", required=True)
        parser.add_argument(f"--review-{round_id}-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0080(
            release_sha=args.release_sha,
            substrate_90_sha256=args.substrate_90_sha256,
            substrate_120_path=args.substrate_120,
            substrate_120_sha256=args.substrate_120_sha256,
            scale_geometry_path=args.scale_geometry,
            scale_geometry_sha256=args.scale_geometry_sha256,
            model_120_path=args.model_120,
            model_120_sha256=args.model_120_sha256,
            receipt_120_path=args.receipt_120,
            receipt_120_sha256=args.receipt_120_sha256,
            r0065_review_path=args.review_0065,
            r0065_review_sha256=args.review_0065_sha256,
            r0076_review_path=args.review_0076,
            r0076_review_sha256=args.review_0076_sha256,
            r0079_review_path=args.review_0079,
            r0079_review_sha256=args.review_0079_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
