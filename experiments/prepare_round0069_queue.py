#!/usr/bin/env python3
"""Prepare, but never launch, the balanced 45M evaluation queue."""
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
from basemap.round0053_program import validate_control_substrate
from basemap.round0064_evaluation import validate_train_bundle
from basemap.round0065_substrates import validate_scale_substrate
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


ROUND_ID = "0069"
ROUND_ROOT = "/data/latent-basemap/runs/round-0069"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0069-2026-07-27.md")
SUBSTRATE_30 = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
SUBSTRATE_45 = (
    "/data/latent-basemap/runs/round-0065/queue/artifacts/"
    "balanced-45m-int8-substrate/balanced-45m-substrate-v1.json"
)
R0064_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0064/queue/artifacts"
)
R0064_REFERENCE_30 = os.path.join(
    R0064_ARTIFACTS,
    "high-d-reference-30m",
)
R0064_CONTROL_PANEL = os.path.join(
    R0064_ARTIFACTS,
    "panel-r0061-30m",
    "panel.json",
)
R0064_UPPER_MATCHED_PANEL = os.path.join(
    R0064_ARTIFACTS,
    "panel-r0063-on-30m",
    "panel.json",
)
R0064_CONTROL_TRANSFORM = os.path.join(
    R0064_ARTIFACTS,
    "coordinates-r0061-30m",
)
R0064_UPPER_MATCHED_TRANSFORM = os.path.join(
    R0064_ARTIFACTS,
    "coordinates-r0063-on-30m",
)
R0064_MATCHED_SAMPLE = os.path.join(
    R0064_ARTIFACTS,
    "semantic-renders",
    "matched-30m-sample-rows.npy",
)
R0064_SCALE_COMPARISON = os.path.join(
    R0064_ARTIFACTS,
    "scale-comparison",
    "scale-comparison.json",
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
        raise RuntimeError("R0069 remains draft; refuse queue materialization")


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
        "handler_module": "experiments.round0069_nodes",
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


def prepare_round0069(
    *,
    release_sha: str,
    substrate_30_sha256: str,
    substrate_45_sha256: str,
    model_45_path: str,
    model_45_sha256: str,
    receipt_45_path: str,
    receipt_45_sha256: str,
    r0064_review_path: str,
    r0064_review_sha256: str,
    r0065_review_path: str,
    r0065_review_sha256: str,
    r0068_review_path: str,
    r0068_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0069 release SHA must be one full commit")
    substrate30 = validate_control_substrate(
        SUBSTRATE_30,
        expected_sha256=substrate_30_sha256,
    )
    substrate45 = validate_scale_substrate(
        SUBSTRATE_45,
        tier="45m",
        expected_sha256=substrate_45_sha256,
    )
    bundle45 = validate_train_bundle(
        label="r0068-45m",
        model_path=model_45_path,
        model_sha256=model_45_sha256,
        train_receipt_path=receipt_45_path,
        train_receipt_sha256=receipt_45_sha256,
    )
    reference_receipt = expected_input_signature(
        os.path.join(R0064_REFERENCE_30, "reference-receipt.json")
    )
    control_panel = expected_input_signature(R0064_CONTROL_PANEL)
    upper_panel = expected_input_signature(R0064_UPPER_MATCHED_PANEL)
    scale_comparison = expected_input_signature(R0064_SCALE_COMPARISON)
    review64 = _require_review(
        r0064_review_path,
        expected_sha256=r0064_review_sha256,
        required_text=(
            "minilm-balanced-30m-60m-scale-geometry-v1",
            scale_comparison["sha256"],
        ),
    )
    review65 = _require_review(
        r0065_review_path,
        expected_sha256=r0065_review_sha256,
        required_text=(substrate_45_sha256, "balanced-45m"),
    )
    review68 = _require_review(
        r0068_review_path,
        expected_sha256=r0068_review_sha256,
        required_text=(model_45_sha256, receipt_45_sha256),
    )

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0069 evaluation queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "transform_matched": os.path.join(
            artifacts,
            "coordinates-r0068-on-30m",
        ),
        "transform_full": os.path.join(
            artifacts,
            "coordinates-r0068-45m",
        ),
        "reference45": os.path.join(
            artifacts,
            "high-d-reference-45m",
        ),
        "panel_matched": os.path.join(
            artifacts,
            "panel-r0068-on-30m",
        ),
        "panel_full": os.path.join(
            artifacts,
            "panel-r0068-45m",
        ),
        "density": os.path.join(
            artifacts,
            "matched-density-diagnostic",
        ),
        "comparison": os.path.join(
            artifacts,
            "scale-comparison",
        ),
        "ood": os.path.join(artifacts, "ood-r0068-45m"),
        "renders": os.path.join(artifacts, "semantic-renders"),
        "registry": os.path.join(artifacts, "registry"),
    }
    outputs30 = substrate30["manifest"]["outputs"]
    outputs45 = substrate45["manifest"]["outputs"]
    existing_coordinate_files = sorted({
        *glob.glob(os.path.join(
            R0064_CONTROL_TRANSFORM,
            "chunk-*",
            "coordinates.npy",
        )),
        *glob.glob(os.path.join(
            R0064_UPPER_MATCHED_TRANSFORM,
            "chunk-*",
            "coordinates.npy",
        )),
    })
    core = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            r0064_review_path,
            r0065_review_path,
            r0068_review_path,
            SUBSTRATE_30,
            SUBSTRATE_45,
            bundle45["model"]["canonical_path"],
            bundle45["train_receipt"]["canonical_path"],
            MINILM_QUERIES,
            MINILM_QUERY_PROVENANCE,
            *CENTROIDS.values(),
            R0064_SCALE_COMPARISON,
            R0064_CONTROL_PANEL,
            R0064_UPPER_MATCHED_PANEL,
            R0064_MATCHED_SAMPLE,
            os.path.join(
                R0064_CONTROL_TRANSFORM,
                "actual-transform.json",
            ),
            os.path.join(
                R0064_UPPER_MATCHED_TRANSFORM,
                "actual-transform.json",
            ),
            os.path.join(R0064_REFERENCE_30, "reference.npz"),
            os.path.join(R0064_REFERENCE_30, "reference-receipt.json"),
            os.path.join(R0064_REFERENCE_30, "recall50-truth.npy"),
            os.path.join(
                R0064_REFERENCE_30,
                "anchor-substrate-rows.npy",
            ),
            *existing_coordinate_files,
        ]),
    ])
    inputs30 = _dedupe([
        *core,
        *_file_inputs([
            outputs30["int8"]["canonical_path"],
            outputs30["scales"]["canonical_path"],
            outputs30["eligibility"]["canonical_path"],
        ]),
    ])
    inputs45 = _dedupe([
        *core,
        *_file_inputs([
            outputs45["int8"]["canonical_path"],
            outputs45["scales"]["canonical_path"],
            outputs45["eligibility"]["canonical_path"],
        ]),
    ])
    ood_inputs = _dedupe([
        *core,
        *(expected_input_signature(path) for path in _static_ood_paths()),
        *_materialized_chunk_inputs(),
        *_hf_snapshot_file_inputs(),
    ])
    model45 = {
        "model_label": "r0068-45m",
        "model_path": bundle45["model"]["canonical_path"],
        "model_sha256": bundle45["model"]["sha256"],
        "train_receipt_path": bundle45["train_receipt"]["canonical_path"],
        "train_receipt_sha256": bundle45["train_receipt"]["sha256"],
    }
    data30 = {
        "substrate_label": "balanced-30m",
        "row_count": 30_000_000,
        "rows_per_corpus": 10_000_000,
        "row_order": (
            "first 10M rows of each FineWeb/RedPajama/Pile block"
        ),
        "int8_path": outputs30["int8"]["canonical_path"],
        "int8_sha256": outputs30["int8"]["sha256"],
        "scales_path": outputs30["scales"]["canonical_path"],
        "scales_sha256": outputs30["scales"]["sha256"],
        "eligibility_path": outputs30["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs30["eligibility"]["sha256"],
    }
    data45 = {
        "substrate_label": "balanced-45m",
        "row_count": 45_000_000,
        "rows_per_corpus": 15_000_000,
        "row_order": (
            "first 15M rows of each FineWeb/RedPajama/Pile block"
        ),
        "int8_path": outputs45["int8"]["canonical_path"],
        "int8_sha256": outputs45["int8"]["sha256"],
        "scales_path": outputs45["scales"]["canonical_path"],
        "scales_sha256": outputs45["scales"]["sha256"],
        "eligibility_path": outputs45["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs45["eligibility"]["sha256"],
    }
    panel_schema = "round0069-registered-panel-v1"
    jobs = [
        _job(
            node_id="transform_r0068_on_30m",
            action="transform",
            deps=[],
            output=paths["transform_matched"],
            p90_wall_s=180,
            inputs=inputs30,
            gpu=True,
            map_key="r0068-45m-on-30m",
            **model45,
            **data30,
        ),
        _job(
            node_id="transform_r0068_45m",
            action="transform",
            deps=[],
            output=paths["transform_full"],
            p90_wall_s=240,
            inputs=inputs45,
            gpu=True,
            map_key="r0068-45m-on-45m",
            **model45,
            **data45,
        ),
        _job(
            node_id="high_d_reference_45m",
            action="high_d_reference",
            deps=[],
            output=paths["reference45"],
            p90_wall_s=600,
            inputs=inputs45,
            gpu=True,
            reference_schema="round0069-high-d-reference-v1",
            **data45,
        ),
        _job(
            node_id="panel_r0068_on_30m",
            action="panel",
            deps=["transform_r0068_on_30m"],
            output=paths["panel_matched"],
            p90_wall_s=900,
            inputs=inputs30,
            gpu=True,
            map_key="r0068-45m-on-30m",
            transform_output=paths["transform_matched"],
            reference_output=R0064_REFERENCE_30,
            panel_schema=panel_schema,
            **model45,
            **data30,
        ),
        _job(
            node_id="panel_r0068_45m",
            action="panel",
            deps=["transform_r0068_45m", "high_d_reference_45m"],
            output=paths["panel_full"],
            p90_wall_s=1_500,
            inputs=inputs45,
            gpu=True,
            map_key="r0068-45m-on-45m",
            transform_output=paths["transform_full"],
            reference_output=paths["reference45"],
            panel_schema=panel_schema,
            **model45,
            **data45,
        ),
        _job(
            node_id="matched_density_diagnostic",
            action="density_diagnostic",
            deps=["panel_r0068_on_30m"],
            output=paths["density"],
            p90_wall_s=360,
            inputs=inputs30,
            gpu=True,
            reference_output=R0064_REFERENCE_30,
            matched_maps=[
                {
                    "map_key": "r0061-30m-on-30m",
                    "transform_output": R0064_CONTROL_TRANSFORM,
                    "panel_path": R0064_CONTROL_PANEL,
                },
                {
                    "map_key": "r0068-45m-on-30m",
                    "transform_output": paths["transform_matched"],
                    "panel_path": os.path.join(
                        paths["panel_matched"],
                        "panel.json",
                    ),
                },
                {
                    "map_key": "r0063-60m-on-30m",
                    "transform_output": R0064_UPPER_MATCHED_TRANSFORM,
                    "panel_path": R0064_UPPER_MATCHED_PANEL,
                },
            ],
            **data30,
        ),
        _job(
            node_id="scale_comparison",
            action="comparison",
            deps=[
                "panel_r0068_on_30m",
                "panel_r0068_45m",
                "matched_density_diagnostic",
            ],
            output=paths["comparison"],
            p90_wall_s=60,
            inputs=core,
            gpu=False,
            density_diagnostic=paths["density"],
            panels={
                "control_30m": {
                    "path": R0064_CONTROL_PANEL,
                    "key": "r0061-30m-on-30m",
                    "schema": "round0064-registered-panel-v1",
                },
                "treatment_45m_matched": {
                    "path": os.path.join(
                        paths["panel_matched"],
                        "panel.json",
                    ),
                    "key": "r0068-45m-on-30m",
                    "schema": panel_schema,
                },
                "upper_60m_matched": {
                    "path": R0064_UPPER_MATCHED_PANEL,
                    "key": "r0063-60m-on-30m",
                    "schema": "round0064-registered-panel-v1",
                },
                "treatment_45m_full": {
                    "path": os.path.join(
                        paths["panel_full"],
                        "panel.json",
                    ),
                    "key": "r0068-45m-on-45m",
                    "schema": panel_schema,
                },
            },
        ),
        _job(
            node_id="ood_r0068_45m",
            action="ood",
            deps=["transform_r0068_45m"],
            output=paths["ood"],
            p90_wall_s=300,
            inputs=ood_inputs,
            gpu=True,
            map_key="r0068-45m-on-45m",
            transform_output=paths["transform_full"],
            ood_schema="round0069-ood-bundle-v1",
            **model45,
        ),
        _job(
            node_id="matched_renders",
            action="renders",
            deps=[
                "transform_r0068_on_30m",
                "transform_r0068_45m",
            ],
            output=paths["renders"],
            p90_wall_s=180,
            inputs=_dedupe([*inputs30, *inputs45]),
            gpu=False,
            matched_sample_rows=R0064_MATCHED_SAMPLE,
            matched_maps=[
                {
                    "map_key": "r0061-30m-on-30m",
                    "transform_output": R0064_CONTROL_TRANSFORM,
                },
                {
                    "map_key": "r0068-45m-on-30m",
                    "transform_output": paths["transform_matched"],
                },
                {
                    "map_key": "r0063-60m-on-30m",
                    "transform_output": R0064_UPPER_MATCHED_TRANSFORM,
                },
            ],
            full_transform=paths["transform_full"],
            full_int8_path=outputs45["int8"]["canonical_path"],
            full_int8_sha256=outputs45["int8"]["sha256"],
            full_scales_path=outputs45["scales"]["canonical_path"],
            full_scales_sha256=outputs45["scales"]["sha256"],
            full_eligibility_path=outputs45["eligibility"]["canonical_path"],
            full_eligibility_sha256=outputs45["eligibility"]["sha256"],
            **data30,
        ),
        _job(
            node_id="registry_publication",
            action="registry",
            deps=[
                "panel_r0068_45m",
                "ood_r0068_45m",
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
        gpu_hours_cap=1.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0069-scale-evaluation-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0064", "0065", "0068"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-60m-scale-geometry-v1",
        "minilm-balanced-45m-int8-substrate-v1",
        "minilm-balanced-45m-trained-model-seed42-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-30m-45m-60m-scale-geometry-v1",
        "minilm-balanced-45m-map-registry-v1",
    ]
    manifest["training_performed"] = False
    manifest["late_bound_model"] = {
        "model": bundle45["model"],
        "train_receipt": bundle45["train_receipt"],
        "review": review68,
    }
    manifest["reused_reviewed_evidence"] = {
        "r0064_review": review64,
        "r0065_review": review65,
        "r0064_scale_comparison": scale_comparison,
        "reference_30m": reference_receipt,
        "control_30m_panel": control_panel,
        "upper_60m_on_30m_panel": upper_panel,
    }
    manifest["scientific_contract"] = {
        "primary_comparison": (
            "R0061 30M, R0068 45M, and R0063 60M models on the exact "
            "same R0053 retained 30M rows/reference/anchors"
        ),
        "matched_noninferiority_margins": {
            "ffr": 0.02,
            "density": 0.05,
            "purity_k256": 0.05,
            "purity_k1024": 0.05,
            "projection_ffr": 0.02,
        },
        "legacy_density_floor": {
            "value": 0.60,
            "still_reported": True,
            "recalibrated_or_replaced": False,
            "status": (
                "not a sole scale gate because the exact balanced 30M "
                "control also fails it at 0.0991"
            ),
        },
        "advance_directly_to_120m": False,
        "ood": (
            "Dadabase, TREC-COVID, code, science, Latin; map-card evidence, "
            "non-gating"
        ),
    }
    manifest["jobs"] = jobs
    gpu_seconds = sum(
        float(job["p90_wall_s"])
        for job in jobs
        if job["node_policy"]["gpu_required"]
    )
    manifest["p90_gpu_seconds"] = {
        job["id"]: job["p90_wall_s"]
        for job in jobs
        if job["node_policy"]["gpu_required"]
    }
    manifest["p90_gpu_seconds"]["total"] = gpu_seconds
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--substrate-30-sha256", required=True)
    parser.add_argument("--substrate-45-sha256", required=True)
    parser.add_argument("--model-45", required=True)
    parser.add_argument("--model-45-sha256", required=True)
    parser.add_argument("--receipt-45", required=True)
    parser.add_argument("--receipt-45-sha256", required=True)
    for round_id in ("0064", "0065", "0068"):
        parser.add_argument(f"--review-{round_id}", required=True)
        parser.add_argument(f"--review-{round_id}-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0069(
            release_sha=args.release_sha,
            substrate_30_sha256=args.substrate_30_sha256,
            substrate_45_sha256=args.substrate_45_sha256,
            model_45_path=args.model_45,
            model_45_sha256=args.model_45_sha256,
            receipt_45_path=args.receipt_45,
            receipt_45_sha256=args.receipt_45_sha256,
            r0064_review_path=args.review_0064,
            r0064_review_sha256=args.review_0064_sha256,
            r0065_review_path=args.review_0065,
            r0065_review_sha256=args.review_0065_sha256,
            r0068_review_path=args.review_0068,
            r0068_review_sha256=args.review_0068_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
