#!/usr/bin/env python3
"""Prepare the matched/full balanced-90M scale-evaluation queue."""
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


ROUND_ID = "0076"
ROUND_ROOT = "/data/latent-basemap/runs/round-0076"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0076-2026-07-27.md")
SUBSTRATE_30 = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
SUBSTRATE_90 = (
    "/data/latent-basemap/runs/round-0071/queue/artifacts/"
    "balanced-90m-int8-substrate/balanced-90m-substrate-v1.json"
)
R0064_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0064/queue/artifacts"
)
R0069_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0069/queue/artifacts"
)
R0074_ANCHOR = (
    "/data/latent-basemap/runs/round-0074/queue-attempt-2/artifacts/"
    "duplicate-anchor-leverage/duplicate-anchor-leverage.json"
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
R0069_45M_MATCHED_PANEL = os.path.join(
    R0069_ARTIFACTS,
    "panel-r0068-on-30m",
    "panel.json",
)
R0064_60M_MATCHED_PANEL = os.path.join(
    R0064_ARTIFACTS,
    "panel-r0063-on-30m",
    "panel.json",
)
R0064_CONTROL_TRANSFORM = os.path.join(
    R0064_ARTIFACTS,
    "coordinates-r0061-30m",
)
R0069_45M_MATCHED_TRANSFORM = os.path.join(
    R0069_ARTIFACTS,
    "coordinates-r0068-on-30m",
)
R0064_60M_MATCHED_TRANSFORM = os.path.join(
    R0064_ARTIFACTS,
    "coordinates-r0063-on-30m",
)
R0064_MATCHED_SAMPLE = os.path.join(
    R0064_ARTIFACTS,
    "semantic-renders",
    "matched-30m-sample-rows.npy",
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
        raise RuntimeError("R0076 remains draft; refuse queue materialization")


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
        "handler_module": "experiments.round0076_nodes",
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


def prepare_round0076(
    *,
    release_sha: str,
    substrate_30_sha256: str,
    substrate_90_sha256: str,
    model_90_path: str,
    model_90_sha256: str,
    receipt_90_path: str,
    receipt_90_sha256: str,
    anchor_leverage_sha256: str,
    r0069_review_path: str,
    r0069_review_sha256: str,
    r0071_review_path: str,
    r0071_review_sha256: str,
    r0074_review_path: str,
    r0074_review_sha256: str,
    r0075_review_path: str,
    r0075_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0076 release SHA must be one full commit")
    substrate30 = validate_control_substrate(
        SUBSTRATE_30,
        expected_sha256=substrate_30_sha256,
    )
    substrate90 = validate_substrate(
        SUBSTRATE_90,
        expected_sha256=substrate_90_sha256,
    )
    bundle90 = validate_train_bundle(
        label="r0075-90m",
        model_path=model_90_path,
        model_sha256=model_90_sha256,
        train_receipt_path=receipt_90_path,
        train_receipt_sha256=receipt_90_sha256,
    )
    anchor = expected_input_signature(R0074_ANCHOR)
    if anchor["sha256"] != anchor_leverage_sha256:
        raise RuntimeError("R0074 anchor-leverage bytes changed")
    prior_panels = {
        "control_30m": expected_input_signature(R0064_CONTROL_PANEL),
        "rung_45m": expected_input_signature(R0069_45M_MATCHED_PANEL),
        "rung_60m": expected_input_signature(R0064_60M_MATCHED_PANEL),
    }
    reference30 = expected_input_signature(
        os.path.join(R0064_REFERENCE_30, "reference-receipt.json")
    )
    reviews = {
        "0069": _require_review(
            r0069_review_path,
            expected_sha256=r0069_review_sha256,
            required_text=(
                "capability:minilm-balanced-30m-45m-60m-scale-geometry-v1",
                prior_panels["rung_45m"]["sha256"],
            ),
        ),
        "0071": _require_review(
            r0071_review_path,
            expected_sha256=r0071_review_sha256,
            required_text=(
                "capability:minilm-balanced-90m-int8-input-v1",
                substrate_90_sha256,
            ),
        ),
        "0074": _require_review(
            r0074_review_path,
            expected_sha256=r0074_review_sha256,
            required_text=(
                "capability:minilm-30m-density-anchor-leverage-v1",
                anchor_leverage_sha256,
            ),
        ),
        "0075": _require_review(
            r0075_review_path,
            expected_sha256=r0075_review_sha256,
            required_text=(model_90_sha256, receipt_90_sha256),
        ),
    }

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0076 scale evaluation queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "transform_matched": os.path.join(
            artifacts,
            "coordinates-r0075-on-30m",
        ),
        "transform_full": os.path.join(
            artifacts,
            "coordinates-r0075-90m",
        ),
        "reference_full": os.path.join(
            artifacts,
            "high-d-reference-90m",
        ),
        "panel_matched": os.path.join(
            artifacts,
            "panel-r0075-on-30m",
        ),
        "panel_full": os.path.join(
            artifacts,
            "panel-r0075-90m",
        ),
        "comparison": os.path.join(artifacts, "scale-comparison"),
        "ood": os.path.join(artifacts, "ood-r0075-90m"),
        "renders": os.path.join(artifacts, "semantic-renders"),
        "registry": os.path.join(artifacts, "registry"),
    }
    outputs30 = substrate30["manifest"]["outputs"]
    outputs90 = substrate90["manifest"]["outputs"]
    existing_coordinate_files = sorted({
        *glob.glob(os.path.join(
            R0064_CONTROL_TRANSFORM,
            "chunk-*",
            "coordinates.npy",
        )),
        *glob.glob(os.path.join(
            R0069_45M_MATCHED_TRANSFORM,
            "chunk-*",
            "coordinates.npy",
        )),
        *glob.glob(os.path.join(
            R0064_60M_MATCHED_TRANSFORM,
            "chunk-*",
            "coordinates.npy",
        )),
    })
    # These prior coordinate chunks are large immutable inputs used by both
    # the comparison provenance and the CPU render job. Hash each one once
    # during queue preparation, then reuse its signature in both registries.
    existing_coordinate_inputs = _file_inputs(existing_coordinate_files)
    core = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            r0069_review_path,
            r0071_review_path,
            r0074_review_path,
            r0075_review_path,
            SUBSTRATE_30,
            SUBSTRATE_90,
            bundle90["model"]["canonical_path"],
            bundle90["train_receipt"]["canonical_path"],
            MINILM_QUERIES,
            MINILM_QUERY_PROVENANCE,
            *CENTROIDS.values(),
            R0064_CONTROL_PANEL,
            R0069_45M_MATCHED_PANEL,
            R0064_60M_MATCHED_PANEL,
            R0064_MATCHED_SAMPLE,
            os.path.join(R0064_REFERENCE_30, "reference.npz"),
            os.path.join(R0064_REFERENCE_30, "reference-receipt.json"),
            os.path.join(R0064_REFERENCE_30, "recall50-truth.npy"),
            os.path.join(R0064_REFERENCE_30, "anchor-substrate-rows.npy"),
            os.path.join(R0064_CONTROL_TRANSFORM, "actual-transform.json"),
            os.path.join(
                R0069_45M_MATCHED_TRANSFORM,
                "actual-transform.json",
            ),
            os.path.join(
                R0064_60M_MATCHED_TRANSFORM,
                "actual-transform.json",
            ),
            R0074_ANCHOR,
        ]),
        *existing_coordinate_inputs,
    ])
    inputs30 = _dedupe([
        *core,
        *_file_inputs([
            outputs30["int8"]["canonical_path"],
            outputs30["scales"]["canonical_path"],
            outputs30["eligibility"]["canonical_path"],
        ]),
    ])
    inputs90 = _dedupe([
        *core,
        *_file_inputs([
            outputs90["int8"]["canonical_path"],
            outputs90["scales"]["canonical_path"],
            outputs90["eligibility"]["canonical_path"],
        ]),
    ])
    render_inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            R0064_MATCHED_SAMPLE,
            outputs30["eligibility"]["canonical_path"],
            outputs90["eligibility"]["canonical_path"],
            os.path.join(R0064_CONTROL_TRANSFORM, "actual-transform.json"),
            os.path.join(
                R0069_45M_MATCHED_TRANSFORM,
                "actual-transform.json",
            ),
            os.path.join(
                R0064_60M_MATCHED_TRANSFORM,
                "actual-transform.json",
            ),
        ]),
        *existing_coordinate_inputs,
    ])
    ood_inputs = _dedupe([
        *core,
        *(expected_input_signature(path) for path in _static_ood_paths()),
        *_materialized_chunk_inputs(),
        *_hf_snapshot_file_inputs(),
    ])
    model90 = {
        "model_label": "r0075-90m",
        "model_path": bundle90["model"]["canonical_path"],
        "model_sha256": bundle90["model"]["sha256"],
        "train_receipt_path": bundle90["train_receipt"]["canonical_path"],
        "train_receipt_sha256": bundle90["train_receipt"]["sha256"],
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
    data90 = {
        "substrate_label": "balanced-90m",
        "row_count": 90_000_000,
        "rows_per_corpus": 30_000_000,
        "row_order": (
            "first 30M rows of each FineWeb/RedPajama/Pile block"
        ),
        "int8_path": outputs90["int8"]["canonical_path"],
        "int8_sha256": outputs90["int8"]["sha256"],
        "scales_path": outputs90["scales"]["canonical_path"],
        "scales_sha256": outputs90["scales"]["sha256"],
        "eligibility_path": outputs90["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs90["eligibility"]["sha256"],
    }
    jobs = [
        _job(
            node_id="transform_r0075_on_30m",
            action="transform",
            deps=[],
            output=paths["transform_matched"],
            p90_wall_s=240,
            inputs=inputs30,
            gpu=True,
            map_key="r0075-90m-on-30m",
            **model90,
            **data30,
        ),
        _job(
            node_id="transform_r0075_90m",
            action="transform",
            deps=[],
            output=paths["transform_full"],
            p90_wall_s=600,
            inputs=inputs90,
            gpu=True,
            map_key="r0075-90m-on-90m",
            **model90,
            **data90,
        ),
        _job(
            node_id="high_d_reference_90m",
            action="high_d_reference",
            deps=[],
            output=paths["reference_full"],
            p90_wall_s=1_200,
            inputs=inputs90,
            gpu=True,
            reference_schema="round0076-high-d-reference-v1",
            **data90,
        ),
        _job(
            node_id="panel_r0075_on_30m",
            action="panel",
            deps=["transform_r0075_on_30m"],
            output=paths["panel_matched"],
            p90_wall_s=900,
            inputs=inputs30,
            gpu=True,
            map_key="r0075-90m-on-30m",
            transform_output=paths["transform_matched"],
            reference_output=R0064_REFERENCE_30,
            panel_schema="round0076-registered-panel-v1",
            **model90,
            **data30,
        ),
        _job(
            node_id="panel_r0075_90m",
            action="panel",
            deps=["transform_r0075_90m", "high_d_reference_90m"],
            output=paths["panel_full"],
            p90_wall_s=2_400,
            inputs=inputs90,
            gpu=True,
            map_key="r0075-90m-on-90m",
            transform_output=paths["transform_full"],
            reference_output=paths["reference_full"],
            panel_schema="round0076-registered-panel-v1",
            **model90,
            **data90,
        ),
        _job(
            node_id="scale_comparison",
            action="comparison",
            deps=["panel_r0075_on_30m", "panel_r0075_90m"],
            output=paths["comparison"],
            p90_wall_s=60,
            inputs=core,
            gpu=False,
            anchor_leverage=R0074_ANCHOR,
            anchor_leverage_sha256=anchor_leverage_sha256,
            panels={
                "control_30m": {
                    "path": R0064_CONTROL_PANEL,
                    "key": "r0061-30m-on-30m",
                    "schema": "round0064-registered-panel-v1",
                },
                "rung_45m": {
                    "path": R0069_45M_MATCHED_PANEL,
                    "key": "r0068-45m-on-30m",
                    "schema": "round0069-registered-panel-v1",
                },
                "rung_60m": {
                    "path": R0064_60M_MATCHED_PANEL,
                    "key": "r0063-60m-on-30m",
                    "schema": "round0064-registered-panel-v1",
                },
                "treatment_90m_matched": {
                    "path": os.path.join(
                        paths["panel_matched"],
                        "panel.json",
                    ),
                    "key": "r0075-90m-on-30m",
                    "schema": "round0076-registered-panel-v1",
                },
                "treatment_90m_full": {
                    "path": os.path.join(paths["panel_full"], "panel.json"),
                    "key": "r0075-90m-on-90m",
                    "schema": "round0076-registered-panel-v1",
                },
            },
        ),
        _job(
            node_id="ood_r0075_90m",
            action="ood",
            deps=["transform_r0075_90m"],
            output=paths["ood"],
            p90_wall_s=300,
            inputs=ood_inputs,
            gpu=True,
            map_key="r0075-90m-on-90m",
            transform_output=paths["transform_full"],
            ood_schema="round0076-ood-bundle-v1",
            **model90,
        ),
        _job(
            node_id="matched_renders",
            action="renders",
            deps=["transform_r0075_on_30m", "transform_r0075_90m"],
            output=paths["renders"],
            p90_wall_s=180,
            inputs=render_inputs,
            gpu=False,
            matched_sample_rows=R0064_MATCHED_SAMPLE,
            matched_maps=[
                {
                    "map_key": "r0061-30m-on-30m",
                    "transform_output": R0064_CONTROL_TRANSFORM,
                },
                {
                    "map_key": "r0068-45m-on-30m",
                    "transform_output": R0069_45M_MATCHED_TRANSFORM,
                },
                {
                    "map_key": "r0063-60m-on-30m",
                    "transform_output": R0064_60M_MATCHED_TRANSFORM,
                },
                {
                    "map_key": "r0075-90m-on-30m",
                    "transform_output": paths["transform_matched"],
                },
            ],
            full_transform=paths["transform_full"],
            full_row_count=90_000_000,
            full_eligibility_path=outputs90["eligibility"]["canonical_path"],
            full_eligibility_sha256=outputs90["eligibility"]["sha256"],
            eligibility_path=outputs30["eligibility"]["canonical_path"],
            eligibility_sha256=outputs30["eligibility"]["sha256"],
            row_count=30_000_000,
        ),
        _job(
            node_id="registry_publication",
            action="registry",
            deps=[
                "panel_r0075_90m",
                "ood_r0075_90m",
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
        gpu_hours_cap=2.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0076-scale-evaluation-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0069", "0071", "0074", "0075"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-45m-60m-scale-geometry-v1",
        "minilm-balanced-90m-int8-input-v1",
        "minilm-30m-density-anchor-leverage-v1",
        "minilm-balanced-90m-trained-model-seed42-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
        "minilm-balanced-90m-map-registry-v1",
    ]
    manifest["training_performed"] = False
    manifest["late_bound_model"] = {
        "model": bundle90["model"],
        "train_receipt": bundle90["train_receipt"],
        "review": reviews["0075"],
    }
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "substrate_90m": substrate90["signature"],
        "anchor_leverage": anchor,
        "reference_30m": reference30,
        "prior_panels": prior_panels,
    }
    manifest["scientific_contract"] = {
        "primary_comparison": (
            "30M, 45M, 60M, and 90M models on the exact same R0053 "
            "representative rows, reference, and anchors"
        ),
        "90m_noninferiority_controls": ["30m", "60m"],
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
        "full_90m_non_density_checks_required": True,
        "train_120m_without_separate_round": False,
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
    parser.add_argument("--substrate-90-sha256", required=True)
    parser.add_argument("--model-90", required=True)
    parser.add_argument("--model-90-sha256", required=True)
    parser.add_argument("--receipt-90", required=True)
    parser.add_argument("--receipt-90-sha256", required=True)
    parser.add_argument("--anchor-leverage-sha256", required=True)
    for round_id in ("0069", "0071", "0074", "0075"):
        parser.add_argument(f"--review-{round_id}", required=True)
        parser.add_argument(f"--review-{round_id}-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0076(
            release_sha=args.release_sha,
            substrate_30_sha256=args.substrate_30_sha256,
            substrate_90_sha256=args.substrate_90_sha256,
            model_90_path=args.model_90,
            model_90_sha256=args.model_90_sha256,
            receipt_90_path=args.receipt_90,
            receipt_90_sha256=args.receipt_90_sha256,
            anchor_leverage_sha256=args.anchor_leverage_sha256,
            r0069_review_path=args.review_0069,
            r0069_review_sha256=args.review_0069_sha256,
            r0071_review_path=args.review_0071,
            r0071_review_sha256=args.review_0071_sha256,
            r0074_review_path=args.review_0074,
            r0074_review_sha256=args.review_0074_sha256,
            r0075_review_path=args.review_0075,
            r0075_review_sha256=args.review_0075_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
