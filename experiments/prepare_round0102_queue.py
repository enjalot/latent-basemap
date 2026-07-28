#!/usr/bin/env python3
"""Prepare the matched/full balanced-150M scale-evaluation queue."""
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
from basemap.round0086_program import validate_substrate
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


ROUND_ID = "0102"
ROUND_ROOT = "/data/latent-basemap/runs/round-0102"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0102-*.md")
SUBSTRATE_120 = (
    "/data/latent-basemap/runs/round-0065/queue/artifacts/"
    "balanced-120m-int8-substrate/balanced-120m-substrate-v1.json"
)
R0080_ARTIFACTS = (
    "/data/latent-basemap/runs/round-0080/queue/artifacts"
)
R0080_REVIEWED_SHA256 = {
    "coordinates-r0079-120m/actual-transform.json":
        "e7be86da8bb1e325f9ab9a5919bbe43f8dcaf7ff411ace8b041469aebfb57bf5",
    "high-d-reference-120m/reference.npz":
        "8c497eea4ca9f4f116a829036d92205f1e5e60717f39b4fda868c114501aa77a",
    "high-d-reference-120m/reference-receipt.json":
        "870a92e852bb4f5c6268d24c907e408b83ee91fb89913fd3726a92c9450bfeff",
    "high-d-reference-120m/recall50-truth.npy":
        "e4c8a67393897307a16a57315001f8883524f36e349e9215927d127ed381a48d",
    "panel-r0079-120m/panel.json":
        "bb16e22530fd04f488b67ef9632eaede4da6148f494bb8120ec4146d322971c3",
    "scale-comparison/scale-comparison.json":
        "6c1bf66e5691c743eb48d01ef5f0654031cca43cad50363752a6fadfc7c5cf87",
    "semantic-renders/full-120m-sample-rows.npy":
        "f970c92c0ade42646d36ffce011c2489cc97e75da866c2238dfb38afc486b85f",
}
CONTROL_TRANSFORM = os.path.join(
    R0080_ARTIFACTS,
    "coordinates-r0079-120m",
)
CONTROL_REFERENCE = os.path.join(
    R0080_ARTIFACTS,
    "high-d-reference-120m",
)
CONTROL_PANEL = os.path.join(
    R0080_ARTIFACTS,
    "panel-r0079-120m",
    "panel.json",
)
CONTROL_SCALE = os.path.join(
    R0080_ARTIFACTS,
    "scale-comparison",
    "scale-comparison.json",
)
MATCHED_SAMPLE = os.path.join(
    R0080_ARTIFACTS,
    "semantic-renders",
    "full-120m-sample-rows.npy",
)
DENSITY_CALIBRATION = (
    "/data/latent-basemap/runs/round-0085/queue/artifacts/"
    "density-v2-calibration/density-v2-calibration.json"
)
DENSITY_CALIBRATION_SHA256 = (
    "b46161ae5e96df96664d1589b1dbb11d9fcdc2d5e1284b470261749614db3102"
)
DENSITY_V2_FLOOR = 0.041703756293199175


def _reviewed_r0080_signature(relative_path: str) -> dict[str, Any]:
    """Return one exact artifact released by Review 0080."""
    expected_sha256 = R0080_REVIEWED_SHA256[relative_path]
    path = os.path.join(R0080_ARTIFACTS, relative_path)
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(
            f"reviewed R0080 artifact changed: {relative_path}"
        )
    return signature


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


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "R0102 requires exactly one issued round document; "
            f"found {len(candidates)}"
        )
    return candidates[0]


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
        "handler_module": "experiments.round0102_nodes",
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


def prepare_round0102(
    *,
    release_sha: str,
    substrate_120_sha256: str,
    substrate_150_path: str,
    substrate_150_sha256: str,
    model_150_path: str,
    model_150_sha256: str,
    receipt_150_path: str,
    receipt_150_sha256: str,
    r0025_review_path: str,
    r0025_review_sha256: str,
    r0033_review_path: str,
    r0033_review_sha256: str,
    r0080_review_path: str,
    r0080_review_sha256: str,
    r0084_review_path: str,
    r0084_review_sha256: str,
    r0085_review_path: str,
    r0085_review_sha256: str,
    r0101_review_path: str,
    r0101_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    round_file = _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0102 release SHA must be one full commit")
    substrate120 = validate_scale_substrate(
        SUBSTRATE_120,
        tier="120m",
        expected_sha256=substrate_120_sha256,
    )
    substrate150 = validate_substrate(
        substrate_150_path,
        expected_sha256=substrate_150_sha256,
    )
    bundle150 = validate_train_bundle(
        label="r0101-150m",
        model_path=model_150_path,
        model_sha256=model_150_sha256,
        train_receipt_path=receipt_150_path,
        train_receipt_sha256=receipt_150_sha256,
    )
    with open(CONTROL_SCALE, encoding="utf-8") as handle:
        scale_geometry = json.load(handle)
    validate_seal(scale_geometry, label="R0102 R0080 scale geometry")
    scale_signature = expected_input_signature(CONTROL_SCALE)
    if (
        scale_signature["sha256"]
        != R0080_REVIEWED_SHA256[
            "scale-comparison/scale-comparison.json"
        ]
        or scale_geometry.get("schema")
        != "round0080-scale-geometry-comparison-v1"
        or scale_geometry.get("decision", {}).get(
            "120m_supported_as_deliberate_ladder_rung"
        )
        is not True
    ):
        raise RuntimeError("R0080 scale geometry changed")
    reviewed_r0080_inputs = [
        _reviewed_r0080_signature(relative_path)
        for relative_path in R0080_REVIEWED_SHA256
    ]
    control_panel = next(
        signature
        for signature in reviewed_r0080_inputs
        if signature["canonical_path"] == os.path.realpath(CONTROL_PANEL)
    )
    with open(CONTROL_PANEL, encoding="utf-8") as handle:
        control_value = json.load(handle)
    validate_seal(control_value, label="R0102 R0080 full-120M panel")
    if (
        control_value.get("schema") != "round0080-registered-panel-v1"
        or control_value.get("map_key") != "r0079-120m-on-120m"
    ):
        raise RuntimeError("R0080 full-120M control panel changed")
    density_signature = expected_input_signature(DENSITY_CALIBRATION)
    with open(DENSITY_CALIBRATION, encoding="utf-8") as handle:
        density_calibration = json.load(handle)
    validate_seal(density_calibration, label="R0102 R0085 density_v2")
    if (
        density_signature["sha256"] != DENSITY_CALIBRATION_SHA256
        or density_calibration.get("schema")
        != "round0085-density-v2-calibration-v1"
        or float(
            density_calibration.get("floor_calibration", {}).get(
                "registered_floor", -1.0
            )
        )
        != DENSITY_V2_FLOOR
        or density_calibration.get("floor_calibration", {}).get(
            "gating_floor_registered"
        )
        is not True
    ):
        raise RuntimeError("R0085 density_v2 calibration changed")
    reviews = {
        "0025": _require_review(
            r0025_review_path,
            expected_sha256=r0025_review_sha256,
            required_text=(
                "capability:minilm-int8-shards-v1",
                substrate150["manifest"]["outputs"]["int8"]["sha256"],
            ),
        ),
        "0033": _require_review(
            r0033_review_path,
            expected_sha256=r0033_review_sha256,
            required_text=(
                "capability:minilm-150m-row-eligibility-v1",
                substrate150["manifest"]["outputs"]["eligibility"]["sha256"],
            ),
        ),
        "0080": _require_review(
            r0080_review_path,
            expected_sha256=r0080_review_sha256,
            required_text=(
                "capability:minilm-balanced-90m-120m-scale-geometry-v1",
                control_panel["sha256"],
            ),
        ),
        "0084": _require_review(
            r0084_review_path,
            expected_sha256=r0084_review_sha256,
            required_text=(
                "capability:minilm-balanced-90m-seed43-sensitivity-v1",
                "0.0209",
            ),
        ),
        "0085": _require_review(
            r0085_review_path,
            expected_sha256=r0085_review_sha256,
            required_text=(
                "capability:minilm-density-v2-calibration-v1",
                str(DENSITY_V2_FLOOR),
            ),
        ),
        "0101": _require_review(
            r0101_review_path,
            expected_sha256=r0101_review_sha256,
            required_text=(model_150_sha256, receipt_150_sha256),
        ),
    }

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0102 scale evaluation queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "transform_matched": os.path.join(
            artifacts, "coordinates-r0101-on-120m"
        ),
        "transform_full": os.path.join(
            artifacts, "coordinates-r0101-150m"
        ),
        "reference_full": os.path.join(
            artifacts, "high-d-reference-150m"
        ),
        "panel_matched": os.path.join(
            artifacts, "panel-r0101-on-120m"
        ),
        "panel_full": os.path.join(artifacts, "panel-r0101-150m"),
        "density": os.path.join(artifacts, "density-v2"),
        "comparison": os.path.join(artifacts, "scale-comparison"),
        "ood": os.path.join(artifacts, "ood-r0101-150m"),
        "renders": os.path.join(artifacts, "semantic-renders"),
        "registry": os.path.join(artifacts, "registry"),
    }
    outputs120 = substrate120["manifest"]["outputs"]
    outputs150 = substrate150["manifest"]["outputs"]
    control_coordinate_files = sorted(glob.glob(os.path.join(
        CONTROL_TRANSFORM,
        "chunk-*",
        "coordinates.npy",
    )))
    control_coordinate_inputs = _file_inputs(control_coordinate_files)
    core = _dedupe([
        *_file_inputs([
            round_file,
            r0025_review_path,
            r0033_review_path,
            r0080_review_path,
            r0084_review_path,
            r0085_review_path,
            r0101_review_path,
            SUBSTRATE_120,
            substrate_150_path,
            CONTROL_SCALE,
            DENSITY_CALIBRATION,
            bundle150["model"]["canonical_path"],
            bundle150["train_receipt"]["canonical_path"],
            MINILM_QUERIES,
            MINILM_QUERY_PROVENANCE,
            *CENTROIDS.values(),
            MATCHED_SAMPLE,
            os.path.join(CONTROL_REFERENCE, "anchor-substrate-rows.npy"),
        ]),
        *reviewed_r0080_inputs,
        *control_coordinate_inputs,
    ])
    inputs120 = _dedupe([
        *core,
        *_file_inputs([
            outputs120["int8"]["canonical_path"],
            outputs120["scales"]["canonical_path"],
            outputs120["eligibility"]["canonical_path"],
        ]),
    ])
    inputs150 = _dedupe([
        *core,
        *_file_inputs([
            outputs150["int8"]["canonical_path"],
            outputs150["scales"]["canonical_path"],
            outputs150["eligibility"]["canonical_path"],
        ]),
    ])
    ood_inputs = _dedupe([
        *core,
        *(expected_input_signature(path) for path in _static_ood_paths()),
        *_materialized_chunk_inputs(),
        *_hf_snapshot_file_inputs(),
    ])
    model150 = {
        "model_label": "r0101-150m",
        "model_path": bundle150["model"]["canonical_path"],
        "model_sha256": bundle150["model"]["sha256"],
        "train_receipt_path": bundle150["train_receipt"]["canonical_path"],
        "train_receipt_sha256": bundle150["train_receipt"]["sha256"],
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
    data150 = {
        "substrate_label": "balanced-150m",
        "row_count": 150_000_000,
        "rows_per_corpus": 50_000_000,
        "row_order": "first 50M rows of each FineWeb/RedPajama/Pile block",
        "int8_path": outputs150["int8"]["canonical_path"],
        "int8_sha256": outputs150["int8"]["sha256"],
        "scales_path": outputs150["scales"]["canonical_path"],
        "scales_sha256": outputs150["scales"]["sha256"],
        "eligibility_path": outputs150["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs150["eligibility"]["sha256"],
    }
    jobs = [
        _job(
            node_id="transform_r0101_on_120m",
            action="transform",
            deps=[],
            output=paths["transform_matched"],
            p90_wall_s=800,
            inputs=inputs120,
            gpu=True,
            map_key="r0101-150m-on-120m",
            **model150,
            **data120,
        ),
        _job(
            node_id="transform_r0101_150m",
            action="transform",
            deps=[],
            output=paths["transform_full"],
            p90_wall_s=1_000,
            inputs=inputs150,
            gpu=True,
            map_key="r0101-150m-on-150m",
            **model150,
            **data150,
        ),
        _job(
            node_id="high_d_reference_150m",
            action="high_d_reference",
            deps=[],
            output=paths["reference_full"],
            p90_wall_s=2_000,
            inputs=inputs150,
            gpu=True,
            reference_schema="round0102-high-d-reference-v1",
            **data150,
        ),
        _job(
            node_id="panel_r0101_on_120m",
            action="panel",
            deps=["transform_r0101_on_120m"],
            output=paths["panel_matched"],
            p90_wall_s=3_200,
            inputs=inputs120,
            gpu=True,
            map_key="r0101-150m-on-120m",
            transform_output=paths["transform_matched"],
            reference_output=CONTROL_REFERENCE,
            panel_schema="round0102-registered-panel-v1",
            **model150,
            **data120,
        ),
        _job(
            node_id="panel_r0101_150m",
            action="panel",
            deps=["transform_r0101_150m", "high_d_reference_150m"],
            output=paths["panel_full"],
            p90_wall_s=4_000,
            inputs=inputs150,
            gpu=True,
            map_key="r0101-150m-on-150m",
            transform_output=paths["transform_full"],
            reference_output=paths["reference_full"],
            panel_schema="round0102-registered-panel-v1",
            **model150,
            **data150,
        ),
        _job(
            node_id="density_v2",
            action="density_v2",
            deps=[
                "transform_r0101_on_120m",
                "transform_r0101_150m",
                "high_d_reference_150m",
            ],
            output=paths["density"],
            p90_wall_s=1_200,
            inputs=_dedupe([*inputs120, *inputs150]),
            gpu=True,
            registered_floor=DENSITY_V2_FLOOR,
            calibration=DENSITY_CALIBRATION,
            universes=[
                {
                    "key": "matched_120m",
                    "map_key": "r0101-150m-on-120m",
                    "model_sha256": bundle150["model"]["sha256"],
                    "coordinates_path": paths["transform_matched"],
                    "row_count": 120_000_000,
                    "eligibility_path": outputs120["eligibility"][
                        "canonical_path"
                    ],
                    "eligibility_sha256": outputs120["eligibility"]["sha256"],
                    "reference_path": os.path.join(
                        CONTROL_REFERENCE, "reference.npz"
                    ),
                    "reference_receipt_path": os.path.join(
                        CONTROL_REFERENCE, "reference-receipt.json"
                    ),
                    "anchor_rows_path": os.path.join(
                        CONTROL_REFERENCE, "anchor-substrate-rows.npy"
                    ),
                },
                {
                    "key": "full_150m",
                    "map_key": "r0101-150m-on-150m",
                    "model_sha256": bundle150["model"]["sha256"],
                    "coordinates_path": paths["transform_full"],
                    "row_count": 150_000_000,
                    "eligibility_path": outputs150["eligibility"][
                        "canonical_path"
                    ],
                    "eligibility_sha256": outputs150["eligibility"]["sha256"],
                    "reference_path": os.path.join(
                        paths["reference_full"], "reference.npz"
                    ),
                    "reference_receipt_path": os.path.join(
                        paths["reference_full"], "reference-receipt.json"
                    ),
                    "anchor_rows_path": os.path.join(
                        paths["reference_full"], "anchor-substrate-rows.npy"
                    ),
                },
            ],
        ),
        _job(
            node_id="scale_comparison",
            action="comparison",
            deps=[
                "panel_r0101_on_120m",
                "panel_r0101_150m",
                "density_v2",
            ],
            output=paths["comparison"],
            p90_wall_s=60,
            inputs=core,
            gpu=False,
            control_panel=CONTROL_PANEL,
            matched_panel=os.path.join(paths["panel_matched"], "panel.json"),
            full_panel=os.path.join(paths["panel_full"], "panel.json"),
            density_v2=os.path.join(
                paths["density"], "density-v2-evaluation.json"
            ),
        ),
        _job(
            node_id="ood_r0101_150m",
            action="ood",
            deps=["transform_r0101_150m"],
            output=paths["ood"],
            p90_wall_s=400,
            inputs=ood_inputs,
            gpu=True,
            map_key="r0101-150m-on-150m",
            transform_output=paths["transform_full"],
            ood_schema="round0102-ood-bundle-v1",
            **model150,
        ),
        _job(
            node_id="matched_renders",
            action="renders",
            deps=["transform_r0101_on_120m", "transform_r0101_150m"],
            output=paths["renders"],
            p90_wall_s=180,
            inputs=core,
            gpu=False,
            matched_sample_rows=MATCHED_SAMPLE,
            matched_maps=[
                {
                    "map_key": "r0079-120m-on-120m",
                    "transform_output": CONTROL_TRANSFORM,
                },
                {
                    "map_key": "r0101-150m-on-120m",
                    "transform_output": paths["transform_matched"],
                },
            ],
            full_transform=paths["transform_full"],
            matched_eligibility_path=outputs120["eligibility"][
                "canonical_path"
            ],
            matched_eligibility_sha256=outputs120["eligibility"]["sha256"],
            full_eligibility_path=outputs150["eligibility"][
                "canonical_path"
            ],
            full_eligibility_sha256=outputs150["eligibility"]["sha256"],
        ),
        _job(
            node_id="registry_publication",
            action="registry",
            deps=[
                "panel_r0101_150m",
                "ood_r0101_150m",
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
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=4.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0102-scale-evaluation-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = [
        "0025", "0033", "0080", "0084", "0085", "0101",
    ]
    manifest["capability_dependencies"] = [
        "minilm-int8-shards-v1",
        "minilm-150m-row-eligibility-v1",
        "minilm-balanced-90m-120m-scale-geometry-v1",
        "minilm-balanced-90m-seed43-sensitivity-v1",
        "minilm-density-v2-calibration-v1",
        "minilm-balanced-150m-trained-model-seed42-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-120m-150m-scale-geometry-v1",
        "minilm-balanced-150m-map-registry-v1",
    ]
    manifest["training_performed"] = False
    manifest["late_bound_model"] = {
        "model": bundle150["model"],
        "train_receipt": bundle150["train_receipt"],
        "review": reviews["0101"],
    }
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "substrate_120m": substrate120["signature"],
        "substrate_150m": substrate150["signature"],
        "scale_geometry_120m": scale_signature,
        "control_panel_120m": control_panel,
        "density_v2_calibration": density_signature,
    }
    manifest["scientific_contract"] = {
        "primary_comparison": (
            "120M and 150M models on the exact same R0065 representative "
            "rows, R0080 high-D reference, and anchors"
        ),
        "150m_noninferiority_control": "120m",
        "matched_noninferiority_margins": {
            "ffr": 0.02,
            "purity_k256": 0.05,
            "purity_k1024": 0.05,
        },
        "projection_ffr": {
            "decision_gating": False,
            "reason": (
                "R0084 one-seed full-90M contrast was 0.0209, exceeding "
                "the old 0.02 margin without estimating a replacement"
            ),
        },
        "density": {
            "anchors": "representative-only",
            "candidate_universe": "representative-only",
            "metric": "density_v2",
            "registered_floor": DENSITY_V2_FLOOR,
            "legacy_absolute_floor_is_decision_gating": False,
            "threshold_recalibrated": False,
        },
        "full_150m_absolute_checks_required": True,
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
    parser.add_argument("--substrate-120-sha256", required=True)
    parser.add_argument("--substrate-150", required=True)
    parser.add_argument("--substrate-150-sha256", required=True)
    parser.add_argument("--model-150", required=True)
    parser.add_argument("--model-150-sha256", required=True)
    parser.add_argument("--receipt-150", required=True)
    parser.add_argument("--receipt-150-sha256", required=True)
    for round_id in ("0025", "0033", "0080", "0084", "0085", "0101"):
        parser.add_argument(f"--review-{round_id}", required=True)
        parser.add_argument(f"--review-{round_id}-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0102(
            release_sha=args.release_sha,
            substrate_120_sha256=args.substrate_120_sha256,
            substrate_150_path=args.substrate_150,
            substrate_150_sha256=args.substrate_150_sha256,
            model_150_path=args.model_150,
            model_150_sha256=args.model_150_sha256,
            receipt_150_path=args.receipt_150,
            receipt_150_sha256=args.receipt_150_sha256,
            r0025_review_path=args.review_0025,
            r0025_review_sha256=args.review_0025_sha256,
            r0033_review_path=args.review_0033,
            r0033_review_sha256=args.review_0033_sha256,
            r0080_review_path=args.review_0080,
            r0080_review_sha256=args.review_0080_sha256,
            r0084_review_path=args.review_0084,
            r0084_review_sha256=args.review_0084_sha256,
            r0085_review_path=args.review_0085,
            r0085_review_sha256=args.review_0085_sha256,
            r0101_review_path=args.review_0101,
            r0101_review_sha256=args.review_0101_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
