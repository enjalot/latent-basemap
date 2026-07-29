#!/usr/bin/env python3
"""Prepare, but never launch, the matched 30M/60M evaluation queue."""
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
from basemap.round0049_program import validate_substrate_manifest
from basemap.round0053_program import validate_control_substrate
from basemap.round0064_evaluation import validate_train_bundle
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


ROUND_ID = "0064"
ROUND_ROOT = "/data/latent-basemap/runs/round-0064"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0064-2026-07-26.md",
)
SUBSTRATE_30 = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
SUBSTRATE_60 = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "balanced-60m-substrate/balanced-60m-substrate-v1.json"
)


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_released_review(
    path: str,
    *,
    model_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    status = _frontmatter_status(path)
    if status not in {"accepted", "partial"}:
        raise RuntimeError(f"{path} does not release reviewed capabilities")
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    missing = [
        value
        for value in (model_sha256, receipt_sha256)
        if value not in text
    ]
    if missing:
        raise RuntimeError(
            f"{path} does not bind the supplied model/receipt hashes"
        )
    return signature


def _require_issued_round() -> None:
    if _frontmatter_status(ROUND_FILE) != "issued":
        raise RuntimeError("R0064 remains draft; refuse queue materialization")


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
        "handler_module": "experiments.round0064_nodes",
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


def prepare_round0064(
    *,
    release_sha: str,
    substrate_30_sha256: str,
    substrate_60_sha256: str,
    model_30_path: str,
    model_30_sha256: str,
    receipt_30_path: str,
    receipt_30_sha256: str,
    review_30_path: str,
    review_30_sha256: str,
    model_60_path: str,
    model_60_sha256: str,
    receipt_60_path: str,
    receipt_60_sha256: str,
    review_60_path: str,
    review_60_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0064 release SHA must be one full commit")
    _require_issued_round()
    substrate30 = validate_control_substrate(
        SUBSTRATE_30,
        expected_sha256=substrate_30_sha256,
    )
    substrate60 = validate_substrate_manifest(
        SUBSTRATE_60,
        expected_sha256=substrate_60_sha256,
    )
    bundle30 = validate_train_bundle(
        label="r0061-30m",
        model_path=model_30_path,
        model_sha256=model_30_sha256,
        train_receipt_path=receipt_30_path,
        train_receipt_sha256=receipt_30_sha256,
    )
    bundle60 = validate_train_bundle(
        label="r0063-60m",
        model_path=model_60_path,
        model_sha256=model_60_sha256,
        train_receipt_path=receipt_60_path,
        train_receipt_sha256=receipt_60_sha256,
    )
    review30 = _require_released_review(
        review_30_path,
        model_sha256=model_30_sha256,
        receipt_sha256=receipt_30_sha256,
    )
    review60 = _require_released_review(
        review_60_path,
        model_sha256=model_60_sha256,
        receipt_sha256=receipt_60_sha256,
    )
    if review30["sha256"] != review_30_sha256:
        raise RuntimeError("R0061 review bytes changed")
    if review60["sha256"] != review_60_sha256:
        raise RuntimeError("R0063 review bytes changed")

    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0064 queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "transform_control": os.path.join(
            artifacts, "coordinates-r0061-30m"
        ),
        "transform_scaled_matched": os.path.join(
            artifacts, "coordinates-r0063-on-30m"
        ),
        "transform_scaled_full": os.path.join(
            artifacts, "coordinates-r0063-60m"
        ),
        "reference30": os.path.join(artifacts, "high-d-reference-30m"),
        "reference60": os.path.join(artifacts, "high-d-reference-60m"),
        "panel_control": os.path.join(artifacts, "panel-r0061-30m"),
        "panel_scaled_matched": os.path.join(
            artifacts, "panel-r0063-on-30m"
        ),
        "panel_scaled_full": os.path.join(
            artifacts, "panel-r0063-60m"
        ),
        "comparison": os.path.join(artifacts, "scale-comparison"),
        "ood_control": os.path.join(artifacts, "ood-r0061-30m"),
        "ood_scaled": os.path.join(artifacts, "ood-r0063-60m"),
        "renders": os.path.join(artifacts, "semantic-renders"),
        "registry": os.path.join(artifacts, "registry"),
    }
    outputs30 = substrate30["manifest"]["outputs"]
    outputs60 = substrate60["manifest"]["outputs"]
    core = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            review_30_path,
            review_60_path,
            SUBSTRATE_30,
            SUBSTRATE_60,
            bundle30["model"]["canonical_path"],
            bundle30["train_receipt"]["canonical_path"],
            bundle60["model"]["canonical_path"],
            bundle60["train_receipt"]["canonical_path"],
            MINILM_QUERIES,
            MINILM_QUERY_PROVENANCE,
            *CENTROIDS.values(),
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
    inputs60 = _dedupe([
        *core,
        *_file_inputs([
            outputs60["int8"]["canonical_path"],
            outputs60["scales"]["canonical_path"],
            outputs60["eligibility"]["canonical_path"],
        ]),
    ])
    ood = _dedupe([
        *core,
        *(expected_input_signature(path) for path in _static_ood_paths()),
        *_materialized_chunk_inputs(),
        *_hf_snapshot_file_inputs(),
    ])

    model30 = {
        "model_label": "r0061-30m",
        "model_path": bundle30["model"]["canonical_path"],
        "model_sha256": bundle30["model"]["sha256"],
        "train_receipt_path": bundle30["train_receipt"]["canonical_path"],
        "train_receipt_sha256": bundle30["train_receipt"]["sha256"],
    }
    model60 = {
        "model_label": "r0063-60m",
        "model_path": bundle60["model"]["canonical_path"],
        "model_sha256": bundle60["model"]["sha256"],
        "train_receipt_path": bundle60["train_receipt"]["canonical_path"],
        "train_receipt_sha256": bundle60["train_receipt"]["sha256"],
    }
    data30 = {
        "substrate_label": "balanced-30m",
        "row_count": 30_000_000,
        "row_order": "first 10M rows of each FineWeb/RedPajama/Pile block",
        "int8_path": outputs30["int8"]["canonical_path"],
        "int8_sha256": outputs30["int8"]["sha256"],
        "scales_path": outputs30["scales"]["canonical_path"],
        "scales_sha256": outputs30["scales"]["sha256"],
        "eligibility_path": outputs30["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs30["eligibility"]["sha256"],
    }
    data60 = {
        "substrate_label": "balanced-60m",
        "row_count": 60_000_000,
        "row_order": "first 20M rows of each FineWeb/RedPajama/Pile block",
        "int8_path": outputs60["int8"]["canonical_path"],
        "int8_sha256": outputs60["int8"]["sha256"],
        "scales_path": outputs60["scales"]["canonical_path"],
        "scales_sha256": outputs60["scales"]["sha256"],
        "eligibility_path": outputs60["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs60["eligibility"]["sha256"],
    }
    jobs = [
        _job(
            node_id="transform_r0061_30m",
            action="transform",
            deps=[],
            output=paths["transform_control"],
            p90_wall_s=180,
            inputs=inputs30,
            gpu=True,
            map_key="r0061-30m-on-30m",
            **model30,
            **data30,
        ),
        _job(
            node_id="transform_r0063_on_30m",
            action="transform",
            deps=[],
            output=paths["transform_scaled_matched"],
            p90_wall_s=180,
            inputs=inputs30,
            gpu=True,
            map_key="r0063-60m-on-30m",
            **model60,
            **data30,
        ),
        _job(
            node_id="transform_r0063_60m",
            action="transform",
            deps=[],
            output=paths["transform_scaled_full"],
            p90_wall_s=300,
            inputs=inputs60,
            gpu=True,
            map_key="r0063-60m-on-60m",
            **model60,
            **data60,
        ),
        _job(
            node_id="high_d_reference_30m",
            action="high_d_reference",
            deps=[],
            output=paths["reference30"],
            p90_wall_s=600,
            inputs=inputs30,
            gpu=True,
            **data30,
        ),
        _job(
            node_id="panel_r0061_30m",
            action="panel",
            deps=["transform_r0061_30m", "high_d_reference_30m"],
            output=paths["panel_control"],
            p90_wall_s=1_200,
            inputs=inputs30,
            gpu=True,
            map_key="r0061-30m-on-30m",
            transform_output=paths["transform_control"],
            reference_output=paths["reference30"],
            **model30,
            **data30,
        ),
        _job(
            node_id="panel_r0063_on_30m",
            action="panel",
            deps=["transform_r0063_on_30m", "high_d_reference_30m"],
            output=paths["panel_scaled_matched"],
            p90_wall_s=1_200,
            inputs=inputs30,
            gpu=True,
            map_key="r0063-60m-on-30m",
            transform_output=paths["transform_scaled_matched"],
            reference_output=paths["reference30"],
            **model60,
            **data30,
        ),
        _job(
            node_id="high_d_reference_60m",
            action="high_d_reference",
            deps=[],
            output=paths["reference60"],
            p90_wall_s=900,
            inputs=inputs60,
            gpu=True,
            **data60,
        ),
        _job(
            node_id="panel_r0063_60m",
            action="panel",
            deps=["transform_r0063_60m", "high_d_reference_60m"],
            output=paths["panel_scaled_full"],
            p90_wall_s=1_800,
            inputs=inputs60,
            gpu=True,
            map_key="r0063-60m-on-60m",
            transform_output=paths["transform_scaled_full"],
            reference_output=paths["reference60"],
            **model60,
            **data60,
        ),
        _job(
            node_id="scale_comparison",
            action="comparison",
            deps=[
                "panel_r0061_30m",
                "panel_r0063_on_30m",
                "panel_r0063_60m",
            ],
            output=paths["comparison"],
            p90_wall_s=60,
            inputs=core,
            gpu=False,
            matched_control_panel=paths["panel_control"],
            scaled_matched_panel=paths["panel_scaled_matched"],
            scaled_full_panel=paths["panel_scaled_full"],
        ),
        _job(
            node_id="ood_r0061_30m",
            action="ood",
            deps=["transform_r0061_30m"],
            output=paths["ood_control"],
            p90_wall_s=600,
            inputs=ood,
            gpu=True,
            map_key="r0061-30m-on-30m",
            transform_output=paths["transform_control"],
            **model30,
        ),
        _job(
            node_id="ood_r0063_60m",
            action="ood",
            deps=["transform_r0063_60m"],
            output=paths["ood_scaled"],
            p90_wall_s=600,
            inputs=ood,
            gpu=True,
            map_key="r0063-60m-on-60m",
            transform_output=paths["transform_scaled_full"],
            **model60,
        ),
        _job(
            node_id="matched_renders",
            action="renders",
            deps=[
                "transform_r0061_30m",
                "transform_r0063_on_30m",
                "transform_r0063_60m",
            ],
            output=paths["renders"],
            p90_wall_s=300,
            inputs=_dedupe([*inputs30, *inputs60]),
            gpu=False,
            control_transform=paths["transform_control"],
            scaled_matched_transform=paths["transform_scaled_matched"],
            scaled_full_transform=paths["transform_scaled_full"],
            full_int8_path=outputs60["int8"]["canonical_path"],
            full_int8_sha256=outputs60["int8"]["sha256"],
            full_scales_path=outputs60["scales"]["canonical_path"],
            full_scales_sha256=outputs60["scales"]["sha256"],
            full_eligibility_path=outputs60["eligibility"]["canonical_path"],
            full_eligibility_sha256=outputs60["eligibility"]["sha256"],
            **data30,
        ),
        _job(
            node_id="registry_publication",
            action="registry",
            deps=[
                "panel_r0061_30m",
                "panel_r0063_60m",
                "ood_r0061_30m",
                "ood_r0063_60m",
                "matched_renders",
            ],
            output=paths["registry"],
            p90_wall_s=300,
            inputs=core,
            gpu=False,
        ),
    ]
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=2.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0064-scale-evaluation-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0049", "0053", "0061", "0063"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-int8-trained-model-seed42-v1",
        "minilm-balanced-60m-trained-model-seed42-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-30m-60m-scale-geometry-v1",
        "minilm-balanced-30m-map-registry-v1",
        "minilm-balanced-60m-map-registry-v1",
    ]
    manifest["training_performed"] = False
    manifest["late_bound_models"] = {
        "r0061_30m": {
            "model": bundle30["model"],
            "train_receipt": bundle30["train_receipt"],
            "review": review30,
        },
        "r0063_60m": {
            "model": bundle60["model"],
            "train_receipt": bundle60["train_receipt"],
            "review": review60,
        },
    }
    manifest["scientific_contract"] = {
        "matched_universe": (
            "both checkpoints scored on exact R0053 retained 30M rows "
            "with one shared high-D reference and anchors"
        ),
        "full_scale_universe": (
            "R0063 checkpoint scored on exact R0049 retained 60M rows"
        ),
        "absolute_selector": {
            "ffr_min": 0.40,
            "density_min": 0.60,
            "purity_ratio_min": 0.50,
            "projection": "strictly beats three-seed untrained floor",
            "recall": "recall@50 > recall@10",
        },
        "matched_noninferiority_margins": {
            "ffr": 0.02,
            "density": 0.05,
            "purity_k256": 0.05,
            "purity_k1024": 0.05,
            "projection_ffr": 0.02,
        },
        "decision": (
            "advance to 120M only when full-60M absolute selector and every "
            "matched noninferiority check pass; otherwise bisect at 45M"
        ),
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
    parser.add_argument("--substrate-60-sha256", required=True)
    for scale in ("30", "60"):
        parser.add_argument(f"--model-{scale}", required=True)
        parser.add_argument(f"--model-{scale}-sha256", required=True)
        parser.add_argument(f"--receipt-{scale}", required=True)
        parser.add_argument(f"--receipt-{scale}-sha256", required=True)
        parser.add_argument(f"--review-{scale}", required=True)
        parser.add_argument(f"--review-{scale}-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0064(
            release_sha=args.release_sha,
            substrate_30_sha256=args.substrate_30_sha256,
            substrate_60_sha256=args.substrate_60_sha256,
            model_30_path=args.model_30,
            model_30_sha256=args.model_30_sha256,
            receipt_30_path=args.receipt_30,
            receipt_30_sha256=args.receipt_30_sha256,
            review_30_path=args.review_30,
            review_30_sha256=args.review_30_sha256,
            model_60_path=args.model_60,
            model_60_sha256=args.model_60_sha256,
            receipt_60_path=args.receipt_60,
            receipt_60_sha256=args.receipt_60_sha256,
            review_60_path=args.review_60,
            review_60_sha256=args.review_60_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
