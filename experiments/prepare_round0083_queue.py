#!/usr/bin/env python3
"""Prepare, but never launch, the fixed 30M graph-recall sensitivity queue."""
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
from basemap.round0034_pipeline import load_canonical_graph
from basemap.round0053_program import (
    ROW_COUNT,
    validate_control_substrate,
)
from basemap.round0055_program import SUCCESSFUL_UPDATES
from basemap.round0083_program import (
    NPROBES,
    PANEL_SCHEMA,
    TRAIN_RECEIPT_SCHEMAS,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.run_round0036_node import (
    CENTROIDS,
    MINILM_QUERIES,
    MINILM_QUERY_PROVENANCE,
)
from experiments.round0054_nodes import _validate_quality
from experiments.round0058_nodes import RECEIPT_SCHEMA as R0058_SCHEMA
from experiments.round0059_nodes import (
    FAISS_WHEEL,
    _load_sealed_json,
)
from experiments.round0060_nodes import (
    QUALIFICATION_SCHEMA as R0060_QUALIFICATION_SCHEMA,
    RUNTIME_SPEC,
)


ROUND_ID = "0083"
ROUND_ROOT = "/data/latent-basemap/runs/round-0083"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0083-2026-07-27.md")
SUBSTRATE = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
QUALITY = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "candidate-quality-30m/balanced-30m-candidate-quality-v1.json"
)
R0058_SWEEP = (
    "/data/latent-basemap/runs/round-0058/queue/artifacts/"
    "nprobe-sweep-60m/balanced-60m-nprobe-sweep-v1.json"
)
R0060_INDEX_ROOT = (
    "/data/latent-basemap/runs/round-0060/queue/artifacts/"
    "gpu-index-balanced-30m"
)
R0060_QUALIFICATION = os.path.join(
    R0060_INDEX_ROOT,
    "gpu-index-qualification-v1.json",
)
FILTERED_INDEX = os.path.join(
    R0060_INDEX_ROOT,
    "balanced-30m-retained.ivfpq",
)
R0060_GRAPH_ROOT = (
    "/data/latent-basemap/runs/round-0060/queue/artifacts/"
    "native-graph-balanced-30m"
)
R0060_GRAPH = os.path.join(R0060_GRAPH_ROOT, "canonical-graph-v1.json")
R0064_ROOT = "/data/latent-basemap/runs/round-0064/queue/artifacts"
R0064_REFERENCE = os.path.join(R0064_ROOT, "high-d-reference-30m")
R0064_BASELINE_PANEL = os.path.join(
    R0064_ROOT,
    "panel-r0061-30m",
    "panel.json",
)
REVIEWS = {
    "0053": os.path.join(LAB_ROOT, "review-0053-2026-07-26.md"),
    "0058": os.path.join(LAB_ROOT, "review-0058-2026-07-26.md"),
    "0060": os.path.join(LAB_ROOT, "review-0060-2026-07-26.md"),
    "0064": os.path.join(LAB_ROOT, "review-0064-2026-07-26.md"),
}
REVIEW_CAPABILITIES = {
    "0053": "capability:minilm-balanced-30m-int8-input-v1",
    "0058": "capability:minilm-balanced-60m-nprobe-calibration-v1",
    "0060": "capability:minilm-balanced-30m-gpu-native-graph-v1",
    "0064": "capability:minilm-balanced-30m-60m-scale-geometry-v1",
}
EXPECTED = {
    "substrate": "d3d553f7a1d36f14a11d63ffee134e3bd6dc9c39b174908307291e4394241245",
    "quality": "239cd2a20d7c2962cd8e674b3b4902668b4dcb0e91ee7bd1c2094a57b45da208",
    "r0058_sweep": "614c5f85aa0bae4fb10be1ad1d56197da116c144312bb2a3aa6eb05538ec9960",
    "r0060_qualification": "48f9b4fe2c8da0dcfa6a7ca639d347b978d857b0c756b881c93f9140322f9495",
    "filtered_index": "a92c05a80d14a45bd5655e8eaeb6c1b97aa737d98fdf14875be4de1c35a2c765",
    "r0060_graph": "74316a3807a13fe763212cb18dcfeb3a30e866be15f5834c3397a59f9815be7e",
    "r0060_targets": "90cd10975a22cfd902bc65d5baf6eb94e0c93eecd14093ae84250ee98807613d",
    "r0064_panel": "c9dfb96b71aebbc7a1b0b2956f3d1d8276faf00f99327b6415250af5d15f3cba",
    "r0064_reference": "71c123df83247dbbd6d8d6d5a2ea79ec12b90d012d638faa0f1281d706834ae3",
    "runtime": "fafb150c22c911c19beb6f24351b4fcfd69a934816f96f5ef6989a22cb2f3097",
    "review_0053": "f06a48c63881a45a2841e1358b89ce2fd610afa4a764a0a6641ce687444c17f5",
    "review_0058": "2ec35a117a9969679c3fce2cbc287b027bbaea414cdaaa45515f3644246058c5",
    "review_0060": "a6a0955adf6699d27cf720801f66464b7808814d93de15d559c31d4683524fc4",
    "review_0064": "f6b77feedfbc6dbef2e098edb88e2e8d9414ebe12ed2a67d18ce05676adcf0a7",
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> None:
    if _frontmatter_status(ROUND_FILE) != "issued":
        raise RuntimeError("R0083 remains draft; refuse queue materialization")


def _require_reviews() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for round_id, path in REVIEWS.items():
        if _frontmatter_status(path) not in {"accepted", "partial"}:
            raise RuntimeError(f"R{round_id} does not release evidence")
        signature = expected_input_signature(path)
        if signature["sha256"] != EXPECTED[f"review_{round_id}"]:
            raise RuntimeError(f"R{round_id} review bytes changed")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        if REVIEW_CAPABILITIES[round_id] not in text:
            raise RuntimeError(
                f"R{round_id} does not release its required capability"
            )
        observed[round_id] = signature
    return observed


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    p90_wall_s: float,
    inputs: list[dict[str, Any]],
    gpu: bool,
    training: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        **extra,
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0083_nodes",
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
            "training_performed": training,
        },
    }


def prepare_round0083(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0083 release SHA must be one full commit")
    reviews = _require_reviews()
    substrate = validate_control_substrate(
        SUBSTRATE,
        expected_sha256=EXPECTED["substrate"],
    )
    quality, quality_signature = _validate_quality(
        QUALITY,
        expected_sha256=EXPECTED["quality"],
        nprobe=64,
    )
    sweep, sweep_signature = _load_sealed_json(
        R0058_SWEEP,
        expected_sha256=EXPECTED["r0058_sweep"],
        schema=R0058_SCHEMA,
    )
    baseline_qualification, baseline_qualification_signature = (
        _load_sealed_json(
            R0060_QUALIFICATION,
            expected_sha256=EXPECTED["r0060_qualification"],
            schema=R0060_QUALIFICATION_SCHEMA,
        )
    )
    graph = load_canonical_graph(
        R0060_GRAPH,
        expected_sha256=EXPECTED["r0060_graph"],
        expected_eligibility_sha256=(
            substrate["manifest"]["outputs"]["eligibility"]["sha256"]
        ),
        row_count=ROW_COUNT,
    )
    filtered = expected_input_signature(FILTERED_INDEX)
    baseline_panel = expected_input_signature(R0064_BASELINE_PANEL)
    reference = expected_input_signature(
        os.path.join(R0064_REFERENCE, "reference.npz")
    )
    runtime = expected_input_signature(RUNTIME_SPEC)
    if (
        filtered["sha256"] != EXPECTED["filtered_index"]
        or graph["manifest"]["outputs"]["targets"]["sha256"]
        != EXPECTED["r0060_targets"]
        or baseline_panel["sha256"] != EXPECTED["r0064_panel"]
        or reference["sha256"] != EXPECTED["r0064_reference"]
        or runtime["sha256"] != EXPECTED["runtime"]
        or sweep.get("rows_by_nprobe", {}).get("32", {}).get(
            "mean_recall_at_15_unambiguous"
        ) != 0.8914062500000001
        or baseline_qualification.get("quality", {}).get(
            "mean_recall_at_15_unambiguous"
        ) != 0.9224609375000001
        or quality.get("recall", {}).get(
            "mean_recall_at_15_unambiguous"
        ) != 0.9224609375000001
    ):
        raise RuntimeError("R0083 issued treatment evidence changed")

    outputs = substrate["manifest"]["outputs"]
    static_paths = [
        ROUND_FILE,
        *REVIEWS.values(),
        SUBSTRATE,
        outputs["int8"]["canonical_path"],
        outputs["scales"]["canonical_path"],
        outputs["eligibility"]["canonical_path"],
        QUALITY,
        R0058_SWEEP,
        R0060_QUALIFICATION,
        FILTERED_INDEX,
        R0060_GRAPH,
        graph["manifest"]["outputs"]["targets"]["canonical_path"],
        graph["manifest"]["outputs"]["degrees"]["canonical_path"],
        RUNTIME_SPEC,
        FAISS_WHEEL,
        R0064_BASELINE_PANEL,
        os.path.join(R0064_REFERENCE, "reference.npz"),
        os.path.join(R0064_REFERENCE, "reference-receipt.json"),
        os.path.join(R0064_REFERENCE, "recall50-truth.npy"),
        os.path.join(R0064_REFERENCE, "anchor-substrate-rows.npy"),
        MINILM_QUERIES,
        MINILM_QUERY_PROVENANCE,
        *CENTROIDS.values(),
    ]
    inputs = _dedupe(_file_inputs(static_paths))
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0083 graph-recall queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    qualification_output = os.path.join(
        artifacts, "graph-recall-qualification"
    )
    qualification_receipt = os.path.join(
        qualification_output,
        "graph-recall-qualification.json",
    )
    paths: dict[int, dict[str, str]] = {}
    for nprobe in NPROBES:
        paths[nprobe] = {
            "graph": os.path.join(
                artifacts, f"native-graph-nprobe{nprobe}"
            ),
            "train": os.path.join(
                artifacts, f"train-nprobe{nprobe}"
            ),
            "coordinates": os.path.join(
                artifacts, f"coordinates-nprobe{nprobe}"
            ),
            "panel": os.path.join(
                artifacts, f"panel-nprobe{nprobe}"
            ),
        }
    comparison_output = os.path.join(
        artifacts, "graph-recall-sensitivity"
    )
    common = {
        "substrate_manifest": SUBSTRATE,
        "substrate_manifest_sha256": EXPECTED["substrate"],
    }
    jobs: list[dict[str, Any]] = [
        _job(
            node_id="qualify_fixed_graph_recall",
            action="qualify",
            deps=[],
            output=qualification_output,
            p90_wall_s=300,
            inputs=inputs,
            gpu=True,
            quality_receipt=QUALITY,
            quality_receipt_sha256=EXPECTED["quality"],
            baseline_qualification=R0060_QUALIFICATION,
            baseline_qualification_sha256=(
                EXPECTED["r0060_qualification"]
            ),
            filtered_index=FILTERED_INDEX,
            runtime_spec=RUNTIME_SPEC,
            runtime_spec_sha256=EXPECTED["runtime"],
            **common,
        )
    ]
    data30 = {
        "substrate_label": "balanced-30m",
        "row_count": ROW_COUNT,
        "rows_per_corpus": 10_000_000,
        "row_order": (
            "first 10M rows of each FineWeb/RedPajama/Pile block"
        ),
        "int8_path": outputs["int8"]["canonical_path"],
        "int8_sha256": outputs["int8"]["sha256"],
        "scales_path": outputs["scales"]["canonical_path"],
        "scales_sha256": outputs["scales"]["sha256"],
        "eligibility_path": outputs["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs["eligibility"]["sha256"],
    }
    for nprobe in NPROBES:
        graph_node = f"build_graph_nprobe{nprobe}"
        train_node = f"train_nprobe{nprobe}_seed42"
        transform_node = f"transform_nprobe{nprobe}"
        panel_node = f"panel_nprobe{nprobe}"
        graph_manifest = os.path.join(
            paths[nprobe]["graph"], "canonical-graph-v1.json"
        )
        train_receipt = os.path.join(
            paths[nprobe]["train"], "train-receipt.json"
        )
        model = os.path.join(paths[nprobe]["train"], "model.pt")
        jobs.extend([
            _job(
                node_id=graph_node,
                action="build_graph",
                deps=["qualify_fixed_graph_recall"],
                output=paths[nprobe]["graph"],
                p90_wall_s=3_600,
                inputs=inputs,
                gpu=True,
                nprobe=nprobe,
                qualification_receipt=qualification_receipt,
                filtered_index=FILTERED_INDEX,
                baseline_targets=(
                    graph["manifest"]["outputs"]["targets"][
                        "canonical_path"
                    ]
                ),
                runtime_spec=RUNTIME_SPEC,
                runtime_spec_sha256=EXPECTED["runtime"],
                **common,
            ),
            _job(
                node_id=train_node,
                action="train",
                deps=[graph_node],
                output=paths[nprobe]["train"],
                p90_wall_s=5_400,
                inputs=inputs,
                gpu=True,
                training=True,
                nprobe=nprobe,
                canonical_graph_manifest=graph_manifest,
                successful_updates=SUCCESSFUL_UPDATES,
                batch_size=8_192,
                production_config_receipt_schema=(
                    "round0083-production-config-receipt-v1"
                ),
                train_receipt_schema=TRAIN_RECEIPT_SCHEMAS[nprobe],
                **common,
            ),
            _job(
                node_id=transform_node,
                action="transform",
                deps=[train_node],
                output=paths[nprobe]["coordinates"],
                p90_wall_s=240,
                inputs=inputs,
                gpu=True,
                nprobe=nprobe,
                map_key=f"r0083-nprobe{nprobe}-on-30m",
                model_label=f"r0083-nprobe{nprobe}",
                model_path=model,
                train_receipt_path=train_receipt,
                **data30,
            ),
            _job(
                node_id=panel_node,
                action="panel",
                deps=[transform_node],
                output=paths[nprobe]["panel"],
                p90_wall_s=900,
                inputs=inputs,
                gpu=True,
                nprobe=nprobe,
                map_key=f"r0083-nprobe{nprobe}-on-30m",
                model_label=f"r0083-nprobe{nprobe}",
                model_path=model,
                train_receipt_path=train_receipt,
                transform_output=paths[nprobe]["coordinates"],
                reference_output=R0064_REFERENCE,
                panel_schema=PANEL_SCHEMA,
                **data30,
            ),
        ])
    jobs.append(_job(
        node_id="compare_graph_recall_sensitivity",
        action="comparison",
        deps=[f"panel_nprobe{value}" for value in NPROBES],
        output=comparison_output,
        p90_wall_s=60,
        inputs=inputs,
        gpu=False,
        baseline_panel=R0064_BASELINE_PANEL,
        qualification_receipt=qualification_receipt,
        panel_nprobe16=os.path.join(paths[16]["panel"], "panel.json"),
        panel_nprobe32=os.path.join(paths[32]["panel"], "panel.json"),
        graph_receipt_nprobe16=os.path.join(
            paths[16]["graph"], "receipt.json"
        ),
        graph_receipt_nprobe32=os.path.join(
            paths[32]["graph"], "receipt.json"
        ),
    ))
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=6.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0083-graph-recall-sensitivity-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = list(REVIEWS)
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-int8-input-v1",
        "minilm-balanced-60m-nprobe-calibration-v1",
        "minilm-balanced-30m-gpu-native-graph-v1",
        "minilm-balanced-30m-60m-scale-geometry-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-30m-graph-recall-sensitivity-v1",
    ]
    manifest["training_performed"] = True
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "substrate": substrate["signature"],
        "quality": quality_signature,
        "r0058_sweep": sweep_signature,
        "r0060_qualification": baseline_qualification_signature,
        "r0060_graph": graph["signature"],
        "r0064_baseline_panel": baseline_panel,
        "r0064_reference": reference,
    }
    manifest["scientific_contract"] = {
        "fixed_nprobes": list(NPROBES),
        "baseline_nprobe": 64,
        "baseline_candidate_recall_at_15_unambiguous": (
            0.9224609375000001
        ),
        "planning_bands_not_admission_gates": {
            "16": [0.82, 0.88],
            "32": [0.87, 0.91],
        },
        "actual_treatment_dose": [
            "sampled exact-truth candidate recall@15",
            "full-graph neighbor-set overlap versus R0060",
        ],
        "held_fixed": [
            "balanced retained-30M substrate",
            "fixed-degree directed k15 graph",
            "uniform retained-source then uniform-destination sampler law",
            "seed 42",
            "500003 successful positive-LR updates",
            "R0064 retained-30M evaluation universe/reference/anchors",
        ],
        "noninferiority_margins": {
            "ffr": 0.02,
            "purity_k256": 0.05,
            "purity_k1024": 0.05,
            "projection_ffr": 0.02,
        },
        "legacy_density_is_diagnostic_only": True,
        "changes_floor_in_this_round": False,
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
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(prepare_round0083(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
