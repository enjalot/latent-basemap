#!/usr/bin/env python3
"""Prepare the coverage-aligned balanced-120M training queue."""
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
from basemap.round0065_substrates import validate_scale_substrate
from basemap.round0079_training import (
    ELIGIBILITY_SUMMARY,
    ROUND_ID,
    ROW_COUNT,
    TIER,
    train_config_from_capabilities,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0079_nodes import _load_scale_evidence


ROUND_ROOT = "/data/latent-basemap/runs/round-0079"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0079-2026-07-27.md",
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
        raise RuntimeError("R0079 remains draft; refuse queue materialization")


def prepare_round0079(
    *,
    release_sha: str,
    scale_geometry_path: str,
    scale_geometry_sha256: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    canonical_graph_manifest_path: str,
    canonical_graph_manifest_sha256: str,
    r0065_review_path: str,
    r0065_review_sha256: str,
    r0076_review_path: str,
    r0076_review_sha256: str,
    r0078_review_path: str,
    r0078_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0079 release SHA must be one full commit")
    evidence_job = {
        "scale_geometry": scale_geometry_path,
        "scale_geometry_sha256": scale_geometry_sha256,
    }
    scale_signature, anchor_signature = _load_scale_evidence(evidence_job)
    substrate = validate_scale_substrate(
        substrate_manifest_path,
        tier=TIER,
        expected_sha256=substrate_manifest_sha256,
    )
    outputs = substrate["manifest"]["outputs"]
    graph = load_canonical_graph(
        canonical_graph_manifest_path,
        expected_sha256=canonical_graph_manifest_sha256,
        expected_eligibility_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    if (
        graph["manifest"].get("round_id") != "0078"
        or graph["manifest"].get("tier") != TIER
        or graph["manifest"].get("inputs", {}).get("substrate")
        != substrate["signature"]
    ):
        raise RuntimeError(
            "R0079 graph does not bind the exact R0065 120M substrate"
        )
    config, config_sha256 = train_config_from_capabilities(
        graph_manifest=graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        substrate_manifest=substrate["manifest"],
        substrate_manifest_path=substrate["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate["signature"]["sha256"],
        scale_geometry_signature=scale_signature,
        anchor_leverage_signature=anchor_signature,
    )
    reviews = {
        "0065": _require_review(
            r0065_review_path,
            expected_sha256=r0065_review_sha256,
            required_text=(
                "capability:minilm-balanced-120m-int8-input-v1",
                substrate_manifest_sha256,
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
        "0078": _require_review(
            r0078_review_path,
            expected_sha256=r0078_review_sha256,
            required_text=(
                "capability:minilm-balanced-120m-gpu-native-graph-v1",
                canonical_graph_manifest_sha256,
            ),
        ),
    }
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0079 GPU queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(artifacts, "train-balanced-120m")
    inputs = _dedupe(_file_inputs([
        ROUND_FILE,
        scale_geometry_path,
        substrate_manifest_path,
        outputs["int8"]["canonical_path"],
        outputs["scales"]["canonical_path"],
        outputs["eligibility"]["canonical_path"],
        canonical_graph_manifest_path,
        graph["manifest"]["outputs"]["targets"]["canonical_path"],
        graph["manifest"]["outputs"]["degrees"]["canonical_path"],
        r0065_review_path,
        r0076_review_path,
        r0078_review_path,
    ]))
    updates = int(config["optimizer"]["successful_positive_lr_updates"])
    p90_seconds = 25_200.0
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=7.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0079-balanced-120m-train-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0065", "0076", "0078"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-120m-int8-input-v1",
        "minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
        "minilm-balanced-120m-gpu-native-graph-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-120m-trained-model-seed42-v1",
    ]
    manifest["training_performed"] = True
    manifest["production_config"] = config
    manifest["production_config_sha256"] = config_sha256
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "scale_geometry": scale_signature,
        "anchor_leverage": anchor_signature,
        "substrate": substrate["signature"],
        "graph": graph["signature"],
    }
    manifest["scientific_contract"] = {
        "tier": TIER,
        "rows": ROW_COUNT,
        "retained_scientific_rows": ELIGIBILITY_SUMMARY[
            "retained_row_count"
        ],
        "successful_updates": updates,
        "coverage_alignment": config["execution"]["coverage_alignment"],
        "source_edge_uniform_equivalence": config["graph"][
            "source_edge_uniform_equivalence"
        ],
        "duplicate_control": config["execution"]["duplicate_control"],
        "runtime_safety": {
            "standalone_canary": False,
            "minimum_updates_per_second": config["execution"][
                "minimum_train_upd_s"
            ],
            "live_performance_windows": config["execution"][
                "performance_windows"
            ],
            "full_run_retry_count": 0,
        },
        "training_wall_only": True,
        "density_threshold_tuned": False,
        "geometry_claim_requires_successor_evaluation": True,
    }
    manifest["jobs"] = [{
        "id": "train_seed42_balanced_120m",
        "action": "train_balanced_120m",
        "handler_module": "experiments.round0079_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            "train_seed42_balanced_120m.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": p90_seconds,
        "scale_geometry": scale_geometry_path,
        "scale_geometry_sha256": scale_geometry_sha256,
        "substrate_manifest": substrate_manifest_path,
        "substrate_manifest_sha256": substrate_manifest_sha256,
        "canonical_graph_manifest": canonical_graph_manifest_path,
        "canonical_graph_manifest_sha256": (
            canonical_graph_manifest_sha256
        ),
        "train_config_sha256": config_sha256,
        "successful_updates": updates,
        "batch_size": config["optimizer"]["batch_size"],
        "node_policy": {
            "gpu_required": True,
            "training_performed": True,
        },
    }]
    manifest["p90_gpu_seconds"] = {
        "train_seed42_balanced_120m": p90_seconds,
        "total": p90_seconds,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--scale-geometry", required=True)
    parser.add_argument("--scale-geometry-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--canonical-graph-manifest", required=True)
    parser.add_argument("--canonical-graph-manifest-sha256", required=True)
    for round_id in ("0065", "0076", "0078"):
        parser.add_argument(f"--r{round_id}-review", required=True)
        parser.add_argument(f"--r{round_id}-review-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0079(
            release_sha=args.release_sha,
            scale_geometry_path=args.scale_geometry,
            scale_geometry_sha256=args.scale_geometry_sha256,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            canonical_graph_manifest_path=args.canonical_graph_manifest,
            canonical_graph_manifest_sha256=(
                args.canonical_graph_manifest_sha256
            ),
            r0065_review_path=args.r0065_review,
            r0065_review_sha256=args.r0065_review_sha256,
            r0076_review_path=args.r0076_review,
            r0076_review_sha256=args.r0076_review_sha256,
            r0078_review_path=args.r0078_review,
            r0078_review_sha256=args.r0078_review_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
