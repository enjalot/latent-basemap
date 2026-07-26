#!/usr/bin/env python3
"""Prepare the coverage-aligned selected-tier training queue."""
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
from basemap.round0065_substrates import (
    SUBSETS,
    validate_scale_substrate,
)
from basemap.round0066_quality import load_scale_decision
from basemap.round0068_training import (
    ROUND_ID,
    train_config_from_capabilities,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0068"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(
    LAB_ROOT,
    "round-0068-2026-07-26.md",
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
        raise RuntimeError("R0068 remains draft; refuse queue materialization")


def prepare_round0068(
    *,
    release_sha: str,
    scale_comparison_path: str,
    scale_comparison_sha256: str,
    substrate_manifest_path: str,
    substrate_manifest_sha256: str,
    canonical_graph_manifest_path: str,
    canonical_graph_manifest_sha256: str,
    r0064_review_path: str,
    r0064_review_sha256: str,
    r0065_review_path: str,
    r0065_review_sha256: str,
    r0067_review_path: str,
    r0067_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0068 release SHA must be one full commit")
    decision = load_scale_decision(
        scale_comparison_path,
        expected_sha256=scale_comparison_sha256,
    )
    tier = decision["tier"]
    spec = SUBSETS[tier]
    row_count = int(spec["row_count"])
    substrate = validate_scale_substrate(
        substrate_manifest_path,
        tier=tier,
        expected_sha256=substrate_manifest_sha256,
    )
    outputs = substrate["manifest"]["outputs"]
    graph = load_canonical_graph(
        canonical_graph_manifest_path,
        expected_sha256=canonical_graph_manifest_sha256,
        expected_eligibility_sha256=outputs["eligibility"]["sha256"],
        row_count=row_count,
    )
    config, config_sha256 = train_config_from_capabilities(
        tier=tier,
        graph_manifest=graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        substrate_manifest=substrate["manifest"],
        substrate_manifest_path=substrate["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate["signature"]["sha256"],
    )
    review64 = _require_review(
        r0064_review_path,
        expected_sha256=r0064_review_sha256,
        required_text=(scale_comparison_sha256, tier),
    )
    review65 = _require_review(
        r0065_review_path,
        expected_sha256=r0065_review_sha256,
        required_text=(substrate_manifest_sha256, f"balanced-{tier}"),
    )
    review67 = _require_review(
        r0067_review_path,
        expected_sha256=r0067_review_sha256,
        required_text=(canonical_graph_manifest_sha256, tier),
    )
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0068 GPU queue",
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    output = os.path.join(
        artifacts,
        f"train-balanced-{tier}",
    )
    inputs = _dedupe([
        *_file_inputs([
            ROUND_FILE,
            scale_comparison_path,
            substrate_manifest_path,
            outputs["int8"]["canonical_path"],
            outputs["scales"]["canonical_path"],
            outputs["eligibility"]["canonical_path"],
            canonical_graph_manifest_path,
            graph["manifest"]["outputs"]["targets"]["canonical_path"],
            graph["manifest"]["outputs"]["degrees"]["canonical_path"],
            r0064_review_path,
            r0065_review_path,
            r0067_review_path,
        ]),
    ])
    cap = 3.0 if tier == "45m" else 7.5
    p90 = 10_800.0 if tier == "45m" else 27_000.0
    updates = int(
        config["optimizer"]["successful_positive_lr_updates"]
    )
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=cap,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0068-selected-tier-train-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0064", "0065", "0067"]
    manifest["capability_dependencies"] = [
        "minilm-balanced-30m-60m-scale-geometry-v1",
        f"minilm-balanced-{tier}-int8-input-v1",
        f"minilm-balanced-{tier}-gpu-native-graph-v1",
    ]
    manifest["capabilities_produced"] = [
        f"minilm-balanced-{tier}-trained-model-seed42-v1",
    ]
    manifest["training_performed"] = True
    manifest["production_config"] = config
    manifest["production_config_sha256"] = config_sha256
    manifest["late_bound_selection"] = {
        "tier": tier,
        "scale_comparison": decision["signature"],
        "substrate": substrate["signature"],
        "graph": graph["signature"],
        "reviews": {
            "0064": review64,
            "0065": review65,
            "0067": review67,
        },
    }
    manifest["scientific_contract"] = {
        "tier_selected_only_by_r0064": tier,
        "rows": row_count,
        "retained_scientific_rows": config["graph"][
            "positive_source_rows"
        ],
        "successful_updates": updates,
        "coverage_alignment": config["execution"]["coverage_alignment"],
        "source_edge_uniform_equivalence": config["graph"][
            "source_edge_uniform_equivalence"
        ],
        "runtime_safety": {
            "standalone_canary": False,
            "minimum_updates_per_second": config["execution"][
                "minimum_train_upd_s"
            ],
            "live_performance_windows": config["execution"][
                "performance_windows"
            ],
        },
        "training_wall_only": True,
        "geometry_claim_requires_successor_evaluation": True,
    }
    manifest["jobs"] = [{
        "id": f"train_seed42_balanced_{tier}",
        "action": "train_selected_tier",
        "handler_module": "experiments.round0068_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(
            artifacts,
            f"train_seed42_balanced_{tier}.done.json",
        ),
        "expected_inputs": inputs,
        "p90_wall_s": p90,
        "tier": tier,
        "scale_comparison": scale_comparison_path,
        "scale_comparison_sha256": scale_comparison_sha256,
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
        f"train_seed42_balanced_{tier}": p90,
        "total": p90,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--scale-comparison", required=True)
    parser.add_argument("--scale-comparison-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument("--canonical-graph-manifest", required=True)
    parser.add_argument("--canonical-graph-manifest-sha256", required=True)
    for round_id in ("0064", "0065", "0067"):
        parser.add_argument(f"--r{round_id}-review", required=True)
        parser.add_argument(f"--r{round_id}-review-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0068(
            release_sha=args.release_sha,
            scale_comparison_path=args.scale_comparison,
            scale_comparison_sha256=args.scale_comparison_sha256,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_sha256=args.substrate_manifest_sha256,
            canonical_graph_manifest_path=args.canonical_graph_manifest,
            canonical_graph_manifest_sha256=(
                args.canonical_graph_manifest_sha256
            ),
            r0064_review_path=args.r0064_review,
            r0064_review_sha256=args.r0064_review_sha256,
            r0065_review_path=args.r0065_review,
            r0065_review_sha256=args.r0065_review_sha256,
            r0067_review_path=args.r0067_review,
            r0067_review_sha256=args.r0067_review_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
