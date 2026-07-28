#!/usr/bin/env python3
"""Prepare, but never launch, the paired balanced-90M seed contrast."""
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
from basemap.round0053_program import validate_control_substrate
from basemap.round0064_evaluation import validate_train_bundle
from basemap.round0071_substrate import (
    ELIGIBILITY_SUMMARY,
    ROW_COUNT,
    validate_substrate,
)
from basemap.round0075_training import SUCCESSFUL_UPDATES
from basemap.round0084_program import (
    CONFIG_SCHEMA,
    FULL_KEY,
    MATCHED_KEY,
    MODEL_LABEL,
    PANEL_SCHEMA,
    ROUND_ID,
    SEED,
    seed43_config_from_seed42,
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


ROUND_ROOT = "/data/latent-basemap/runs/round-0084"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0084-2026-07-27.md")
SUBSTRATE_30 = (
    "/data/latent-basemap/runs/round-0053/queue/artifacts/"
    "balanced-30m-int8-substrate/balanced-30m-int8-substrate-v1.json"
)
SUBSTRATE_90 = (
    "/data/latent-basemap/runs/round-0071/queue/artifacts/"
    "balanced-90m-int8-substrate/balanced-90m-substrate-v1.json"
)
GRAPH_90 = (
    "/data/latent-basemap/runs/round-0073/queue/artifacts/"
    "native-graph-balanced-90m/canonical-graph-v1.json"
)
SCALE_GEOMETRY = (
    "/data/latent-basemap/runs/round-0069/queue/artifacts/"
    "scale-comparison/scale-comparison.json"
)
ANCHOR_LEVERAGE = (
    "/data/latent-basemap/runs/round-0074/queue-attempt-2/artifacts/"
    "duplicate-anchor-leverage/duplicate-anchor-leverage.json"
)
BASELINE_TRAIN_ROOT = (
    "/data/latent-basemap/runs/round-0075/queue/artifacts/"
    "train-balanced-90m"
)
BASELINE_MODEL = os.path.join(BASELINE_TRAIN_ROOT, "model.pt")
BASELINE_RECEIPT = os.path.join(BASELINE_TRAIN_ROOT, "train-receipt.json")
REFERENCE_30 = (
    "/data/latent-basemap/runs/round-0064/queue/artifacts/"
    "high-d-reference-30m"
)
R0076_ROOT = (
    "/data/latent-basemap/runs/round-0076/queue-attempt-2/artifacts"
)
REFERENCE_90 = os.path.join(R0076_ROOT, "high-d-reference-90m")
BASELINE_MATCHED_PANEL = os.path.join(
    R0076_ROOT, "panel-r0075-on-30m", "panel.json"
)
BASELINE_FULL_PANEL = os.path.join(
    R0076_ROOT, "panel-r0075-90m", "panel.json"
)
REVIEWS = {
    "0071": os.path.join(LAB_ROOT, "review-0071-2026-07-27.md"),
    "0073": os.path.join(LAB_ROOT, "review-0073-2026-07-27.md"),
    "0075": os.path.join(LAB_ROOT, "review-0075-2026-07-27.md"),
    "0076": os.path.join(LAB_ROOT, "review-0076-2026-07-27.md"),
}
CAPABILITIES = {
    "0071": "capability:minilm-balanced-90m-int8-input-v1",
    "0073": "capability:minilm-balanced-90m-gpu-native-graph-v1",
    "0075": "capability:minilm-balanced-90m-trained-model-seed42-v1",
    "0076": "capability:minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
}
EXPECTED = {
    "review_0071": "9e98617b5059c8eca2bb4b31d588b91118564f2e146f6362059edc457d9daee5",
    "review_0073": "5b6897f89cdf792185f10b9a20cfefd562610c059cbe3339bd231a537d23efeb",
    "review_0075": "0f94c3631d22982ef7111881633584ce7a5cbd01954bfe30a290ddbbde6d1e02",
    "review_0076": "d987d15d33bd49119efa279d1e375c73d4edab19c020b21304fe9103415958b5",
    "substrate_30": "d3d553f7a1d36f14a11d63ffee134e3bd6dc9c39b174908307291e4394241245",
    "substrate_90": "032e3c6396e26e0f2ff0db81f764330e4e84175d337d164ab63ae9c7ddeec6d2",
    "graph_90": "d8ec25e2887926d11af6da7b6c6c4bf07d1fa9adedfc9f84d2c1c5baf07fcef5",
    "scale_geometry": "5eec5ce7135c19bc75044c476b09591950b2d2bc951b79b480074646daa0f587",
    "anchor_leverage": "4f2f64a38754791d640b6d101f2cc8bdbe8d17f2fcf5723290ab48dd663bd97a",
    "baseline_model": "197af726596c8959c1f0d107ae7430df721452e340d64155ffdc618e3cb855ee",
    "baseline_receipt": "9b1383f7139c7803460d9a6148ba4afe40a142a519517fb8aae46b4aad6d2235",
    "baseline_matched_panel": "57218f736d80cb495141c702b9d7815b10c428b8ae703ddc0be12f10abd732a2",
    "baseline_full_panel": "351c131a61bd5f9ff6d570aab04170a447b88426f21ddf28f35bb43ce048db72",
    "reference_30": "71c123df83247dbbd6d8d6d5a2ea79ec12b90d012d638faa0f1281d706834ae3",
    "reference_90": "9cf81ea4e9e3f44367e3781f98d70eae0a3e25974c39079d4d546c9126040c18",
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> None:
    if _frontmatter_status(ROUND_FILE) != "issued":
        raise RuntimeError("R0084 remains draft; refuse queue materialization")


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
        if CAPABILITIES[round_id] not in text:
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
        "handler_module": "experiments.round0084_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(
            os.path.dirname(output), f"{node_id}.done.json"
        ),
        "expected_inputs": inputs,
        "p90_wall_s": float(p90_wall_s),
        "node_policy": {
            "gpu_required": gpu,
            "training_performed": training,
        },
    }


def prepare_round0084(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0084 release SHA must be one full commit")
    reviews = _require_reviews()
    substrate30 = validate_control_substrate(
        SUBSTRATE_30,
        expected_sha256=EXPECTED["substrate_30"],
    )
    substrate90 = validate_substrate(
        SUBSTRATE_90,
        expected_sha256=EXPECTED["substrate_90"],
    )
    graph = load_canonical_graph(
        GRAPH_90,
        expected_sha256=EXPECTED["graph_90"],
        expected_eligibility_sha256=(
            substrate90["manifest"]["outputs"]["eligibility"]["sha256"]
        ),
        row_count=ROW_COUNT,
    )
    baseline = validate_train_bundle(
        label="r0075-90m",
        model_path=BASELINE_MODEL,
        model_sha256=EXPECTED["baseline_model"],
        train_receipt_path=BASELINE_RECEIPT,
        train_receipt_sha256=EXPECTED["baseline_receipt"],
    )
    config, config_sha256 = seed43_config_from_seed42(
        baseline["production_config"],
        graph_manifest=graph["manifest"],
        graph_manifest_path=graph["signature"]["canonical_path"],
        graph_manifest_sha256=graph["signature"]["sha256"],
        substrate_manifest=substrate90["manifest"],
        substrate_manifest_path=substrate90["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate90["signature"]["sha256"],
    )
    fixed_signatures = {
        "scale_geometry": expected_input_signature(SCALE_GEOMETRY),
        "anchor_leverage": expected_input_signature(ANCHOR_LEVERAGE),
        "baseline_matched_panel": expected_input_signature(
            BASELINE_MATCHED_PANEL
        ),
        "baseline_full_panel": expected_input_signature(BASELINE_FULL_PANEL),
        "reference_30": expected_input_signature(
            os.path.join(REFERENCE_30, "reference.npz")
        ),
        "reference_90": expected_input_signature(
            os.path.join(REFERENCE_90, "reference.npz")
        ),
    }
    if any(
        fixed_signatures[key]["sha256"] != EXPECTED[key]
        for key in fixed_signatures
    ):
        raise RuntimeError("R0084 fixed evaluation/training evidence changed")

    outputs30 = substrate30["manifest"]["outputs"]
    outputs90 = substrate90["manifest"]["outputs"]
    # All large substrate/graph/model artifacts were authenticated by the
    # validators above. Carry those exact signatures into the runner
    # manifest instead of rereading roughly 50 GB solely to reconstruct the
    # same dictionaries.
    validated_inputs = [
        *reviews.values(),
        substrate30["signature"],
        outputs30["int8"],
        outputs30["scales"],
        outputs30["eligibility"],
        substrate90["signature"],
        outputs90["int8"],
        outputs90["scales"],
        outputs90["eligibility"],
        graph["signature"],
        graph["manifest"]["outputs"]["targets"],
        graph["manifest"]["outputs"]["degrees"],
        baseline["model"],
        baseline["train_receipt"],
        *fixed_signatures.values(),
    ]
    remaining_paths = [
        ROUND_FILE,
        *[
            os.path.join(root, name)
            for root in (REFERENCE_30, REFERENCE_90)
            for name in (
                "reference-receipt.json",
                "recall50-truth.npy",
                "anchor-substrate-rows.npy",
            )
        ],
        MINILM_QUERIES,
        MINILM_QUERY_PROVENANCE,
        *CENTROIDS.values(),
    ]
    inputs = _dedupe([
        *validated_inputs,
        *_file_inputs(remaining_paths),
    ])
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0084 seed-sensitivity queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    paths = {
        "train": os.path.join(artifacts, "train-seed43-balanced-90m"),
        "matched_coordinates": os.path.join(
            artifacts, "coordinates-seed43-on-30m"
        ),
        "full_coordinates": os.path.join(
            artifacts, "coordinates-seed43-90m"
        ),
        "matched_panel": os.path.join(artifacts, "panel-seed43-on-30m"),
        "full_panel": os.path.join(artifacts, "panel-seed43-90m"),
        "comparison": os.path.join(artifacts, "seed43-sensitivity"),
    }
    model = os.path.join(paths["train"], "model.pt")
    train_receipt = os.path.join(paths["train"], "train-receipt.json")
    data30 = {
        "substrate_label": "balanced-30m",
        "row_count": 30_000_000,
        "rows_per_corpus": 10_000_000,
        "row_order": "first 10M rows of each corpus block",
        "int8_path": outputs30["int8"]["canonical_path"],
        "int8_sha256": outputs30["int8"]["sha256"],
        "scales_path": outputs30["scales"]["canonical_path"],
        "scales_sha256": outputs30["scales"]["sha256"],
        "eligibility_path": outputs30["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs30["eligibility"]["sha256"],
    }
    data90 = {
        "substrate_label": "balanced-90m",
        "row_count": ROW_COUNT,
        "rows_per_corpus": 30_000_000,
        "row_order": "first 30M rows of each corpus block",
        "int8_path": outputs90["int8"]["canonical_path"],
        "int8_sha256": outputs90["int8"]["sha256"],
        "scales_path": outputs90["scales"]["canonical_path"],
        "scales_sha256": outputs90["scales"]["sha256"],
        "eligibility_path": outputs90["eligibility"]["canonical_path"],
        "eligibility_sha256": outputs90["eligibility"]["sha256"],
    }
    late_model = {
        "model_label": MODEL_LABEL,
        "model_path": model,
        "train_receipt_path": train_receipt,
    }
    jobs = [
        _job(
            node_id="train_seed43_balanced_90m",
            action="train",
            deps=[],
            output=paths["train"],
            p90_wall_s=13_500,
            inputs=inputs,
            gpu=True,
            training=True,
            release_sha=release_sha,
            scale_geometry=SCALE_GEOMETRY,
            scale_geometry_sha256=EXPECTED["scale_geometry"],
            anchor_leverage=ANCHOR_LEVERAGE,
            anchor_leverage_sha256=EXPECTED["anchor_leverage"],
            substrate_manifest=SUBSTRATE_90,
            substrate_manifest_sha256=EXPECTED["substrate_90"],
            canonical_graph_manifest=GRAPH_90,
            canonical_graph_manifest_sha256=EXPECTED["graph_90"],
            baseline_model=BASELINE_MODEL,
            baseline_model_sha256=EXPECTED["baseline_model"],
            baseline_train_receipt=BASELINE_RECEIPT,
            baseline_train_receipt_sha256=EXPECTED["baseline_receipt"],
            train_config_sha256=config_sha256,
            successful_updates=SUCCESSFUL_UPDATES,
            batch_size=config["optimizer"]["batch_size"],
        ),
        _job(
            node_id="transform_seed43_on_30m",
            action="transform",
            deps=["train_seed43_balanced_90m"],
            output=paths["matched_coordinates"],
            p90_wall_s=240,
            inputs=inputs,
            gpu=True,
            map_key=MATCHED_KEY,
            **late_model,
            **data30,
        ),
        _job(
            node_id="transform_seed43_90m",
            action="transform",
            deps=["train_seed43_balanced_90m"],
            output=paths["full_coordinates"],
            p90_wall_s=600,
            inputs=inputs,
            gpu=True,
            map_key=FULL_KEY,
            **late_model,
            **data90,
        ),
        _job(
            node_id="panel_seed43_on_30m",
            action="panel",
            deps=["transform_seed43_on_30m"],
            output=paths["matched_panel"],
            p90_wall_s=900,
            inputs=inputs,
            gpu=True,
            map_key=MATCHED_KEY,
            transform_output=paths["matched_coordinates"],
            reference_output=REFERENCE_30,
            panel_schema=PANEL_SCHEMA,
            **late_model,
            **data30,
        ),
        _job(
            node_id="panel_seed43_90m",
            action="panel",
            deps=["transform_seed43_90m"],
            output=paths["full_panel"],
            p90_wall_s=2_400,
            inputs=inputs,
            gpu=True,
            map_key=FULL_KEY,
            transform_output=paths["full_coordinates"],
            reference_output=REFERENCE_90,
            panel_schema=PANEL_SCHEMA,
            **late_model,
            **data90,
        ),
        _job(
            node_id="compare_seed43_sensitivity",
            action="comparison",
            deps=["panel_seed43_on_30m", "panel_seed43_90m"],
            output=paths["comparison"],
            p90_wall_s=60,
            inputs=inputs,
            gpu=False,
            seed42_matched_panel=BASELINE_MATCHED_PANEL,
            seed43_matched_panel=os.path.join(
                paths["matched_panel"], "panel.json"
            ),
            seed42_full_panel=BASELINE_FULL_PANEL,
            seed43_full_panel=os.path.join(paths["full_panel"], "panel.json"),
        ),
    ]
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=5.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0084-seed43-sensitivity-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = list(REVIEWS)
    manifest["capability_dependencies"] = [
        "minilm-balanced-90m-int8-input-v1",
        "minilm-balanced-90m-gpu-native-graph-v1",
        "minilm-balanced-90m-trained-model-seed42-v1",
        "minilm-balanced-30m-45m-60m-90m-scale-geometry-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-90m-seed43-sensitivity-v1",
    ]
    manifest["training_performed"] = True
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "substrate_30": substrate30["signature"],
        "substrate_90": substrate90["signature"],
        "graph_90": graph["signature"],
        "seed42_model": baseline["model"],
        "seed42_train_receipt": baseline["train_receipt"],
        **fixed_signatures,
    }
    manifest["production_config"] = config
    manifest["production_config_sha256"] = config_sha256
    manifest["scientific_contract"] = {
        "treatment": "seed 43 versus reviewed seed 42",
        "only_intended_training_difference": "random seed",
        "held_fixed": [
            "balanced retained-90M substrate",
            "reviewed R0073 fixed-degree directed k15 graph",
            "host-int8 canonical sampler and negative law",
            "1493293 successful positive-LR updates",
            "architecture, batch, precision, and optimizer schedule",
            "matched-30M and full-90M evaluation universes",
            "high-D references and representative anchors",
        ],
        "deliverable": (
            "signed and absolute per-metric seed43-minus-seed42 contrasts"
        ),
        "one_contrast_is_not_variance_or_error_bar": True,
        "changes_ladder_decision": False,
        "legacy_density_is_diagnostic_only": True,
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
    print(prepare_round0084(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
