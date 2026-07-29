#!/usr/bin/env python3
"""Materialize, but never launch, the Round 0043 nested geometry queue."""
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
from basemap.round0036_pipeline import validate_seal
from basemap.round0043_program import (
    COORDINATE_RECEIPT_SHA256,
    COORDINATE_ROOT,
    ELIGIBILITY_PATH,
    R0025_MANIFEST,
    R0025_MANIFEST_SHA256,
    R0036_PANEL,
    ROUND_ID,
    RUNG_WIDTHS,
    rung_label,
    validate_manifest_universe,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_FILE = os.path.join(
    LAB_ROOT, "round-0043-2026-07-24.md"
)
ROUND_ROOT = "/data/latent-basemap/runs/round-0043"


def _require_issued_round() -> None:
    with open(ROUND_FILE, encoding="utf-8") as handle:
        head = handle.read(4096)
    statuses = re.findall(r"(?m)^status:\s*([^\s]+)\s*$", head)
    if statuses != ["issued"]:
        raise RuntimeError(
            f"R0043 requires one issued status; observed {statuses}"
        )


def _coordinate_inputs() -> list[dict[str, Any]]:
    receipt_path = os.path.join(
        COORDINATE_ROOT, "actual-transform.json"
    )
    signature = expected_input_signature(receipt_path)
    if signature["sha256"] != COORDINATE_RECEIPT_SHA256:
        raise RuntimeError("accepted R0036 coordinate receipt changed")
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0036 transform receipt")
    members = receipt["coordinate_stream"]["ordered_chunks"]
    outputs = [signature]
    for item in members:
        path = os.path.realpath(os.path.join(
            COORDINATE_ROOT,
            f"chunk-{int(item['chunk_index']):05d}",
            "coordinates.npy",
        ))
        value = {
            "canonical_path": path,
            "kind": "file",
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }
        if (
            not os.path.isfile(path)
            or os.path.getsize(path) != value["bytes"]
        ):
            raise RuntimeError("R0036 coordinate member changed size")
        outputs.append(value)
    return outputs


def _reviewed_int8_inputs() -> list[dict[str, Any]]:
    manifest_signature = expected_input_signature(R0025_MANIFEST)
    if manifest_signature["sha256"] != R0025_MANIFEST_SHA256:
        raise RuntimeError("accepted R0025 int8 manifest changed")
    with open(R0025_MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_seal(manifest, label="R0025 int8 manifest")
    universe = validate_manifest_universe(manifest)
    return [
        manifest_signature,
        universe["int8"],
        universe["scales"],
    ]


def prepare_round0043(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
    gpu_hours_cap: float = 3.0,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0043 release SHA must be one full commit")
    if not 0.0 < gpu_hours_cap <= 3.0:
        raise ValueError("R0043 GPU-hour cap must be in (0, 3.0]")
    _require_issued_round()
    queue_root = create_fresh_directory(
        queue_root, label="Round 0043 queue"
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    protocol = _file_inputs([
        ROUND_FILE,
        os.path.join(LAB_ROOT, "review-0025-2026-07-20.md"),
        os.path.join(LAB_ROOT, "review-0033-2026-07-22.md"),
        os.path.join(LAB_ROOT, "review-0036-2026-07-23.md"),
        R0036_PANEL,
        ELIGIBILITY_PATH,
    ])
    inputs = _dedupe([
        *protocol,
        *_reviewed_int8_inputs(),
        *_coordinate_inputs(),
    ])
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=gpu_hours_cap,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0043-nested-geometry-queue-v1"
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0025", "0033", "0036"]
    manifest["capability_dependencies"] = [
        "minilm-int8-shards-v1",
        "minilm-150m-row-eligibility-v1",
        "diagnostic:R0036-150m-model-coordinates",
    ]
    manifest["capabilities_produced"] = [
        "150m-nested-candidate-geometry-diagnostic-v1"
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "fixed": [
            "R0036 model",
            "R0036 150M coordinates",
            "R0025 int8 representation",
            "R0033 exact representative policy",
            "panel v2.2 distance formulas",
        ],
        "balanced_rungs": {
            "030m": "first 10M rows per corpus",
            "060m": "first 20M rows per corpus",
            "120m": "first 40M rows per corpus",
            "150m": "first 50M rows per corpus",
        },
        "anchor_views": [
            "fixed anchors drawn once from the balanced 30M representatives",
            "fresh representative anchors over each complete rung",
        ],
        "metrics": ["ffr", "recall_at_10", "density"],
        "no_training": True,
        "no_sampler_or_scale_recipe_selection": True,
    }
    p90 = {
        10_000_000: 600.0,
        20_000_000: 1_200.0,
        40_000_000: 2_400.0,
        50_000_000: 3_000.0,
    }
    jobs = []
    rung_outputs: dict[str, str] = {}
    prior: list[str] = []
    for width in RUNG_WIDTHS:
        label = rung_label(width)
        job_id = f"score_nested_{label}"
        output = os.path.join(artifacts, label)
        rung_outputs[label] = output
        jobs.append({
            "id": job_id,
            "action": "score_rung",
            "handler_module": "experiments.round0043_nodes",
            "handler_callable": "run_job",
            "deps": list(prior),
            "outputs": [output],
            "done_marker": os.path.join(
                artifacts, f"{job_id}.done.json"
            ),
            "expected_inputs": inputs,
            "p90_wall_s": p90[width],
            "per_corpus_rows": width,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        })
        prior = [job_id]
    aggregate = os.path.join(artifacts, "aggregate")
    jobs.append({
        "id": "aggregate_nested_geometry",
        "action": "aggregate",
        "handler_module": "experiments.round0043_nodes",
        "handler_callable": "run_job",
        "deps": [f"score_nested_{rung_label(width)}" for width in RUNG_WIDTHS],
        "outputs": [aggregate],
        "done_marker": os.path.join(
            artifacts, "aggregate_nested_geometry.done.json"
        ),
        "expected_inputs": inputs,
        "p90_wall_s": 60.0,
        "rung_outputs": rung_outputs,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    })
    manifest["p90_gpu_seconds"] = {
        rung_label(width): p90[width] for width in RUNG_WIDTHS
    }
    manifest["p90_gpu_seconds"]["total"] = sum(p90.values())
    manifest["jobs"] = jobs
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
    parser.add_argument(
        "--gpu-hours-cap",
        type=float,
        default=3.0,
        help="Per-attempt cap; lower this on retry to preserve the round cap.",
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0043(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            gpu_hours_cap=args.gpu_hours_cap,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
