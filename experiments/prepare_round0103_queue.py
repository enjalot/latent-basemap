#!/usr/bin/env python3
"""Prepare the CPU-only full-768 diverse-Jina substrate queue."""
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
from basemap.round0087_inventory import DIMENSION, TARGET_ROWS
from basemap.round0103_substrate import (
    ELIGIBILITY_PATH,
    ELIGIBILITY_SHA256,
    EXCLUDED_ROWS,
    INVENTORY_IDENTITY,
    INVENTORY_PATH,
    INVENTORY_SHA256,
    RECONSTRUCTION_COSINE_P01_FLOOR,
    RETAINED_ROWS,
    ROUND_ID,
    SAMPLE_ROWS,
    SAMPLE_SEED,
    validate_inventory,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0103"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-cpu-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0103-*.md")


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "R0103 requires exactly one issued round document; "
            f"found {len(candidates)}"
        )
    return candidates[0]


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
        raise RuntimeError(f"{path} does not bind the required capability")
    return signature


def _recorded_source_inputs(
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for item in selection["ranges"]:
        shard = item["shard"]
        signature = {
            "canonical_path": os.path.realpath(str(shard["canonical_path"])),
            "kind": "file",
            "bytes": int(shard["bytes"]),
            "sha256": str(shard["sha256"]),
        }
        previous = inputs.setdefault(signature["canonical_path"], signature)
        if previous != signature:
            raise RuntimeError("R0087 inventory has conflicting shard hashes")
    return list(inputs.values())


def prepare_round0103(
    *,
    release_sha: str,
    r0038_review_path: str,
    r0038_review_sha256: str,
    r0087_review_path: str,
    r0087_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    round_file = _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0103 release SHA must be one full commit")
    inventory = validate_inventory()
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    if eligibility["sha256"] != ELIGIBILITY_SHA256:
        raise RuntimeError("R0087 eligibility bytes changed")
    reviews = {
        "0038": _require_review(
            r0038_review_path,
            expected_sha256=r0038_review_sha256,
            required_text=(
                "capability:jina-mrl-two-seed-decision-v1",
                "reject-384d",
            ),
        ),
        "0087": _require_review(
            r0087_review_path,
            expected_sha256=r0087_review_sha256,
            required_text=(
                "capability:jina-diverse-25m-inventory-v1",
                INVENTORY_IDENTITY,
                ELIGIBILITY_SHA256,
            ),
        ),
    }
    inputs = _dedupe([
        expected_input_signature(round_file),
        inventory["signature"],
        eligibility,
        *reviews.values(),
        *_recorded_source_inputs(inventory["selection"]),
    ])
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0103 CPU substrate queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    output = os.path.join(
        artifacts,
        "jina-diverse-25m-full768-int8-substrate",
    )
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.0,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["schema"] = "round0103-diverse-jina-substrate-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "cpu-io-heavy"
    manifest["required_reviews"] = ["0038", "0087"]
    manifest["capability_dependencies"] = [
        "jina-mrl-two-seed-decision-v1",
        "jina-diverse-25m-inventory-v1",
    ]
    manifest["capabilities_produced"] = [
        "jina-diverse-25m-full768-int8-substrate-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        "review_0038": reviews["0038"],
        "review_0087": reviews["0087"],
        "inventory": inventory["signature"],
        "eligibility": eligibility,
    }
    manifest["scientific_contract"] = {
        "rows": TARGET_ROWS,
        "dimension": DIMENSION,
        "embedding_prompt": "raw",
        "source_dtype": "<f2",
        "output_dtype": "|i1",
        "scale_dtype": "<f2",
        "retained_rows": RETAINED_ROWS,
        "excluded_exact_copy_rows": EXCLUDED_ROWS,
        "inventory_sha256": INVENTORY_SHA256,
        "inventory_identity_sha256": INVENTORY_IDENTITY,
        "eligibility_sha256": ELIGIBILITY_SHA256,
        "quantization": (
            "row-local symmetric signed int8 with exact stored fp16 scale"
        ),
        "reconstruction_sample": {
            "seed": SAMPLE_SEED,
            "retained_rows": SAMPLE_ROWS,
            "cosine_p01_floor": RECONSTRUCTION_COSINE_P01_FLOOR,
        },
        "labels": [
            "dataset_id",
            "english_corpus_id",
            "language_id",
        ],
        "no_dimension_truncation": True,
        "no_renormalization": True,
        "no_prompt_application": True,
        "no_graph_search_training_or_evaluation": True,
        "may_overlap_host_int8_training": False,
    }
    node_id = "stage_jina_diverse_25m_full768_int8"
    manifest["jobs"] = [{
        "id": node_id,
        "action": "stage_full768_int8",
        "handler_module": "experiments.round0103_nodes",
        "handler_callable": "run_job",
        "deps": [],
        "outputs": [output],
        "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
        "expected_inputs": inputs,
        "p90_wall_s": 1_800.0,
        "inventory": INVENTORY_PATH,
        "inventory_sha256": INVENTORY_SHA256,
        "node_policy": {
            "gpu_required": False,
            "training_performed": False,
        },
    }]
    manifest["p90_gpu_seconds"] = {"total": 0.0}
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    for round_id in ("0038", "0087"):
        parser.add_argument(f"--review-{round_id}", required=True)
        parser.add_argument(f"--review-{round_id}-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0103(
            release_sha=args.release_sha,
            r0038_review_path=args.review_0038,
            r0038_review_sha256=args.review_0038_sha256,
            r0087_review_path=args.review_0087,
            r0087_review_sha256=args.review_0087_sha256,
            queue_root=args.queue_root,
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
