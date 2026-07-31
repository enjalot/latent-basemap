#!/usr/bin/env python3
"""Prepare the retained-only diverse-Jina search qualification queue."""
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
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0105_search import (
    BENCHMARK_REPEATS,
    BENCHMARK_WARMUP_ROWS,
    BOUNDARY_TIE_ATOL,
    DIMENSION,
    ELIGIBILITY_PATH,
    ELIGIBILITY_SHA256,
    EVERY_GROUP_MEAN_FLOOR,
    GLOBAL_MEAN_FLOOR,
    GROUPS,
    INDEX_TRAIN_ROWS,
    INDEX_TRAIN_SAMPLE_SHA256,
    INDEX_TRAIN_SEED,
    K,
    NLIST,
    POLICY_GRID,
    PQ_BITS,
    PQ_M,
    QUALITY_GROUP_IDS_SHA256,
    QUALITY_ROWS,
    QUALITY_ROWS_PER_GROUP,
    QUALITY_SAMPLE_SHA256,
    QUALITY_SEED,
    RETAINED_ROWS,
    ROUND_ID,
    ROW_COUNT,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0105"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0105-*.md")

REVIEW_DEFAULTS = {
    "0087": (
        os.path.join(LAB_ROOT, "review-0087-2026-07-28.md"),
        "61ab9268899c2edc47519bdbe4efeea65a54f0c9fda52bd89e7cad0dafd9d483",
        ("capability:jina-diverse-25m-inventory-v1",),
    ),
    "0103": (
        os.path.join(LAB_ROOT, "review-0103-2026-07-29.md"),
        "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51",
        ("capability:jina-diverse-25m-full768-int8-substrate-v1",),
    ),
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4_096)
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
            f"R0105 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str | None,
    required_text: tuple[str, ...],
) -> dict[str, Any]:
    if _frontmatter_status(path) != "accepted":
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if expected_sha256 and signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if any(value not in text for value in required_text):
        raise RuntimeError(f"{path} lacks required capability evidence")
    return signature


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    p90_wall_s: float,
    inputs: list[dict[str, Any]],
    **values: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0105_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(
            os.path.dirname(output), f"{node_id}.done.json"
        ),
        "expected_inputs": inputs,
        "p90_wall_s": p90_wall_s,
        "node_policy": {
            "gpu_required": True,
            "training_performed": False,
        },
        **values,
    }


def prepare_round0105(
    *,
    release_sha: str,
    r0104_review_path: str,
    r0104_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0105 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            path,
            expected_sha256=sha,
            required_text=required,
        )
        for round_id, (path, sha, required) in REVIEW_DEFAULTS.items()
    }
    reviews["0104"] = _require_review(
        r0104_review_path,
        expected_sha256=r0104_review_sha256,
        required_text=(
            "capability:jina-full768-host-int8-training-validation-v1",
        ),
    )
    substrate = validate_substrate_manifest(verify_payloads=False)
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    if eligibility["sha256"] != ELIGIBILITY_SHA256:
        raise RuntimeError("R0087 eligibility bytes changed")

    queue_root = create_fresh_directory(
        queue_root, label="R0105 retained-only search queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        substrate["payloads"]["labels"],
        eligibility,
    ])
    index_output = os.path.join(artifacts, "retained-index")
    qualification_output = os.path.join(artifacts, "search-qualification")
    index_path = os.path.join(
        index_output, "jina-diverse-25m-retained.ivfpq"
    )
    index_receipt = os.path.join(index_output, "index-receipt.json")

    jobs = [
        _job(
            node_id="build_retained_search_index",
            action="build_index",
            deps=[],
            output=index_output,
            p90_wall_s=1_800.0,
            inputs=expected_inputs,
        ),
        _job(
            node_id="qualify_retained_search",
            action="qualify_index",
            deps=["build_retained_search_index"],
            output=qualification_output,
            p90_wall_s=3_600.0,
            inputs=expected_inputs,
            index=index_path,
            index_receipt=index_receipt,
        ),
    ]
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=2.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0105-retained-search-qualification-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0087", "0103", "0104"]
    manifest["capability_dependencies"] = [
        "jina-diverse-25m-inventory-v1",
        "jina-diverse-25m-full768-int8-substrate-v1",
        "jina-full768-host-int8-training-validation-v1",
    ]
    manifest["capabilities_produced"] = [
        "jina-diverse-25m-full768-search-qualified-v1"
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "row_count": ROW_COUNT,
        "retained_rows": RETAINED_ROWS,
        "dimension": DIMENSION,
        "index": {
            "class": "GpuIndexIVFPQ with persisted CPU IndexIVFPQ",
            "nlist": NLIST,
            "pq_m": PQ_M,
            "pq_bits": PQ_BITS,
            "metric": "inner product over normalized native int8-plus-scale",
            "physically_filtered": True,
            "global_ids_preserved": True,
        },
        "index_training": {
            "rows": INDEX_TRAIN_ROWS,
            "seed": INDEX_TRAIN_SEED,
            "sample_sha256": INDEX_TRAIN_SAMPLE_SHA256,
        },
        "quality": {
            "k": K,
            "seed": QUALITY_SEED,
            "rows_per_group": QUALITY_ROWS_PER_GROUP,
            "rows": QUALITY_ROWS,
            "groups": list(GROUPS),
            "sample_sha256": QUALITY_SAMPLE_SHA256,
            "group_ids_sha256": QUALITY_GROUP_IDS_SHA256,
            "boundary_tie_atol": BOUNDARY_TIE_ATOL,
            "boundary_ties_excluded_from_denominators": True,
            "global_mean_recall_floor": GLOBAL_MEAN_FLOOR,
            "every_group_mean_recall_floor": EVERY_GROUP_MEAN_FLOOR,
            "policy_grid": [
                {"nprobe": nprobe, "shortlist_width": width}
                for nprobe, width in POLICY_GRID
            ],
        },
        "performance_selector": (
            "lowest median complete search-plus-native-exact-rerank "
            "seconds/query among passing cells; tie by smaller width then nprobe"
        ),
        "performance_benchmark": {
            "same_stratified_queries": QUALITY_ROWS,
            "warmup_rows": BENCHMARK_WARMUP_ROWS,
            "repeats": BENCHMARK_REPEATS,
        },
        "no_graph": True,
        "no_map_training": True,
        "no_map_decision": True,
    }
    manifest["jobs"] = jobs
    manifest["p90_gpu_seconds"] = {
        "build_retained_search_index": 1_800.0,
        "qualify_retained_search": 3_600.0,
        "total": 5_400.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0104-review", required=True)
    parser.add_argument("--r0104-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0105(
        release_sha=args.release_sha,
        r0104_review_path=args.r0104_review,
        r0104_review_sha256=args.r0104_review_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
