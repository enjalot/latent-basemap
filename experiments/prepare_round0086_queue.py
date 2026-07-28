#!/usr/bin/env python3
"""Prepare, but never launch, reviewed-gated 150M staging and search."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0049_program import (
    DIMENSION,
    INDEX_PATH,
    INDEX_SHA256,
)
from basemap.round0086_program import (
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    RETAINED_ROWS,
    ROUND_ID,
    ROW_COUNT,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)
from experiments.round0059_nodes import FAISS_WHEEL
from experiments.round0081_nodes import (
    BENCHMARK_REPEATS,
    BENCHMARK_ROWS,
    RUNTIME_SPEC,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0086"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0086-2026-07-27.md")
R0025_MANIFEST = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/"
    "int8-shards/int8-shards-v1.json"
)
INT8 = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/"
    "int8-shards/minilm-int8-150m/embeddings.i8"
)
SCALES = (
    "/data/latent-basemap/runs/round-0025/queue/artifacts/"
    "int8-shards/minilm-int8-150m/scales.f16"
)
R0033_RECEIPT = (
    "/data/latent-basemap/runs/round-0033/queue/artifacts/"
    "eligibility/receipt.json"
)
ELIGIBILITY = (
    "/data/latent-basemap/runs/round-0033/queue/artifacts/"
    "eligibility/minilm-150m-row-eligibility-v1.npz"
)
REVIEWS = {
    "0025": os.path.join(LAB_ROOT, "review-0025-2026-07-20.md"),
    "0033": os.path.join(LAB_ROOT, "review-0033-2026-07-22.md"),
    "0049": os.path.join(LAB_ROOT, "review-0049-2026-07-26.md"),
    "0082": os.path.join(LAB_ROOT, "review-0082-2026-07-27.md"),
}
EXPECTED = {
    "r0025_manifest": "38c3847f2811725d571d4861a74864598faa4c76f56caf81a5d3a89cdb4a3f7d",
    "int8": "2171e4bf3c21e7156435b4b4021ca62b2ef8a57d9404b2764e6e968d210b7090",
    "scales": "d282d4f5a5abbe17e981d957fce1cd9e227cbd67aa3262803542d496dbbecb49",
    "r0033_receipt": "595070c1f52f07589f8770f5a0432dd6abdcf84b1adb7950dd4fbd77cdaacc77",
    "eligibility": "cd9738d1cb35b7847923ec24e343583ac91dea4d76381ec28c8c2c8bf6412aca",
    "review_0025": "41cff7ee77ba5333c4ce1dece4ab2282edcae0f3da5c77c98a6581db0beadc14",
    "review_0033": "fa237b7e64f6273e0d2b81c95009e9e5e1805d9965bbececc411075cda671e51",
    "review_0049": "12c94acf168fea0f984b7c5f69c8b1a3a397fc49e9bc96a3935cb3132ea073d9",
    "review_0082": "219b2258a089806c77b13f04522cd73e485305d6b49ea149caf17239d41f5d85",
    "index": INDEX_SHA256,
    "runtime": "fafb150c22c911c19beb6f24351b4fcfd69a934816f96f5ef6989a22cb2f3097",
}
CAPABILITIES = {
    "0025": "capability:minilm-int8-shards-v1",
    "0033": "capability:minilm-150m-row-eligibility-v1",
    "0049": "capability:minilm-balanced-60m-candidate-quality-v1",
    "0082": "capability:minilm-balanced-120m-gpu-ivfpq-search-confirmed-v1",
}


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _require_issued_round() -> None:
    if _frontmatter_status(ROUND_FILE) != "issued":
        raise RuntimeError("R0086 remains draft; refuse queue materialization")


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
        raise RuntimeError(f"{path} does not bind required evidence")
    return signature


def _load_advance(path: str, *, expected_sha256: str) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError("R0080 scale-comparison bytes changed")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    body = {
        key: item for key, item in value.items()
        if key != "identity_sha256"
    }
    if (
        value.get("schema") != "round0080-scale-geometry-comparison-v1"
        or value.get("round_id") != "0080"
        or value.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or value.get("decision", {}).get(
            "120m_supported_as_deliberate_ladder_rung"
        )
        is not True
    ):
        raise RuntimeError("R0080 does not advance the ladder to 150M")
    return signature


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
        "handler_module": "experiments.round0086_nodes",
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
            "training_performed": False,
        },
    }


def prepare_round0086(
    *,
    release_sha: str,
    r0080_review_path: str,
    r0080_review_sha256: str,
    scale_geometry_path: str,
    scale_geometry_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    _require_issued_round()
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0086 release SHA must be one full commit")
    reviews = {
        round_id: _require_review(
            path,
            expected_sha256=EXPECTED[f"review_{round_id}"],
            required_text=(CAPABILITIES[round_id],),
        )
        for round_id, path in REVIEWS.items()
    }
    reviews["0080"] = _require_review(
        r0080_review_path,
        expected_sha256=r0080_review_sha256,
        required_text=(
            "capability:minilm-balanced-90m-120m-scale-geometry-v1",
            scale_geometry_sha256,
        ),
    )
    advance = _load_advance(
        scale_geometry_path,
        expected_sha256=scale_geometry_sha256,
    )
    fixed = {
        "r0025_manifest": expected_input_signature(R0025_MANIFEST),
        "int8": expected_input_signature(INT8),
        "scales": expected_input_signature(SCALES),
        "r0033_receipt": expected_input_signature(R0033_RECEIPT),
        "eligibility": expected_input_signature(ELIGIBILITY),
        "index": expected_input_signature(INDEX_PATH),
        "runtime": expected_input_signature(RUNTIME_SPEC),
    }
    if any(
        fixed[key]["sha256"] != EXPECTED[key]
        for key in fixed
    ):
        raise RuntimeError("R0086 reviewed source bytes changed")
    if (
        fixed["int8"]["bytes"] != ROW_COUNT * DIMENSION
        or fixed["scales"]["bytes"] != ROW_COUNT * 2
    ):
        raise RuntimeError("R0086 150M source geometry changed")

    inputs = _dedupe(_file_inputs([
        ROUND_FILE,
        *REVIEWS.values(),
        r0080_review_path,
        scale_geometry_path,
        R0025_MANIFEST,
        INT8,
        SCALES,
        R0033_RECEIPT,
        ELIGIBILITY,
        INDEX_PATH,
        RUNTIME_SPEC,
        FAISS_WHEEL,
    ]))
    queue_root = create_fresh_directory(
        queue_root,
        label="Round 0086 staging/search queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    substrate_output = os.path.join(artifacts, "balanced-150m-substrate")
    substrate_manifest = os.path.join(
        substrate_output, "balanced-150m-substrate-v1.json"
    )
    filter_output = os.path.join(artifacts, "filtered-index-150m")
    filtered_index = os.path.join(
        filter_output, "balanced-150m-retained.ivfpq"
    )
    filter_receipt = os.path.join(filter_output, "filter-receipt.json")
    qualification_output = os.path.join(
        artifacts, "gpu-ivfpq-policy-qualification-150m"
    )
    jobs = [
        _job(
            node_id="stage_balanced_150m_manifest",
            action="stage",
            deps=[],
            output=substrate_output,
            p90_wall_s=180,
            inputs=inputs,
            gpu=False,
            int8_path=INT8,
            int8_sha256=EXPECTED["int8"],
            scales_path=SCALES,
            scales_sha256=EXPECTED["scales"],
            eligibility_path=ELIGIBILITY,
            eligibility_sha256=EXPECTED["eligibility"],
            r0025_manifest=R0025_MANIFEST,
            r0025_manifest_sha256=EXPECTED["r0025_manifest"],
            r0033_receipt=R0033_RECEIPT,
            r0033_receipt_sha256=EXPECTED["r0033_receipt"],
        ),
        _job(
            node_id="filter_150m_representative_index",
            action="filter",
            deps=["stage_balanced_150m_manifest"],
            output=filter_output,
            p90_wall_s=300,
            inputs=inputs,
            gpu=False,
            substrate_manifest=substrate_manifest,
        ),
        _job(
            node_id="qualify_balanced_150m_gpu_ivfpq_policy",
            action="qualify",
            deps=["filter_150m_representative_index"],
            output=qualification_output,
            p90_wall_s=1_800,
            inputs=inputs,
            gpu=True,
            substrate_manifest=substrate_manifest,
            filtered_index=filtered_index,
            filter_receipt=filter_receipt,
            runtime_spec=RUNTIME_SPEC,
            runtime_spec_sha256=EXPECTED["runtime"],
        ),
    ]
    manifest = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["schema"] = "round0086-balanced-150m-staging-search-queue-v1"
    manifest["repo_root"] = RELEASE_ROOT
    manifest["queue_class"] = "gpu-research"
    manifest["required_reviews"] = ["0025", "0033", "0049", "0080", "0082"]
    manifest["capability_dependencies"] = [
        "minilm-int8-shards-v1",
        "minilm-150m-row-eligibility-v1",
        "minilm-balanced-60m-candidate-quality-v1",
        "minilm-balanced-90m-120m-scale-geometry-v1",
        "minilm-balanced-120m-gpu-ivfpq-search-confirmed-v1",
    ]
    manifest["capabilities_produced"] = [
        "minilm-balanced-150m-int8-input-v1",
        "minilm-balanced-150m-gpu-ivfpq-search-qualified-v1",
    ]
    manifest["training_performed"] = False
    manifest["reviewed_inputs"] = {
        **{f"review_{key}": value for key, value in reviews.items()},
        "r0080_scale_geometry": advance,
        **fixed,
    }
    manifest["scientific_contract"] = {
        "rows": ROW_COUNT,
        "retained_rows": RETAINED_ROWS,
        "substrate_is_reference_only_no_payload_copy": True,
        "eligibility_is_exact_reviewed_r0033_full_universe": True,
        "policy_grid": [
            {"nprobe": nprobe, "shortlist_width": width}
            for nprobe, width in POLICY_GRID
        ],
        "quality_selector": (
            "mean unambiguous exact-reranked recall@15 at least 0.90"
        ),
        "performance_selector": (
            "lowest median three-repeat 10000-query search-plus-rerank "
            "wall among passing cells"
        ),
        "sample_rows": 4_096,
        "sample_seed": 86,
        "benchmark_rows": BENCHMARK_ROWS,
        "benchmark_repeats": BENCHMARK_REPEATS,
        "r0083_does_not_change_floor_in_place": True,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "no_graph": True,
        "no_training": True,
        "no_scale_decision": True,
    }
    manifest["jobs"] = jobs
    manifest["p90_gpu_seconds"] = {
        "qualify_balanced_150m_gpu_ivfpq_policy": 1_800.0,
        "total": 1_800.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0080-review", required=True)
    parser.add_argument("--r0080-review-sha256", required=True)
    parser.add_argument("--scale-geometry", required=True)
    parser.add_argument("--scale-geometry-sha256", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(prepare_round0086(
        release_sha=args.release_sha,
        r0080_review_path=args.r0080_review,
        r0080_review_sha256=args.r0080_review_sha256,
        scale_geometry_path=args.scale_geometry,
        scale_geometry_sha256=args.scale_geometry_sha256,
        queue_root=args.queue_root,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
