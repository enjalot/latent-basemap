#!/usr/bin/env python3
"""Prepare the paired 2M raw/document Jina embedding queue for R0112."""
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
from basemap.round0112_prompt_substrate import (
    BATCH_SIZE,
    CHUNK_ROWS,
    COMPUTE_DTYPE,
    CONVENTIONS,
    EMBED_MINIMUM_PAIRED_ROWS_PER_S,
    EMBED_WARNING_PAIRED_ROWS_PER_S,
    ELIGIBILITY_PATH,
    ELIGIBILITY_SHA256,
    MODEL_ROOT,
    OUTPUT_DTYPE,
    PROMPT_PREFIX,
    ROUND_ID,
    SLICE_ROWS,
    SUBSTRATE_SCHEMA,
    expected_slice_ranges,
    first2m_layout,
    model_member_signatures,
    source_contract,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0112"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0112-*.md")
REVIEW_DEFAULTS = {
    "0103": (
        "review-0103-2026-07-29.md",
        "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51",
        "capability:jina-diverse-25m-full768-int8-substrate-v1",
    ),
    "0104": (
        "review-0104-2026-07-29.md",
        "febc1033d4edcfdf75e48f77065d8236ef36dde261434d3f1bb557cab48b6cde",
        "capability:jina-full768-host-int8-training-validation-v1",
    ),
}


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
            f"R0112 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    capability: str,
) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError(f"{path} does not release reviewed evidence")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if capability not in text:
        raise RuntimeError(f"{path} does not release {capability}")
    return signature


def _authenticated_layout() -> list[dict[str, Any]]:
    layout = first2m_layout()
    authenticated: list[dict[str, Any]] = []
    for item in layout:
        text = expected_input_signature(str(item["text_path"]))
        embedding = expected_input_signature(
            str(item["embedding"]["canonical_path"])
        )
        if (
            embedding["bytes"] != item["embedding"]["bytes"]
            or embedding["sha256"] != item["embedding"]["sha256"]
        ):
            raise RuntimeError("R0112 historical embedding shard changed")
        authenticated.append(
            {
                **item,
                "text": text,
                "embedding": embedding,
            }
        )
    return authenticated


def prepare_round0112(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0112 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            os.path.join(LAB_ROOT, name),
            expected_sha256=sha256,
            capability=capability,
        )
        for round_id, (name, sha256, capability) in REVIEW_DEFAULTS.items()
    }
    layout = _authenticated_layout()
    model_members = model_member_signatures()
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    if eligibility["sha256"] != ELIGIBILITY_SHA256:
        raise RuntimeError("R0112 duplicate eligibility bytes changed")
    model_inputs = [
        {
            key: value
            for key, value in member.items()
            if key != "model_relative_path"
        }
        for member in model_members
    ]
    base_inputs = _dedupe(
        [
            expected_input_signature(round_file),
            *reviews.values(),
            eligibility,
            *model_inputs,
        ]
    )
    final_inputs = _dedupe(
        [expected_input_signature(round_file), *reviews.values(), eligibility]
    )

    queue_root = create_fresh_directory(
        queue_root, label="R0112 paired prompt embedding queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    jobs: list[dict[str, Any]] = []
    slice_outputs: list[str] = []
    for index, (start, stop) in enumerate(expected_slice_ranges()):
        node_id = f"embed_paired_slice_{index:02d}"
        output = os.path.join(
            artifacts, f"paired-embedding-slice-{start:07d}-{stop:07d}"
        )
        slice_outputs.append(output)
        slice_layout = [
            item
            for item in layout
            if int(item["global_row_stop"]) > start
            and int(item["global_row_start"]) < stop
        ]
        slice_inputs = _dedupe(
            [
                *base_inputs,
                *[item["text"] for item in slice_layout],
                *[item["embedding"] for item in slice_layout],
            ]
        )
        jobs.append(
            {
                "id": node_id,
                "action": "embed_paired_slice",
                "handler_module": "experiments.round0112_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [output],
                "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
                "expected_inputs": slice_inputs,
                "p90_wall_s": 3_900.0,
                "source_row_start": start,
                "source_row_stop": stop,
                "authenticated_source_layout": slice_layout,
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": False,
                },
            }
        )
    final_node = "finalize_dual_prompt_substrate"
    final_output = os.path.join(
        artifacts, "jina-fineweb-2m-dual-prompt-embedding-substrate"
    )
    jobs.append(
        {
            "id": final_node,
            "action": final_node,
            "handler_module": "experiments.round0112_nodes",
            "handler_callable": "run_job",
            "deps": [job["id"] for job in jobs],
            "outputs": [final_output],
            "done_marker": os.path.join(artifacts, f"{final_node}.done.json"),
            "expected_inputs": final_inputs,
            "p90_wall_s": 300.0,
            "slice_receipts": [
                os.path.join(output, "slice-receipt.json")
                for output in slice_outputs
            ],
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        }
    )

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=5.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0112-paired-prompt-embedding-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = ["0103", "0104"]
    queue["capability_dependencies"] = [
        "jina-diverse-25m-full768-int8-substrate-v1",
        "jina-full768-host-int8-training-validation-v1",
    ]
    queue["capabilities_produced"] = [SUBSTRATE_SCHEMA]
    queue["training_performed"] = False
    queue["scientific_contract"] = {
        **source_contract(),
        "model_root": MODEL_ROOT,
        "model_member_count": len(model_members),
        "conventions": list(CONVENTIONS),
        "control_prompt": "",
        "treatment_prompt": PROMPT_PREFIX,
        "compute_dtype": COMPUTE_DTYPE,
        "output_dtype": OUTPUT_DTYPE.str,
        "batch_size": BATCH_SIZE,
        "performance_guard": {
            "evaluate_after_paired_source_rows": 2 * CHUNK_ROWS,
            "minimum_paired_rows_per_s": EMBED_MINIMUM_PAIRED_ROWS_PER_S,
            "warning_paired_rows_per_s": EMBED_WARNING_PAIRED_ROWS_PER_S,
        },
        "slice_rows": SLICE_ROWS,
        "atomic_chunk_rows": CHUNK_ROWS,
        "paired_invariant": (
            "fresh local SentenceTransformer embeddings over identical ordered "
            "texts; the literal Document prefix is the only arm difference"
        ),
        "duplicate_policy": (
            "store all rows for embedding/SAE reuse; derive one shared "
            "cohort-local representative selector from R0087's complete exact "
            "family table so no family is erased when its 25M representative "
            "lies outside the first 2M"
        ),
        "graph_training_or_quality_claim": False,
        "production_or_sae_readiness_claim": False,
        "thresholds_tunable_after_treatment": False,
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {
        **{
            job["id"]: job["p90_wall_s"]
            for job in jobs
            if job["node_policy"]["gpu_required"]
        },
        "total": 4 * 3_900.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "queue_manifest": prepare_round0112(
                    release_sha=args.release_sha,
                    queue_root=args.queue_root,
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
