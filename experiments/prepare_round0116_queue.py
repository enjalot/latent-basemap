#!/usr/bin/env python3
"""Prepare, but never launch, canonical prompted-Jina production for R0116."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
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
    first2m_layout,
    model_member_signatures,
)
from basemap.round0116_prompted_corpus import (
    CAPABILITY,
    CHUNK_ROWS,
    CORPUS_ROWS,
    DATASET_GLOBAL_OFFSETS,
    DATASET_ROWS,
    EMBED_MINIMUM_ROWS_PER_S,
    EMBED_WARNING_ROWS_PER_S,
    INVENTORY_MANIFEST_PATH,
    NEW_ROWS,
    PERFORMANCE_GUARD_ROWS,
    PROMPT_PREFIX,
    R0114_MANIFEST_PATH,
    ROUND_ID,
    WORK_RANGES,
    canonical_source_layout,
    clip_layout,
    environment_freeze_receipt,
    load_reused_manifest,
    model_contract,
    production_payload_bytes,
    required_free_bytes,
    source_manifest_summary,
    validate_reused_mapping,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0116"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0116-*.md")
OUTPUT_NAMESPACE = (
    "canonical-jina-document-native8192-english-v1"
)
REVIEW_DEFAULTS = {
    "0087": (
        "review-0087-2026-07-28.md",
        "61ab9268899c2edc47519bdbe4efeea65a54f0c9fda52bd89e7cad0dafd9d483",
        "capability:jina-diverse-25m-inventory-v1",
    ),
    "0114": (
        "review-0114-2026-07-30.md",
        "610a9abb93f3fb6908a018d855f81feecc1045e261c007a3ca13ad8379eec4b9",
        "capability:jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    ),
    # R0115 is program-state evidence, not a capability dependency.  Binding
    # it makes the stale "unmaterialized fallback" premise impossible to
    # revive and does not supersede its accepted prompt-map contrast.
    "0115": (
        "review-0115-2026-07-30.md",
        "cbc6ad74773624a0fd8ea966f5a1e9cd37be120b554a0ca56c28011720d3bb02",
        "capability:jina-fineweb-2m-prompt-map-contrast-v1",
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
            f"R0116 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_review(
    path: str,
    *,
    expected_sha256: str,
    capability: str,
) -> dict[str, Any]:
    if _frontmatter_status(path) != "accepted":
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if capability not in text:
        raise RuntimeError(f"{path} does not release {capability}")
    return signature


def _node_p90(rows: int) -> float:
    # The accepted R0112 receipts measured 252.90--254.15 convention rows/s.
    # Budget at 200 rows/s plus five minutes of model/source/receipt overhead.
    return float(rows / 200.0 + 300.0)


def prepare_round0116(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0116 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        round_id: _require_review(
            os.path.join(LAB_ROOT, name),
            expected_sha256=sha256,
            capability=capability,
        )
        for round_id, (name, sha256, capability) in REVIEW_DEFAULTS.items()
    }
    layout, inventory_signature = canonical_source_layout()
    reused, reused_signature = load_reused_manifest()
    r0114_source_lineage = first2m_layout()
    reused_mapping = validate_reused_mapping(
        layout,
        reused,
        r0114_source_lineage=r0114_source_lineage,
    )
    model_members = model_member_signatures()
    environment_freeze = environment_freeze_receipt()
    model_inputs = [
        {
            key: value
            for key, value in member.items()
            if key != "model_relative_path"
        }
        for member in model_members
    ]

    disk = shutil.disk_usage("/data")
    required = required_free_bytes()
    if disk.free < required:
        raise RuntimeError(
            "R0116 disk preflight failed: "
            f"{disk.free:,} free bytes < {required:,} required"
        )

    queue_root = create_fresh_directory(
        queue_root,
        label="R0116 canonical prompted-Jina production queue",
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    namespace = ensure_data_directory(
        os.path.join(artifacts, OUTPUT_NAMESPACE),
        label="R0116 prompted output namespace",
    )
    jobs: list[dict[str, Any]] = []
    receipt_paths: list[str] = []
    round_and_reviews = _dedupe(
        [
            expected_input_signature(round_file),
            *reviews.values(),
            inventory_signature,
        ]
    )
    for node_id, dataset, start, stop in WORK_RANGES:
        node_layout = clip_layout(
            layout,
            dataset=dataset,
            start=start,
            stop=stop,
        )
        output = os.path.join(namespace, node_id)
        receipt_path = os.path.join(output, "node-receipt.json")
        receipt_paths.append(receipt_path)
        inputs = _dedupe(
            [
                *round_and_reviews,
                *model_inputs,
                *[dict(item["text"]) for item in node_layout],
            ]
        )
        jobs.append(
            {
                "id": node_id,
                "action": "embed_document_rows",
                "handler_module": "experiments.round0116_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [output],
                "done_marker": os.path.join(
                    artifacts, f"{node_id}.done.json"
                ),
                "expected_inputs": inputs,
                "p90_wall_s": _node_p90(stop - start),
                "dataset": dataset,
                "dataset_row_start": start,
                "dataset_row_stop": stop,
                "corpus_global_row_start": (
                    DATASET_GLOBAL_OFFSETS[dataset] + start
                ),
                "corpus_global_row_stop": (
                    DATASET_GLOBAL_OFFSETS[dataset] + stop
                ),
                "authenticated_source_layout": node_layout,
                "environment_freeze": environment_freeze,
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": False,
                },
            }
        )

    reused_chunks = list(
        reused["conventions"]["document"]["chunks"]
    )
    final_id = "finalize_canonical_prompted_english"
    final_output = os.path.join(
        artifacts, "jina-document-english-fineweb-rpj-5p727m-v1"
    )
    final_inputs = _dedupe(
        [
            *round_and_reviews,
            reused_signature,
            *[dict(item) for item in reused_chunks],
        ]
    )
    jobs.append(
        {
            "id": final_id,
            "action": "finalize_canonical_prompted_english",
            "handler_module": "experiments.round0116_nodes",
            "handler_callable": "run_job",
            "deps": [item[0] for item in WORK_RANGES],
            "outputs": [final_output],
            "done_marker": os.path.join(
                artifacts, f"{final_id}.done.json"
            ),
            "expected_inputs": final_inputs,
            "p90_wall_s": 600.0,
            "node_receipts": receipt_paths,
            "reused_manifest": R0114_MANIFEST_PATH,
            "canonical_source_layout": layout,
            "r0114_source_lineage": r0114_source_lineage,
            "reused_prefix_mapping": reused_mapping,
            "environment_freeze": environment_freeze,
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
        gpu_hours_cap=6.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0116-canonical-prompted-corpus-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-data-production"
    queue["required_reviews"] = list(REVIEW_DEFAULTS)
    queue["capability_dependencies"] = [
        "jina-diverse-25m-inventory-v1",
        "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    ]
    queue["program_state_reviews"] = {
        "0115": (
            "accepted prompt-map contrast is preserved and is not superseded"
        )
    }
    queue["capabilities_produced"] = [CAPABILITY]
    queue["training_performed"] = False
    queue["output_namespace"] = namespace
    queue["scientific_contract"] = {
        "purpose": "canonical prompted embedding data production only",
        "datasets": {
            dataset: {
                "rows": rows,
                "corpus_global_offset": DATASET_GLOBAL_OFFSETS[dataset],
            }
            for dataset, rows in DATASET_ROWS.items()
        },
        "corpus_rows": CORPUS_ROWS,
        "reused_reviewed_fineweb_rows": 2_000_000,
        "new_gpu_rows": NEW_ROWS,
        "model": model_contract(),
        "atomic_chunk_rows_maximum": CHUNK_ROWS,
        "work_ranges": [
            {
                "node_id": node_id,
                "dataset": dataset,
                "dataset_row_range": [start, stop],
            }
            for node_id, dataset, start, stop in WORK_RANGES
        ],
        "source_manifest": source_manifest_summary(layout),
        "inventory_manifest": inventory_signature,
        "reused_manifest": reused_signature,
        "reused_prefix_mapping": reused_mapping,
        "performance_calibration": {
            "source": "accepted R0112/R0114 receipts",
            "r0112_measured_document_convention_rows_per_s_range": [
                252.89684892474966,
                254.15208030721982,
            ],
            "expected_document_rows_per_s": 253.0,
            "p90_budget_document_rows_per_s": 200.0,
            "minimum_document_rows_per_s": EMBED_MINIMUM_ROWS_PER_S,
            "warning_document_rows_per_s": EMBED_WARNING_ROWS_PER_S,
            "guard_evaluated_after_rows": PERFORMANCE_GUARD_ROWS,
        },
        "disk_preflight": {
            "filesystem": "/data",
            "free_bytes_observed": disk.free,
            "new_payload_bytes": production_payload_bytes(),
            "required_free_bytes": required,
            "reused_prefix_copied": False,
            "passed": True,
        },
        "environment_freeze": environment_freeze,
        "prompt_prefix": PROMPT_PREFIX,
        "source_order": [FINEWEB, REDPAJAMA],
        "no_graph": True,
        "no_map": True,
        "no_training": True,
        "no_quality_claim": True,
        "no_production_promotion": True,
        "no_complete_sae_corpus_claim": True,
    }
    queue["jobs"] = jobs
    gpu_p90 = {
        job["id"]: float(job["p90_wall_s"])
        for job in jobs
        if job["node_policy"]["gpu_required"]
    }
    queue["p90_gpu_seconds"] = {
        **gpu_p90,
        "total": sum(gpu_p90.values()),
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root",
        default=os.path.join(ROUND_ROOT, "queue"),
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "queue_manifest": prepare_round0116(
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
