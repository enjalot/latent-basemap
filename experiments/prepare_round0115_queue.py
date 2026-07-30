#!/usr/bin/env python3
"""Prepare the minimal R0115 correction queue for the frozen R0113 contrast."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0113_prompt_contrast import (
    ARMS,
    BASELINE_EXCLUDED_ROWS,
    EXCLUDED_ROWS,
    GRAPH_K,
    GRAPH_NPROBE,
    NONINFERIORITY_RATIO,
    POLISH_QUERY_ROWS,
    PROMPT_UNION_EXTRA_EXCLUDED_ROWS,
    PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256,
    QUERY_CANDIDATES,
    QUERY_ROWS,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    read_sealed,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ID = "0115"
ROUND_ROOT = "/data/latent-basemap/runs/round-0115"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0115-*.md")
R0113_ASSEMBLY = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/compact-arrays"
)
R0113_QUERY = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/query-reserve"
)


def _frontmatter_status(path: str) -> str | None:
    with open(path, encoding="utf-8") as handle:
        text = handle.read(4096)
    match = re.search(r"(?m)^status:\s*[\"']?([^\s\"']+)", text)
    return match.group(1) if match else None


def _issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter_status(path) == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0115 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_review(path: str, expected_sha256: str) -> dict[str, Any]:
    if _frontmatter_status(path) not in {"accepted", "partial"}:
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    return signature


def _recovery_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    assembly_path = os.path.join(R0113_ASSEMBLY, "assembly-manifest.json")
    query_path = os.path.join(R0113_QUERY, "query-reserve-receipt.json")
    assembly = read_sealed(assembly_path, label="R0113 compact assembly")
    query = read_sealed(query_path, label="R0113 query reserve")
    polish = query["ood"]["pol_Latn"]
    inputs = [
        expected_input_signature(assembly_path),
        assembly["mapping"],
        assembly["source_text_hash_index"],
        assembly["source_prompt_family_discovery"],
        assembly["retained_duplicate_audit"],
        assembly["substrate"],
        *[assembly["outputs"][arm] for arm in ARMS],
        expected_input_signature(query_path),
        query["query_rows"],
        query["source_text_row_hashes"],
        *[query["outputs"][arm] for arm in ARMS],
        polish["query_rows"],
        polish["source_text_row_hashes"],
        *[polish["outputs"][arm] for arm in ARMS],
    ]
    return _dedupe(inputs), assembly, query


def prepare_round0115(
    *,
    release_sha: str,
    r0113_review: str,
    r0113_review_sha256: str,
    r0114_review: str,
    r0114_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0115 release SHA must be one full commit")
    round_file = _issued_round()
    reviews = {
        "0113": _accepted_review(r0113_review, r0113_review_sha256),
        "0114": _accepted_review(r0114_review, r0114_review_sha256),
    }
    recovery_inputs, assembly, query = _recovery_inputs()
    base_inputs = _dedupe(
        [
            expected_input_signature(round_file),
            *reviews.values(),
            *recovery_inputs,
        ]
    )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        queue_root, label="R0115 corrected paired prompt-map queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    graph_outputs = {
        arm: os.path.join(artifacts, arm, "graph") for arm in ARMS
    }
    selection_output = os.path.join(artifacts, "query-selection")
    train_outputs = {
        arm: os.path.join(artifacts, arm, "train") for arm in ARMS
    }
    score_outputs = {
        arm: os.path.join(artifacts, arm, "evaluation") for arm in ARMS
    }
    decision_output = os.path.join(artifacts, "decision")

    jobs: list[dict[str, Any]] = []
    for arm in ARMS:
        jobs.append(
            {
                "id": f"build_{arm}_graph",
                "action": "build_arm_graph",
                "handler_module": "experiments.round0113_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [graph_outputs[arm]],
                "done_marker": os.path.join(
                    artifacts, f"build_{arm}_graph.done.json"
                ),
                "expected_inputs": base_inputs,
                "p90_wall_s": 600.0,
                "arm": arm,
                "assembly_output": R0113_ASSEMBLY,
                "query_output": R0113_QUERY,
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": False,
                },
            }
        )
    jobs.append(
        {
            "id": "select_matched_clean_queries",
            "action": "select_matched_queries",
            "handler_module": "experiments.round0113_nodes",
            "handler_callable": "run_job",
            "deps": [f"build_{arm}_graph" for arm in ARMS],
            "outputs": [selection_output],
            "done_marker": os.path.join(
                artifacts, "select_matched_clean_queries.done.json"
            ),
            "expected_inputs": base_inputs,
            "p90_wall_s": 120.0,
            "query_output": R0113_QUERY,
            "graph_outputs": graph_outputs,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        }
    )
    for arm in ARMS:
        graph_manifest = os.path.join(
            graph_outputs[arm], "graph-manifest.json"
        )
        jobs.append(
            {
                "id": f"train_{arm}_map",
                "action": "train_arm",
                "handler_module": "experiments.round0113_nodes",
                "handler_callable": "run_job",
                "deps": [
                    f"build_{arm}_graph",
                    "select_matched_clean_queries",
                ],
                "outputs": [train_outputs[arm]],
                "done_marker": os.path.join(
                    artifacts, f"train_{arm}_map.done.json"
                ),
                "expected_inputs": base_inputs,
                "p90_wall_s": 5_400.0,
                "arm": arm,
                "assembly_output": R0113_ASSEMBLY,
                "graph_manifest": graph_manifest,
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": True,
                },
            }
        )
    for arm in ARMS:
        jobs.append(
            {
                "id": f"evaluate_{arm}_map",
                "action": "evaluate_arm",
                "handler_module": "experiments.round0113_nodes",
                "handler_callable": "run_job",
                "deps": [f"train_{arm}_map"],
                "outputs": [score_outputs[arm]],
                "done_marker": os.path.join(
                    artifacts, f"evaluate_{arm}_map.done.json"
                ),
                "expected_inputs": base_inputs,
                "p90_wall_s": 1_200.0,
                "arm": arm,
                "assembly_output": R0113_ASSEMBLY,
                "query_output": R0113_QUERY,
                "query_selection_output": selection_output,
                "graph_manifest": os.path.join(
                    graph_outputs[arm], "graph-manifest.json"
                ),
                "train_output": train_outputs[arm],
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": False,
                },
            }
        )
    jobs.append(
        {
            "id": "decide_prompt_contrast",
            "action": "decide_prompt_contrast",
            "handler_module": "experiments.round0113_nodes",
            "handler_callable": "run_job",
            "deps": [f"evaluate_{arm}_map" for arm in ARMS],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_prompt_contrast.done.json"
            ),
            "expected_inputs": base_inputs,
            "p90_wall_s": 120.0,
            "score_outputs": score_outputs,
            "graph_outputs": graph_outputs,
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
    queue.update(
        {
            "schema": "round0115-corrected-paired-prompt-map-queue-v1",
            "repo_root": RELEASE_ROOT,
            "queue_class": "gpu-research",
            "required_reviews": ["0113", "0114"],
            "capability_dependencies": [
                "jina-full768-host-int8-training-validation-v1",
                "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
            ],
            "capabilities_produced": [
                "jina-fineweb-2m-prompt-map-contrast-v1",
                "jina-fineweb-2m-document-prompt-map-transfer-v1",
            ],
            "training_performed": True,
            "recovery": {
                "supersedes_failed_round": "0113",
                "reuse_only_successful_nodes": [
                    "assemble_compact_prompt_arrays",
                    "embed_dual_prompt_query_reserve",
                ],
                "reuse_assembly_identity": assembly["identity_sha256"],
                "reuse_query_identity": query["identity_sha256"],
                "discard_partial_failed_graph": True,
                "scientific_contract_unchanged": True,
                "only_code_change": (
                    "encode panel-v2 data identity with its strict "
                    "ordered_shards schema and import the existing "
                    "weighted-positive accounting constant"
                ),
            },
            "scientific_contract": {
                "rows_stored_per_arm": 2_000_000,
                "retained_representatives_per_arm": RETAINED_ROWS,
                "duplicate_exclusions": EXCLUDED_ROWS,
                "r0114_baseline_duplicate_exclusions": BASELINE_EXCLUDED_ROWS,
                "prompt_union_extra_exclusions": (
                    PROMPT_UNION_EXTRA_EXCLUDED_ROWS
                ),
                "prompt_union_extra_exclusions_sha256": (
                    PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256
                ),
                "dimension": 768,
                "arms": list(ARMS),
                "graph": {
                    "k": GRAPH_K,
                    "fixed_nprobe": GRAPH_NPROBE,
                    "shared_compact_ids": True,
                    "separate_graph_bytes": True,
                    "identical_builder_parameters_and_seeds": True,
                },
                "training": {
                    "seed": 42,
                    "successful_updates_per_arm": SUCCESSFUL_UPDATES,
                    "same_recipe_and_sampler": True,
                },
                "queries": {
                    "reserve": QUERY_CANDIDATES,
                    "selected": QUERY_ROWS,
                    "polish_ood_queries": POLISH_QUERY_ROWS,
                    "matched_projection_primary": True,
                    "polish_ood_prompt_contrast": "diagnostic-only",
                },
                "document_noninferiority_ratio": NONINFERIORITY_RATIO,
                "projection_ffr": "diagnostic-only",
                "one_seed_screen": True,
                "thresholds_tunable_after_treatment": False,
            },
            "jobs": jobs,
            "p90_gpu_seconds": {
                "build_raw_graph": 600.0,
                "build_document_graph": 600.0,
                "train_raw_map": 5_400.0,
                "train_document_map": 5_400.0,
                "evaluate_raw_map": 1_200.0,
                "evaluate_document_map": 1_200.0,
                "total": 14_400.0,
            },
        }
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0113-review", required=True)
    parser.add_argument("--r0113-review-sha256", required=True)
    parser.add_argument("--r0114-review", required=True)
    parser.add_argument("--r0114-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "queue_manifest": prepare_round0115(
                    release_sha=args.release_sha,
                    r0113_review=args.r0113_review,
                    r0113_review_sha256=args.r0113_review_sha256,
                    r0114_review=args.r0114_review,
                    r0114_review_sha256=args.r0114_review_sha256,
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
