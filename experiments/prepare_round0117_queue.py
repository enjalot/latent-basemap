#!/usr/bin/env python3
"""Prepare the seed-43 replicate of the accepted R0115 prompt contrast."""
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


ROUND_ID = "0117"
TRAINING_SEED = 43
ROUND_ROOT = "/data/latent-basemap/runs/round-0117"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0117-*.md")
R0113_ASSEMBLY = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/compact-arrays"
)
R0113_QUERY = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/query-reserve"
)
R0115_QUEUE_ROOT = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2"
)
R0115_QUEUE_SHA256 = (
    "91b30c0cdad3dcb0e15f851e017107b79279cb442eafd826bda2edc5d43684c7"
)
R0115_DECISION_SHA256 = (
    "65172c2b445391d3e30799e833479d359196f2593d846369fdb0fc76f6e5b24c"
)
R0115_GRAPH_SHA256 = {
    "raw": "b39a705bf5f426777c33c5941607738a4e0070969f8892234ef42b94b077973c",
    "document": (
        "3c617463f308ae756c3256cea83180af70a5d2492cb7c09617f0ee6881917912"
    ),
}
R0115_QUERY_SELECTION_SHA256 = (
    "7b8eaaa82f2ae1484510f8cb4422d4169c8dc0390387409ea01fc3d13292dbf8"
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
            f"R0117 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_review(path: str, expected_sha256: str) -> dict[str, Any]:
    if _frontmatter_status(path) != "accepted":
        raise RuntimeError(f"{path} is not an accepted review")
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{path} bytes changed")
    return signature


def _require_signature(
    path: str, expected_sha256: str, *, label: str
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise RuntimeError(f"{label} bytes changed")
    return signature


def _accepted_r0115_inputs() -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    str,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    queue_path = os.path.join(R0115_QUEUE_ROOT, "queue.json")
    queue_signature = _require_signature(
        queue_path, R0115_QUEUE_SHA256, label="R0115 queue"
    )
    decision_path = os.path.join(
        R0115_QUEUE_ROOT, "artifacts", "decision", "decision.json"
    )
    decision_signature = _require_signature(
        decision_path, R0115_DECISION_SHA256, label="R0115 decision"
    )
    decision = read_sealed(decision_path, label="R0115 paired decision")
    if (
        decision.get("round_id") != "0115"
        or (decision.get("registered_decision") or {}).get("passed") is not False
    ):
        raise RuntimeError("R0115 accepted decision contract changed")

    graph_outputs = {
        arm: os.path.join(R0115_QUEUE_ROOT, "artifacts", arm, "graph")
        for arm in ARMS
    }
    graphs: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = [queue_signature, decision_signature]
    for arm in ARMS:
        manifest_path = os.path.join(
            graph_outputs[arm], "graph-manifest.json"
        )
        manifest_signature = _require_signature(
            manifest_path,
            R0115_GRAPH_SHA256[arm],
            label=f"R0115 {arm} graph manifest",
        )
        graph = read_sealed(
            manifest_path, label=f"R0115 {arm} graph manifest"
        )
        if (
            graph.get("round_id") != "0115"
            or graph.get("arm") != arm
            or int(graph.get("retained_rows", -1)) != RETAINED_ROWS
            or int(graph.get("k", -1)) != GRAPH_K
            or int(
                (graph.get("search_qualification") or {}).get(
                    "selected_nprobe", -1
                )
            )
            != GRAPH_NPROBE
        ):
            raise RuntimeError(f"R0115 {arm} graph contract changed")
        graphs[arm] = graph
        inputs.extend(
            [
                manifest_signature,
                graph["graph"],
                graph["high_d_reference"],
                graph["topology_probe"],
                graph["query_training_copy_mask"],
                graph["polish_query_training_copy_mask"],
            ]
        )

    selection_output = os.path.join(
        R0115_QUEUE_ROOT, "artifacts", "query-selection"
    )
    selection_path = os.path.join(selection_output, "query-selection.json")
    selection_signature = _require_signature(
        selection_path,
        R0115_QUERY_SELECTION_SHA256,
        label="R0115 query selection",
    )
    selection = read_sealed(selection_path, label="R0115 query selection")
    if (
        selection.get("round_id") != "0115"
        or int(selection.get("selected_rows", -1)) != QUERY_ROWS
        or selection.get("selected_before_training") is not True
        or (selection.get("graphs") or {})
        != {
            arm: expected_input_signature(
                os.path.join(graph_outputs[arm], "graph-manifest.json")
            )
            for arm in ARMS
        }
    ):
        raise RuntimeError("R0115 query-selection contract changed")
    inputs.extend(
        [selection_signature, selection["positions"], selection["global_rows"]]
    )

    assembly_path = os.path.join(R0113_ASSEMBLY, "assembly-manifest.json")
    query_path = os.path.join(R0113_QUERY, "query-reserve-receipt.json")
    assembly_signature = expected_input_signature(assembly_path)
    query_signature = expected_input_signature(query_path)
    assembly = read_sealed(assembly_path, label="R0113 compact assembly")
    query = read_sealed(query_path, label="R0113 query reserve")
    polish = query["ood"]["pol_Latn"]
    if (
        any(
            graphs[arm]["assembly"] != assembly_signature
            or graphs[arm]["query_reserve"] != query_signature
            for arm in ARMS
        )
        or selection["query_reserve"] != query_signature
    ):
        raise RuntimeError("R0115 graph/query lineage changed")
    inputs.extend(
        [
            assembly_signature,
            assembly["mapping"],
            assembly["source_text_hash_index"],
            assembly["source_prompt_family_discovery"],
            assembly["retained_duplicate_audit"],
            assembly["substrate"],
            *[assembly["outputs"][arm] for arm in ARMS],
            query_signature,
            query["query_rows"],
            query["source_text_row_hashes"],
            *[query["outputs"][arm] for arm in ARMS],
            polish["query_rows"],
            polish["source_text_row_hashes"],
            *[polish["outputs"][arm] for arm in ARMS],
        ]
    )
    return (
        _dedupe(inputs),
        graph_outputs,
        selection_output,
        decision_path,
        assembly,
        query,
    )


def prepare_round0117(
    *,
    release_sha: str,
    r0115_review: str,
    r0115_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0117 release SHA must be one full commit")
    round_file = _issued_round()
    review = _accepted_review(r0115_review, r0115_review_sha256)
    (
        reused_inputs,
        graph_outputs,
        query_selection_output,
        prior_decision,
        assembly,
        query,
    ) = _accepted_r0115_inputs()
    base_inputs = _dedupe(
        [
            expected_input_signature(round_file),
            review,
            *reused_inputs,
        ]
    )

    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        queue_root, label="R0117 seed-43 paired prompt-map queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
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
                "id": f"train_{arm}_map_seed43",
                "action": "train_arm",
                "handler_module": "experiments.round0113_nodes",
                "handler_callable": "run_job",
                "deps": [],
                "outputs": [train_outputs[arm]],
                "done_marker": os.path.join(
                    artifacts, f"train_{arm}_map_seed43.done.json"
                ),
                "expected_inputs": base_inputs,
                "p90_wall_s": 5_000.0,
                "arm": arm,
                "training_seed": TRAINING_SEED,
                "graph_execution_round_id": "0115",
                "assembly_output": R0113_ASSEMBLY,
                "graph_manifest": os.path.join(
                    graph_outputs[arm], "graph-manifest.json"
                ),
                "node_policy": {
                    "gpu_required": True,
                    "training_performed": True,
                },
            }
        )
    for arm in ARMS:
        jobs.append(
            {
                "id": f"evaluate_{arm}_map_seed43",
                "action": "evaluate_arm",
                "handler_module": "experiments.round0113_nodes",
                "handler_callable": "run_job",
                "deps": [f"train_{arm}_map_seed43"],
                "outputs": [score_outputs[arm]],
                "done_marker": os.path.join(
                    artifacts, f"evaluate_{arm}_map_seed43.done.json"
                ),
                "expected_inputs": base_inputs,
                "p90_wall_s": 300.0,
                "arm": arm,
                "training_seed": TRAINING_SEED,
                "graph_execution_round_id": "0115",
                "assembly_output": R0113_ASSEMBLY,
                "query_output": R0113_QUERY,
                "query_selection_output": query_selection_output,
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
            "id": "decide_seed43_prompt_contrast",
            "action": "decide_prompt_contrast",
            "handler_module": "experiments.round0113_nodes",
            "handler_callable": "run_job",
            "deps": [f"evaluate_{arm}_map_seed43" for arm in ARMS],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_seed43_prompt_contrast.done.json"
            ),
            "expected_inputs": base_inputs,
            "p90_wall_s": 120.0,
            "training_seed": TRAINING_SEED,
            "score_outputs": score_outputs,
            "graph_outputs": graph_outputs,
            "prior_decision": prior_decision,
            "prior_decision_sha256": R0115_DECISION_SHA256,
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
        gpu_hours_cap=4.0,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update(
        {
            "schema": "round0117-seed43-paired-prompt-map-queue-v1",
            "repo_root": RELEASE_ROOT,
            "queue_class": "gpu-research",
            "required_reviews": ["0115"],
            "capability_dependencies": [
                "jina-fineweb-2m-prompt-map-contrast-v1",
            ],
            "capabilities_produced": [
                "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
            ],
            "training_performed": True,
            "reuse": {
                "accepted_round": "0115",
                "accepted_queue_sha256": R0115_QUEUE_SHA256,
                "accepted_seed42_decision_sha256": R0115_DECISION_SHA256,
                "graph_manifest_sha256": R0115_GRAPH_SHA256,
                "query_selection_sha256": R0115_QUERY_SELECTION_SHA256,
                "assembly_identity": assembly["identity_sha256"],
                "query_reserve_identity": query["identity_sha256"],
                "rebuild_graphs": False,
                "reselect_queries": False,
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
                    "exact_r0115_bytes_reused": True,
                },
                "training": {
                    "seed": TRAINING_SEED,
                    "successful_updates_per_arm": SUCCESSFUL_UPDATES,
                    "same_recipe_and_sampler_as_r0115": True,
                    "negative_row_pairs_identical_across_arms": True,
                },
                "queries": {
                    "reserve": QUERY_CANDIDATES,
                    "selected": QUERY_ROWS,
                    "polish_ood_queries": POLISH_QUERY_ROWS,
                    "exact_r0115_selection_reused": True,
                    "matched_projection_primary": True,
                    "polish_ood_prompt_contrast": "diagnostic-only",
                },
                "document_noninferiority_ratio": NONINFERIORITY_RATIO,
                "projection_ffr": "diagnostic-only",
                "seed42_decision": "failed-noninferiority",
                "cross_seed_interpretation": {
                    "seed43_fails": "confirmed-negative-two-seed",
                    "seed43_passes": "seed-sensitive-mixed",
                },
                "thresholds_tunable_after_treatment": False,
            },
            "cpu_handoff_smoke": {
                "required_before_release": True,
                "maximum_wall_s": 120.0,
                "command": (
                    "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "
                    "PYTHONPATH=. .venv/bin/python -m pytest -q "
                    "-p no:cacheprovider "
                    "tests/test_round0117_seed43_prompt_contrast.py "
                    "tests/test_round0117_cpu_smoke.py"
                ),
                "path": "train -> seal -> checkpoint reload -> panel",
            },
            "jobs": jobs,
            "p90_gpu_seconds": {
                "train_raw_map_seed43": 5_000.0,
                "train_document_map_seed43": 5_000.0,
                "evaluate_raw_map_seed43": 300.0,
                "evaluate_document_map_seed43": 300.0,
                "total": 10_600.0,
            },
        }
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0115-review", required=True)
    parser.add_argument("--r0115-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "queue_manifest": prepare_round0117(
                    release_sha=args.release_sha,
                    r0115_review=args.r0115_review,
                    r0115_review_sha256=args.r0115_review_sha256,
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
