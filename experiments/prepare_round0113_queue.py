#!/usr/bin/env python3
"""Prepare the paired raw/document map contrast queue for R0113."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

import numpy as np

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
from basemap.round0113_prompt_contrast import (
    ARMS,
    BASELINE_EXCLUDED_ROWS,
    GRAPH_K,
    GRAPH_NPROBE,
    NONINFERIORITY_RATIO,
    POLISH_HISTORICAL_EMBEDDING_PATH,
    POLISH_HISTORICAL_EMBEDDING_SHA256,
    POLISH_HISTORICAL_MANIFEST_PATH,
    POLISH_QUERY_ROWS,
    POLISH_SOURCE_ROWS,
    POLISH_TEXT_PATH,
    EXCLUDED_ROWS,
    PROMPT_UNION_EXTRA_EXCLUDED_ROWS,
    PROMPT_UNION_EXTRA_EXCLUSIONS_SHA256,
    QUERY_CANDIDATES,
    QUERY_ROWS,
    RETAINED_ROWS,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    load_substrate_manifest,
    query_candidate_rows,
    query_source_layout,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0113"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0113-*.md")
R0104_REVIEW = os.path.join(LAB_ROOT, "review-0104-2026-07-29.md")
R0104_REVIEW_SHA256 = (
    "febc1033d4edcfdf75e48f77065d8236ef36dde261434d3f1bb557cab48b6cde"
)


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
            f"R0113 requires exactly one issued round; found {len(candidates)}"
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


def _query_source_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, _report = query_candidate_rows()
    layout = query_source_layout(rows)
    inputs: list[dict[str, Any]] = []
    for item in layout:
        inputs.extend((dict(item["embedding"]), dict(item["text"])))
    inputs.extend(
        [
            {
                key: value
                for key, value in member.items()
                if key != "model_relative_path"
            }
            for member in model_member_signatures()
        ]
    )
    return _dedupe(inputs), layout


def _source_text_layout() -> list[dict[str, Any]]:
    authenticated: list[dict[str, Any]] = []
    for item in first2m_layout():
        text = expected_input_signature(str(item["text_path"]))
        authenticated.append(
            {
                key: value
                for key, value in item.items()
                if key != "text_path"
            }
            | {"text": text}
        )
    return authenticated


def _polish_source() -> dict[str, dict[str, Any]]:
    source = {
        "historical_embedding": expected_input_signature(
            POLISH_HISTORICAL_EMBEDDING_PATH
        ),
        "manifest": expected_input_signature(POLISH_HISTORICAL_MANIFEST_PATH),
        "text": expected_input_signature(POLISH_TEXT_PATH),
    }
    historical = source["historical_embedding"]
    if historical["sha256"] != POLISH_HISTORICAL_EMBEDDING_SHA256:
        raise RuntimeError("R0113 Polish historical embedding bytes changed")
    values = np.load(
        POLISH_HISTORICAL_EMBEDDING_PATH, mmap_mode="r", allow_pickle=False
    )
    if (
        values.shape != (POLISH_SOURCE_ROWS, 768)
        or values.dtype != np.float16
    ):
        raise RuntimeError("R0113 Polish historical embedding geometry changed")
    import pyarrow.parquet as pq

    if int(pq.ParquetFile(POLISH_TEXT_PATH).metadata.num_rows) != POLISH_SOURCE_ROWS:
        raise RuntimeError("R0113 Polish source text row count changed")
    return source


def prepare_round0113(
    *,
    release_sha: str,
    r0114_review_path: str,
    r0114_review_sha256: str,
    substrate_manifest: str,
    substrate_manifest_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0113 release SHA must be one full commit")
    round_file = _require_issued_round()
    reviews = {
        "0104": _require_review(
            R0104_REVIEW,
            expected_sha256=R0104_REVIEW_SHA256,
            capability="capability:jina-full768-host-int8-training-validation-v1",
        ),
        "0114": _require_review(
            r0114_review_path,
            expected_sha256=r0114_review_sha256,
            capability=(
                "capability:"
                "jina-fineweb-2m-dual-prompt-native8192-substrate-v2"
            ),
        ),
    }
    substrate = load_substrate_manifest(
        substrate_manifest,
        expected_sha256=substrate_manifest_sha256,
        verify_chunks=False,
    )
    source_text_layout = _source_text_layout()
    base_inputs = _dedupe(
        [
            expected_input_signature(round_file),
            *reviews.values(),
            substrate["signature"],
            substrate["selector"],
        ]
    )
    assembly_inputs = _dedupe(
        [
            *base_inputs,
            *substrate["chunks"]["raw"],
            *substrate["chunks"]["document"],
            *[item["text"] for item in source_text_layout],
        ]
    )
    query_source_inputs, query_layout = _query_source_inputs()
    polish_source = _polish_source()
    query_inputs = _dedupe(
        [*base_inputs, *query_source_inputs, *polish_source.values()]
    )

    queue_root = create_fresh_directory(
        queue_root, label="R0113 paired prompt-map queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    query_output = os.path.join(artifacts, "query-reserve")
    assembly_output = os.path.join(artifacts, "compact-arrays")
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
    jobs: list[dict[str, Any]] = [
        {
            "id": "embed_dual_prompt_query_reserve",
            "action": "embed_query_reserve",
            "handler_module": "experiments.round0113_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [query_output],
            "done_marker": os.path.join(
                artifacts, "embed_dual_prompt_query_reserve.done.json"
            ),
            "expected_inputs": query_inputs,
            "authenticated_query_layout": query_layout,
            "p90_wall_s": 600.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "polish_source": polish_source,
        },
        {
            "id": "assemble_compact_prompt_arrays",
            "action": "assemble_compact_arrays",
            "handler_module": "experiments.round0113_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [assembly_output],
            "done_marker": os.path.join(
                artifacts, "assemble_compact_prompt_arrays.done.json"
            ),
            "expected_inputs": assembly_inputs,
            "p90_wall_s": 900.0,
            "substrate_manifest": substrate_manifest,
            "substrate_manifest_sha256": substrate_manifest_sha256,
            "source_text_layout": source_text_layout,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        },
    ]
    # Prove the raw-derived selector also removes any prompt-induced exact
    # collisions before the first GPU-required node.
    jobs.reverse()
    for arm in ARMS:
        jobs.append(
            {
                "id": f"build_{arm}_graph",
                "action": "build_arm_graph",
                "handler_module": "experiments.round0113_nodes",
                "handler_callable": "run_job",
                "deps": [
                    "embed_dual_prompt_query_reserve",
                    "assemble_compact_prompt_arrays",
                ],
                "outputs": [graph_outputs[arm]],
                "done_marker": os.path.join(
                    artifacts, f"build_{arm}_graph.done.json"
                ),
                "expected_inputs": base_inputs,
                "p90_wall_s": 600.0,
                "arm": arm,
                "assembly_output": assembly_output,
                "query_output": query_output,
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
            "query_output": query_output,
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
                "assembly_output": assembly_output,
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
                "assembly_output": assembly_output,
                "query_output": query_output,
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
    queue["schema"] = "round0113-paired-prompt-map-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = ["0104", "0114"]
    queue["capability_dependencies"] = [
        "jina-full768-host-int8-training-validation-v1",
        "jina-fineweb-2m-dual-prompt-native8192-substrate-v2",
    ]
    queue["capabilities_produced"] = [
        "jina-fineweb-2m-prompt-map-contrast-v1",
        "jina-fineweb-2m-document-prompt-map-transfer-v1",
    ]
    queue["training_performed"] = True
    queue["scientific_contract"] = {
        "rows_stored_per_arm": 2_000_000,
        "retained_representatives_per_arm": RETAINED_ROWS,
        "duplicate_exclusions": EXCLUDED_ROWS,
        "r0114_baseline_duplicate_exclusions": BASELINE_EXCLUDED_ROWS,
        "prompt_union_extra_exclusions": PROMPT_UNION_EXTRA_EXCLUDED_ROWS,
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
            "clean_in_both_arms_before_training": True,
            "training_disjointness": [
                "complete source-text UTF-8 bytes",
                "complete stored embedding-row bytes",
            ],
            "within_panel_uniqueness": [
                "complete source-text UTF-8 bytes",
                "complete stored embedding-row bytes in both arms",
            ],
            "matched_projection_primary": True,
            "cross_convention_projection_diagnostic": True,
            "polish_ood_queries": POLISH_QUERY_ROWS,
            "polish_ood_prompt_contrast": "diagnostic-only",
        },
        "decision_metrics": [
            "ffr",
            "density",
            "recall_at_10",
            "oos_recall_at_10",
            "oos_recall_at_50",
        ],
        "document_noninferiority_ratio": NONINFERIORITY_RATIO,
        "projection_ffr": "diagnostic-only",
        "one_seed_screen": True,
        "production_or_complete_sae_claim": False,
        "thresholds_tunable_after_treatment": False,
    }
    queue["jobs"] = jobs
    queue["p90_gpu_seconds"] = {
        "embed_dual_prompt_query_reserve": 600.0,
        "build_raw_graph": 600.0,
        "build_document_graph": 600.0,
        "train_raw_map": 5_400.0,
        "train_document_map": 5_400.0,
        "evaluate_raw_map": 1_200.0,
        "evaluate_document_map": 1_200.0,
        "total": 15_000.0,
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0114-review", required=True)
    parser.add_argument("--r0114-review-sha256", required=True)
    parser.add_argument("--substrate-manifest", required=True)
    parser.add_argument("--substrate-manifest-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "queue_manifest": prepare_round0113(
                    release_sha=args.release_sha,
                    r0114_review_path=args.r0114_review,
                    r0114_review_sha256=args.r0114_review_sha256,
                    substrate_manifest=args.substrate_manifest,
                    substrate_manifest_sha256=args.substrate_manifest_sha256,
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
