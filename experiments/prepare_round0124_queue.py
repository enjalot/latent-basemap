#!/usr/bin/env python3
"""Prepare the independent R0124 native 2M Jina degree-bridge queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0113_prompt_contrast import (
    QUERY_ROWS,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    read_sealed,
)
from basemap.round0124_degree_bridge import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    GRAPH_DEGREE,
    GRAPH_SEARCH_NEIGHBORS,
    MATERIAL_DENSITY_DEGRADATION,
    NATIVE_ANCHOR_SEED,
    NATIVE_DENSITY_ANCHORS,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.prepare_round0119_queue import (
    _clean_terminal,
    _document,
    _frontmatter_list,
)
ROUND_ID = "0124"
ROUND_ROOT = "/data/latent-basemap/runs/round-0124"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0124-*.md")

R0113_ASSEMBLY = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/compact-arrays"
)
R0113_QUERY = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/query-reserve"
)
R0115_ROOT = "/data/latent-basemap/runs/round-0115/queue-attempt-2"
R0115_CONTROL_GRAPH = os.path.join(
    R0115_ROOT, "artifacts/raw/graph/graph-manifest.json"
)
R0115_CONTROL_TRAIN = os.path.join(
    R0115_ROOT, "artifacts/raw/train/train-receipt.json"
)
R0115_CONTROL_SCORE = os.path.join(
    R0115_ROOT, "artifacts/raw/evaluation/score.json"
)
R0115_QUERY_SELECTION = os.path.join(
    R0115_ROOT, "artifacts/query-selection"
)
R0115_QUEUE = os.path.join(R0115_ROOT, "queue.json")
R0115_TERMINAL = os.path.join(R0115_ROOT, "runner-terminal.json")

R0106_GRAPH_CONTEXT = (
    "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts/"
    "canonical-fuzzy-graph/graph-manifest.json"
)
R0108_CORE_CONTEXT = (
    "/data/latent-basemap/runs/round-0108/queue-attempt-3/artifacts/"
    "core-geometry/core-geometry.json"
)

REQUIRED_CAPABILITIES = {
    "0106": "jina-diverse-25m-full768-fuzzy-graph-v1",
    "0108": "jina-diverse-25m-map-registry-v1",
    "0115": "jina-fineweb-2m-prompt-map-contrast-v1",
}


def _issued_round() -> str:
    candidates = []
    for path in sorted(glob.glob(ROUND_FILE_GLOB)):
        frontmatter, _text = _document(path)
        if frontmatter.get("status") == "issued":
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0124 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_review(
    path: str,
    expected_sha256: str,
    *,
    round_id: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    frontmatter, _text = _document(path)
    release = f"capability:{REQUIRED_CAPABILITIES[round_id]}"
    if (
        signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
        or release
        not in _frontmatter_list(
            frontmatter,
            "releases",
            label=f"R{round_id} review",
        )
    ):
        raise RuntimeError(f"R{round_id} review is not exact and accepted")
    result_name = frontmatter.get("result") or ""
    if (
        os.path.basename(result_name) != result_name
        or not re.fullmatch(
            rf"result-{round_id}-[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.md",
            result_name,
        )
    ):
        raise RuntimeError(f"R{round_id} review result binding is malformed")
    result_path = os.path.join(os.path.dirname(path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter, _result_text = _document(result_path)
    capability = REQUIRED_CAPABILITIES[round_id]
    release_commit = frontmatter.get("verified_release_commit")
    if (
        result_signature["sha256"] != frontmatter.get("result_sha256")
        or result_frontmatter.get("round_id") != round_id
        or result_frontmatter.get("status") != "complete"
        or result_frontmatter.get("release_commit")
        != release_commit
        or not re.fullmatch(r"[0-9a-f]{40}", release_commit or "")
        or capability
        not in _frontmatter_list(
            result_frontmatter,
            "capabilities_produced",
            label=f"R{round_id} result",
        )
    ):
        raise RuntimeError(f"R{round_id} review does not close its result")
    return {
        "review": signature,
        "result": result_signature,
        "release_commit": release_commit,
    }


def _read_sealed_signature(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    value = read_sealed(path, label=label)
    return value, signature


def _inputs(
    *,
    reviews: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assembly_path = os.path.join(R0113_ASSEMBLY, "assembly-manifest.json")
    query_path = os.path.join(R0113_QUERY, "query-reserve-receipt.json")
    selection_path = os.path.join(
        R0115_QUERY_SELECTION, "query-selection.json"
    )
    assembly, assembly_signature = _read_sealed_signature(
        assembly_path, label="R0113 compact assembly"
    )
    query, query_signature = _read_sealed_signature(
        query_path, label="R0113 query reserve"
    )
    selection, selection_signature = _read_sealed_signature(
        selection_path, label="R0115 query selection"
    )
    control, control_signature = _read_sealed_signature(
        R0115_CONTROL_GRAPH, label="R0115 raw graph control"
    )
    control_train, control_train_signature = _read_sealed_signature(
        R0115_CONTROL_TRAIN, label="R0115 raw train control"
    )
    control_score, control_score_signature = _read_sealed_signature(
        R0115_CONTROL_SCORE, label="R0115 raw native score"
    )
    graph_context, graph_context_signature = _read_sealed_signature(
        R0106_GRAPH_CONTEXT, label="R0106 25M k15 graph context"
    )
    core_context, core_context_signature = _read_sealed_signature(
        R0108_CORE_CONTEXT, label="R0108 25M native-density context"
    )
    terminal = _clean_terminal(
        R0115_QUEUE,
        R0115_TERMINAL,
        round_id="0115",
        expected_release_sha=str(reviews["0115"]["release_commit"]),
    )
    control_coordinates = (
        (control_score.get("coordinates") or {}).get("training") or {}
    )
    control_panel = control_score.get("panel") or {}
    core_density = ((core_context.get("metrics") or {}).get("density_v2") or {})
    if (
        assembly.get("retained_rows") != RETAINED_ROWS
        or selection.get("selected_rows") != QUERY_ROWS
        or selection.get("selected_before_training") is not True
        or control.get("round_id") != "0115"
        or control.get("release_sha") != reviews["0115"]["release_commit"]
        or control.get("arm") != "raw"
        or control.get("k") != 50
        or control_train.get("round_id") != "0115"
        or control_train.get("release_sha")
        != reviews["0115"]["release_commit"]
        or control_train.get("arm") != "raw"
        or control_train.get("graph_manifest") != control_signature
        or control_score.get("schema")
        != "round0113-prompt-arm-score-v1"
        or control_score.get("round_id") != "0115"
        or control_score.get("release_sha")
        != reviews["0115"]["release_commit"]
        or control_score.get("arm") != "raw"
        or control_score.get("graph_manifest") != control_signature
        or control_score.get("train_receipt") != control_train_signature
        or control_score.get("high_d_reference")
        != control.get("high_d_reference")
        or not isinstance(control_coordinates, Mapping)
        or control_coordinates.get("sha256")
        != "ac79a548bbd237937e9c561c169b88f38852b6591cc5a84e8946047dce7f07f2"
        or control_panel.get("n") != RETAINED_ROWS
        or control_panel.get("n_anchors") != NATIVE_DENSITY_ANCHORS
        or control_panel.get("anchor_seed") != NATIVE_ANCHOR_SEED
        or control_panel.get("k_density") != GRAPH_DEGREE
        or control_panel.get("density") != 0.2304
        or graph_context.get("schema")
        != "round0106-jina-diverse-25m-fuzzy-graph-v1"
        or graph_context.get("round_id") != "0106"
        or graph_context.get("release_sha")
        != reviews["0106"]["release_commit"]
        or graph_context.get("k_real") != GRAPH_DEGREE
        or graph_context.get("n_neighbors_including_self")
        != GRAPH_SEARCH_NEIGHBORS
        or core_context.get("schema")
        != "round0108-diverse-jina-core-geometry-v1"
        or core_context.get("round_id") != "0108"
        or core_context.get("graph_manifest") != graph_context_signature
        or core_density.get("correlation") != 0.15773929111469354
    ):
        raise RuntimeError("R0124 frozen predecessor evidence changed")
    polish = query["ood"]["pol_Latn"]
    return _dedupe(
        [
            assembly_signature,
            assembly["mapping"],
            assembly["outputs"]["raw"],
            assembly["outputs"]["document"],
            assembly["source_text_hash_index"],
            assembly["source_prompt_family_discovery"],
            assembly["retained_duplicate_audit"],
            query_signature,
            query["query_rows"],
            query["source_text_row_hashes"],
            query["outputs"]["raw"],
            query["outputs"]["document"],
            polish["outputs"]["raw"],
            polish["outputs"]["document"],
            polish["query_rows"],
            polish["source_text_row_hashes"],
            selection_signature,
            selection["positions"],
            selection["global_rows"],
            control_signature,
            control_train_signature,
            control_train["production_config"],
            control_train["model"],
            control_score_signature,
            control_coordinates,
            control["topology_probe"],
            control["high_d_reference"],
            control["query_training_copy_mask"],
            control["polish_query_training_copy_mask"],
            graph_context_signature,
            core_context_signature,
            *terminal.values(),
        ]
    )


def prepare_round0124(
    *,
    release_sha: str,
    r0106_review: str,
    r0106_review_sha256: str,
    r0108_review: str,
    r0108_review_sha256: str,
    r0115_review: str,
    r0115_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0124 release SHA must be one full commit")
    round_file = _issued_round()
    reviews = {
        "0106": _accepted_review(
            r0106_review, r0106_review_sha256, round_id="0106"
        ),
        "0108": _accepted_review(
            r0108_review, r0108_review_sha256, round_id="0108"
        ),
        "0115": _accepted_review(
            r0115_review, r0115_review_sha256, round_id="0115"
        ),
    }
    inputs = _dedupe(
        [
            expected_input_signature(round_file),
            *(
                signature
                for evidence in reviews.values()
                for signature in (evidence["review"], evidence["result"])
            ),
            *_inputs(reviews=reviews),
        ]
    )
    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        queue_root, label="R0124 graph-degree bridge queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    graph_output = os.path.join(artifacts, "k15-graph")
    train_output = os.path.join(artifacts, "k15-train")
    diagnostic_output = os.path.join(artifacts, "core-ood-diagnostics")
    density_output = os.path.join(artifacts, "native-density-contrast")
    decision_output = os.path.join(artifacts, "degree-bridge-decision")
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    common = {
        "expected_inputs": inputs,
        "assembly_output": R0113_ASSEMBLY,
        "query_output": R0113_QUERY,
        "query_selection_output": R0115_QUERY_SELECTION,
        "r0115_control_graph_manifest": R0115_CONTROL_GRAPH,
        "r0115_control_train_receipt": R0115_CONTROL_TRAIN,
        "r0115_control_score": R0115_CONTROL_SCORE,
        "r0115_release_sha": reviews["0115"]["release_commit"],
        "r0106_context_graph": expected_input_signature(
            R0106_GRAPH_CONTEXT
        ),
        "r0108_context_core": expected_input_signature(
            R0108_CORE_CONTEXT
        ),
    }
    jobs = [
        {
            "id": "build_k15_graph",
            "action": "build_k15_graph",
            "handler_module": "experiments.round0124_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [graph_output],
            "done_marker": os.path.join(artifacts, "build_k15_graph.done.json"),
            "p90_wall_s": 300.0,
            **common,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            "id": "train_k15_treatment",
            "action": "train_k15_treatment",
            "handler_module": "experiments.round0124_nodes",
            "handler_callable": "run_job",
            "deps": ["build_k15_graph"],
            "outputs": [train_output],
            "done_marker": os.path.join(
                artifacts, "train_k15_treatment.done.json"
            ),
            "p90_wall_s": 5_400.0,
            "graph_manifest": graph_manifest,
            **common,
            "node_policy": {
                "gpu_required": True,
                "training_performed": True,
            },
        },
        {
            "id": "evaluate_core_ood",
            "action": "evaluate_core_ood",
            "handler_module": "experiments.round0124_nodes",
            "handler_callable": "run_job",
            "deps": ["train_k15_treatment"],
            "outputs": [diagnostic_output],
            "done_marker": os.path.join(
                artifacts, "evaluate_core_ood.done.json"
            ),
            "p90_wall_s": 300.0,
            "arm": "raw",
            "graph_manifest": graph_manifest,
            "train_output": train_output,
            **common,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            "id": "score_native_density",
            "action": "score_native_density",
            "handler_module": "experiments.round0124_nodes",
            "handler_callable": "run_job",
            "deps": ["evaluate_core_ood"],
            "outputs": [density_output],
            "done_marker": os.path.join(
                artifacts, "score_native_density.done.json"
            ),
            "p90_wall_s": 120.0,
            "graph_manifest": graph_manifest,
            "train_output": train_output,
            "diagnostic_output": diagnostic_output,
            **common,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
        },
        {
            "id": "decide_degree_bridge",
            "action": "decide_degree_bridge",
            "handler_module": "experiments.round0124_nodes",
            "handler_callable": "run_job",
            "deps": ["score_native_density"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts, "decide_degree_bridge.done.json"
            ),
            "p90_wall_s": 60.0,
            "diagnostic_output": diagnostic_output,
            "density_output": density_output,
            "expected_inputs": inputs,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=2.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update(
        {
            "schema": "round0124-fineweb-2m-degree-bridge-queue-v1",
            "repo_root": RELEASE_ROOT,
            "queue_class": "gpu-research",
            "required_reviews": ["0106", "0108", "0115"],
            "capability_dependencies": [
                "jina-fineweb-2m-prompt-map-contrast-v1"
            ],
            "capabilities_produced": [
                "jina-fineweb-2m-native-k15-degree-bridge-v1"
            ],
            "training_performed": True,
            "scientific_contract": {
                "control": (
                    "exact re-score of R0115 raw seed-42 k50 native "
                    "coordinates"
                ),
                "treatment_trained": "raw seed-42 k15 only",
                "changed_factor": "fuzzy graph neighbor degree only",
                "population_rows": RETAINED_ROWS,
                "graph_nonself_neighbors": GRAPH_DEGREE,
                "graph_search_neighbors_including_self": (
                    GRAPH_SEARCH_NEIGHBORS
                ),
                "successful_updates": SUCCESSFUL_UPDATES,
                "native_reference": (
                    "exact R0115 raw 4,000-anchor high-D reference"
                ),
                "paired_bootstrap": {
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                    "ci_level": BOOTSTRAP_CI_LEVEL,
                    "material_density_degradation": (
                        MATERIAL_DENSITY_DEGRADATION
                    ),
                },
                "legacy_density_floor_used": False,
                "normal_core_ood_diagnostics": True,
                "diagnostics_can_rescue_or_fail_selector": False,
            },
            "jobs": jobs,
            "p90_gpu_seconds": {
                "build_k15_graph": 300.0,
                "train_k15_treatment": 5_400.0,
                "evaluate_core_ood": 300.0,
                "score_native_density": 120.0,
                "total": 6_120.0,
            },
        }
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0106-review", required=True)
    parser.add_argument("--r0106-review-sha256", required=True)
    parser.add_argument("--r0108-review", required=True)
    parser.add_argument("--r0108-review-sha256", required=True)
    parser.add_argument("--r0115-review", required=True)
    parser.add_argument("--r0115-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0124(
        release_sha=args.release_sha,
        r0106_review=args.r0106_review,
        r0106_review_sha256=args.r0106_review_sha256,
        r0108_review=args.r0108_review,
        r0108_review_sha256=args.r0108_review_sha256,
        r0115_review=args.r0115_review,
        r0115_review_sha256=args.r0115_review_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
