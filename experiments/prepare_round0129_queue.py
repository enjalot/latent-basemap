#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0129 seed-43 replicate."""
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
    train_config as r0117_train_config,
)
from basemap.round0124_degree_bridge import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECISION_SCHEMA as R0124_DECISION_SCHEMA,
    GRAPH_DEGREE,
    GRAPH_SEARCH_NEIGHBORS,
    MATERIAL_DENSITY_DEGRADATION,
    NATIVE_ANCHOR_SEED,
    NATIVE_DENSITY_ANCHORS,
    OUTCOME_INCONCLUSIVE as R0124_INCONCLUSIVE_OUTCOME,
)
from basemap.round0129_degree_replicate import (
    CAPABILITY,
    ROUND_ID,
    TRAINING_SEED,
    graph_provenance,
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


ROUND_ROOT = "/data/latent-basemap/runs/round-0129"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0129-*.md")
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
R0115_QUERY_SELECTION = os.path.join(
    R0115_ROOT, "artifacts/query-selection"
)
R0117_ROOT = "/data/latent-basemap/runs/round-0117/queue"
R0117_QUEUE = os.path.join(R0117_ROOT, "queue.json")
R0117_TERMINAL = os.path.join(R0117_ROOT, "runner-terminal.json")
R0117_CONTROL_TRAIN = os.path.join(
    R0117_ROOT, "artifacts/raw/train/train-receipt.json"
)
R0117_CONTROL_SCORE = os.path.join(
    R0117_ROOT, "artifacts/raw/evaluation/score.json"
)
R0124_ROOT = "/data/latent-basemap/runs/round-0124/queue-attempt-2"
R0124_QUEUE = os.path.join(R0124_ROOT, "queue.json")
R0124_TERMINAL = os.path.join(R0124_ROOT, "runner-terminal.json")
R0124_DECISION = os.path.join(
    R0124_ROOT, "artifacts/degree-bridge-decision/decision.json"
)

GPU_HOURS_CAP = 2.5
P90_TRAIN_SECONDS = 5_400.0
P90_DIAGNOSTIC_SECONDS = 300.0
P90_DENSITY_SECONDS = 120.0
P90_GPU_TOTAL_SECONDS = (
    P90_TRAIN_SECONDS + P90_DIAGNOSTIC_SECONDS + P90_DENSITY_SECONDS
)


def _issued_round() -> str:
    candidates = []
    for path in sorted(glob.glob(ROUND_FILE_GLOB)):
        frontmatter, _text = _document(path)
        if frontmatter.get("status") == "issued":
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0129 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_r0117_review(
    path: str, expected_sha256: str
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    frontmatter, _text = _document(path)
    capability = "jina-fineweb-2m-prompt-map-seed43-contrast-v1"
    if (
        signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != "0117"
        or frontmatter.get("status") != "accepted"
        or f"capability:{capability}"
        not in _frontmatter_list(frontmatter, "releases", label="R0117 review")
    ):
        raise RuntimeError("Review 0117 is not exact and accepted")
    result_name = frontmatter.get("result") or ""
    if (
        os.path.basename(result_name) != result_name
        or not re.fullmatch(r"result-0117-[0-9]{4}-[0-9]{2}-[0-9]{2}\.md", result_name)
    ):
        raise RuntimeError("Review 0117 result binding is malformed")
    result_path = os.path.join(os.path.dirname(path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter, _result_text = _document(result_path)
    release = frontmatter.get("verified_release_commit")
    if (
        result_signature["sha256"] != frontmatter.get("result_sha256")
        or result_frontmatter.get("round_id") != "0117"
        or result_frontmatter.get("status") != "complete"
        or result_frontmatter.get("release_commit") != release
        or not re.fullmatch(r"[0-9a-f]{40}", release or "")
        or capability
        not in _frontmatter_list(
            result_frontmatter,
            "capabilities_produced",
            label="R0117 result",
        )
    ):
        raise RuntimeError("Review 0117 does not close its exact result")
    return {
        "review": signature,
        "result": result_signature,
        "release_commit": release,
    }


def _require_inconclusive_r0124_review(
    review_path: str,
    *,
    expected_sha256: str,
    queue_path: str = R0124_QUEUE,
    terminal_path: str = R0124_TERMINAL,
    decision_path: str = R0124_DECISION,
) -> dict[str, dict[str, Any]]:
    """Authenticate the sealed selector outcome; review prose is irrelevant."""
    review_signature = expected_input_signature(review_path)
    review_frontmatter, _review_text = _document(review_path)
    if (
        review_signature["sha256"] != expected_sha256
        or review_frontmatter.get("round_id") != "0124"
        or review_frontmatter.get("status") != "accepted"
        or "capability:jina-fineweb-2m-native-k15-degree-bridge-v1"
        not in _frontmatter_list(
            review_frontmatter, "releases", label="R0124 review"
        )
    ):
        raise RuntimeError("Review 0124 is not accepted at the expected bytes")
    result_name = review_frontmatter.get("result") or ""
    result_sha256 = review_frontmatter.get("result_sha256") or ""
    if (
        os.path.basename(result_name) != result_name
        or not re.fullmatch(r"result-0124-[0-9]{4}-[0-9]{2}-[0-9]{2}\.md", result_name)
        or not re.fullmatch(r"[0-9a-f]{64}", result_sha256)
    ):
        raise RuntimeError("Review 0124 does not bind one result document")
    result_path = os.path.join(os.path.dirname(review_path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter, result_text = _document(result_path)
    result_release = result_frontmatter.get("release_commit") or ""
    result_queue = (result_frontmatter.get("queue_manifest") or "").removeprefix(
        "gsv:"
    )
    if (
        result_signature["sha256"] != result_sha256
        or result_frontmatter.get("round_id") != "0124"
        or result_frontmatter.get("status") != "complete"
        or review_frontmatter.get("verified_release_commit") != result_release
        or not re.fullmatch(r"[0-9a-f]{40}", result_release)
        or os.path.realpath(result_queue) != os.path.realpath(queue_path)
    ):
        raise RuntimeError("Accepted Review 0124 result binding changed")

    queue_signature = expected_input_signature(queue_path)
    terminal_signature = expected_input_signature(terminal_path)
    decision_signature = expected_input_signature(decision_path)
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    decision = read_sealed(decision_path, label="R0124 degree decision")
    jobs = queue.get("jobs") or []
    required_jobs = [str(job.get("id") or "") for job in jobs]
    decision_jobs = [job for job in jobs if job.get("id") == "decide_degree_bridge"]
    nodes = terminal.get("nodes") or []
    node_ids = [str(node.get("node") or "") for node in nodes]
    if (
        queue.get("schema")
        != "round0124-fineweb-2m-degree-bridge-retry-queue-v1"
        or queue.get("round_id") != "0124"
        or queue.get("release_sha") != result_release
        or not required_jobs
        or any(not value for value in required_jobs)
        or len(set(required_jobs)) != len(required_jobs)
        or len(decision_jobs) != 1
        or decision_jobs[0].get("outputs")
        != [os.path.dirname(os.path.realpath(decision_path))]
        or terminal.get("schema") != "slim-runner-terminal-v3"
        or terminal.get("round_id") != "0124"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("required_jobs") != required_jobs
        or sorted(terminal.get("completed_jobs") or []) != sorted(required_jobs)
        or len(terminal.get("completed_jobs") or []) != len(required_jobs)
        or sorted(node_ids) != sorted(required_jobs)
        or len(node_ids) != len(required_jobs)
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
        or terminal.get("gpu_wall_accounting_complete") is not True
        or terminal.get("boundary_problems") != []
        or any(node.get("validation_problems") != [] for node in nodes)
        or (terminal.get("release_checkout") or {}).get("head")
        != result_release
        or (terminal.get("release_checkout_at_finish") or {}).get("head")
        != result_release
        or decision.get("schema") != R0124_DECISION_SCHEMA
        or decision.get("round_id") != "0124"
        or decision.get("release_sha") != result_release
        or decision.get("retry_provenance") != queue.get("retry_provenance")
        or decision.get("capabilities_produced")
        != ["jina-fineweb-2m-native-k15-degree-bridge-v1"]
        or result_frontmatter.get("queue_manifest_sha256")
        != queue_signature["sha256"]
        or decision_signature["sha256"] not in result_text
        or queue_signature["sha256"] not in result_text
        or terminal_signature["sha256"] not in result_text
    ):
        raise RuntimeError("R0124 execution/result linkage changed")
    outcome = (decision.get("registered_selector") or {}).get("outcome")
    if outcome != R0124_INCONCLUSIVE_OUTCOME:
        raise RuntimeError(
            "R0124 sealed decision outcome is not the inconclusive branch"
        )
    return {
        "review": review_signature,
        "result": result_signature,
        "queue": queue_signature,
        "terminal": terminal_signature,
        "decision": decision_signature,
    }


def _scientific_inputs(
    *, r0117: Mapping[str, Any]
) -> list[dict[str, Any]]:
    assembly_path = os.path.join(R0113_ASSEMBLY, "assembly-manifest.json")
    query_path = os.path.join(R0113_QUERY, "query-reserve-receipt.json")
    selection_path = os.path.join(R0115_QUERY_SELECTION, "query-selection.json")
    assembly = read_sealed(assembly_path, label="R0113 compact assembly")
    query = read_sealed(query_path, label="R0113 query reserve")
    selection = read_sealed(selection_path, label="R0115 query selection")
    graph = read_sealed(R0115_CONTROL_GRAPH, label="R0115 raw k49 graph")
    train = read_sealed(R0117_CONTROL_TRAIN, label="R0117 raw train")
    score = read_sealed(R0117_CONTROL_SCORE, label="R0117 raw score")
    terminal = _clean_terminal(
        R0117_QUEUE,
        R0117_TERMINAL,
        round_id="0117",
        expected_release_sha=str(r0117["release_commit"]),
    )
    graph_signature = expected_input_signature(R0115_CONTROL_GRAPH)
    train_signature = expected_input_signature(R0117_CONTROL_TRAIN)
    score_signature = expected_input_signature(R0117_CONTROL_SCORE)
    production_path = str(train.get("production_config", {}).get("canonical_path") or "")
    with open(production_path, encoding="utf-8") as handle:
        production = json.load(handle)
    control_config, control_sha = r0117_train_config(
        "raw",
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=int(graph["retained_rows"]),
        seed=TRAINING_SEED,
    )
    coordinates = (score.get("coordinates") or {}).get("training") or {}
    panel = score.get("panel") or {}
    if (
        assembly.get("retained_rows") != RETAINED_ROWS
        or selection.get("selected_rows") != QUERY_ROWS
        or selection.get("selected_before_training") is not True
        or graph.get("round_id") != "0115"
        or graph.get("arm") != "raw"
        or graph.get("k") != 50
        or train.get("round_id") != "0117"
        or train.get("release_sha") != r0117["release_commit"]
        or train.get("training_seed") != TRAINING_SEED
        or train.get("graph_manifest") != graph_signature
        or train.get("production_config_sha256") != control_sha
        or production.get("config") != control_config
        or score.get("round_id") != "0117"
        or score.get("release_sha") != r0117["release_commit"]
        or score.get("training_seed") != TRAINING_SEED
        or score.get("train_receipt") != train_signature
        or score.get("graph_manifest") != graph_signature
        or not isinstance(coordinates, Mapping)
        or panel.get("n") != RETAINED_ROWS
        or panel.get("n_anchors") != NATIVE_DENSITY_ANCHORS
        or panel.get("anchor_seed") != NATIVE_ANCHOR_SEED
        or panel.get("k_density") != GRAPH_DEGREE
        or panel.get("density") != 0.2116
    ):
        raise RuntimeError("R0129 frozen R0117 control evidence changed")
    polish = query["ood"]["pol_Latn"]
    return _dedupe(
        [
            expected_input_signature(assembly_path),
            assembly["mapping"],
            assembly["outputs"]["raw"],
            assembly["source_text_hash_index"],
            assembly["source_prompt_family_discovery"],
            assembly["retained_duplicate_audit"],
            expected_input_signature(query_path),
            query["query_rows"],
            query["source_text_row_hashes"],
            query["outputs"]["raw"],
            query["outputs"]["document"],
            polish["outputs"]["raw"],
            polish["outputs"]["document"],
            polish["query_rows"],
            polish["source_text_row_hashes"],
            expected_input_signature(selection_path),
            selection["positions"],
            selection["global_rows"],
            graph_signature,
            graph["graph"],
            graph["topology_probe"],
            graph["high_d_reference"],
            graph["query_training_copy_mask"],
            graph["polish_query_training_copy_mask"],
            train_signature,
            train["production_config"],
            train["model"],
            score_signature,
            coordinates,
            *terminal.values(),
        ]
    )


def prepare_round0129(
    *,
    release_sha: str,
    r0117_review: str,
    r0117_review_sha256: str,
    r0124_review: str,
    r0124_review_sha256: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0129 release SHA must be one full commit")
    round_file = _issued_round()
    r0117 = _accepted_r0117_review(r0117_review, r0117_review_sha256)
    trigger = _require_inconclusive_r0124_review(
        r0124_review,
        expected_sha256=r0124_review_sha256,
    )
    provenance = graph_provenance()
    inputs = _dedupe(
        [
            expected_input_signature(round_file),
            r0117["review"],
            r0117["result"],
            *trigger.values(),
            *_scientific_inputs(r0117=r0117),
            *provenance["evidence"].values(),
        ]
    )
    ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        queue_root, label="R0129 seed-43 degree replicate queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    train_output = os.path.join(artifacts, "k15-seed43-train")
    diagnostic_output = os.path.join(artifacts, "core-ood-diagnostics")
    density_output = os.path.join(artifacts, "native-density-contrast")
    decision_output = os.path.join(artifacts, "degree-replicate-decision")
    common = {
        "expected_inputs": inputs,
        "training_seed": TRAINING_SEED,
        "graph_provenance": provenance,
        "graph_manifest": provenance["evidence"]["graph_manifest"]["canonical_path"],
        "assembly_output": R0113_ASSEMBLY,
        "query_output": R0113_QUERY,
        "query_selection_output": R0115_QUERY_SELECTION,
        "r0115_control_graph_manifest": R0115_CONTROL_GRAPH,
        "r0115_release_sha": "3b6ed28e1801e13228c78e05cf992a30e398a678",
        "r0117_control_train_receipt": R0117_CONTROL_TRAIN,
        "r0117_control_score": R0117_CONTROL_SCORE,
        "r0117_release_sha": r0117["release_commit"],
        "r0124_inconclusive_decision": trigger["decision"],
    }
    jobs = [
        {
            "id": "train_k15_seed43",
            "action": "train_k15_seed43",
            "handler_module": "experiments.round0129_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [train_output],
            "done_marker": os.path.join(artifacts, "train_k15_seed43.done.json"),
            "p90_wall_s": P90_TRAIN_SECONDS,
            **common,
            "node_policy": {"gpu_required": True, "training_performed": True},
        },
        {
            "id": "evaluate_core_ood",
            "action": "evaluate_core_ood",
            "handler_module": "experiments.round0129_nodes",
            "handler_callable": "run_job",
            "deps": ["train_k15_seed43"],
            "outputs": [diagnostic_output],
            "done_marker": os.path.join(artifacts, "evaluate_core_ood.done.json"),
            "p90_wall_s": P90_DIAGNOSTIC_SECONDS,
            "arm": "raw",
            "train_output": train_output,
            **common,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            "id": "score_native_density",
            "action": "score_native_density",
            "handler_module": "experiments.round0129_nodes",
            "handler_callable": "run_job",
            "deps": ["evaluate_core_ood"],
            "outputs": [density_output],
            "done_marker": os.path.join(artifacts, "score_native_density.done.json"),
            "p90_wall_s": P90_DENSITY_SECONDS,
            "train_output": train_output,
            "diagnostic_output": diagnostic_output,
            **common,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            "id": "decide_degree_replicate",
            "action": "decide_degree_replicate",
            "handler_module": "experiments.round0129_nodes",
            "handler_callable": "run_job",
            "deps": ["score_native_density"],
            "outputs": [decision_output],
            "done_marker": os.path.join(artifacts, "decide_degree_replicate.done.json"),
            "p90_wall_s": 60.0,
            "train_output": train_output,
            "diagnostic_output": diagnostic_output,
            "density_output": density_output,
            **common,
            "node_policy": {"gpu_required": False, "training_performed": False},
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_CAP,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update(
        {
            "schema": "round0129-seed43-native-degree-replicate-queue-v1",
            "repo_root": RELEASE_ROOT,
            "queue_class": "gpu-research",
            "required_reviews": ["0117", "0124"],
            "capability_dependencies": [
                "jina-fineweb-2m-prompt-map-seed43-contrast-v1",
                "jina-fineweb-2m-native-k15-degree-bridge-v1",
            ],
            "capabilities_produced": [CAPABILITY],
            "training_performed": True,
            "conditional_trigger": {
                "source_round": "0124",
                "required_structured_outcome": R0124_INCONCLUSIVE_OUTCOME,
                "review": trigger["review"],
                "result": trigger["result"],
                "queue": trigger["queue"],
                "terminal": trigger["terminal"],
                "decision": trigger["decision"],
                "prose_can_release": False,
            },
            "scientific_contract": {
                "control": "exact accepted R0117 raw seed-43 k49 map",
                "treatment": "fresh raw seed-43 k15 map",
                "changed_factor": "fuzzy graph neighbor degree only",
                "population_rows": RETAINED_ROWS,
                "graph_nonself_neighbors": GRAPH_DEGREE,
                "graph_search_neighbors_including_self": GRAPH_SEARCH_NEIGHBORS,
                "graph_reuse": {
                    "policy": "exact immutable successful R0124 attempt-1 graph",
                    "provenance": provenance,
                    "rebuild_permitted": False,
                },
                "training_seed": TRAINING_SEED,
                "successful_updates": SUCCESSFUL_UPDATES,
                "non_graph_config_equal": True,
                "sampling_mechanism_equal_conditioned_on_graph": True,
                "positive_edge_distribution_equal": False,
                "registered_distributional_intervention": (
                    "weighted graph topology/edge population/weights induced "
                    "by k49-to-k15"
                ),
                "negative_sampling_distribution_equal": True,
                "identical_realized_negative_pairs_claimed": False,
                "identical_realized_draws_claimed": False,
                "actual_pre_update_initial_state_hook_required": True,
                "native_reference": "exact R0115/R0117 4,000-anchor high-D reference",
                "paired_bootstrap": {
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                    "ci_level": BOOTSTRAP_CI_LEVEL,
                    "material_density_degradation": MATERIAL_DENSITY_DEGRADATION,
                },
                "core_and_ood_diagnostics": "diagnostic-only",
                "diagnostics_can_rescue_or_fail_selector": False,
                "legacy_density_floor_used": False,
            },
            "cpu_handoff_smoke": {
                "required_before_issue": True,
                "maximum_wall_s": 120.0,
                "cuda_visible_devices": "empty",
                "path": "train -> seal -> checkpoint reload -> panel",
            },
            "jobs": jobs,
            "p90_gpu_seconds": {
                "train_k15_seed43": P90_TRAIN_SECONDS,
                "evaluate_core_ood": P90_DIAGNOSTIC_SECONDS,
                "score_native_density": P90_DENSITY_SECONDS,
                "total": P90_GPU_TOTAL_SECONDS,
                "estimate_status": "conservative-pending-r0124-retry-receipt",
            },
        }
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0117-review", required=True)
    parser.add_argument("--r0117-review-sha256", required=True)
    parser.add_argument("--r0124-review", required=True)
    parser.add_argument("--r0124-review-sha256", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    path = prepare_round0129(
        release_sha=args.release_sha,
        r0117_review=args.r0117_review,
        r0117_review_sha256=args.r0117_review_sha256,
        r0124_review=args.r0124_review,
        r0124_review_sha256=args.r0124_review_sha256,
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
