#!/usr/bin/env python3
"""Prepare, but never launch, the conditional R0148 English-anchor queue."""
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
from basemap.round0105_search import ELIGIBILITY_PATH
from basemap.round0108_evaluation import validate_seal
from basemap.round0148_english_anchor import (
    CAPABILITY,
    DENSITY_V2_FLOOR,
    OOD_METRICS,
    REGISTERED_GROUP_COUNTS,
    ROUND_ID,
    build_subset_plan,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0132_queue import _frontmatter
from experiments.round0132_nodes import GRAPH_PART_NAMES


ROUND_ROOT = "/data/latent-basemap/runs/round-0148"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0148-*.md")
R0147_REVIEW_GLOB = os.path.join(LAB_ROOT, "review-0147-*.md")
R0132_REVIEW = os.path.join(LAB_ROOT, "review-0132-2026-08-01.md")

R0147_QUEUE = "/data/latent-basemap/runs/round-0147/queue/queue.json"
R0147_TERMINAL = "/data/latent-basemap/runs/round-0147/queue/runner-terminal.json"
R0147_DECISION = os.path.join(
    "/data/latent-basemap/runs/round-0147/queue/artifacts",
    "jina-2m-historical-row-policy-duplicate-control-v1",
    "decision.json",
)
R0132_QUEUE = "/data/latent-basemap/runs/round-0132/queue/queue.json"
R0132_TERMINAL = "/data/latent-basemap/runs/round-0132/queue/runner-terminal.json"
R0132_ARTIFACTS = "/data/latent-basemap/runs/round-0132/queue/artifacts"
CONTROL_GRAPH = os.path.join(R0132_ARTIFACTS, "half-fuzzy-graph", "graph-manifest.json")
CONTROL_TRAIN = os.path.join(R0132_ARTIFACTS, "train-half-seed42")

GPU_HOURS_MINIMUM = 2.0
GPU_HOURS_EXPECTED = 2.5
GPU_HOURS_P90 = 3.2
GPU_HOURS_MAXIMUM = 4.5

P90_NODE_SECONDS = {
    "build_index": 180.0,
    "qualify_search": 300.0,
    # R0148's middle part has 6.20M rows.  R0132 measured 635 s for its
    # largest 4.54M-row part, so the size-calibrated p90 is about 870 s.
    "graph_part": 900.0,
    "train": 7_200.0,
    "transform": 180.0,
    "functional_density": 300.0,
    "ood": 180.0,
}


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0148 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_positive_r0147_review() -> dict[str, Any]:
    candidates: list[str] = []
    for path in sorted(glob.glob(R0147_REVIEW_GLOB)):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        frontmatter = _frontmatter(path)
        if (
            frontmatter.get("round_id") == "0147"
            and frontmatter.get("status") == "accepted"
            and "capability:jina-2m-historical-row-policy-duplicate-control-v1"
            in text
            and "eligible-historical-row-policy-restores" in text
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            "R0148 requires exactly one accepted positive R0147 review"
        )
    return expected_input_signature(candidates[0])


def _require_positive_r0147_decision() -> dict[str, Any]:
    """Bind the exact branch artifact instead of inferring it from review prose."""
    with open(R0147_DECISION, encoding="utf-8") as handle:
        decision = json.load(handle)
    validate_seal(decision, label="R0147 row-policy decision")
    if (
        decision.get("schema")
        != "round0147-historical-row-policy-decision-v1"
        or decision.get("round_id") != "0147"
        or decision.get("capability")
        != "jina-2m-historical-row-policy-duplicate-control-v1"
        or decision.get("outcome")
        != "eligible-historical-row-policy-restores"
        or decision.get("duplicate_control_compatible_with_restoration") is not True
        or decision.get("diverse_scale_transfer_claimed") is not False
    ):
        raise RuntimeError("R0148 requires the exact positive R0147 decision")
    return expected_input_signature(R0147_DECISION)


def _require_r0132_review() -> dict[str, Any]:
    with open(R0132_REVIEW, encoding="utf-8") as handle:
        text = handle.read()
    if (
        _frontmatter(R0132_REVIEW).get("status") != "accepted"
        or "capability:jina-diverse-12p5m-25m-scale-policy-geometry-v1"
        not in text
    ):
        raise RuntimeError("accepted R0132 capability is unavailable")
    return expected_input_signature(R0132_REVIEW)


def _require_clean_execution(
    queue_path: str, terminal_path: str, *, round_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    queue_signature = expected_input_signature(queue_path)
    terminal_signature = expected_input_signature(terminal_path)
    with open(queue_path, encoding="utf-8") as handle:
        queue = json.load(handle)
    with open(terminal_path, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        queue.get("round_id") != round_id
        or terminal.get("round_id") != round_id
        or terminal.get("verdict") != "succeeded"
        or terminal.get("completed_jobs") != terminal.get("required_jobs")
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
    ):
        raise RuntimeError(f"R{round_id} execution is not clean and terminal")
    return queue, queue_signature, terminal_signature


def _job(
    *,
    node_id: str,
    action: str,
    deps: list[str],
    output: str,
    expected_inputs: list[dict[str, Any]],
    p90_wall_s: float,
    gpu: bool,
    training: bool = False,
    **values: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "action": action,
        "handler_module": "experiments.round0148_nodes",
        "handler_callable": "run_job",
        "deps": deps,
        "outputs": [output],
        "done_marker": os.path.join(os.path.dirname(output), f"{node_id}.done.json"),
        "expected_inputs": _dedupe(expected_inputs),
        "p90_wall_s": p90_wall_s,
        "node_policy": {
            "gpu_required": gpu,
            "training_performed": training,
        },
        **values,
    }


def prepare_round0148(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0148 release SHA must be one full commit")
    round_file = _require_issued_round()
    if _frontmatter(round_file).get("base_commit") != release_sha:
        raise RuntimeError("R0148 round base_commit differs from release")
    r0147_review = _require_positive_r0147_review()
    r0147_decision = _require_positive_r0147_decision()
    r0132_review = _require_r0132_review()
    r0147_queue, r0147_queue_signature, r0147_terminal = _require_clean_execution(
        R0147_QUEUE, R0147_TERMINAL, round_id="0147"
    )
    r0132_queue, r0132_queue_signature, r0132_terminal = _require_clean_execution(
        R0132_QUEUE, R0132_TERMINAL, round_id="0132"
    )
    r0147_jobs = {str(job["action"]): job for job in r0147_queue["jobs"]}
    r0132_jobs = {str(job["action"]): job for job in r0132_queue["jobs"]}
    functional_source = r0147_jobs["functional_panel"]
    ood_source = r0132_jobs["score_matched_ood"]
    native_source = r0132_jobs["score_matched_native"]

    substrate = validate_substrate_manifest(verify_payloads=False)
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    control_graph = expected_input_signature(CONTROL_GRAPH)
    # Written explicitly so every model-bearing member is bound.
    control_inputs = [
        control_graph,
        *(
            expected_input_signature(os.path.join(CONTROL_TRAIN, name))
            for name in ("train-receipt.json", "production-config.json", "model.pt")
        ),
    ]
    common = _dedupe([
        expected_input_signature(round_file),
        r0147_review,
        r0147_decision,
        r0132_review,
        r0147_queue_signature,
        r0147_terminal,
        r0132_queue_signature,
        r0132_terminal,
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        substrate["payloads"]["labels"],
        eligibility,
        *control_inputs,
        *functional_source["expected_inputs"],
        *ood_source["expected_inputs"],
        *native_source["expected_inputs"],
    ])

    queue_root = create_fresh_directory(
        queue_root, label="R0148 conditional English-anchor rescue queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    subset = os.path.join(artifacts, "english-anchor-subset")
    index_output = os.path.join(artifacts, "english-anchor-search-index")
    index_path = os.path.join(index_output, "english-anchor-12p5m.ivfpq")
    qualification = os.path.join(artifacts, "english-anchor-search-qualification")
    parts = {
        part: os.path.join(artifacts, f"english-anchor-graph-part-{part}")
        for part in GRAPH_PART_NAMES
    }
    graph_output = os.path.join(artifacts, "english-anchor-fuzzy-graph")
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    train_output = os.path.join(artifacts, "train-english-anchor-seed42")
    transform_output = os.path.join(artifacts, "english-anchor-coordinates")
    functional_output = os.path.join(artifacts, "functional-density-panel")
    ood_output = os.path.join(artifacts, "matched-ood")
    decision_output = os.path.join(artifacts, "decision")

    jobs: list[dict[str, Any]] = []
    jobs.append(_job(
        node_id="select_english_anchor_subset",
        action="select_english_anchor_subset",
        deps=[],
        output=subset,
        expected_inputs=common,
        p90_wall_s=1_200.0,
        gpu=False,
        eligibility_sha256=eligibility["sha256"],
    ))
    jobs.append(_job(
        node_id="build_english_anchor_search_index",
        action="build_english_anchor_search_index",
        deps=["select_english_anchor_subset"],
        output=index_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["build_index"],
        gpu=True,
        subset_output=subset,
    ))
    jobs.append(_job(
        node_id="qualify_english_anchor_search",
        action="qualify_english_anchor_search",
        deps=["build_english_anchor_search_index"],
        output=qualification,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["qualify_search"],
        gpu=True,
        subset_output=subset,
        index_output=index_output,
        index=index_path,
    ))
    for part in GRAPH_PART_NAMES:
        jobs.append(_job(
            node_id=f"build_english_anchor_graph_part_{part}",
            action="build_english_anchor_graph_part",
            deps=["qualify_english_anchor_search"],
            output=parts[part],
            expected_inputs=common,
            p90_wall_s=P90_NODE_SECONDS["graph_part"],
            gpu=True,
            part=part,
            subset_output=subset,
            index_output=index_output,
            index=index_path,
            qualification_output=qualification,
        ))
    part_ids = [f"build_english_anchor_graph_part_{part}" for part in GRAPH_PART_NAMES]
    jobs.append(_job(
        node_id="assemble_english_anchor_graph",
        action="assemble_english_anchor_graph",
        deps=part_ids,
        output=graph_output,
        expected_inputs=common,
        p90_wall_s=1_200.0,
        gpu=False,
        subset_output=subset,
        part_outputs=parts,
    ))
    jobs.append(_job(
        node_id="train_english_anchor_map",
        action="train_english_anchor_map",
        deps=["assemble_english_anchor_graph"],
        output=train_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["train"],
        gpu=True,
        training=True,
        release_sha=release_sha,
        graph_release_sha=release_sha,
        graph_manifest=graph_manifest,
        graph_manifest_late_bound_from="assemble_english_anchor_graph",
    ))
    jobs.append(_job(
        node_id="transform_english_anchor_map",
        action="transform_english_anchor_map",
        deps=["train_english_anchor_map"],
        output=transform_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["transform"],
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
    ))
    shared_panel = {
        key: functional_source[key]
        for key in (
            "source",
            "query_embeddings",
            "shared_reference_receipt",
            "high_d_reference",
            "query_truth",
            "centroids",
        )
    }
    controls = {
        "control_train_output": CONTROL_TRAIN,
        "control_graph_manifest": CONTROL_GRAPH,
        "control_graph_manifest_sha256": control_graph["sha256"],
    }
    jobs.append(_job(
        node_id="score_english_anchor_function_density",
        action="score_english_anchor_function_density",
        deps=["transform_english_anchor_map"],
        output=functional_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["functional_density"],
        gpu=True,
        subset_output=subset,
        train_output=train_output,
        graph_manifest=graph_manifest,
        r0108_calibration=expected_input_signature(native_source["stale_calibration"]),
        **controls,
        **shared_panel,
    ))
    jobs.append(_job(
        node_id="score_english_anchor_ood",
        action="score_english_anchor_ood",
        deps=["train_english_anchor_map"],
        output=ood_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["ood"],
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
        selection=ood_source["selection"],
        selection_sha256=ood_source["selection_sha256"],
        language_sources=ood_source["language_sources"],
        diagnostic_sources=ood_source["diagnostic_sources"],
        **controls,
    ))
    jobs.append(_job(
        node_id="decide_english_anchor_rescue",
        action="decide_english_anchor_rescue",
        deps=["score_english_anchor_function_density", "score_english_anchor_ood"],
        output=decision_output,
        expected_inputs=common,
        p90_wall_s=60.0,
        gpu=False,
        functional_output=functional_output,
        ood_output=ood_output,
        train_output=train_output,
        graph_manifest=graph_manifest,
        r0108_calibration=expected_input_signature(native_source["stale_calibration"]),
        **controls,
    ))

    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0148-english-anchor-rescue-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0132", "0147"],
        "capability_dependencies": [
            "jina-diverse-12p5m-25m-scale-policy-geometry-v1",
            "jina-2m-historical-row-policy-duplicate-control-v1",
        ],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "scientific_contract": {
            "question": (
                "does retaining all eligible English representatives and a "
                "proportional 19-language remainder at fixed 12,474,331 rows "
                "transfer R0147 restoration without breaking density or OOD?"
            ),
            "population_plan": build_subset_plan(REGISTERED_GROUP_COUNTS),
            "functional_floors": "unchanged R0140 restoration floors",
            "density_v2_floor": DENSITY_V2_FLOOR,
            "density_floor_recalibration": False,
            "matched_ood_metrics": list(OOD_METRICS),
            "matched_ood_retention": 0.97,
            "all_function_density_and_ood_gates_required": True,
            "25m_transfer_claimed": False,
            "map_registry_state_changed": False,
        },
        "gpu_hours": {
            "minimum": GPU_HOURS_MINIMUM,
            "expected": GPU_HOURS_EXPECTED,
            "p90": GPU_HOURS_P90,
            "maximum": GPU_HOURS_MAXIMUM,
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(prepare_round0148(release_sha=args.release_sha, queue_root=args.queue_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
