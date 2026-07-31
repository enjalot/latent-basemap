#!/usr/bin/env python3
"""Prepare, but never launch, the independent R0132 scale-policy queue."""
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
from basemap.round0036_pipeline import CoordinateStream, Round0036Error
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0105_search import ELIGIBILITY_PATH
from basemap.round0132_scale_bridge import (
    DECISION_SCHEMA,
    DENSITY_BOOTSTRAP_DRAWS,
    DENSITY_BOOTSTRAP_SEED,
    DENSITY_CI_LEVEL,
    DENSITY_COMPARISON_ATOL,
    DENSITY_NONINFERIORITY_MARGIN,
    FFR_ALLOWED_DECREASE,
    FULL_RETAINED_ROWS,
    GRAPH_K,
    HALF_RETAINED_ROWS,
    METRIC_RETENTION,
    NATIVE_ANCHORS_PER_GROUP,
    NATIVE_ANCHOR_SEED,
    OUTCOME_DENSITY_REGRESSION,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_INVALID,
    OUTCOME_QUALITY_REGRESSION,
    OUTCOME_SUPPORTED,
    ROUND_ID,
    SEARCH_ANCHORS_PER_GROUP,
    SEARCH_GLOBAL_RECALL_FLOOR,
    SEARCH_GROUP_RECALL_FLOOR,
    SEARCH_NPROBE,
    SEARCH_SHORTLIST_WIDTH,
    SUBSET_NAMESPACE,
    assert_no_conditional_branch_dependency,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
)
from experiments.round0132_nodes import GRAPH_PART_NAMES


ROUND_ROOT = "/data/latent-basemap/runs/round-0132"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0132-*.md")

R0106_ROOT = "/data/latent-basemap/runs/round-0106/queue-attempt-3/artifacts"
FULL_GRAPH_MANIFEST = os.path.join(
    R0106_ROOT, "canonical-fuzzy-graph", "graph-manifest.json"
)
FULL_MAPPING = os.path.join(
    R0106_ROOT, "canonical-fuzzy-graph", "compact-to-global.i64.npy"
)
FULL_TRAIN_OUTPUT = (
    "/data/latent-basemap/runs/round-0107/queue/artifacts/"
    "train-diverse-jina-25m"
)
FULL_MODEL = os.path.join(FULL_TRAIN_OUTPUT, "model.pt")
R0108_ROOT = "/data/latent-basemap/runs/round-0108/queue-attempt-3"
R0108_QUEUE = os.path.join(R0108_ROOT, "queue.json")
R0108_TERMINAL = os.path.join(R0108_ROOT, "runner-terminal.json")
R0108_SELECTION = os.path.join(R0108_ROOT, "inputs", "registered-selections.npz")
R0108_CALIBRATION = os.path.join(
    R0108_ROOT,
    "artifacts",
    "jina-density-calibration",
    "jina-density-calibration.json",
)
R0108_TRANSFORM = os.path.join(R0108_ROOT, "artifacts", "coordinates")
R0108_TRANSFORM_RECEIPT = os.path.join(R0108_TRANSFORM, "actual-transform.json")

GPU_HOURS_MINIMUM = 1.8
GPU_HOURS_EXPECTED = 2.3
GPU_HOURS_P90 = 3.1
GPU_HOURS_MAXIMUM = 4.5
P90_GPU_TOTAL_SECONDS = 11_160.0

P90_NODE_SECONDS = {
    "build_half_search_index": 180.0,
    "qualify_fixed_search": 300.0,
    "graph_part": 450.0,
    "train_half_map": 8_000.0,
    "transform_half_map": 300.0,
    "score_matched_native": 600.0,
    "score_matched_ood": 430.0,
}

REVIEW_DEFAULTS = {
    "0087": (
        "review-0087-2026-07-28.md",
        "61ab9268899c2edc47519bdbe4efeea65a54f0c9fda52bd89e7cad0dafd9d483",
        "capability:jina-diverse-25m-inventory-v1",
    ),
    "0103": (
        "review-0103-2026-07-29.md",
        "c6c4f780c20cc34c7707132581ffaaf8daa8cc7ea9eb1cee3f76e128b6c37a51",
        "capability:jina-diverse-25m-full768-int8-substrate-v1",
    ),
    "0105": (
        "review-0105-2026-07-29.md",
        "084722e2641667333a673a8d9473da11d0aee1f97bca59e2ed646499e4169b96",
        "capability:jina-diverse-25m-full768-search-qualified-v1",
    ),
    "0106": (
        "review-0106-2026-07-29.md",
        "f00a8391cc47f038993b40337cbe71e07536d305015597ea2e39eed9ca116e1f",
        "capability:jina-diverse-25m-full768-fuzzy-graph-v1",
    ),
    "0107": (
        "review-0107-2026-07-30.md",
        "efac370df53f11cd50a3aad4fe8e18c9683bc84faa34ba783f5a342fc00a17ba",
        "capability:jina-diverse-25m-full768-trained-map-seed42-v1",
    ),
    "0108": (
        "review-0108-2026-07-30.md",
        "5ad9fbcf9307552862cff32ae7a86b771cb88f395c71b19f8f9c5b486dc476ee",
        "capability:jina-diverse-25m-map-registry-v1",
    ),
    "0118": (
        "review-0118-2026-07-31.md",
        "a52707eebbeecba739bc8bc60bd36f0e02ef29ff5f74028c8d721fcbef0fe103",
        "capability:jina-diverse-25m-seed44-map-registry-v1",
    ),
    "0119": (
        "review-0119-2026-07-31.md",
        "4b614d633aab4e09c98edcebbca57c724ef74175e609ab1c4ad78de66cafa81c",
        "capability:jina-density-failure-localization-v1",
    ),
}


def _frontmatter(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.startswith("---\n"):
        raise RuntimeError(f"missing frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RuntimeError(f"unterminated frontmatter: {path}")
    output: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            output[key.strip()] = value.strip().strip("\"'")
    return output


def _require_issued_round() -> str:
    candidates = [
        path
        for path in sorted(glob.glob(ROUND_FILE_GLOB))
        if _frontmatter(path).get("status") == "issued"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0132 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _require_round_release(round_file: str, release_sha: str) -> None:
    if _frontmatter(round_file).get("base_commit") != release_sha:
        raise RuntimeError(
            "R0132 issued round base_commit must equal the materialized release SHA"
        )


def _require_review(
    path: str,
    *,
    round_id: str,
    expected_sha256: str,
    capability: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    frontmatter = _frontmatter(path)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if (
        signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != round_id
        or frontmatter.get("status") != "accepted"
        or capability not in text
    ):
        raise RuntimeError(f"Review {round_id} is not the required acceptance")
    return signature


def _require_clean_r0108() -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind only the accepted R0108 execution, with no later-round premise."""
    queue_signature = expected_input_signature(R0108_QUEUE)
    terminal_signature = expected_input_signature(R0108_TERMINAL)
    with open(R0108_QUEUE, encoding="utf-8") as handle:
        queue = json.load(handle)
    with open(R0108_TERMINAL, encoding="utf-8") as handle:
        terminal = json.load(handle)
    if (
        queue.get("round_id") != "0108"
        or terminal.get("round_id") != "0108"
        or terminal.get("verdict") != "succeeded"
        or terminal.get("completed_jobs") != terminal.get("required_jobs")
        or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
        or terminal.get("queue_manifest_sha256_at_finish")
        != queue_signature["sha256"]
        or terminal.get("queue_manifest_unchanged") is not True
        or terminal.get("release_checkout_unchanged") is not True
    ):
        raise RuntimeError("R0108 is not a clean accepted execution")
    return queue_signature, terminal_signature


def _accepted_transform_signatures(
    transform_root: str = R0108_TRANSFORM,
    *,
    expected_rows: int = FULL_RETAINED_ROWS,
    expected_members: int = 5,
) -> list[dict[str, Any]]:
    """Authenticate and bind the receipt plus every coordinate-stream member."""
    receipt_path = os.path.join(transform_root, "actual-transform.json")
    try:
        receipt_signature = expected_input_signature(receipt_path)
        stream = CoordinateStream(
            transform_root,
            expected_receipt_sha256=receipt_signature["sha256"],
        )
    except (OSError, Round0036Error, ValueError) as exc:
        raise RuntimeError(
            "accepted R0108 coordinate stream failed authentication"
        ) from exc
    expected_paths = [
        os.path.realpath(
            os.path.join(
                transform_root,
                f"chunk-{index:05d}",
                "coordinates.npy",
            )
        )
        for index in range(expected_members)
    ]
    if (
        len(stream) != expected_rows
        or len(stream.shard_paths) != expected_members
        or [os.path.realpath(path) for path in stream.shard_paths]
        != expected_paths
    ):
        raise RuntimeError("accepted R0108 coordinate stream coverage changed")
    return _dedupe([
        receipt_signature,
        *(expected_input_signature(path) for path in stream.shard_paths),
    ])


def _accepted_control_signatures(
    *,
    full_graph_manifest: str = FULL_GRAPH_MANIFEST,
    full_mapping: str = FULL_MAPPING,
    full_train_output: str = FULL_TRAIN_OUTPUT,
    selection_path: str = R0108_SELECTION,
    calibration_path: str = R0108_CALIBRATION,
    transform_root: str = R0108_TRANSFORM,
    expected_transform_rows: int = FULL_RETAINED_ROWS,
    expected_transform_members: int = 5,
) -> list[dict[str, Any]]:
    """Return every immutable accepted-25M control input for queue binding."""
    direct_paths = [
        full_graph_manifest,
        full_mapping,
        os.path.join(full_train_output, "train-receipt.json"),
        os.path.join(full_train_output, "production-config.json"),
        os.path.join(full_train_output, "model.pt"),
        selection_path,
        calibration_path,
    ]
    try:
        direct = [expected_input_signature(path) for path in direct_paths]
    except (OSError, ValueError) as exc:
        raise RuntimeError("accepted 25M control input is missing or invalid") from exc
    transform = _accepted_transform_signatures(
        transform_root,
        expected_rows=expected_transform_rows,
        expected_members=expected_transform_members,
    )
    return _dedupe([*direct, *transform])


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
        "handler_module": "experiments.round0132_nodes",
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


def prepare_round0132(
    *,
    release_sha: str,
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0132 release SHA must be one full commit")
    round_file = _require_issued_round()
    _require_round_release(round_file, release_sha)
    reviews = {
        round_id: _require_review(
            os.path.join(LAB_ROOT, name),
            round_id=round_id,
            expected_sha256=digest,
            capability=capability,
        )
        for round_id, (name, digest, capability) in REVIEW_DEFAULTS.items()
    }
    assert_no_conditional_branch_dependency(tuple(reviews))

    r0108_queue_signature, r0108_terminal_signature = _require_clean_r0108()
    if os.path.realpath(r0108_queue_signature["canonical_path"]) != os.path.realpath(
        R0108_QUEUE
    ):
        raise RuntimeError("accepted R0108 queue path changed")
    with open(R0108_QUEUE, encoding="utf-8") as handle:
        r0108_queue = json.load(handle)
    source_jobs = {str(job["action"]): job for job in r0108_queue["jobs"]}
    ood_source = source_jobs["score_ood"]

    queue_root = create_fresh_directory(
        queue_root, label="R0132 matched 12.5M-to-25M scale-policy queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    substrate = validate_substrate_manifest(verify_payloads=False)
    eligibility = expected_input_signature(ELIGIBILITY_PATH)
    common = _dedupe([
        expected_input_signature(round_file),
        *reviews.values(),
        r0108_queue_signature,
        r0108_terminal_signature,
        substrate["signature"],
        substrate["payloads"]["int8"],
        substrate["payloads"]["scales"],
        substrate["payloads"]["labels"],
        eligibility,
    ])

    full_graph = expected_input_signature(FULL_GRAPH_MANIFEST)
    full_mapping = expected_input_signature(FULL_MAPPING)
    selection = expected_input_signature(R0108_SELECTION)
    accepted_controls = _accepted_control_signatures()
    full_transform_receipt = next(
        signature
        for signature in accepted_controls
        if os.path.realpath(signature["canonical_path"])
        == os.path.realpath(R0108_TRANSFORM_RECEIPT)
    )

    subset_output = os.path.join(artifacts, "half-subset")
    index_output = os.path.join(artifacts, "half-search-index")
    index_path = os.path.join(index_output, "jina-diverse-12p5m.ivfpq")
    qualification_output = os.path.join(artifacts, "half-search-qualification")
    part_outputs = {
        part: os.path.join(artifacts, f"half-graph-part-{part}")
        for part in GRAPH_PART_NAMES
    }
    graph_output = os.path.join(artifacts, "half-fuzzy-graph")
    graph_manifest = os.path.join(graph_output, "graph-manifest.json")
    train_output = os.path.join(artifacts, "train-half-seed42")
    transform_output = os.path.join(artifacts, "half-coordinates")
    native_output = os.path.join(artifacts, "matched-native")
    ood_output = os.path.join(artifacts, "matched-ood")
    decision_output = os.path.join(artifacts, "decision")

    subset_job = _job(
        node_id="select_half_subset",
        action="select_half_subset",
        deps=[],
        output=subset_output,
        expected_inputs=common,
        p90_wall_s=1_200.0,
        gpu=False,
        eligibility_sha256=eligibility["sha256"],
    )
    index_job = _job(
        node_id="build_half_search_index",
        action="build_half_search_index",
        deps=["select_half_subset"],
        output=index_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["build_half_search_index"],
        gpu=True,
        subset_output=subset_output,
    )
    qualification_job = _job(
        node_id="qualify_fixed_search",
        action="qualify_fixed_search",
        deps=["build_half_search_index"],
        output=qualification_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["qualify_fixed_search"],
        gpu=True,
        subset_output=subset_output,
        index_output=index_output,
        index=index_path,
    )
    graph_jobs: list[dict[str, Any]] = []
    for part in GRAPH_PART_NAMES:
        graph_jobs.append(_job(
            node_id=f"build_half_graph_part_{part}",
            action="build_half_graph_part",
            deps=["qualify_fixed_search"],
            output=part_outputs[part],
            expected_inputs=common,
            p90_wall_s=P90_NODE_SECONDS["graph_part"],
            gpu=True,
            part=part,
            subset_output=subset_output,
            index_output=index_output,
            index=index_path,
            qualification_output=qualification_output,
        ))
    graph_job_ids = [str(job["id"]) for job in graph_jobs]
    assemble_job = _job(
        node_id="assemble_half_graph",
        action="assemble_half_graph",
        deps=graph_job_ids,
        output=graph_output,
        expected_inputs=common,
        p90_wall_s=1_200.0,
        gpu=False,
        subset_output=subset_output,
        part_outputs=part_outputs,
    )
    train_job = _job(
        node_id="train_half_map",
        action="train_half_map",
        deps=["assemble_half_graph"],
        output=train_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["train_half_map"],
        gpu=True,
        training=True,
        release_sha=release_sha,
        graph_release_sha=release_sha,
        graph_manifest=graph_manifest,
        graph_manifest_late_bound_from="assemble_half_graph",
    )
    transform_job = _job(
        node_id="transform_half_map",
        action="transform_half_map",
        deps=["train_half_map"],
        output=transform_output,
        expected_inputs=common,
        p90_wall_s=P90_NODE_SECONDS["transform_half_map"],
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
    )
    native_job = _job(
        node_id="score_matched_native",
        action="score_matched_native",
        deps=["transform_half_map"],
        output=native_output,
        expected_inputs=[*common, *accepted_controls],
        p90_wall_s=P90_NODE_SECONDS["score_matched_native"],
        gpu=True,
        subset_output=subset_output,
        train_output=train_output,
        graph_manifest=graph_manifest,
        transform_output=transform_output,
        full_transform_output=R0108_TRANSFORM,
        full_transform_receipt_sha256=full_transform_receipt["sha256"],
        full_mapping=FULL_MAPPING,
        full_mapping_sha256=full_mapping["sha256"],
        eligibility=ELIGIBILITY_PATH,
        stale_calibration=R0108_CALIBRATION,
    )
    ood_inputs = _dedupe([
        *common,
        *accepted_controls,
        *ood_source["language_sources"].values(),
        *ood_source["diagnostic_sources"].values(),
    ])
    ood_job = _job(
        node_id="score_matched_ood",
        action="score_matched_ood",
        deps=["train_half_map"],
        output=ood_output,
        expected_inputs=ood_inputs,
        p90_wall_s=P90_NODE_SECONDS["score_matched_ood"],
        gpu=True,
        train_output=train_output,
        graph_manifest=graph_manifest,
        full_train_output=FULL_TRAIN_OUTPUT,
        full_graph_manifest=FULL_GRAPH_MANIFEST,
        full_graph_manifest_sha256=full_graph["sha256"],
        selection=R0108_SELECTION,
        selection_sha256=selection["sha256"],
        language_sources=ood_source["language_sources"],
        diagnostic_sources=ood_source["diagnostic_sources"],
    )
    decision_job = _job(
        node_id="decide_scale_policy",
        action="decide_scale_policy",
        deps=["score_matched_native", "score_matched_ood"],
        output=decision_output,
        expected_inputs=common,
        p90_wall_s=60.0,
        gpu=False,
        native_output=native_output,
        ood_output=ood_output,
        train_output=train_output,
        graph_manifest=graph_manifest,
    )
    jobs = [
        subset_job,
        index_job,
        qualification_job,
        *graph_jobs,
        assemble_job,
        train_job,
        transform_job,
        native_job,
        ood_job,
        decision_job,
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue["schema"] = "round0132-matched-scale-policy-queue-v1"
    queue["repo_root"] = RELEASE_ROOT
    queue["queue_class"] = "gpu-research"
    queue["required_reviews"] = list(REVIEW_DEFAULTS)
    queue["capability_dependencies"] = [
        "jina-diverse-25m-inventory-v1",
        "jina-diverse-25m-full768-int8-substrate-v1",
        "jina-diverse-25m-full768-search-qualified-v1",
        "jina-diverse-25m-full768-fuzzy-graph-v1",
        "jina-diverse-25m-full768-trained-map-seed42-v1",
        "jina-diverse-25m-map-registry-v1",
        "jina-diverse-25m-seed44-map-registry-v1",
        "jina-density-failure-localization-v1",
    ]
    queue["capabilities_produced"] = [
        "jina-diverse-12p5m-25m-scale-policy-geometry-v1"
    ]
    queue["training_performed"] = True
    queue["scientific_contract"] = {
        "causal_question": (
            "under one frozen raw-Jina graph/training policy, does scaling a "
            "source-proportional duplicate-controlled population from "
            "12,474,331 to 24,948,663 degrade matched geometry or OOD recall?"
        ),
        "estimand": (
            "scale-policy bundle: population plus induced graph plus "
            "coverage-aligned horizon; explicitly not a pure-N effect"
        ),
        "subset": {
            "rows": HALF_RETAINED_ROWS,
            "groups": 22,
            "quota": "integer largest remainder; GROUPS order tie break",
            "within_group": "lowest full SHA-256 rank",
            "namespace_hex": SUBSET_NAMESPACE.hex(),
            "prefix_selection": False,
        },
        "search": {
            "nprobe": SEARCH_NPROBE,
            "shortlist_width": SEARCH_SHORTLIST_WIDTH,
            "anchors_per_group": SEARCH_ANCHORS_PER_GROUP,
            "global_recall_floor": SEARCH_GLOBAL_RECALL_FLOOR,
            "every_group_recall_floor": SEARCH_GROUP_RECALL_FLOOR,
            "widen_or_sweep_after_failure": False,
        },
        "graph": {
            "k_nonself": GRAPH_K,
            "fresh_subset_graph": True,
            "resumable_parts": list(GRAPH_PART_NAMES),
            "fuzzy_and_tconorm_semantics": "exact R0106",
        },
        "training": {
            "seed": 42,
            "successful_updates": "ceil(actual directed fuzzy edges / 409)",
            "horizon_computed_before_launch": True,
            "coverage_aligned": True,
        },
        "matched_native": {
            "universe": "exact U12 for both maps",
            "anchors_per_group": NATIVE_ANCHORS_PER_GROUP,
            "anchor_seed": NATIVE_ANCHOR_SEED,
            "density_bootstrap_draws": DENSITY_BOOTSTRAP_DRAWS,
            "density_bootstrap_seed": DENSITY_BOOTSTRAP_SEED,
            "density_ci_level": DENSITY_CI_LEVEL,
            "density_noninferiority_margin": DENSITY_NONINFERIORITY_MARGIN,
            "density_comparison_atol": DENSITY_COMPARISON_ATOL,
            "native_global_ffr": "registered noninferiority gate",
            "ffr_allowed_decrease": FFR_ALLOWED_DECREASE,
            "recall_retention": METRIC_RETENTION,
        },
        "matched_ood": {
            "gates": [
                "FineWeb recall50 retains 0.97",
                "Polish recall50 retains 0.97",
                "median 19 in-mix language recall50 retains 0.97",
            ],
            "ood_projection_ffr": "diagnostic-only",
            "trec-covid": "diagnostic-only",
            "dadabase": "diagnostic-only",
        },
        "stale_absolute_jina_floor": "diagnostic-only",
        "outcomes": [
            OUTCOME_SUPPORTED,
            OUTCOME_DENSITY_REGRESSION,
            OUTCOME_QUALITY_REGRESSION,
            OUTCOME_INCONCLUSIVE,
            OUTCOME_INVALID,
        ],
        "decision_schema": DECISION_SCHEMA,
        "atlas_quality_or_production_claim": False,
        "one_seed_limitation": (
            "seed-42 matched contrast only; no seed-variance robustness claim"
        ),
        "conditional_branch_evidence_is_dependency": False,
    }
    queue["gpu_hours"] = {
        "minimum": GPU_HOURS_MINIMUM,
        "expected": GPU_HOURS_EXPECTED,
        "p90": GPU_HOURS_P90,
        "maximum": GPU_HOURS_MAXIMUM,
    }
    queue["p90_gpu_seconds"] = {
        **P90_NODE_SECONDS,
        "graph_parts_total": len(GRAPH_PART_NAMES) * P90_NODE_SECONDS["graph_part"],
        "total": P90_GPU_TOTAL_SECONDS,
    }
    queue["jobs"] = jobs
    assert_no_conditional_branch_dependency(queue["required_reviews"])
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(prepare_round0132(release_sha=args.release_sha, queue_root=args.queue_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
