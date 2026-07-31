#!/usr/bin/env python3
"""Prepare, but never launch, conditional no-training R0123."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0108_evaluation import validate_seal
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
from experiments.round0123_nodes import (
    ASSEMBLY_SHA256,
    COMPACT_ROWS,
    DECISION_SCHEMA,
    DIMENSION,
    FRESH_HIGH_D_REFERENCE_SHA256,
    FRESH_INPUT_SHA256,
    FRESH_MODEL_SHA256,
    LEGACY_MODEL_SHA256,
    MAPPING_SHA256,
    PANEL_SCHEMA,
    R0104_SOURCE_PAYLOAD_SHA256,
    R0115_RESULT_SHA256,
    R0115_REVIEW_SHA256,
    R0122_RELEASE_SHA,
    R0122_REQUIRED_OUTCOME,
)
from experiments.round0122_nodes import (
    DECISION_SCHEMA as R0122_DECISION_SCHEMA,
    R0115_NATIVE_HIGH_D_REFERENCE_KEY,
    SCORE_SCHEMA as R0122_SCORE_SCHEMA,
)


ROUND_ID = "0123"
ROUND_ROOT = "/data/latent-basemap/runs/round-0123"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
RUN_ENVIRONMENT_PREFIX = os.path.join(RELEASE_ROOT, ".venv")
RUN_PYTHON = os.path.join(RUN_ENVIRONMENT_PREFIX, "bin", "python")
ROUND_FILE_GLOB = os.path.join(LAB_ROOT, "round-0123-*.md")

R0122_QUEUE = "/data/latent-basemap/runs/round-0122/queue/queue.json"
R0122_TERMINAL = (
    "/data/latent-basemap/runs/round-0122/queue/runner-terminal.json"
)
R0122_SCORE = (
    "/data/latent-basemap/runs/round-0122/queue/artifacts/"
    "density-provenance-bridge/density-bridge-panel.json"
)
R0122_DECISION = (
    "/data/latent-basemap/runs/round-0122/queue/artifacts/"
    "density-provenance-bridge-decision/density-bridge-decision.json"
)
R0122_CAPABILITY = (
    "capability:jina-density-provenance-representation-bridge-v1"
)

ASSEMBLY_MANIFEST = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/"
    "compact-arrays/assembly-manifest.json"
)
COMPACT_MAPPING = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/"
    "compact-arrays/compact-to-global.i64.npy"
)
FRESH_INPUT = (
    "/data/latent-basemap/runs/round-0113/queue/artifacts/"
    "compact-arrays/raw-compact.f16"
)
FRESH_HIGH_D_REFERENCE = (
    "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
    "raw/graph/high-d-reference.npz"
)


def _require_dedicated_run_environment() -> None:
    """Fail before discovery or writes unless preparation uses the run venv."""
    observed_python = os.path.abspath(sys.executable)
    observed_prefix = os.path.abspath(sys.prefix)
    if (
        observed_python != RUN_PYTHON
        or observed_prefix != RUN_ENVIRONMENT_PREFIX
    ):
        raise RuntimeError(
            "R0123 queue preparation must use the dedicated run "
            f"environment: python={RUN_PYTHON}, "
            f"prefix={RUN_ENVIRONMENT_PREFIX}; observed "
            f"python={observed_python}, prefix={observed_prefix}"
        )


def _issued_round() -> str:
    candidates = []
    for path in sorted(glob.glob(ROUND_FILE_GLOB)):
        frontmatter, _ = _document(path)
        if frontmatter.get("status") == "issued":
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"R0123 requires exactly one issued round; found {len(candidates)}"
        )
    return candidates[0]


def _accepted_r0122(
    path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    review_signature = expected_input_signature(path)
    frontmatter, review_text = _document(path)
    if (
        review_signature["sha256"] != expected_sha256
        or frontmatter.get("round_id") != "0122"
        or frontmatter.get("status") != "accepted"
        or frontmatter.get("verified_release_commit") != R0122_RELEASE_SHA
        or R0122_CAPABILITY
        not in _frontmatter_list(
            frontmatter, "releases", label="R0122 review"
        )
    ):
        raise RuntimeError("R0122 review is not exact and accepted")

    result_name = frontmatter.get("result") or ""
    if (
        os.path.basename(result_name) != result_name
        or re.fullmatch(
            r"result-0122-[0-9]{4}-[0-9]{2}-[0-9]{2}\.md",
            result_name,
        )
        is None
    ):
        raise RuntimeError("R0122 review result binding is invalid")
    result_path = os.path.join(os.path.dirname(path), result_name)
    result_signature = expected_input_signature(result_path)
    result_frontmatter, result_text = _document(result_path)
    queue_signature = expected_input_signature(R0122_QUEUE)
    if (
        result_signature["sha256"] != frontmatter.get("result_sha256")
        or result_frontmatter.get("round_id") != "0122"
        or result_frontmatter.get("status") != "complete"
        or result_frontmatter.get("release_commit") != R0122_RELEASE_SHA
        or result_frontmatter.get("queue_manifest")
        != f"gsv:{R0122_QUEUE}"
        or result_frontmatter.get("queue_manifest_sha256")
        != queue_signature["sha256"]
        or R0122_CAPABILITY.removeprefix("capability:")
        not in _frontmatter_list(
            result_frontmatter,
            "capabilities_produced",
            label="R0122 result",
        )
    ):
        raise RuntimeError("R0122 accepted review does not close its result")

    runtime = _clean_terminal(
        R0122_QUEUE,
        R0122_TERMINAL,
        round_id="0122",
        expected_release_sha=R0122_RELEASE_SHA,
    )
    score_signature = expected_input_signature(R0122_SCORE)
    decision_signature = expected_input_signature(R0122_DECISION)
    evidence_text = review_text + "\n" + result_text
    for label, signature in (
        ("queue", queue_signature),
        ("terminal", runtime["terminal"]),
        ("score", score_signature),
        ("decision", decision_signature),
    ):
        if signature["sha256"] not in evidence_text:
            raise RuntimeError(
                f"R0122 accepted evidence does not bind {label}"
            )

    with open(R0122_SCORE, encoding="utf-8") as handle:
        score = json.load(handle)
    with open(R0122_DECISION, encoding="utf-8") as handle:
        decision = json.load(handle)
    validate_seal(score, label="R0122 density provenance bridge panel")
    validate_seal(decision, label="R0122 density provenance bridge decision")
    if (
        score.get("schema") != R0122_SCORE_SCHEMA
        or score.get("round_id") != "0122"
        or score.get("release_sha") != R0122_RELEASE_SHA
        or score.get("training_performed") is not False
        or decision.get("schema") != R0122_DECISION_SCHEMA
        or decision.get("round_id") != "0122"
        or decision.get("release_sha") != R0122_RELEASE_SHA
        or decision.get("score") != score_signature
        or decision.get("outcome") != R0122_REQUIRED_OUTCOME
        or decision.get("evaluation_path_material") is not False
        or decision.get("boundary_localized") is not True
        or decision.get("single_factor_cause_localized") is not False
        or decision.get("native_training_geometry_declared_bad") is not False
        or decision.get("production_transfer_claimed") is not False
        or decision.get("training_performed") is not False
    ):
        raise RuntimeError(
            "R0122 accepted decision does not release R0123's branch"
        )
    return {
        "review": review_signature,
        "result": result_signature,
        "queue": queue_signature,
        "terminal": runtime["terminal"],
        "score": score_signature,
        "decision": decision_signature,
        "evidence_text": evidence_text,
    }


def _exact_signature(
    signature: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    actual = expected_input_signature(
        str(signature.get("canonical_path") or "")
    )
    if actual != dict(signature):
        raise RuntimeError(f"{label} bytes changed")
    return actual


def _queue_lineage(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    with open(R0122_QUEUE, encoding="utf-8") as handle:
        queue = json.load(handle)
    score_jobs = [
        job
        for job in queue.get("jobs") or []
        if job.get("action") == "score_density_provenance_bridge"
    ]
    if (
        queue.get("schema")
        != "round0122-jina-density-provenance-bridge-queue-v1"
        or queue.get("round_id") != "0122"
        or queue.get("release_sha") != R0122_RELEASE_SHA
        or queue.get("training_performed") is not False
        or len(score_jobs) != 1
    ):
        raise RuntimeError("R0122 accepted queue structure changed")
    score_job = score_jobs[0]
    r0104 = score_job.get("r0104_model_bundles")
    replay = score_job.get("r0119_replay_model_bundles")
    if (
        not isinstance(r0104, list)
        or len(r0104) != 2
        or r0104[0].get("key")
        != "r0104_fp16_seed42_full_transform"
        or (r0104[0].get("model") or {}).get("sha256")
        != LEGACY_MODEL_SHA256
        or not isinstance(replay, list)
        or len(replay) != 2
        or replay[0].get("key") != "current_2m_seed42"
        or replay[0].get("group") != "current_2m"
        or replay[0].get("arm") != "raw"
        or replay[0].get("seed") != 42
        or (replay[0].get("model") or {}).get("sha256")
        != FRESH_MODEL_SHA256
        or (replay[0].get("accepted_review") or {}).get("sha256")
        != R0115_REVIEW_SHA256
        or (replay[0].get("accepted_result") or {}).get("sha256")
        != R0115_RESULT_SHA256
    ):
        raise RuntimeError("R0122 crossed-map source bundles changed")
    legacy_bundle = dict(r0104[0])
    fresh_bundle = dict(replay[0])
    for label, bundle in (
        ("R0104 legacy map", legacy_bundle),
        ("R0115 fresh map", fresh_bundle),
    ):
        for field in ("train_receipt", "production_config", "model"):
            _exact_signature(bundle[field], label=f"{label} {field}")
    for label, field in (
        ("R0115 accepted review", "accepted_review"),
        ("R0115 accepted result", "accepted_result"),
    ):
        _exact_signature(fresh_bundle[field], label=label)

    legacy_train_path = legacy_bundle["train_receipt"]["canonical_path"]
    with open(legacy_train_path, encoding="utf-8") as handle:
        legacy_train = json.load(handle)
    validate_seal(legacy_train, label="R0104 fp16 train receipt")
    shared_signature = _exact_signature(
        legacy_train["shared_evidence"],
        label="R0104 shared evidence",
    )
    with open(
        shared_signature["canonical_path"], encoding="utf-8"
    ) as handle:
        shared = json.load(handle)
    validate_seal(shared, label="R0104 shared evidence")
    source_proof = shared.get("source_prefix_proof")
    if (
        shared.get("schema") != "round0104-paired-shared-evidence-v2"
        or not isinstance(source_proof, Mapping)
        or source_proof.get("schema")
        != "round0104-r0103-first2m-source-proof-v2"
        or source_proof.get("rows") != 2_000_000
        or source_proof.get("dimension") != DIMENSION
        or source_proof.get("dtype") != "<f2"
        or source_proof.get("payload_sha256")
        != R0104_SOURCE_PAYLOAD_SHA256
        or source_proof.get("cross_round_row_equivalence_claimed") is not False
        or source_proof.get("segments")
        != (legacy_train.get("exact_execution_receipt") or {}).get(
            "source_segments"
        )
    ):
        raise RuntimeError("R0104 exact source-prefix lineage changed")
    source_segments = source_proof.get("segments")
    if not isinstance(source_segments, list) or not source_segments:
        raise RuntimeError("R0104 exact source segments are missing")
    source_shards = [
        _exact_signature(
            segment["shard"],
            label=f"R0104 source segment {index}",
        )
        for index, segment in enumerate(source_segments)
    ]

    assembly_signature = expected_input_signature(ASSEMBLY_MANIFEST)
    mapping_signature = expected_input_signature(COMPACT_MAPPING)
    fresh_input_signature = expected_input_signature(FRESH_INPUT)
    fresh_reference_signature = expected_input_signature(
        FRESH_HIGH_D_REFERENCE
    )
    if (
        assembly_signature["sha256"] != ASSEMBLY_SHA256
        or mapping_signature["sha256"] != MAPPING_SHA256
        or fresh_input_signature["sha256"] != FRESH_INPUT_SHA256
        or fresh_reference_signature["sha256"]
        != FRESH_HIGH_D_REFERENCE_SHA256
        or score_job.get("r0115_native_high_d_reference")
        != fresh_reference_signature
    ):
        raise RuntimeError("R0113/R0115 compact input lineage changed")
    with open(ASSEMBLY_MANIFEST, encoding="utf-8") as handle:
        assembly = json.load(handle)
    validate_seal(assembly, label="R0113 compact assembly")
    if (
        assembly.get("schema")
        != "round0113-compact-prompt-arrays-v1"
        or assembly.get("round_id") != "0113"
        or assembly.get("retained_rows") != COMPACT_ROWS
        or assembly.get("dimension") != DIMENSION
        or assembly.get("mapping") != mapping_signature
        or (assembly.get("outputs") or {}).get("raw")
        != fresh_input_signature
        or assembly.get("paired_row_population_identical") is not True
    ):
        raise RuntimeError("R0113 compact assembly semantics changed")
    with open(
        fresh_bundle["train_receipt"]["canonical_path"], encoding="utf-8"
    ) as handle:
        fresh_train = json.load(handle)
    validate_seal(fresh_train, label="R0115 raw train receipt")
    if fresh_train.get("assembly") != assembly_signature:
        raise RuntimeError("R0115 raw map no longer binds the compact assembly")

    with np.load(FRESH_HIGH_D_REFERENCE, allow_pickle=False) as archive:
        reference_key = str(np.asarray(archive["key"]).item())
        anchor_count = len(np.asarray(archive["anchor_ids"]))
    if (
        reference_key != R0115_NATIVE_HIGH_D_REFERENCE_KEY
        or anchor_count != 4_000
    ):
        raise RuntimeError("R0115 native reference identity changed")

    evidence_text = str(evidence["evidence_text"])
    for label, signature in (
        ("R0104 legacy train receipt", legacy_bundle["train_receipt"]),
        ("R0104 legacy model", legacy_bundle["model"]),
        ("R0115 fresh train receipt", fresh_bundle["train_receipt"]),
        ("R0115 fresh model", fresh_bundle["model"]),
        ("R0115 native high-D reference", fresh_reference_signature),
    ):
        # The accepted R0122 result/review binds all selected map/reference
        # evidence. Production configs are transitively sealed by train receipts.
        if signature["sha256"] not in evidence_text:
            raise RuntimeError(
                f"R0122 accepted evidence does not bind {label}"
            )
    return {
        "legacy_model_bundle": legacy_bundle,
        "fresh_model_bundle": fresh_bundle,
        "r0104_shared_evidence": shared_signature,
        "r0104_source_segments": source_segments,
        "source_shards": source_shards,
        "assembly_manifest": assembly_signature,
        "compact_mapping": mapping_signature,
        "fresh_input": fresh_input_signature,
        "fresh_high_d_reference": fresh_reference_signature,
        "r0115_accepted_review": dict(fresh_bundle["accepted_review"]),
        "r0115_accepted_result": dict(fresh_bundle["accepted_result"]),
    }


def prepare_round0123(
    *,
    release_sha: str,
    r0122_review: tuple[str, str],
    queue_root: str = os.path.join(ROUND_ROOT, "queue"),
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
        raise ValueError("R0123 release SHA must be one full commit")

    # This is intentionally first. A wrong interpreter must fail before round
    # discovery and, most importantly, before any queue directory is created.
    _require_dedicated_run_environment()
    round_file = _issued_round()
    evidence = _accepted_r0122(
        r0122_review[0], r0122_review[1]
    )
    lineage = _queue_lineage(evidence)
    common_inputs = _dedupe([
        expected_input_signature(round_file),
        *[
            dict(evidence[field])
            for field in (
                "review",
                "result",
                "queue",
                "terminal",
                "score",
                "decision",
            )
        ],
        *[
            dict(lineage["legacy_model_bundle"][field])
            for field in ("train_receipt", "production_config", "model")
        ],
        *[
            dict(lineage["fresh_model_bundle"][field])
            for field in ("train_receipt", "production_config", "model")
        ],
        lineage["r0104_shared_evidence"],
        *lineage["source_shards"],
        lineage["assembly_manifest"],
        lineage["compact_mapping"],
        lineage["fresh_input"],
        lineage["fresh_high_d_reference"],
        lineage["r0115_accepted_review"],
        lineage["r0115_accepted_result"],
    ])

    queue_root = create_fresh_directory(
        queue_root, label="R0123 crossed-representation queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    score_output = os.path.join(
        artifacts, "crossed-representation-density-panel"
    )
    decision_output = os.path.join(
        artifacts, "crossed-representation-alignment-decision"
    )
    jobs = [
        {
            "id": "score_crossed_representation_density",
            "action": "score_crossed_representation_density",
            "handler_module": "experiments.round0123_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "outputs": [score_output],
            "done_marker": os.path.join(
                artifacts,
                "score_crossed_representation_density.done.json",
            ),
            "expected_inputs": common_inputs,
            "p90_wall_s": 600.0,
            "node_policy": {
                "gpu_required": True,
                "training_performed": False,
            },
            "r0122_evidence": {
                field: dict(evidence[field])
                for field in (
                    "review",
                    "result",
                    "queue",
                    "terminal",
                    "score",
                    "decision",
                )
            },
            "legacy_model_bundle": lineage["legacy_model_bundle"],
            "fresh_model_bundle": lineage["fresh_model_bundle"],
            "r0104_shared_evidence": lineage[
                "r0104_shared_evidence"
            ],
            "r0104_source_segments": lineage["r0104_source_segments"],
            "assembly_manifest": lineage["assembly_manifest"],
            "compact_mapping": lineage["compact_mapping"],
            "fresh_input": lineage["fresh_input"],
            "fresh_high_d_reference": lineage[
                "fresh_high_d_reference"
            ],
        },
        {
            "id": "decide_crossed_representation_alignment",
            "action": "decide_crossed_representation_alignment",
            "handler_module": "experiments.round0123_nodes",
            "handler_callable": "run_job",
            "deps": ["score_crossed_representation_density"],
            "outputs": [decision_output],
            "done_marker": os.path.join(
                artifacts,
                "decide_crossed_representation_alignment.done.json",
            ),
            "expected_inputs": _dedupe([
                expected_input_signature(round_file),
                evidence["review"],
                evidence["result"],
                evidence["decision"],
            ]),
            "p90_wall_s": 60.0,
            "node_policy": {
                "gpu_required": False,
                "training_performed": False,
            },
            "score_output": score_output,
        },
    ]
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0123-crossed-representation-density-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-research",
        "required_reviews": ["0122"],
        "capability_dependencies": [
            "jina-density-provenance-representation-bridge-v1",
        ],
        "capabilities_produced": [
            "jina-density-crossed-representation-alignment-v1",
        ],
        "training_performed": False,
        "conditional_admission": {
            "round": "0122",
            "required_outcome": R0122_REQUIRED_OUTCOME,
            "accepted_review_required": True,
            "alternative_outcomes_refuse_queue_preparation": True,
        },
        "scientific_contract": {
            "purpose": (
                "test whether the R0104 and R0115 seed42 maps each retain "
                "higher density fidelity on their aligned embedding "
                "representation than on the crossed representation"
            ),
            "population": (
                "exact common R0113 1993761 compact FineWeb IDs and mapping"
            ),
            "maps": [
                "R0104 fp16 seed42",
                "R0115 raw seed42",
            ],
            "input_representations": [
                (
                    "R0104 first-2M source rows selected by the exact R0113 "
                    "compact-to-global mapping"
                ),
                "R0113 fresh local raw compact fp16",
            ],
            "cells": [
                "legacy_map__legacy_input",
                "legacy_map__fresh_input",
                "fresh_map__legacy_input",
                "fresh_map__fresh_input",
            ],
            "anchors": (
                "exact accepted R0115 4000 compact anchor IDs reused in "
                "all four cells"
            ),
            "high_d_reference_rule": (
                "score each cell against the high-D k15 radii belonging to "
                "that cell's input representation"
            ),
            "selector": (
                "paired shared-anchor bootstrap of each map's "
                "matched-minus-crossed density correlation; positive only "
                "when the central 99% interval is strictly above zero"
            ),
            "historical_absolute_floor_applied": False,
            "native_quality_claim": False,
            "single_factor_cause_claim": False,
            "production_transfer_claim": False,
            "training_performed": False,
        },
        "p90_gpu_seconds": {
            "score_crossed_representation_density": 600.0,
            "total": 600.0,
        },
        "estimate_basis": {
            "expected_gpu_hours": 0.05,
            "p90_gpu_hours": 0.17,
            "hard_cap_gpu_hours": 0.5,
            "basis": (
                "one exact legacy high-D k15 radius pass, four compact "
                "full-map transforms, and four exact 2D k15 radius passes; "
                "no graph build and no training"
            ),
        },
        "output_schemas": {
            "panel": PANEL_SCHEMA,
            "decision": DECISION_SCHEMA,
        },
        "jobs": jobs,
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--r0122-review", required=True)
    parser.add_argument("--r0122-review-sha256", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    path = prepare_round0123(
        release_sha=args.release_sha,
        r0122_review=(
            args.r0122_review,
            args.r0122_review_sha256,
        ),
        queue_root=args.queue_root,
    )
    print(json.dumps({"queue_manifest": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
