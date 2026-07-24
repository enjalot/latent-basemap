#!/usr/bin/env python3
"""Prepare, but never launch, the two-cell Round 0038 Jina completion."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_file,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0038_program import (
    CANARY_MINIMUM_UPDATES_PER_S,
    CELL_LABELS,
    CENTROIDS,
    DIMENSIONS,
    GRAPH_BYTES,
    GRAPH_PATH,
    GRAPH_SHA256,
    PREFIX_PAYLOAD_SHA256,
    QUERY_ROWS,
    TRAIN_BYTES,
    TRAIN_PATH,
    TRAIN_SHA256,
    graph_manifest_for_dimension,
    parse_cell,
    query_row_ids,
    train_config_for_cell,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0038"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0038-2026-07-24.md")
R0037_REVIEW_PATH = os.path.join(
    LAB_ROOT, "review-0037-2026-07-23.md")
R0037_DECISION_ROOT = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/decision"
)
R0037_DECISION_PATH = os.path.join(
    R0037_DECISION_ROOT, "mrl-seed42-screen-v1.json")
R0037_SHARED_ROOT = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts/shared-reference"
)
R0037_CELL_ROOT = (
    "/data/latent-basemap/runs/round-0037/queue/artifacts"
)

# Fixed predecessor artifacts are content-bound here rather than inferred from
# whatever happens to be present when a queue is materialized.
R0037_STATIC_ARTIFACTS = {
    R0037_REVIEW_PATH: (
        "8192d5478c63c1e961283c398370619144bfa97828aabcecbbd56ed7fbdb39a1",
        13_683,
    ),
    R0037_DECISION_PATH: (
        "874cfff88b56097cb8488913710f31436dd7655267d57c1474e57ad9c6d7ab5d",
        8_987,
    ),
    os.path.join(R0037_SHARED_ROOT, "receipt.json"): (
        "2213a8dbdd9f435aa57fa333de33f5066f62c717b7a6accbc666d0b9af3342ac",
        3_885,
    ),
    os.path.join(R0037_SHARED_ROOT, "high-d-reference.npz"): (
        "2ca06df954730c60c5810ee38567c40ca9040149c68d95f2b79ddc29d3f06dd6",
        176_971_608,
    ),
    os.path.join(R0037_SHARED_ROOT, "literal-prefix-proof.json"): (
        "5e09447679e9e4c27b2dfc47ced1ce58bd68dc2036924e40b02fa75873c0e432",
        1_163,
    ),
    os.path.join(R0037_SHARED_ROOT, "oos-query-embeddings-768.npy"): (
        "3d2998a047a6effd032def6a2f1fa5f9ef96af1ba3afd9153ab2508ca504259a",
        61_440_128,
    ),
    os.path.join(R0037_SHARED_ROOT, "oos-query-truth-k10.npz"): (
        "5e82db6faac8f9f3a8224e21b5df52989a6ca5469cfe628c5f48da907f297f04",
        1_612_980,
    ),
    os.path.join(
        R0037_CELL_ROOT, "d768_s42", "train", "train-receipt.json"): (
        "1f13e1bbc53acda62099ad0539c300b4fe9c192d9626e3e35848c061d806a674",
        7_064,
    ),
    os.path.join(
        R0037_CELL_ROOT, "d768_s42", "panel", "panel.json"): (
        "ce31e67541ea445eb9da35ffaeff1e6e5cac3da765f853586d1d4e0ed97d74a6",
        6_549,
    ),
    os.path.join(
        R0037_CELL_ROOT, "d384_s42", "train", "train-receipt.json"): (
        "647aa3e9dcf3dd9e91e427267674e2c8cf61c3c12c00781d182d5b096fcdda99",
        7_111,
    ),
    os.path.join(
        R0037_CELL_ROOT, "d384_s42", "panel", "panel.json"): (
        "ecbd83a77fe56a11d010a7ec145c15da02b01f41325041d48df0db63d95809b8",
        6_574,
    ),
}
HANDLER_MODULE = "experiments.round0038_nodes"
HANDLER_CALLABLES = {
    "round0038_sampler_canary": "run_sampler_canary",
    "round0038_train": "run_train",
    "round0038_transform": "run_transform",
    "round0038_score": "run_score",
    "round0038_decision": "run_decision",
}
GPU_P90_SECONDS = 11_280.0


def _require_issued_round(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError(f"Round 0038 frontmatter is missing: {path}")
    statuses: list[str] = []
    closed = False
    for line in lines[1:]:
        if line.strip() == "---":
            closed = True
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "status":
            statuses.append(value.strip().strip("\"'"))
    if not closed or len(statuses) != 1:
        raise RuntimeError(
            f"Round 0038 frontmatter has ambiguous status: {path}")
    if statuses[0] != "issued":
        raise RuntimeError(
            "Round 0038 queue materialization requires status: issued; "
            f"observed {statuses[0]!r}")


def _generated_inputs(
    queue_root: str,
) -> tuple[dict[int, dict[str, str]], str]:
    root = ensure_data_directory(os.path.join(queue_root, "inputs"))
    manifests: dict[int, dict[str, str]] = {}
    for dimension in DIMENSIONS:
        path = os.path.join(root, f"graph-manifest-d{dimension}.json")
        atomic_write_new_json(
            path, graph_manifest_for_dimension(dimension), immutable=True)
        manifests[dimension] = {
            "path": path,
            "sha256": sha256_file(path),
        }
    query_path = os.path.join(root, "oos-query-row-ids.npy")
    atomic_save_new_npy(query_path, query_row_ids(), immutable=True)
    return manifests, query_path


def _static_inputs(
    *,
    manifests: dict[int, dict[str, str]],
    query_ids_path: str,
) -> list[dict[str, Any]]:
    files = [
        ROUND_FILE,
        TRAIN_PATH,
        GRAPH_PATH,
        *(item["path"] for item in CENTROIDS.values()),
        *(item["path"] for item in manifests.values()),
        query_ids_path,
        *R0037_STATIC_ARTIFACTS,
    ]
    inputs = _dedupe(_file_inputs(files))
    observed = {item["canonical_path"]: item for item in inputs}
    expected = {
        TRAIN_PATH: (TRAIN_SHA256, TRAIN_BYTES),
        GRAPH_PATH: (GRAPH_SHA256, GRAPH_BYTES),
        **{
            item["path"]: (item["sha256"], item["bytes"])
            for item in CENTROIDS.values()
        },
        **R0037_STATIC_ARTIFACTS,
    }
    mismatches = {}
    for path, (sha, size) in expected.items():
        item = observed.get(os.path.realpath(path), {})
        if item.get("sha256") != sha or item.get("bytes") != size:
            mismatches[path] = {
                "expected_sha256": sha,
                "observed_sha256": item.get("sha256"),
                "expected_bytes": size,
                "observed_bytes": item.get("bytes"),
            }
    if mismatches:
        raise RuntimeError(f"Round 0038 static input tuple changed: {mismatches}")
    return inputs


def _jobs(
    *,
    artifacts: str,
    inputs: list[dict[str, Any]],
    manifests: dict[int, dict[str, str]],
    query_ids_path: str,
) -> list[dict[str, Any]]:
    canary = os.path.join(artifacts, "canary")
    cell_paths = {
        label: {
            "train": os.path.join(artifacts, label, "train"),
            "transform": os.path.join(artifacts, label, "transform"),
            "panel": os.path.join(artifacts, label, "panel"),
        }
        for label in CELL_LABELS
    }

    def job(
        node_id: str,
        handler: str,
        deps: list[str],
        output: str,
        p90: float,
        *,
        gpu: bool = True,
        training: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        if handler not in HANDLER_CALLABLES:
            raise ValueError(f"unknown Round 0038 handler: {handler}")
        return {
            "id": node_id,
            "handler": handler,
            "handler_module": HANDLER_MODULE,
            "handler_callable": HANDLER_CALLABLES[handler],
            "deps": deps,
            "done_marker": os.path.join(artifacts, f"{node_id}.done.json"),
            "outputs": [output],
            "expected_inputs": inputs,
            "p90_wall_s": float(p90),
            "node_policy": {
                "gpu_required": gpu,
                "training_performed": training,
            },
            **extra,
        }

    def json_payload(value: Any) -> Any:
        return json.loads(canonical_json(value))

    control = "d768_s43"
    control_manifest = manifests[768]
    control_config, control_sha = train_config_for_cell(
        control,
        graph_manifest_path=control_manifest["path"],
        graph_manifest_sha256=control_manifest["sha256"],
    )
    jobs: list[dict[str, Any]] = [
        job(
            "mrl_sampler_canary",
            "round0038_sampler_canary",
            [],
            canary,
            300.0,
            training=True,
            cell=control,
            graph_manifest_path=control_manifest["path"],
            graph_manifest_sha256=control_manifest["sha256"],
            production_config=json_payload(control_config),
            production_config_sha256=control_sha,
            minimum_updates_per_s=CANARY_MINIMUM_UPDATES_PER_S,
        )
    ]
    predecessor = "mrl_sampler_canary"
    train_p90 = {768: 5100.0, 384: 4800.0}
    for label in CELL_LABELS:
        dimension, _ = parse_cell(label)
        graph_manifest = manifests[dimension]
        config, config_sha = train_config_for_cell(
            label,
            graph_manifest_path=graph_manifest["path"],
            graph_manifest_sha256=graph_manifest["sha256"],
        )
        common = {
            "cell": label,
            "graph_manifest_path": graph_manifest["path"],
            "graph_manifest_sha256": graph_manifest["sha256"],
            "production_config": json_payload(config),
            "production_config_sha256": config_sha,
        }
        train_id = f"train_{label}"
        transform_id = f"transform_{label}"
        score_id = f"score_{label}"
        jobs.extend([
            job(
                train_id,
                "round0038_train",
                [predecessor],
                cell_paths[label]["train"],
                train_p90[dimension],
                training=True,
                canary_output=canary,
                **common,
            ),
            job(
                transform_id,
                "round0038_transform",
                [train_id],
                cell_paths[label]["transform"],
                240.0,
                train_output=cell_paths[label]["train"],
                shared_reference_output=R0037_SHARED_ROOT,
                query_ids_path=query_ids_path,
                **common,
            ),
            job(
                score_id,
                "round0038_score",
                [transform_id],
                cell_paths[label]["panel"],
                300.0,
                train_output=cell_paths[label]["train"],
                transform_output=cell_paths[label]["transform"],
                shared_reference_output=R0037_SHARED_ROOT,
                query_ids_path=query_ids_path,
                **common,
            ),
        ])
        predecessor = score_id
    seed42 = {
        label: {
            "train_receipt": os.path.join(
                R0037_CELL_ROOT, label, "train", "train-receipt.json"),
            "panel": os.path.join(
                R0037_CELL_ROOT, label, "panel", "panel.json"),
        }
        for label in ("d768_s42", "d384_s42")
    }
    jobs.append(job(
        "mrl_seed43_completion",
        "round0038_decision",
        [f"score_{label}" for label in CELL_LABELS],
        os.path.join(artifacts, "decision"),
        180.0,
        gpu=False,
        cell_outputs=cell_paths,
        seed42_cell_outputs=seed42,
        shared_reference_output=R0037_SHARED_ROOT,
        prior_screen_path=R0037_DECISION_PATH,
    ))
    return jobs


def prepare_round0038(release_sha: str) -> str:
    _require_issued_round(ROUND_FILE)
    round_root = ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="R0038 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    manifests, query_ids_path = _generated_inputs(queue_root)
    inputs = _static_inputs(
        manifests=manifests, query_ids_path=query_ids_path)
    manifest = _base_manifest(
        round_id="0038",
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=4.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0037"]
    manifest["capability_dependencies"] = ["jina-mrl-seed42-screen-v1"]
    manifest["capabilities_produced"] = ["jina-mrl-two-seed-decision-v1"]
    manifest["training_performed"] = True
    manifest["scientific_contract"] = {
        "cells": list(CELL_LABELS),
        "candidate_dimension": 384,
        "shared_graph": expected_input_signature(GRAPH_PATH),
        "shared_full_768d_graph_truth": True,
        "literal_2m_prefix_payload_sha256": PREFIX_PAYLOAD_SHA256,
        "oos_query_ids": expected_input_signature(query_ids_path),
        "query_rows": QUERY_ROWS,
        "graph_manifests": manifests,
        "successful_positive_lr_updates_per_cell": 500_000,
        "canary_minimum_updates_per_s": CANARY_MINIMUM_UPDATES_PER_S,
        "gpu_p90_seconds": GPU_P90_SECONDS,
        "registered_adoption_rule": {
            "mean_oos_proj_ffr_ratio_min": 0.90,
            "mean_transductive_ffr_floor": (
                "768d seed mean minus 768d max-minus-min seed spread"
            ),
        },
        "reviewed_r0037": expected_input_signature(R0037_REVIEW_PATH),
        "r0037_screen": expected_input_signature(R0037_DECISION_PATH),
        "reused_shared_reference": expected_input_signature(
            os.path.join(R0037_SHARED_ROOT, "receipt.json")),
        "outlier_tail_receipt_required": True,
    }
    manifest["jobs"] = _jobs(
        artifacts=artifacts,
        inputs=inputs,
        manifests=manifests,
        query_ids_path=query_ids_path,
    )
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        {"queue_manifest": prepare_round0038(args.release_sha)},
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
