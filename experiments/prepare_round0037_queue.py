#!/usr/bin/env python3
"""Prepare, but never launch, the three-cell Round 0037 Jina MRL screen."""
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
from basemap.round0037_program import (
    CANARY_MINIMUM_UPDATES_PER_S,
    CELL_LABELS,
    CENTROIDS,
    DIMENSIONS,
    GRAPH_BYTES,
    GRAPH_PATH,
    GRAPH_SHA256,
    PREFIX_PAYLOAD_SHA256,
    QUERY_ROWS,
    SOURCE_4M_BYTES,
    SOURCE_4M_PATH,
    SOURCE_4M_SHA256,
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


ROUND_ROOT = "/data/latent-basemap/runs/round-0037"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0037-2026-07-22.md")
R0027_REVIEW_PATH = os.path.join(
    LAB_ROOT, "review-0027-2026-07-22.md")
R0027_REVIEW_SHA256 = (
    "958b17f3c17c6004ae8fa75c3b09bf53599eb7053574441af5afd8446194f543"
)
R0027_REVIEW_BYTES = 12_382
HANDLER_MODULE = "experiments.round0037_nodes"
HANDLER_CALLABLES = {
    "round0037_sampler_canary": "run_sampler_canary",
    "round0037_shared_reference": "run_shared_reference",
    "round0037_train": "run_train",
    "round0037_transform": "run_transform",
    "round0037_score": "run_score",
    "round0037_decision": "run_decision",
}
GPU_P90_SECONDS = 17_720.0


def _require_issued_round(path: str) -> None:
    """Refuse to materialize a queue before its pre-registration is issued."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError(f"Round 0037 frontmatter is missing: {path}")
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
            f"Round 0037 frontmatter has ambiguous status: {path}")
    if statuses[0] != "issued":
        raise RuntimeError(
            "Round 0037 queue materialization requires status: issued; "
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
        R0027_REVIEW_PATH,
        TRAIN_PATH,
        SOURCE_4M_PATH,
        GRAPH_PATH,
        *(item["path"] for item in CENTROIDS.values()),
        *(item["path"] for item in manifests.values()),
        query_ids_path,
    ]
    inputs = _dedupe(_file_inputs(files))
    observed = {item["canonical_path"]: item for item in inputs}
    expected = {
        TRAIN_PATH: (TRAIN_SHA256, TRAIN_BYTES),
        SOURCE_4M_PATH: (SOURCE_4M_SHA256, SOURCE_4M_BYTES),
        GRAPH_PATH: (GRAPH_SHA256, GRAPH_BYTES),
        R0027_REVIEW_PATH: (R0027_REVIEW_SHA256, R0027_REVIEW_BYTES),
        **{
            item["path"]: (item["sha256"], item["bytes"])
            for item in CENTROIDS.values()
        },
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
        raise RuntimeError(f"Round 0037 static input tuple changed: {mismatches}")
    return inputs


def _jobs(
    *,
    artifacts: str,
    inputs: list[dict[str, Any]],
    manifests: dict[int, dict[str, str]],
    query_ids_path: str,
) -> list[dict[str, Any]]:
    shared = os.path.join(artifacts, "shared-reference")
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
            raise ValueError(f"unknown Round 0037 handler: {handler}")
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

    control = "d768_s42"
    control_manifest = manifests[768]
    control_config, control_sha = train_config_for_cell(
        control,
        graph_manifest_path=control_manifest["path"],
        graph_manifest_sha256=control_manifest["sha256"],
    )
    jobs: list[dict[str, Any]] = [
        job(
            "mrl_sampler_canary",
            "round0037_sampler_canary",
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
        ),
        job(
            "shared_score_reference",
            "round0037_shared_reference",
            ["mrl_sampler_canary"],
            shared,
            1200.0,
            query_ids_path=query_ids_path,
        ),
    ]
    predecessor = "shared_score_reference"
    train_p90 = {768: 5100.0, 384: 4800.0, 256: 4700.0}
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
                "round0037_train",
                [predecessor],
                cell_paths[label]["train"],
                train_p90[dimension],
                training=True,
                canary_output=canary,
                **common,
            ),
            job(
                transform_id,
                "round0037_transform",
                [train_id],
                cell_paths[label]["transform"],
                240.0,
                train_output=cell_paths[label]["train"],
                shared_reference_output=shared,
                query_ids_path=query_ids_path,
                **common,
            ),
            job(
                score_id,
                "round0037_score",
                [transform_id],
                cell_paths[label]["panel"],
                300.0,
                train_output=cell_paths[label]["train"],
                transform_output=cell_paths[label]["transform"],
                shared_reference_output=shared,
                query_ids_path=query_ids_path,
                **common,
            ),
        ])
        predecessor = score_id
    jobs.append(job(
        "mrl_seed42_screen",
        "round0037_decision",
        [f"score_{label}" for label in CELL_LABELS],
        os.path.join(artifacts, "decision"),
        120.0,
        gpu=False,
        cell_outputs=cell_paths,
        shared_reference_output=shared,
    ))
    return jobs


def prepare_round0037(release_sha: str) -> str:
    _require_issued_round(ROUND_FILE)
    round_root = ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="R0037 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    manifests, query_ids_path = _generated_inputs(queue_root)
    inputs = _static_inputs(
        manifests=manifests, query_ids_path=query_ids_path)
    manifest = _base_manifest(
        round_id="0037",
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=6.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0027"]
    manifest["capability_dependencies"] = []
    manifest["capabilities_produced"] = ["jina-mrl-seed42-screen-v1"]
    manifest["training_performed"] = True
    manifest["scientific_contract"] = {
        "cells": list(CELL_LABELS),
        "shared_graph": expected_input_signature(GRAPH_PATH),
        "shared_full_768d_graph_truth": True,
        "literal_2m_prefix_payload_sha256": PREFIX_PAYLOAD_SHA256,
        "oos_query_ids": expected_input_signature(query_ids_path),
        "query_rows": QUERY_ROWS,
        "graph_manifests": manifests,
        "successful_positive_lr_updates_per_cell": 500_000,
        "canary_minimum_updates_per_s": CANARY_MINIMUM_UPDATES_PER_S,
        "gpu_p90_seconds": GPU_P90_SECONDS,
        "screen_only_no_adoption": True,
        "reviewed_r0027_terminal_calibration": expected_input_signature(
            R0027_REVIEW_PATH),
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
        {"queue_manifest": prepare_round0037(args.release_sha)},
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
