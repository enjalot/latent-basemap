#!/usr/bin/env python3
"""Prepare, but never launch, the Round 0040 representative-only rescore."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap import round0014_program as minilm
from basemap import round0027_program as jina
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    RUN_ROOT,
    _base_manifest,
    _dedupe,
    _file_inputs,
    _materialized_chunk_inputs,
)
from experiments.round0040_nodes import (
    JINA_CELLS,
    JINA_CENSUS_SOURCE,
    JINA_CENSUS_SOURCE_SHA256,
    JINA_DECISION,
    JINA_QUERY_EMBEDDINGS,
    MINILM_CAP_PATH,
    MINILM_CELLS,
    MINILM_QUERIES,
    MINILM_QUERY_PROVENANCE,
    R0036_PANEL,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0040"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0040-2026-07-24.md")
HANDLER_MODULE = "experiments.round0040_nodes"
HANDLERS = {
    "jina_census": "run_jina_census",
    "jina_representative_rescore": "run_jina_rescore",
    "minilm_representative_reference": "run_minilm_reference",
    "minilm_representative_rescore": "run_minilm_rescore",
    "duplicate_controlled_comparison": "run_comparison",
}
GPU_P90_SECONDS = 6_300.0


def _require_issued_round(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError(f"Round 0040 frontmatter is missing: {path}")
    statuses: list[str] = []
    closed = False
    for line in lines[1:]:
        if line.strip() == "---":
            closed = True
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "status":
            statuses.append(value.strip().strip("\"'"))
    if not closed or statuses != ["issued"]:
        raise RuntimeError(
            f"Round 0040 requires one status: issued; observed {statuses}"
        )


def _known_file(
    path: str, *, sha256: str, size: int
) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if canonical != path or os.path.islink(path):
        raise ValueError(f"Round 0040 input path is not canonical: {path}")
    if os.path.getsize(path) != int(size):
        raise ValueError(f"Round 0040 input size changed: {path}")
    return {
        "canonical_path": canonical,
        "kind": "file",
        "bytes": int(size),
        "sha256": str(sha256),
    }


def _stream_inputs(root: str) -> list[dict[str, Any]]:
    receipt_path = os.path.join(root, "actual-transform.json")
    receipt_signature = expected_input_signature(receipt_path)
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    capability = receipt.get("stream_capability", {}).get(
        "capability_payload", {}
    )
    ordered = capability.get("ordered_chunks")
    if not isinstance(ordered, list) or len(ordered) != 30:
        raise RuntimeError(
            f"Round 0040 coordinate stream changed: {receipt_path}"
        )
    chunks = []
    for position, item in enumerate(ordered):
        path = os.path.join(
            root, f"chunk-{int(item['chunk_index']):05d}", "coordinates.npy"
        )
        if (
            item.get("chunk_index") != position
            or item.get("global_row_start") != position * 1_000_000
            or item.get("global_row_stop") != (position + 1) * 1_000_000
        ):
            raise RuntimeError(
                f"Round 0040 coordinate order changed: {receipt_path}"
            )
        chunks.append(_known_file(
            path,
            sha256=str(item["sha256"]),
            size=int(item["size_bytes"]),
        ))
    return [
        receipt_signature,
        *chunks,
        expected_input_signature(
            os.path.join(root, "heldout-query-coordinates.npy")
        ),
    ]


def _static_inputs() -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = [
        *_file_inputs([
            ROUND_FILE,
            JINA_QUERY_EMBEDDINGS,
            JINA_DECISION,
            MINILM_CAP_PATH,
            MINILM_QUERIES,
            MINILM_QUERY_PROVENANCE,
            minilm.ACCEPTED_MANIFEST,
            R0036_PANEL,
            *(item["path"] for item in jina.CENTROIDS.values()),
            minilm.CENTROIDS_K256_PATH,
            minilm.CENTROIDS_K1024_PATH,
            *(cell["coordinates"] for cell in JINA_CELLS.values()),
            *(cell["query_coordinates"] for cell in JINA_CELLS.values()),
            *(cell["prior_panel"] for cell in JINA_CELLS.values()),
            *(cell["prior_panel"] for cell in MINILM_CELLS.values()),
        ]),
        _known_file(
            JINA_CENSUS_SOURCE,
            sha256=JINA_CENSUS_SOURCE_SHA256,
            size=jina.TRAIN_BYTES,
        ),
        *_materialized_chunk_inputs(),
    ]
    for cell in MINILM_CELLS.values():
        inputs.extend(_stream_inputs(cell["coordinates"]))
    return _dedupe(inputs)


def _jobs(
    *,
    artifacts: str,
    inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    census = os.path.join(artifacts, "jina-census")
    jina_rescore = os.path.join(artifacts, "jina-rescore")
    minilm_reference = os.path.join(artifacts, "minilm-reference")
    minilm_rescore = os.path.join(artifacts, "minilm-rescore")
    comparison = os.path.join(artifacts, "comparison")

    def job(
        node_id: str,
        deps: list[str],
        output: str,
        p90: float,
        *,
        gpu: bool,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "id": node_id,
            "handler": node_id,
            "handler_module": HANDLER_MODULE,
            "handler_callable": HANDLERS[node_id],
            "deps": deps,
            "done_marker": os.path.join(
                artifacts, f"{node_id}.done.json"
            ),
            "outputs": [output],
            "expected_inputs": inputs,
            "p90_wall_s": float(p90),
            "node_policy": {
                "gpu_required": gpu,
                "training_performed": False,
            },
            **extra,
        }

    return [
        job(
            "jina_census",
            [],
            census,
            180.0,
            gpu=False,
        ),
        job(
            "jina_representative_rescore",
            ["jina_census"],
            jina_rescore,
            900.0,
            gpu=True,
            census_output=census,
        ),
        job(
            "minilm_representative_reference",
            ["jina_representative_rescore"],
            minilm_reference,
            1_800.0,
            gpu=True,
        ),
        job(
            "minilm_representative_rescore",
            ["minilm_representative_reference"],
            minilm_rescore,
            3_600.0,
            gpu=True,
            reference_output=minilm_reference,
        ),
        job(
            "duplicate_controlled_comparison",
            [
                "jina_representative_rescore",
                "minilm_representative_rescore",
            ],
            comparison,
            120.0,
            gpu=False,
            jina_output=jina_rescore,
            minilm_output=minilm_rescore,
        ),
    ]


def prepare_round0040(release_sha: str) -> str:
    _require_issued_round(ROUND_FILE)
    round_root = ensure_data_directory(ROUND_ROOT)
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="Round 0040 queue"
    )
    artifacts = ensure_data_directory(
        os.path.join(queue_root, "artifacts")
    )
    inputs = _static_inputs()
    manifest = _base_manifest(
        round_id="0040",
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=2.5,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0038", "0039"]
    manifest["capability_dependencies"] = [
        "jina-mrl-two-seed-decision-v1",
        "30m-update-budget-response-v1",
    ]
    manifest["capabilities_produced"] = [
        "duplicate-controlled-panel-v1"
    ]
    manifest["training_performed"] = False
    manifest["scientific_contract"] = {
        "geometry_unit": "one-exact-nonzero-encoded-vector",
        "product_row_policy": (
            "preserve family membership; exact copies inherit representative "
            "coordinates outside the primary scientific panel"
        ),
        "jina_cells": list(JINA_CELLS),
        "minilm_cells": list(MINILM_CELLS),
        "high_d_reference_reuse": {
            "jina": "one representative full-768d reference for four maps",
            "minilm": "one representative 384d reference for three maps",
        },
        "projection": (
            "one representative-corpus OOS truth and one low-D projection "
            "search per map; omit unrelated kNN/random floor baselines"
        ),
        "r0036_150m": "context-only-not-a-matched-scale-cell",
        "no_training": True,
        "gpu_p90_seconds": GPU_P90_SECONDS,
    }
    manifest["jobs"] = _jobs(artifacts=artifacts, inputs=inputs)
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    print(prepare_round0040(args.release_sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
