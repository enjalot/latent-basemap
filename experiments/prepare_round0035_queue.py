#!/usr/bin/env python3
"""Materialize the two-node standalone R0035 Common Corpus OOD queue."""
from __future__ import annotations

import argparse
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
from experiments.common_corpus_ood_round0035 import (
    BASE_SEED,
    CANARY_MIN_MEAN_COSINE,
    CANARY_ROWS,
    CHUNK_SCRIPT,
    COORDINATE_RECEIPT,
    CORPUS_ROWS,
    EMBED_SCRIPT,
    EXACT_SHA256,
    INPUT_PACK_MANIFEST,
    MINILM_QUERIES,
    MODEL_PATH,
    PROBES,
    PROBE_ROWS,
    QUERY_ROWS,
    R0019_REVIEW,
    R0028_REVIEW,
    SENTENCE_MODEL_SNAPSHOT,
    SENTENCE_MODEL_SNAPSHOT_SHA256,
)
from experiments.prepare_round0020_0022_queues import (
    LAB_ROOT,
    _base_manifest,
    _dedupe,
    _hf_snapshot_file_inputs,
    _materialized_chunk_inputs,
)


ROUND_ROOT = "/data/latent-basemap/runs/round-0035"


def scientific_contract() -> dict[str, Any]:
    return {
        "training_performed": False,
        "gpu_time_estimate": {
            "expected_hours": 0.05,
            "conservative_max_hours": 0.25,
            "basis": "R0028 canary/panel timings scaled to three 50k-by-500 probes plus exact-input hashing",
        },
        "map": {
            "label": "r0019",
            "model_sha256": EXACT_SHA256[MODEL_PATH],
            "coordinate_receipt_sha256": EXACT_SHA256[COORDINATE_RECEIPT],
        },
        "accepted_semantics": {
            "r0019_review_sha256": EXACT_SHA256[R0019_REVIEW],
            "r0028_review_sha256": EXACT_SHA256[R0028_REVIEW],
            "probe_and_query_source_dtype": "float32",
            "projection_compute_dtype": "float32",
            "cosine_compute_dtype": "float32",
            "tf32_allowed": False,
            "sentence_model_snapshot": SENTENCE_MODEL_SNAPSHOT,
            "sentence_model_snapshot_sha256": SENTENCE_MODEL_SNAPSHOT_SHA256,
        },
        "probes": sorted(PROBES),
        "split": {
            "seed": BASE_SEED,
            "source_rows": PROBE_ROWS,
            "corpus_rows": CORPUS_ROWS,
            "query_rows": QUERY_ROWS,
            "query_disjoint_from_corpus": True,
            "selection": "collection-and-purpose-bound RandomState without replacement",
        },
        "text_canary": {
            "rows_per_probe": CANARY_ROWS,
            "minimum_mean_cosine": CANARY_MIN_MEAN_COSINE,
            "requires_full_ordered_sidecar_to_parquet_metadata_match": True,
            "mapping_or_model_mismatch_policy": "publish blocker then fail panel closed",
        },
        "matched_control": {
            "corpus": "R0013 accepted 30M materialized fp16 input pack",
            "queries": "held-out MiniLM fp32 query pool",
            "shape_exact": [CORPUS_ROWS, QUERY_ROWS],
        },
        "scoring": {
            "true_neighbors": "top-10 exact brute-force fp32 cosine",
            "map_neighbors": "top-495 exact brute-force fp32 L2 (1% of corpus)",
            "retention": "probe FFR / shape-matched control FFR",
            "verdicts": {"pass": ">=0.7", "amber": "[0.5,0.7)", "failure": "<0.5"},
        },
    }


def _exact_files(round_file: str) -> list[dict[str, Any]]:
    paths = [
        round_file,
        MODEL_PATH,
        COORDINATE_RECEIPT,
        INPUT_PACK_MANIFEST,
        MINILM_QUERIES,
        R0019_REVIEW,
        R0028_REVIEW,
        EMBED_SCRIPT,
        CHUNK_SCRIPT,
    ]
    for probe in PROBES.values():
        paths.extend([probe.vectors, probe.ids, probe.manifest, probe.chunks])
    inputs = [expected_input_signature(path) for path in paths]
    by_path = {item["canonical_path"]: item for item in inputs}
    expected = dict(EXACT_SHA256)
    for probe in PROBES.values():
        expected.update(
            {
                probe.vectors: probe.vectors_sha256,
                probe.ids: probe.ids_sha256,
                probe.manifest: probe.manifest_sha256,
                probe.chunks: probe.chunks_sha256,
            }
        )
    mismatches = {
        path: {
            "expected": digest,
            "observed": by_path.get(os.path.realpath(path), {}).get("sha256"),
        }
        for path, digest in expected.items()
        if by_path.get(os.path.realpath(path), {}).get("sha256") != digest
    }
    if mismatches:
        raise RuntimeError(f"R0035 exact evidence/input tuple changed: {mismatches}")
    return inputs


def jobs(*, artifacts: str, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canary = os.path.join(artifacts, "canary")
    panel = os.path.join(artifacts, "panel")
    return [
        {
            "id": "common_corpus_source_model_canary",
            "handler_module": "experiments.common_corpus_ood_round0035",
            "handler_callable": "run_canary_job",
            "deps": [],
            "done_marker": os.path.join(
                artifacts, "common_corpus_source_model_canary.done.json"
            ),
            "outputs": [canary],
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            "id": "common_corpus_ood_panel",
            "handler_module": "experiments.common_corpus_ood_round0035",
            "handler_callable": "run_panel_job",
            "deps": ["common_corpus_source_model_canary"],
            "done_marker": os.path.join(artifacts, "common_corpus_ood_panel.done.json"),
            "outputs": [panel],
            "canary_output": canary,
            "expected_inputs": inputs,
            "p90_wall_s": 600.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
    ]


def prepare(release_sha: str, *, date: str = "2026-07-22") -> str:
    round_file = os.path.join(LAB_ROOT, f"round-0035-{date}.md")
    queue_root = create_fresh_directory(
        os.path.join(ensure_data_directory(ROUND_ROOT), "queue"), label="R0035 queue"
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe(
        [
            *_exact_files(round_file),
            *_materialized_chunk_inputs(),
            *_hf_snapshot_file_inputs(),
        ]
    )
    manifest = _base_manifest(
        round_id="0035",
        release_sha=release_sha,
        round_file=round_file,
        queue_root=queue_root,
        gpu_hours_cap=0.25,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["required_reviews"] = ["0013", "0019", "0028"]
    manifest["capability_dependencies"] = [
        "30m-input-pack-v1",
        "30m-minilm-map-seed42-duplicate-cap",
        "universality-panel-v1",
    ]
    manifest["capabilities_produced"] = ["common-corpus-ood-panel-v1"]
    manifest["scientific_contract"] = scientific_contract()
    manifest["jobs"] = jobs(artifacts=artifacts, inputs=inputs)
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--date", default="2026-07-22")
    args = parser.parse_args(argv)
    print(prepare(args.release_sha, date=args.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
