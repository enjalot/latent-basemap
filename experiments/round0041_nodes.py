"""CPU-only handlers for Round 0041."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import atomic_write_new_json
from basemap.round0041_program import (
    R0021_TRAIN_RECEIPT,
    R0030_TRAIN_RECEIPT,
    ROUND_ID,
    build_graph,
    read_training_semantics,
)


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def run_build_and_audit(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    started = time.monotonic()
    graph = build_graph(job["outputs"][0])
    summary = graph["summary"]
    histogram = {
        int(degree): int(count)
        for degree, count in summary["degree_histogram"].items()
    }
    positive_degrees = [
        degree for degree, count in histogram.items()
        if degree > 0 and count > 0
    ]
    r0021 = read_training_semantics(R0021_TRAIN_RECEIPT)
    r0030 = read_training_semantics(R0030_TRAIN_RECEIPT)
    expected_r0021 = {
        "pipeline": "device_uniform",
        "sampler_class": "DeviceEdgeSampler",
        "positive_sampling": "uniform",
        "positive_source_sampling": (
            "uniform_retained_rows_then_fixed_k_slot_with_replacement"
        ),
        "positive_destinations": "original_graph_rows",
    }
    expected_r0030 = {
        "pipeline": "hybrid",
        "sampler_class": "HostStreamEdgeSampler",
        "positive_sampling": "uniform",
        "positive_source_sampling": (
            "uniform_over_retained_source_edges_with_replacement"
        ),
        "positive_destinations": "original_graph_rows",
    }
    for label, observed, expected in (
        ("R0021", r0021, expected_r0021),
        ("R0030", r0030, expected_r0030),
    ):
        mismatch = {
            key: {"expected": value, "observed": observed.get(key)}
            for key, value in expected.items()
            if observed.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"{label} sampler receipt changed: {mismatch}")

    audit_body = {
        "schema": "round0041-graph-sampler-semantics-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "optimizer_updates": 0,
        "canonical_graph": graph["manifest"],
        "canonical_graph_summary": summary,
        "historical_receipts": {
            "r0021_fixed_k_source_normalized": r0021,
            "r0030_fuzzy_edge_uniform": r0030,
        },
        "semantic_matrix": {
            "r0021": {
                "topology": "source-major-IVF-PQ-k15",
                "source_law": "uniform-positive-source",
                "destination_law": "uniform-one-of-15-original-row-ids",
                "duplicate_sources": "excluded",
                "duplicate_destinations": "not-canonicalized",
            },
            "r0030": {
                "topology": "symmetrized-fuzzy-variable-degree",
                "source_law": "degree-proportional-via-uniform-directed-edge",
                "destination_law": "selected-directed-edge-original-row-id",
                "duplicate_sources": "excluded",
                "duplicate_destinations": "not-canonicalized",
            },
            "r0034_150m_and_r0041_target": {
                "topology": "source-major-IVF-PQ-k15",
                "source_law": "uniform-positive-source",
                "destination_law": (
                    "uniform-valid-canonical-destination-after-copy-map-and-"
                    "zero-self-repeat-drop"
                ),
                "duplicate_sources": "excluded",
                "duplicate_destinations": "mapped-to-representative",
            },
        },
        "canonical_degree_exposure": {
            "positive_degree_min": min(positive_degrees),
            "positive_degree_max": max(positive_degrees),
            "edge_uniform_max_to_min_source_exposure_ratio": (
                max(positive_degrees) / min(positive_degrees)
            ),
            "source_normalized_max_to_min_source_exposure_ratio": 1.0,
        },
        "decision": {
            "direct_r0030_to_r0034_scale_comparison_valid": False,
            "reason": (
                "topology, source sampling law, and duplicate-destination "
                "policy all differ"
            ),
            "next_training_cell": (
                "30M canonical graph, uniform positive source then uniform "
                "valid canonical destination, seed42, 500k successful updates"
            ),
            "isolates_against": (
                "R0021 isolates destination canonicalization while preserving "
                "fixed-k topology, source law, data, seed, model, and horizon"
            ),
        },
        "wall_seconds": time.monotonic() - started,
    }
    audit = _seal(audit_body)
    audit_path = os.path.join(
        job["outputs"][0], "sampler-semantics-audit-v1.json"
    )
    atomic_write_new_json(audit_path, audit, immutable=True)
    receipt = _seal({
        "schema": "round0041-build-receipt-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "optimizer_updates": 0,
        "graph_manifest": graph["manifest"],
        "audit": expected_input_signature(audit_path),
        "passed": True,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(job["outputs"][0], "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}
