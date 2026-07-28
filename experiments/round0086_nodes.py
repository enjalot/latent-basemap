"""Stage reviewed 150M identities, filter the source index, and qualify search."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0049_program import (
    DIMENSION,
    INDEX_PATH,
    INDEX_SHA256,
)
from basemap.round0086_program import (
    EXCLUDED_ROWS,
    FILTER_RECEIPT_SCHEMA,
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    ROW_COUNT,
    SPEC,
    SUBSTRATE_SCHEMA,
    TIER,
    Round0086Error,
    seal,
    select_cell,
    validate_substrate,
)
from experiments import round0077_nodes as index_builder
from experiments import round0081_nodes as qualification


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0086Error(f"{path} is not a JSON object")
    return value


def _validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {
        key: item
        for key, item in value.items()
        if key != "identity_sha256"
    }
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0086Error(f"{label} seal changed")


def run_stage(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0086 balanced-150M substrate manifest",
    )
    int8 = expected_input_signature(str(job["int8_path"]))
    scales = expected_input_signature(str(job["scales_path"]))
    eligibility_signature = expected_input_signature(
        str(job["eligibility_path"])
    )
    r0025_manifest = expected_input_signature(str(job["r0025_manifest"]))
    r0033_receipt = expected_input_signature(str(job["r0033_receipt"]))
    if (
        int8["sha256"] != job["int8_sha256"]
        or int8["bytes"] != ROW_COUNT * DIMENSION
        or scales["sha256"] != job["scales_sha256"]
        or scales["bytes"] != ROW_COUNT * 2
        or eligibility_signature["sha256"] != job["eligibility_sha256"]
        or r0025_manifest["sha256"] != job["r0025_manifest_sha256"]
        or r0033_receipt["sha256"] != job["r0033_receipt_sha256"]
    ):
        raise Round0086Error("reviewed 150M input bytes changed")
    eligibility = load_int8_eligibility(
        eligibility_signature["canonical_path"],
        expected_sha256=eligibility_signature["sha256"],
        row_count=ROW_COUNT,
    )
    summary = eligibility["metadata"]["summary"]
    if any(
        int(summary.get(key, -1)) != value
        for key, value in SPEC["eligibility_summary"].items()
    ):
        raise Round0086Error("reviewed R0033 census changed")
    body = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "tier": TIER,
        "row_count": ROW_COUNT,
        "dimension": DIMENSION,
        "global_150m_intervals": [[0, ROW_COUNT]],
        "outputs": {
            "int8": int8,
            "scales": scales,
            "eligibility": eligibility_signature,
        },
        "eligibility_summary": summary,
        "source_receipts": {
            "r0025_manifest": r0025_manifest,
            "r0033_receipt": r0033_receipt,
        },
        "reference_only_no_payload_copy": True,
        "training_performed": False,
        "optimizer_updates": 0,
    }
    manifest = seal(body)
    path = os.path.join(output, "balanced-150m-substrate-v1.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(path)}


def run_filter(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0086 physically filtered 150M index",
    )
    substrate = validate_substrate(str(job["substrate_manifest"]))
    excluded = np.asarray(
        substrate["eligibility"]["excluded_rows"], dtype=np.int64
    )
    source = expected_input_signature(INDEX_PATH)
    if (
        source["sha256"] != INDEX_SHA256
        or len(excluded) != EXCLUDED_ROWS
        or ROW_COUNT - len(excluded) != RETAINED_ROWS
    ):
        raise Round0086Error("R0086 source index/eligibility changed")

    previous = {
        "intervals": index_builder.INTERVALS,
        "row_count": index_builder.ROW_COUNT,
        "summary": index_builder.ELIGIBILITY_SUMMARY,
        "error": index_builder.Round0077Error,
    }
    index_builder.INTERVALS = ((0, ROW_COUNT),)
    index_builder.ROW_COUNT = ROW_COUNT
    index_builder.ELIGIBILITY_SUMMARY = {
        "excluded_row_count": EXCLUDED_ROWS,
        "retained_row_count": RETAINED_ROWS,
    }
    index_builder.Round0077Error = Round0086Error
    index_path = os.path.join(output, "balanced-150m-retained.ivfpq")
    started = time.monotonic()
    try:
        filtered, performance = index_builder._build_filtered_index(
            faiss=faiss,
            destination_path=index_path,
            excluded_global=excluded,
        )
    finally:
        index_builder.INTERVALS = previous["intervals"]
        index_builder.ROW_COUNT = previous["row_count"]
        index_builder.ELIGIBILITY_SUMMARY = previous["summary"]
        index_builder.Round0077Error = previous["error"]
    if int(filtered.ntotal) != RETAINED_ROWS:
        raise Round0086Error("filtered 150M candidate count changed")
    body = {
        "schema": FILTER_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "source_index": source,
        "filtered_index": expected_input_signature(index_path),
        "performance": {
            **performance,
            "total_wall_seconds": time.monotonic() - started,
        },
        "training_performed": False,
        "optimizer_updates": 0,
    }
    receipt = seal(body)
    path = os.path.join(output, "filter-receipt.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_qualification(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    substrate = validate_substrate(str(job["substrate_manifest"]))
    filter_receipt = _read_json(str(job["filter_receipt"]))
    _validate_seal(filter_receipt, label="R0086 filter receipt")
    filtered = expected_input_signature(str(job["filtered_index"]))
    if (
        filter_receipt.get("schema") != FILTER_RECEIPT_SCHEMA
        or filter_receipt.get("round_id") != ROUND_ID
        or filter_receipt.get("release_sha")
        != active["manifest"]["release_sha"]
        or filter_receipt.get("substrate") != substrate["signature"]
        or filter_receipt.get("filtered_index") != filtered
    ):
        raise Round0086Error("late-bound R0086 filtered index changed")
    bound_job = {
        **job,
        "substrate_manifest_sha256": substrate["signature"]["sha256"],
        "filtered_index_sha256": filtered["sha256"],
    }
    previous = {
        name: getattr(qualification, name)
        for name in (
            "TIER",
            "SPEC",
            "ROW_COUNT",
            "INTERVALS",
            "ELIGIBILITY_SUMMARY",
            "QUALITY_SEED",
            "MEAN_RECALL_FLOOR",
            "POLICY_GRID",
            "QUALIFICATION_SCHEMA",
            "ROUND_ID",
            "Round0081Error",
            "_selected_cell",
            "validate_scale_substrate",
        )
    }
    qualification.TIER = TIER
    qualification.SPEC = SPEC
    qualification.ROW_COUNT = ROW_COUNT
    qualification.INTERVALS = ((0, ROW_COUNT),)
    qualification.ELIGIBILITY_SUMMARY = {
        "excluded_row_count": EXCLUDED_ROWS,
        "retained_row_count": RETAINED_ROWS,
    }
    qualification.QUALITY_SEED = 86
    qualification.MEAN_RECALL_FLOOR = MEAN_RECALL_FLOOR
    qualification.POLICY_GRID = POLICY_GRID
    qualification.QUALIFICATION_SCHEMA = QUALIFICATION_SCHEMA
    qualification.ROUND_ID = ROUND_ID
    qualification.Round0081Error = Round0086Error
    qualification._selected_cell = select_cell
    qualification.validate_scale_substrate = validate_substrate
    try:
        return qualification.run_qualification(active, bound_job)
    finally:
        for name, value in previous.items():
            setattr(qualification, name, value)


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0086Error("R0086 handler requires its exact round/job")
    action = str(job.get("action"))
    if action == "stage":
        return run_stage(active, job)
    if action == "filter":
        return run_filter(active, job)
    if action == "qualify":
        return run_qualification(active, job)
    raise Round0086Error(f"unknown R0086 action {action!r}")
