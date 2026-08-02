"""Build the CPU-only R0151 prefix/drop-only scale census."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0105_search import GROUPS, group_ranges
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0132_scale_bridge import SUBSET_SCHEMA
from basemap.round0150_seed_replay import CAPABILITY as R0150_CAPABILITY
from basemap.round0151_scale_census import (
    CAPABILITY,
    EXPECTED_DROPPED_ROWS,
    EXPECTED_RETAINED_ROWS,
    EXPECTED_U12_OVERLAP,
    FULL_RAW_ROWS,
    RAW_PREFIX_TARGET,
    ROUND_ID,
    Round0151Error,
    build_prefix_drop_mapping,
    compare_to_u12,
)


def _read_json(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0151Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value, signature


def _require_signature(expected: Mapping[str, Any], *, label: str) -> str:
    path = str(expected.get("canonical_path") or "")
    if expected_input_signature(path) != dict(expected):
        raise Round0151Error(f"{label} bytes changed")
    return path


def run_census(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0151 prefix/drop-only scale census"
    )
    started = time.monotonic()
    activation, activation_signature = _read_json(
        _require_signature(job["r0150_decision"], label="R0150 decision"),
        label="accepted R0150 decision",
    )
    if (
        activation.get("round_id") != "0150"
        or activation.get("capability") != R0150_CAPABILITY
        or activation.get("outcome")
        != "drop-only-restoration-replicates-across-seeds"
        or activation.get("drop_only_scale_candidate_released") is not True
    ):
        raise Round0151Error("R0150 did not release the scale-census branch")

    inventory, inventory_signature = _read_json(
        _require_signature(job["inventory"], label="R0087 inventory"),
        label="accepted R0087 inventory",
    )
    if (
        inventory.get("round_id") != "0087"
        or inventory.get("capability") != "jina-diverse-25m-inventory-v1"
        or inventory.get("capability_ready") is not True
        or inventory.get("selection", {}).get("selected_rows") != FULL_RAW_ROWS
        or inventory.get("duplicate_control", {}).get("eligibility")
        != dict(job["eligibility"])
    ):
        raise Round0151Error("R0087 inventory binding changed")
    eligibility_path = _require_signature(job["eligibility"], label="R0087 eligibility")
    with np.load(eligibility_path, allow_pickle=False) as archive:
        excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)

    substrate, substrate_signature = _read_json(
        _require_signature(job["substrate"], label="R0103 substrate"),
        label="accepted R0103 substrate",
    )
    ranges = group_ranges(substrate)
    mapping, group_ids, census = build_prefix_drop_mapping(ranges, excluded)

    u12_manifest, u12_manifest_signature = _read_json(
        _require_signature(job["u12_manifest"], label="R0132 U12 manifest"),
        label="accepted R0132 U12 manifest",
    )
    if (
        u12_manifest.get("round_id") != "0132"
        or u12_manifest.get("schema") != SUBSET_SCHEMA
        or u12_manifest.get("mapping") != dict(job["u12_mapping"])
    ):
        raise Round0151Error("R0132 U12 binding changed")
    u12_path = _require_signature(job["u12_mapping"], label="R0132 U12 mapping")
    u12 = np.load(u12_path, mmap_mode="r", allow_pickle=False)
    comparison = compare_to_u12(mapping, u12)
    if (
        census["full_raw_rows"] != FULL_RAW_ROWS
        or census["raw_prefix_target"] != RAW_PREFIX_TARGET
        or census["retained_rows"] != EXPECTED_RETAINED_ROWS
        or census["dropped_rows"] != EXPECTED_DROPPED_ROWS
        or comparison["overlap_rows"] != EXPECTED_U12_OVERLAP
        or comparison["distinct"] is not True
    ):
        raise Round0151Error("registered prefix/drop-only census changed")

    mapping_path = os.path.join(output, "compact-to-global.i64.npy")
    group_ids_path = os.path.join(output, "compact-group-ids.u8.npy")
    atomic_save_new_npy(mapping_path, mapping, immutable=True)
    atomic_save_new_npy(group_ids_path, group_ids, immutable=True)
    mapping_signature = expected_input_signature(mapping_path)
    group_ids_signature = expected_input_signature(group_ids_path)
    groups = {
        group: {
            **census["groups"][group],
            "selected_rows_sha256": ordered_array_sha256(
                mapping[group_ids == group_id]
            ),
        }
        for group_id, group in enumerate(GROUPS)
    }
    receipt = seal({
        "schema": "round0151-diverse-prefix-drop-only-census-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "activation": activation_signature,
        "inventory": inventory_signature,
        "eligibility": dict(job["eligibility"]),
        "substrate": substrate_signature,
        "u12_manifest": u12_manifest_signature,
        "u12_mapping": dict(job["u12_mapping"]),
        "selector": {
            "allocation": "integer-largest-remainder-over-raw-25m-groups",
            "allocation_tie_break": "registered-GROUPS order",
            "within_group": "ascending raw global-row prefix",
            "duplicate_policy": "drop R0087-ineligible rows without replacement",
            "replacement_rows": 0,
            "map_outcomes_observed": False,
        },
        "raw_prefix_target": census["raw_prefix_target"],
        "retained_rows": census["retained_rows"],
        "dropped_rows": census["dropped_rows"],
        "replacement_rows": 0,
        "quotas": census["quotas"],
        "groups": groups,
        "mapping": mapping_signature,
        "group_ids": group_ids_signature,
        "u12_comparison": comparison,
        "checks": {
            "every_group_present": True,
            "prefixes_before_exclusion": True,
            "no_replacement": True,
            "mapping_strictly_increasing": True,
            "different_from_r0132_u12": True,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "graph_built": False,
        "map_outcomes_observed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "census.json"), receipt, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0151Error("R0151 handler requires its exact queue manifest")
    if job.get("action") != "build_prefix_drop_census":
        raise Round0151Error(f"unknown R0151 action: {job.get('action')}")
    run_census(active, job)
