"""Qualify a conservative lower-recall policy on the reviewed 150M index."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import atomic_write_new_json
from basemap.round0086_program import (
    EXCLUDED_ROWS,
    FILTER_RECEIPT_SCHEMA,
    RETAINED_ROWS,
    ROW_COUNT,
    SPEC,
    TIER,
    validate_substrate,
)
from basemap.round0093_policy import (
    DECISION_SCHEMA,
    LOWER_POLICY_GRID,
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    QUALIFICATION_SCHEMA,
    ROUND_ID,
    Round0093Error,
    seal,
    select_cell,
    validate_r0083_sensitivity,
    validate_r0084_stability,
    validate_r0086_qualification,
)
from experiments import round0081_nodes as qualification


def _load_filter_receipt(
    path: str,
    *,
    expected_sha256: str,
    substrate_signature: Mapping[str, Any],
    filtered_signature: Mapping[str, Any],
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0093Error("R0086 filter-receipt bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    if (
        receipt.get("schema") != FILTER_RECEIPT_SCHEMA
        or receipt.get("round_id") != "0086"
        or receipt.get("substrate") != substrate_signature
        or receipt.get("filtered_index") != filtered_signature
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
    ):
        raise Round0093Error("R0086 filtered-index evidence changed")
    return {
        "receipt": receipt,
        "signature": signature,
    }


def run_qualification(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    r0083 = validate_r0083_sensitivity(
        str(job["r0083_sensitivity"]),
        expected_sha256=str(job["r0083_sensitivity_sha256"]),
    )
    r0084 = validate_r0084_stability(
        str(job["r0084_seed_contrast"]),
        expected_sha256=str(job["r0084_seed_contrast_sha256"]),
    )
    r0086 = validate_r0086_qualification(
        str(job["r0086_qualification"]),
        expected_sha256=str(job["r0086_qualification_sha256"]),
    )
    r0086_receipt = r0086["receipt"]

    # The shared qualification authenticates the 57.9 GB substrate pair and
    # filtered index. Do not hash them once here and immediately repeat that
    # I/O inside the same process.
    bound_job = dict(job)
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
    qualification.Round0081Error = Round0093Error
    qualification._selected_cell = select_cell
    qualification.validate_scale_substrate = validate_substrate
    try:
        generic = qualification.run_qualification(active, bound_job)
    finally:
        for name, value in previous.items():
            setattr(qualification, name, value)

    selected = select_cell(generic)
    if (
        generic.get("schema") != QUALIFICATION_SCHEMA
        or generic.get("round_id") != ROUND_ID
        or generic.get("validity_passed") is not True
        or selected is None
        or generic.get("selected") != selected
        or float(selected.get("mean_recall_at_15_unambiguous", -1.0))
        < MEAN_RECALL_FLOOR
    ):
        raise Round0093Error("lower-recall policy qualification did not pass")
    substrate_signature = generic["substrate"]
    filtered = generic["filtered_index"]
    if (
        r0086_receipt.get("substrate") != substrate_signature
        or r0086_receipt.get("filtered_index") != filtered
    ):
        raise Round0093Error(
            "R0086 fallback policy does not bind the qualified 150M index"
        )
    filter_receipt = _load_filter_receipt(
        str(job["filter_receipt"]),
        expected_sha256=str(job["filter_receipt_sha256"]),
        substrate_signature=substrate_signature,
        filtered_signature=filtered,
    )
    decision_body = {
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "tier": TIER,
        "registered_mean_recall_floor": MEAN_RECALL_FLOOR,
        "validity_passed": True,
        "selected": selected,
        "selected_from_new_lower_cost_grid": (
            (
                int(selected["nprobe"]),
                int(selected["shortlist_width"]),
            )
            in LOWER_POLICY_GRID
        ),
        "fallback_r0086_selected": r0086_receipt["selected"],
        "qualification": generic["receipt"],
        "substrate": substrate_signature,
        "filtered_index": filtered,
        "filter_receipt": filter_receipt["signature"],
        "r0083_sensitivity": r0083["signature"],
        "r0084_stability_screen": {
            "signature": r0084["signature"],
            "matched_absolute_deltas": r0084[
                "matched_absolute_deltas"
            ],
            "margins": r0084["margins"],
            "one_contrast_is_not_variance_or_error_bar": True,
        },
        "r0086_fallback_qualification": r0086["signature"],
        "selection_semantics": (
            "fastest measured registered cell meeting mean unambiguous "
            "exact-reranked recall@15 >= 0.84; ties by shortlist then nprobe"
        ),
        "full_150m_map_evaluation_still_required": True,
        "changes_prior_artifacts_in_place": False,
        "training_performed": False,
        "optimizer_updates": 0,
    }
    decision = seal(decision_body)
    decision_path = os.path.join(
        str(job["outputs"][0]),
        "lower-recall-policy-decision.json",
    )
    atomic_write_new_json(decision_path, decision, immutable=True)
    return {
        **decision,
        "receipt": expected_input_signature(decision_path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        active.get("manifest", {}).get("round_id") != ROUND_ID
        or job is None
        or job.get("action") != "qualify_lower_recall_150m_policy"
    ):
        raise Round0093Error("R0093 handler requires its exact round/job")
    return run_qualification(active, job)
