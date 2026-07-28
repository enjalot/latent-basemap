"""Conservative lower-recall search policy for the balanced-150M graph."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0086_program import (
    POLICY_GRID as R0086_POLICY_GRID,
    QUALIFICATION_SCHEMA as R0086_QUALIFICATION_SCHEMA,
)
from .round0086_program import select_cell as select_r0086_cell


ROUND_ID = "0093"
TIER = "150m"
MEAN_RECALL_FLOOR = 0.84
LOWER_POLICY_GRID = (
    (32, 128),
    (64, 128),
    (96, 128),
    (32, 256),
    (64, 256),
    (96, 256),
    (32, 384),
    (64, 384),
    (96, 384),
)
FALLBACK_POLICY_GRID = tuple(R0086_POLICY_GRID)
POLICY_GRID = LOWER_POLICY_GRID + FALLBACK_POLICY_GRID
QUALIFICATION_SCHEMA = (
    "round0093-balanced-150m-lower-recall-policy-qualification-v1"
)
DECISION_SCHEMA = "round0093-balanced-150m-lower-recall-policy-decision-v1"
R0083_SCHEMA = "round0083-graph-recall-sensitivity-v1"
R0084_SCHEMA = "round0084-seed43-sensitivity-contrast-v1"
STABILITY_MARGINS = {
    "ffr": 0.02,
    "projection_ffr": 0.02,
    "purity_k256": 0.05,
    "purity_k1024": 0.05,
}


class Round0093Error(RuntimeError):
    """The preregistered lower-recall qualification contract changed."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def _load_sealed(
    path: str,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0093Error(f"{label} bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0093Error(f"{label} is not a JSON object")
    body = {
        key: item
        for key, item in value.items()
        if key != "identity_sha256"
    }
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0093Error(f"{label} seal changed")
    return value, signature


def validate_r0083_sensitivity(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Require the exact direct low-recall treatment that motivates R0093."""
    receipt, signature = _load_sealed(
        path,
        expected_sha256=expected_sha256,
        label="R0083 graph-recall sensitivity",
    )
    cell = (receipt.get("cells") or {}).get("16") or {}
    noninferiority = cell.get("noninferiority_vs_r0061") or {}
    checks = cell.get("full_30m_non_density_checks") or {}
    decision = receipt.get("decision") or {}
    if (
        receipt.get("schema") != R0083_SCHEMA
        or receipt.get("round_id") != "0083"
        or receipt.get("training_performed") is not True
        or cell.get("nprobe") != 16
        or cell.get("passed") is not True
        or float(
            cell.get("candidate_recall_at_15_unambiguous", -1.0)
        )
        < MEAN_RECALL_FLOOR
        or not noninferiority
        or any(
            (noninferiority.get(metric) or {}).get("passed") is not True
            for metric in STABILITY_MARGINS
        )
        or not checks
        or any(value is not True for value in checks.values())
        or decision.get("verdict")
        != "insensitive-through-lowest-tested-recall"
        or decision.get("changes_frozen_floor_in_this_round") is not False
        or float(decision.get("lowest_passing_measured_recall", -1.0))
        != float(cell["candidate_recall_at_15_unambiguous"])
    ):
        raise Round0093Error(
            "R0083 does not support the preregistered 0.84 floor candidate"
        )
    return {
        "receipt": receipt,
        "signature": signature,
        "supporting_cell": cell,
    }


def validate_r0084_stability(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Use one contrast only as a bounded stability screen, never an error bar."""
    receipt, signature = _load_sealed(
        path,
        expected_sha256=expected_sha256,
        label="R0084 seed contrast",
    )
    interpretation = receipt.get("interpretation") or {}
    matched = (
        (receipt.get("paired_metric_contrasts") or {}).get("matched")
        or {}
    )
    full_checks = receipt.get("full_90m_non_density_checks") or {}
    observed = {
        metric: float(
            (matched.get(metric) or {}).get("absolute_delta", float("inf"))
        )
        for metric in STABILITY_MARGINS
    }
    if (
        receipt.get("schema") != R0084_SCHEMA
        or receipt.get("round_id") != "0084"
        or receipt.get("training_performed") is not True
        or interpretation.get("one_paired_seed_contrast") is not True
        or interpretation.get("estimates_variance") is not False
        or interpretation.get("establishes_error_bar") is not False
        or interpretation.get("changes_ladder_decision") is not False
        or set(full_checks) != {"seed42", "seed43"}
        or any(
            not checks
            or any(value is not True for value in checks.values())
            for checks in full_checks.values()
        )
        or any(
            observed[metric] > margin
            for metric, margin in STABILITY_MARGINS.items()
        )
    ):
        raise Round0093Error(
            "R0084 does not pass the preregistered descriptive stability screen"
        )
    return {
        "receipt": receipt,
        "signature": signature,
        "matched_absolute_deltas": observed,
        "margins": dict(STABILITY_MARGINS),
        "interpretation": (
            "one observed contrast passed a conservative screen; this is "
            "not a variance estimate or error bar"
        ),
    }


def validate_r0086_qualification(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Authenticate the old-floor qualification used as the fallback policy."""
    receipt, signature = _load_sealed(
        path,
        expected_sha256=expected_sha256,
        label="R0086 policy qualification",
    )
    selected = receipt.get("selected")
    checks = receipt.get("checks") or {}
    if (
        receipt.get("schema") != R0086_QUALIFICATION_SCHEMA
        or receipt.get("round_id") != "0086"
        or receipt.get("validity_passed") is not True
        or receipt.get("training_performed") is not False
        or float((receipt.get("quality") or {}).get("floor", -1.0)) != 0.90
        or selected is None
        or selected != select_r0086_cell(receipt)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise Round0093Error("R0086 fallback qualification changed")
    return {
        "receipt": receipt,
        "signature": signature,
    }


def select_cell(receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    """Select the fastest measured passing cell with fixed tie-breaks."""
    cells = receipt.get("cells") or {}
    passing = [
        cells.get(f"nprobe-{nprobe}-width-{width}")
        for nprobe, width in POLICY_GRID
    ]
    passing = [
        value
        for value in passing
        if isinstance(value, dict)
        and value.get("passes_mean_floor") is True
        and isinstance(value.get("benchmark"), dict)
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda value: (
            float(value["benchmark"]["median_wall_seconds_per_query"]),
            int(value["shortlist_width"]),
            int(value["nprobe"]),
        ),
    )


def load_decision(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Authenticate the reviewed-consumer-facing R0093 policy decision."""
    receipt, signature = _load_sealed(
        path,
        expected_sha256=expected_sha256,
        label="R0093 lower-recall policy decision",
    )
    selected = receipt.get("selected") or {}
    if (
        receipt.get("schema") != DECISION_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("tier") != TIER
        or float(receipt.get("registered_mean_recall_floor", -1.0))
        != MEAN_RECALL_FLOOR
        or receipt.get("validity_passed") is not True
        or selected.get("passes_mean_floor") is not True
        or float(selected.get("mean_recall_at_15_unambiguous", -1.0))
        < MEAN_RECALL_FLOOR
        or receipt.get("training_performed") is not False
    ):
        raise Round0093Error("R0093 lower-recall decision changed")
    return {
        "receipt": receipt,
        "signature": signature,
    }
