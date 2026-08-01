"""Pure row-policy contract for a conditional 12.5M English-anchor rung.

This module prepares no launch authority.  It defines the diverse-scale
analogue that becomes scientifically relevant only if independent Review
0147 accepts ``eligible-historical-row-policy-restores``.  The candidate keeps
the accepted R0132 12,474,331-row total fixed, retains every eligible row from
the three English source corpora, and allocates the remaining seats
proportionally across all 19 language groups.

Language rows reuse R0132's exact within-group SHA-256 ranking namespace.  The
candidate language selection is therefore a strict subset of R0132 U12, while
the candidate English selection is a strict superset.  That makes the two
populations nested around one exact matched intersection instead of creating
an unrelated second sample.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0105_search import GROUPS
from .round0132_scale_bridge import (
    METRIC_RETENTION,
    FULL_RETAINED_ROWS,
    HALF_RETAINED_ROWS,
    SUBSET_NAMESPACE,
    largest_remainder_quotas,
)
from .round0147_row_policy import RESTORATION_FLOORS, metric_view


ROUND_ID = "0148"
CAPABILITY = "jina-diverse-12p5m-english-anchor-row-policy-v1"
SUBSET_SCHEMA = "round0148-english-anchor-subset-plan-v1"

# R0108 registered this floor before any current diverse-Jina map was scored.
# R0134 explicitly blocks recalibration, and the owner-issued campaign requires
# the unchanged density-v2 floor for the first 12.5M rescue rung.
DENSITY_V2_FLOOR = 0.17589389755990817
OOD_METRICS = (
    "fineweb_recall_at_50_of_high10",
    "polish_recall_at_50_of_high10",
    "in_mix_median_recall_at_50_of_high10",
)

OUTCOME_RESCUED = "english-anchor-12p5m-restores-atlas-quality"
OUTCOME_FUNCTIONAL = "english-anchor-12p5m-functional-restoration-failed"
OUTCOME_DENSITY = "english-anchor-12p5m-density-v2-failed"
OUTCOME_OOD = "english-anchor-12p5m-ood-noninferiority-failed"
OUTCOME_INVALID = "invalid-execution"

ENGLISH_GROUPS = GROUPS[:3]
LANGUAGE_GROUPS = GROUPS[3:]


class Round0148Error(RuntimeError):
    """The conditional English-anchor population contract is malformed."""


def _allocate_language_seats(
    group_counts: Mapping[str, int],
    *,
    target: int,
) -> dict[str, int]:
    """Allocate an exact target proportionally over registered languages."""
    counts = {group: int(group_counts[group]) for group in LANGUAGE_GROUPS}
    if any(value <= 0 for value in counts.values()):
        raise Round0148Error("every language group must be nonempty")
    total = sum(counts.values())
    if not 0 < target < total:
        raise Round0148Error("language-seat target is outside its population")
    quotas = {
        group: (target * counts[group]) // total
        for group in LANGUAGE_GROUPS
    }
    remaining = target - sum(quotas.values())
    ranked = sorted(
        LANGUAGE_GROUPS,
        key=lambda group: (
            -((target * counts[group]) % total),
            GROUPS.index(group),
        ),
    )
    for group in ranked[:remaining]:
        quotas[group] += 1
    if (
        sum(quotas.values()) != target
        or any(quotas[group] <= 0 for group in LANGUAGE_GROUPS)
        or any(quotas[group] > counts[group] for group in LANGUAGE_GROUPS)
    ):
        raise Round0148Error("language largest-remainder allocation did not close")
    return quotas


def english_anchor_quotas(
    group_counts: Mapping[str, int],
    *,
    target: int = HALF_RETAINED_ROWS,
) -> dict[str, int]:
    """Return the exact nested 12.5M English-anchor group quotas.

    All eligible English representatives are retained.  Remaining seats are
    apportioned over the language groups by integer largest remainder.  Reuse
    of the R0132 rank namespace is part of the policy, so each language quota
    must fit inside the accepted R0132 U12 quota.
    """
    if set(group_counts) != set(GROUPS):
        raise Round0148Error("English-anchor group-count keys changed")
    counts = {group: int(group_counts[group]) for group in GROUPS}
    if any(value <= 0 for value in counts.values()):
        raise Round0148Error("every source group must be nonempty")
    if sum(counts.values()) != FULL_RETAINED_ROWS:
        raise Round0148Error("full retained population changed")
    english_rows = sum(counts[group] for group in ENGLISH_GROUPS)
    language_target = int(target) - english_rows
    if not english_rows < target < FULL_RETAINED_ROWS:
        raise Round0148Error("English-anchor target cannot retain all English rows")
    language_quotas = _allocate_language_seats(
        counts,
        target=language_target,
    )
    quotas = {
        group: counts[group] if group in ENGLISH_GROUPS else language_quotas[group]
        for group in GROUPS
    }
    u12 = largest_remainder_quotas(counts, target=target)
    if (
        sum(quotas.values()) != target
        or any(quotas[group] < u12[group] for group in ENGLISH_GROUPS)
        or any(quotas[group] > u12[group] for group in LANGUAGE_GROUPS)
    ):
        raise Round0148Error("English-anchor/U12 nesting did not close")
    return quotas


def ranking_namespace(group: str) -> bytes:
    """Return the exact accepted R0132 group-ranking namespace."""
    if group not in GROUPS:
        raise Round0148Error(f"unknown registered source group: {group!r}")
    return SUBSET_NAMESPACE + group.encode("utf-8") + b"\0"


def build_subset_plan(group_counts: Mapping[str, int]) -> dict[str, Any]:
    """Seal arithmetic and nesting facts before any new map is observed."""
    counts = {group: int(group_counts[group]) for group in GROUPS}
    quotas = english_anchor_quotas(counts)
    u12 = largest_remainder_quotas(counts)
    common = {
        group: min(quotas[group], u12[group])
        for group in GROUPS
    }
    replaced = sum(
        quotas[group] - u12[group]
        for group in ENGLISH_GROUPS
    )
    removed = sum(
        u12[group] - quotas[group]
        for group in LANGUAGE_GROUPS
    )
    if replaced != removed or sum(common.values()) <= 0:
        raise Round0148Error("nested population replacement arithmetic changed")
    body = {
        "schema": SUBSET_SCHEMA,
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "target_rows": HALF_RETAINED_ROWS,
        "group_counts": counts,
        "u12_quotas": u12,
        "english_anchor_quotas": quotas,
        "common_intersection_quotas": common,
        "common_intersection_rows": sum(common.values()),
        "english_rows_added_vs_u12": replaced,
        "language_rows_removed_vs_u12": removed,
        "selector": {
            "english": "retain every R0087-eligible representative",
            "languages": (
                "integer-largest-remainder seats; lowest exact SHA-256 ranks "
                "under the accepted R0132 per-group namespace"
            ),
            "namespace_hex": SUBSET_NAMESPACE.hex(),
            "map_outcomes_observed": False,
        },
        "nesting": {
            "english_anchor_contains_u12_english": True,
            "u12_contains_english_anchor_languages": True,
            "matched_intersection_exactly_precomputable": True,
        },
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def english_anchor_decision(
    *,
    validity_checks: Mapping[str, bool],
    candidate_functional_cell: Mapping[str, Any],
    density_v2: float,
    control_ood: Mapping[str, float],
    candidate_ood: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the frozen first-rung rescue selector without tuning branches.

    Functional restoration is absolute against the pre-R0140 floors rather
    than a post-hoc comparison with R0132.  Density uses the unchanged R0108
    floor.  OOD is a matched same-probe comparison against the accepted R0132
    U12 seed-42 map; all three registered summaries must retain at least 97%.
    """
    validity = {str(key): bool(value) for key, value in validity_checks.items()}
    if not validity:
        raise Round0148Error("at least one execution validity check is required")
    functional = metric_view(candidate_functional_cell)
    if set(functional) != set(RESTORATION_FLOORS):
        raise Round0148Error("candidate functional metric set changed")
    density = float(density_v2)
    control = {key: float(control_ood[key]) for key in OOD_METRICS}
    candidate = {key: float(candidate_ood[key]) for key in OOD_METRICS}
    finite = [*functional.values(), density, *control.values(), *candidate.values()]
    if not all(math.isfinite(value) for value in finite):
        raise Round0148Error("rescue selector metrics must be finite")
    if any(value <= 0 for value in control.values()):
        raise Round0148Error("matched OOD control values must be positive")

    functional_checks = {
        key: functional[key] >= float(floor)
        for key, floor in RESTORATION_FLOORS.items()
    }
    ood_ratios = {
        key: candidate[key] / control[key]
        for key in OOD_METRICS
    }
    ood_checks = {
        key: ood_ratios[key] >= METRIC_RETENTION
        for key in OOD_METRICS
    }
    density_pass = density >= DENSITY_V2_FLOOR

    if not all(validity.values()):
        outcome = OUTCOME_INVALID
        next_action = "repair-invalid-execution-without-interpreting-map-quality"
    elif not all(functional_checks.values()):
        outcome = OUTCOME_FUNCTIONAL
        next_action = "do-not-run-25m-rescue-localize-population-dose"
    elif not density_pass:
        outcome = OUTCOME_DENSITY
        next_action = "do-not-run-25m-rescue-keep-density-floor-frozen"
    elif not all(ood_checks.values()):
        outcome = OUTCOME_OOD
        next_action = "do-not-run-25m-rescue-localize-english-language-tradeoff"
    else:
        outcome = OUTCOME_RESCUED
        next_action = "review-before-preregistering-25m-row-policy-transfer"

    return {
        "schema": "round0148-english-anchor-rescue-decision-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "validity_checks": validity,
        "functional": {
            "metrics": functional,
            "floors": dict(RESTORATION_FLOORS),
            "checks": functional_checks,
            "passed": all(functional_checks.values()),
        },
        "density_v2": {
            "value": density,
            "registered_floor": DENSITY_V2_FLOOR,
            "passed": density_pass,
        },
        "matched_ood": {
            "control_r0132_u12": control,
            "candidate_english_anchor": candidate,
            "candidate_over_control": ood_ratios,
            "retention_floor": METRIC_RETENTION,
            "checks": ood_checks,
            "passed": all(ood_checks.values()),
        },
        "outcome": outcome,
        "next_action": next_action,
        "registered_density_floor_changed": False,
        "map_registry_state_changed": False,
        "production_or_publishing_claimed": False,
        "twenty_five_million_transfer_claimed": False,
    }
