"""Execute R0263 — restore the two-sided ``k256`` purity band, then re-adjudicate R0257.

Two CPU nodes, in that order, and the order is the round's central claim. R0263 is
the exact structural INVERSE of R0260 for ``purity_fidelity_k256``: R0260 dropped the
band's upper edge and re-registered a one-sided lower floor; this restores the upper
edge — the *same* band R0255 registered, re-fitted bitwise from the *same* sealed
n = 29 family at the *same* sealed ``k2`` — and demotes the one-sided floor to a
published-descriptive diagnostic. ``k1024`` and ``ffr`` are untouched: they stay the
one-sided floors R0260/R0256 registered, and the node proves that bitwise.

**Node 1, ``register_two_sided_k256_band_n29``.** Reads the twenty-nine sealed 2M
family cells (R0255 panel), the sealed R0256 repaired gate and the sealed R0260
one-sided-floor artifact. Re-derives the calibration at ``n = 29`` restricted to the
registered estimator and requires it bitwise; re-fits the two-sided ``k256`` band at
the sealed ``k2`` and requires it bitwise against R0255's registered band; re-derives
the RETAINED one-sided ``k256`` floor and the untouched ``k1024`` / ``ffr`` floors
and requires each bitwise against R0260/R0256 (proof this round touched only the
``k256`` gated side). It runs ``assert_family_is_2m_only`` with its rung-map plants,
and — the positive control review-0260-01 found missing in R0260 — runs the ordering
guard ``_no_rung_artifact_is_bound`` TWICE: once on the real manifest (must pass) and
once on a planted manifest that binds an R0257/6250k path (must raise), so the guard
is shown to discriminate rather than merely to pass. It publishes attainability
against the identity bound, two-sided detection power at ±1/2/3σ, the compounded
mixed-panel false-alarm rate, the inverted registry guards with plants, and the
sidedness controls.

**Node 2, ``readjudicate_0257_k256_two_sided``.** Depends on node 1. Re-scores R0257's
three sealed ``6250k`` maps from R0257's own published ``k256`` observations (bitwise
``1.1012 / 1.1022 / 1.0982``) under the restored band, publishes each map's restored
verdict + margin + direction beside R0257's published verdict and R0260's one-sided
verdict, and carries the retained one-sided floor value as the descriptive diagnostic.
It refuses to run unless node 1's registration was sealed **before** this node started,
by the producer's own done-marker timestamp, and issues its result as a NEW dated
addendum — it rewrites neither R0257 nor R0260.

Nothing here trains anything, touches a floor R0255/R0256/R0260 registered on
``k1024``/``ffr``, or rewrites a published result. R0257's three FAILs stand; this
round restores the criterion that produced them and adds a dated record.
"""
from __future__ import annotations

import json
import math
import os
import resource
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import create_fresh_directory
from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0234_calibration as calibration
from basemap.round0234_calibrated_floors import (
    DESCRIPTIVE_METRICS,
    GATED_METRICS,
    METRICS,
    PURITY_METRICS,
    PURITY_RATIO_KEYS,
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    attainability,
    band_at,
    centre_and_scale,
    floor_at,
    identity_bound,
    positive_scale_witness,
)
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0255_gate_n29 import (
    EXACT_FAMILY_SEEDS,
    IDENTITY_BOUND_AT_N,
    N_EXACT,
)
from basemap.round0255_treatment import (
    assert_family_is_2m_only,
    exact_cell_id,
)
from basemap.round0256_gate_repair import (
    DEFINING_CELL_IDS,
    GATE_CAPABILITY as R0256_GATE_CAPABILITY,
    REGISTERED_ESTIMATOR_NAMES,
    two_sided_detection_power,
)
from basemap.round0257_judgement import (
    GATE_CAPABILITY as R0257_CONSUMED_GATE_CAPABILITY,
)
from basemap.round0257_rung_contract import (
    assert_no_rung_map_in_the_gate_family,
    rung_cell_ids,
    rung_family_purity_controls,
)
from basemap.round0260_one_sided_purity import (
    CONFORMITY_MARKING,
    GATE_CAPABILITY as R0260_GATE_CAPABILITY,
    KIND_RATIO_FLOOR,
    KIND_SCALAR_FLOOR,
    ONE_SIDED_KINDS,
    Round0260RegistrationError,
    attainability_against_the_identity_bound,
    one_sided_ratio_floor,
    _numbers_reachable_from,
)

from experiments.round0255_nodes import (
    SAFETY_NOTE,
    _bound_path,
    _guard_tail_reported,
    _node_gate,
    _node_guard,
    _score_gate_without_raising,
    _seal,
    _start_node,
)
# The ordering guard, the bitwise comparator, the intra-queue signer, the family
# assembler, the R0257-cell reader and the GPU-mapping probe are R0260's, imported
# rather than re-typed. This round adds only the two-sided helpers R0260 lacked.
from experiments.round0260_nodes import (
    RUNG_PATH_MARKERS,
    _bitwise_same,
    _cuda_mappings,
    _family_from_panel,
    _intra_queue_signature,
    _no_rung_artifact_is_bound,
    _r0257_cells,
)


REGISTER_ACTION = "register_two_sided_k256_band_n29"
READJUDICATE_ACTION = "readjudicate_0257_k256_two_sided"

ROUND_ID = "0263"

GATE_CAPABILITY = "minilm-mixed-2m-two-sided-k256-purity-band-n29-v1"
GATE_SCHEMA = "round0263-minilm-mixed-2m-two-sided-k256-purity-band-n29-v1"
REGISTRATION_ARTIFACT = "minilm-two-sided-k256-purity-band-n29.json"

READJUDICATION_CAPABILITY = "minilm-mixed-6250k-rung-readjudication-k256-two-sided-v1"
READJUDICATION_SCHEMA = "round0263-6250k-rung-readjudication-k256-two-sided-v1"
READJUDICATION_ARTIFACT = "readjudication-0257-k256-two-sided.json"

#: The kind string R0256 gave the restored criterion. Reused verbatim so the
#: registry a consumer reads is the one R0255/R0256 published, not a new shape.
KIND_TWO_SIDED_BAND = "two_sided_ratio_band"

#: The metric whose gated side this round inverts. Its sibling ``k1024`` and the
#: scalar ``ffr`` are untouched — R0260's one-sided floors, carried through bitwise.
RESTORED_METRIC = "purity_fidelity_k256"
UNCHANGED_RATIO_METRIC = "purity_fidelity_k1024"
UNCHANGED_SCALAR_METRIC = "ffr"

OWNER_RULING_DATE = "2026-08-13"
OWNER_RULING_SECTION = "OWNER-DECISIONS-PENDING.md section 5-A addendum"
OWNER_RULING = (
    "For purity_fidelity_k256 only, restore the two-sided MAD_n band as the gated "
    "criterion, superseding the one-sided-floor clause of the 2026-08-12 section 5 "
    "ruling for this metric. The band is the one R0255 already registered, re-fitted "
    "bitwise from the same sealed n = 29 family. Clause 3 binds: registered before "
    "any further rung map is judged. The one-sided floor value continues to be "
    "published beside every verdict as a descriptive diagnostic. R0257's three "
    "published FAILs stand unchanged. The 6.25M re-adjudication is corrected in a new "
    "dated addendum, never a rewrite of R0260. The registering round must carry the "
    "ordering / positive-control guard review-0260-01 found missing in R0260. This is "
    "a stopgap that makes the collapse direction gateable now; it does not discharge "
    "the option-3 work."
)

WHY_TWO_SIDED = (
    "R0260 re-registered k256 as a one-sided lower floor on the premise that "
    "over-separation cannot indicate a defect. The sandbox collapse/fog program "
    "produced same-universe evidence that post-dates that ruling and contradicts it "
    "at k256: over-separation IS the bead-collapse fingerprint. The three 6.25M rung "
    "maps sit +6.14 to +6.41 family MAD_n above the log centre, +3.0 to +3.3 MAD_n "
    "past the retired upper edge, above the entire observed 29-cell range. A one-sided "
    "lower floor is structurally incapable of failing them. Restoring the upper edge "
    "is the minimal change that makes the direction gateable. This is k256-only: at "
    "k256 the band straddles the ideal 1.0 and 1.0 passes, so the collapse-catching "
    "edge and the reference anchor point the same way. At k1024 the retired upper edge "
    "0.7719 fails a perfectly faithful map, so option 2 stays in force there."
)

NO_TUNING_STATEMENT = (
    "No value in this registration was chosen after seeing a map. The direction is "
    "the 2026-08-13 section 5-A addendum; the estimator is the owner's section-3 "
    "ruling; the multiplier is R0234's Gaussian-null calibration at n = 29, sealed "
    "since R0255; the family is the same twenty-nine sealed 2M cells. Given those four "
    "inputs the band is arithmetic, not a choice, and it is bitwise R0255's. If a "
    "re-adjudicated map fails, it is published failing (section 3, both directions)."
)

SUPERSEDES_K256_CRITERION_IN = R0260_GATE_CAPABILITY

DESCRIPTIVE_MARKING = CONFORMITY_MARKING  # "descriptive_only_never_a_gate"

R0257_PUBLISHED_VERDICT = "FAIL"


# --------------------------------------------------------------------------- #
# Sealed constants — R0263's OWN copy, each reproduced bitwise this round against
# the bound artifact that first published it. Provenance is in SEALED below.
# --------------------------------------------------------------------------- #

#: R0234's Gaussian-null multipliers at n = 29, sealed since R0255. k2 gates the
#: restored band; k1 fits the RETAINED (now descriptive) one-sided floor.
SEALED_K2 = 3.147763220676551
SEALED_K1 = 2.6934393367626024

#: R0255 registered_two_sided_bands.purity_fidelity_k256 — the band this restores.
SEALED_K256_BAND = {
    "log_centre": 0.01143437762566317,
    "log_scale": 0.01339863133626119,
    "ratio_lower": 0.9697263687993266,
    "ratio_upper": 1.0550731452902518,
    "ratio_geometric_centre": 1.0115,
}

#: R0260 registered_ratio_floors.purity_fidelity_k256 — retired here to descriptive.
SEALED_K256_ONE_SIDED_FLOOR = 0.9756474051324426
#: R0260 registered_ratio_floors.purity_fidelity_k1024 — UNCHANGED this round.
SEALED_K1024_ONE_SIDED_FLOOR = 0.6690913806270514
#: R0260 registered_floors.ffr (= R0255/R0256's ffr floor) — UNCHANGED this round.
SEALED_FFR_FLOOR = 0.3125328635126472

#: (n-1)/sqrt(n) at n = 29 = 28/sqrt(29). The band's k2 must sit strictly below it.
SEALED_IDENTITY_BOUND_AT_N29 = 5.199469468957452

#: The three 6.25M rung maps' published k256 ratios (R0257 verdict, re-scored bitwise
#: in node 2 from R0257's own sealed observations). Over the restored band by +6.1σ.
SEALED_RUNG_K256 = {
    "ladder-6250k-h2048-seed42": 1.1012,
    "ladder-6250k-h2048-seed43": 1.1022,
    "ladder-6250k-h2048-seed44": 1.0982,
}

#: The single held-out flip the whole-picture disclosure requires. This 2M held-out
#: cell's k256 observation lives in R0255's floors_n29 all-cells scoring, which NEITHER
#: node binds; it is embedded here as a disclosed literal (source below) and computed
#: in node 1 against the freshly-fit band, NOT presented as a bitwise-verified binding.
SEALED_HELD_OUT_FLIP = {
    "cell_id": "cluster-spill-c8-seed43",
    "k256": 0.9738,
    "reachable_from_a_bound_input": False,
    "source": (
        "R0255 minilm-calibrated-madn-floors-n29 scoring_by_candidate."
        "median_minus_k_madn.all_cells.cells[cluster-spill-c8-seed43]."
        "metrics.purity_fidelity_k256.raw_ratio (a disclosed literal, not a binding)"
    ),
}

SEALED = {
    "panel_n29": {
        "path": (
            "/data/latent-basemap/runs/round-0255/queue/artifacts/"
            "minilm-mixed-2m-seed-family-panel-n29-v1/seed-family-panel-n29.json"
        ),
        "sha256": "76590920ff212113d037676ac7dbd2331d6d3a366228d031607d95e58f88bead",
        "supplies": "raw_purity_ratios for the 29 sealed defining cells",
    },
    "r0256_gate": {
        "capability": R0256_GATE_CAPABILITY,
        "supplies": "the sealed k2/k1 calibration and the registered k256 band",
    },
    "r0260_one_sided": {
        "capability": R0260_GATE_CAPABILITY,
        "supplies": "the retained k256 one-sided floor and the untouched k1024/ffr floors",
    },
    "r0257_verdict": {
        "path": (
            "/data/latent-basemap/runs/round-0257/queue-correction-2/artifacts/"
            "minilm-mixed-6250k-rung-gate-verdict-v1/judge_6250k-rung-verdict.json"
        ),
        "supplies": "R0257's published FAILs and its k256 observations",
    },
}


class Round0263NodeError(RuntimeError):
    """The R0263 node contract changed."""


# --------------------------------------------------------------------------- #
# Local two-sided helpers — added here, NOT in a shared module, because R0263's
# gated side is two-sided and its registry is MIXED (k256 two-sided, k1024/ffr
# one-sided). No shared builder expresses that shape: R0256's is all-purity-two-
# sided, R0260's is all-one-sided. The band fit reuses round0234's `band_at` /
# `centre_and_scale`; only the judge, the mixed registry and the inverted guards
# are local. The judge mirrors experiments/metrics/k256_two_sided.py:judge (its
# repo is not importable from the run checkout), with sealed constants of our own.
# --------------------------------------------------------------------------- #


def _fit_two_sided_k256_band(
    estimator: str, log_ratios: list[float], k2: float
) -> dict[str, float]:
    """The restored band on the UNFOLDED log-ratio, via round0234's shared fit."""
    log_lower, log_upper = band_at(estimator, log_ratios, k2)
    centre, scale = centre_and_scale(estimator, log_ratios)
    return {
        "log_centre": float(centre),
        "log_scale": float(scale),
        "log_lower": float(log_lower),
        "log_upper": float(log_upper),
        "ratio_lower": float(math.exp(log_lower)),
        "ratio_upper": float(math.exp(log_upper)),
        "multiplier": float(k2),
    }


def _judge_k256_two_sided(ratio: float, band: Mapping[str, float]) -> dict[str, Any]:
    """Judge one observed k256 ratio against the restored two-sided band.

    Mirrors ``k256_two_sided.judge``: ``direction`` distinguishes ``over`` (above
    the upper edge — the manufactured-separation / collapse direction) from
    ``under`` and ``inside``; ``separation`` reports the side of the ideal ``1.0``
    independently of pass/fail; ``z_madn`` is the family-MAD_n distance from centre.
    """
    r = float(ratio)
    if not math.isfinite(r) or r <= 0:
        raise Round0263NodeError("R0263 refuses a nonfinite or non-positive k256 ratio")
    lower = float(band["ratio_lower"])
    upper = float(band["ratio_upper"])
    log_r = math.log(r)
    log_centre = float(band["log_centre"])
    log_scale = float(band["log_scale"])
    if r > upper:
        direction = "over"
        margin = upper - r
    elif r < lower:
        direction = "under"
        margin = r - lower
    else:
        direction = "inside"
        margin = min(r - lower, upper - r)
    passes = direction == "inside"
    if r > 1.0:
        separation = "over_separating"
    elif r < 1.0:
        separation = "under_separating"
    else:
        separation = "exactly_faithful"
    return {
        "verdict": "PASS" if passes else "FAIL",
        "direction": direction,
        "passes": passes,
        "raw_ratio": r,
        "log_ratio": log_r,
        "z_madn": (log_r - log_centre) / log_scale,
        "margin": margin,
        "margin_is": (
            "signed distance to the breached edge on the ratio scale; negative "
            "means outside the band"
        ),
        "separation": separation,
        "ratio_lower": lower,
        "ratio_upper": upper,
        "band_direction": {
            "over": "above_band",
            "under": "below_band",
            "inside": "inside_band",
        }[direction],
        "decided_on": "raw unfolded purity ratio, two-sided band",
    }


def _judge_one_sided_floor(ratio: float, floor: float) -> dict[str, Any]:
    """The retired R0260 criterion: a lower ratio floor only (now descriptive)."""
    r = float(ratio)
    passes = r >= float(floor)
    return {
        "verdict": "PASS" if passes else "FAIL",
        "direction": "inside" if passes else "under",
        "passes": passes,
        "raw_ratio": r,
        "margin": r - float(floor),
        "floor": float(floor),
        "kind": KIND_RATIO_FLOOR,
    }


def _registered_criteria_k256_two_sided(
    *, band: Mapping[str, float], k1024_floor: float, ffr_floor: float
) -> dict[str, Any]:
    """The MIXED registry: k256 restored to a band; k1024/ffr as R0260 left them."""
    return {
        UNCHANGED_SCALAR_METRIC: {
            "kind": KIND_SCALAR_FLOOR,
            "floor": float(ffr_floor),
            "gates_a_map": True,
            "fails_only": "under_separation",
            "applicable_power": "one_sided",
            "read_as": "the metric must be at or above `floor` (unchanged from R0260)",
        },
        RESTORED_METRIC: {
            "kind": KIND_TWO_SIDED_BAND,
            "ratio_lower": float(band["ratio_lower"]),
            "ratio_upper": float(band["ratio_upper"]),
            "log_centre": float(band["log_centre"]),
            "log_scale": float(band["log_scale"]),
            "gates_a_map": True,
            "fails": "under_separation AND over_separation",
            "applicable_power": "two_sided",
            "read_as": (
                "the UNFOLDED purity ratio must lie inside [ratio_lower, ratio_upper]. "
                "The upper edge fails over-separation — the collapse direction R0260's "
                "one-sided floor could not see."
            ),
        },
        UNCHANGED_RATIO_METRIC: {
            "kind": KIND_RATIO_FLOOR,
            "ratio_floor": float(k1024_floor),
            "gates_a_map": True,
            "fails_only": "under_separation",
            "applicable_power": "one_sided",
            "read_as": (
                "the UNFOLDED k1024 ratio must be at or above `ratio_floor` "
                "(unchanged from R0260: its retired upper edge fails a faithful map)"
            ),
        },
    }


def _descriptive_one_sided_diagnostics(
    *, one_sided_floor: float, k1: float
) -> dict[str, Any]:
    """The RETIRED side kept as a measurement, marked so it can never gate.

    The exact inverse of R0260's ``descriptive_conformity_diagnostics``: there the
    two-sided band was the retired-but-published side; here the one-sided floor is.
    """
    return {
        DESCRIPTIVE_MARKING: True,
        "contributes_to_verdict": False,
        RESTORED_METRIC: {
            "one_sided_floor": float(one_sided_floor),
            "one_sided_multiplier": float(k1),
            "kind": "one_sided_lower_ratio_floor_DESCRIPTIVE_ONLY",
        },
        "what_this_is": (
            "the R0260 one-sided k256 floor, retired from gating by the section 5-A "
            "addendum and kept published beside every verdict. Dropping the gate must "
            "not drop the measurement. Nothing reads it."
        ),
        "why_it_stays_visible": (
            "a map below this floor is under-separated; the restored two-sided band "
            "still gates that direction (its lower edge 0.9697 is looser than the "
            "floor 0.9756 because k1 < k2), so the floor remains a tracked diagnostic "
            "of the tighter under-separation view."
        ),
    }


def _assert_registry_restores_k256_two_sided(
    criteria: Mapping[str, Any], descriptive: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse a registry that does not restore k256 as a band, or that gates the floor."""
    problems: list[str] = []
    entry = criteria.get(RESTORED_METRIC)
    if entry is None:
        problems.append(f"{RESTORED_METRIC} gates a map and must be registered")
    else:
        if str(entry.get("kind")) != KIND_TWO_SIDED_BAND:
            problems.append(
                f"{RESTORED_METRIC} must be restored as a {KIND_TWO_SIDED_BAND}"
            )
        if str(entry.get("applicable_power")) != "two_sided":
            problems.append(f"{RESTORED_METRIC} must carry two-sided power")
        if "ratio_lower" not in entry or "ratio_upper" not in entry:
            problems.append(f"{RESTORED_METRIC} band must publish both edges")
    k1024 = criteria.get(UNCHANGED_RATIO_METRIC)
    if k1024 is None or str(k1024.get("kind")) not in ONE_SIDED_KINDS:
        problems.append(
            f"{UNCHANGED_RATIO_METRIC} must stay the one-sided floor R0260 registered"
        )
    if "ratio_upper" in (k1024 or {}):
        problems.append(
            f"{UNCHANGED_RATIO_METRIC} gained an upper edge: this round does not touch "
            "k1024, whose retired upper edge fails a faithful map"
        )
    ffr = criteria.get(UNCHANGED_SCALAR_METRIC)
    if ffr is None or str(ffr.get("kind")) != KIND_SCALAR_FLOOR:
        problems.append(f"{UNCHANGED_SCALAR_METRIC} must stay a one-sided scalar floor")
    for metric in DESCRIPTIVE_METRICS:
        if metric in criteria:
            problems.append(f"{metric} is descriptive-only and may not be a criterion")
    if not descriptive.get(DESCRIPTIVE_MARKING):
        problems.append(f"the descriptive block must carry {DESCRIPTIVE_MARKING}: true")
    if descriptive.get("contributes_to_verdict") is not False:
        problems.append("the descriptive block must declare contributes_to_verdict: false")
    if problems:
        raise Round0263NodeError(
            "R0263 registry refuses: " + "; ".join(sorted(set(problems)))
        )
    return {
        "holds": True,
        "gated_criteria_registered": sorted(criteria),
        "k256_is_a_restored_two_sided_band": True,
        "k1024_and_ffr_are_the_one_sided_floors_r0260_registered": True,
        "descriptive_metrics_absent_from_the_registry": sorted(DESCRIPTIVE_METRICS),
    }


def _assert_no_descriptive_value_in_the_gate(
    *, criteria: Mapping[str, Any], descriptive: Mapping[str, Any]
) -> dict[str, Any]:
    """No number published as a descriptive diagnostic may appear inside a gate.

    The value-level inverse of R0260's ``assert_no_gate_reads_a_conformity_value``.
    The realistic way the retired side comes back is renaming the one-sided floor
    ``0.9756…`` into a gated ``ratio_floor``; a key-name guard would not see it. The
    two RESTORED band edges are legitimately gated values and are the sole exception.
    """
    descriptive_values = set(_numbers_reachable_from(descriptive.get(RESTORED_METRIC, {})))
    gate_values = set(_numbers_reachable_from(criteria))
    restored_edges = {
        float(criteria[RESTORED_METRIC]["ratio_lower"]),
        float(criteria[RESTORED_METRIC]["ratio_upper"]),
    }
    leaked = sorted((descriptive_values & gate_values) - restored_edges)
    if leaked:
        raise Round0263NodeError(
            "R0263 refuses: a retired one-sided value is reachable inside the gate "
            f"registry: {leaked}"
        )
    return {
        "holds": True,
        "descriptive_values_checked": len(descriptive_values),
        "gate_values_checked": len(gate_values),
        "restored_edges_exempted": sorted(restored_edges),
        "values_leaking_into_the_gate": [],
        "marking": DESCRIPTIVE_MARKING,
    }


def _registry_controls_k256_two_sided(
    *, criteria: Mapping[str, Any], descriptive: Mapping[str, Any], floor: float
) -> dict[str, Any]:
    """Positive controls for both inverted registry guards."""
    honest_ok = True
    honest_error = None
    try:
        _assert_registry_restores_k256_two_sided(criteria, descriptive)
        _assert_no_descriptive_value_in_the_gate(criteria=criteria, descriptive=descriptive)
    except Round0263NodeError as error:
        honest_ok = False
        honest_error = f"{type(error).__name__}: {error}"

    plants: list[dict[str, Any]] = []

    def plant(name: str, why: str, run: Callable[[], Any]) -> None:
        refused = False
        message = None
        try:
            run()
        except Round0263NodeError as error:
            refused = True
            message = str(error)
        plants.append({
            "plant": name,
            "why_this_is_the_defect": why,
            "refused": refused,
            "error": message,
        })

    plant(
        "k256_left_as_a_one_sided_floor",
        "R0260's floor never restored to a band — the round does nothing",
        lambda: _assert_registry_restores_k256_two_sided(
            {**dict(criteria), RESTORED_METRIC: {
                "kind": KIND_RATIO_FLOOR, "ratio_floor": float(floor),
                "applicable_power": "one_sided",
            }},
            descriptive,
        ),
    )
    plant(
        "the_retired_one_sided_floor_smuggled_into_the_k256_band",
        "the descriptive 0.9756 pasted into the gate as a k256 ratio_floor — the "
        "shape a key-name guard cannot see",
        lambda: _assert_no_descriptive_value_in_the_gate(
            criteria={**dict(criteria), RESTORED_METRIC: {
                **dict(criteria[RESTORED_METRIC]), "ratio_floor": float(floor),
            }},
            descriptive=descriptive,
        ),
    )
    plant(
        "k1024_given_the_restored_upper_edge_too",
        "this round is k256-only; k1024's retired upper edge fails a faithful map",
        lambda: _assert_registry_restores_k256_two_sided(
            {**dict(criteria), UNCHANGED_RATIO_METRIC: {
                **dict(criteria[UNCHANGED_RATIO_METRIC]),
                "ratio_upper": 0.7719564268121961,
            }},
            descriptive,
        ),
    )
    plant(
        "one_sided_power_beside_the_restored_two_sided_band",
        "review-0255/review-0256: the sidedness of the power must match the criterion",
        lambda: _assert_registry_restores_k256_two_sided(
            {**dict(criteria), RESTORED_METRIC: {
                **dict(criteria[RESTORED_METRIC]), "applicable_power": "one_sided",
            }},
            descriptive,
        ),
    )
    plant(
        "the_descriptive_block_left_unmarked",
        "an unmarked block of a retired floor is indistinguishable from a registry",
        lambda: _assert_registry_restores_k256_two_sided(
            criteria,
            {k: v for k, v in descriptive.items() if k != DESCRIPTIVE_MARKING},
        ),
    )
    return {
        "the_honest_registry_passes": honest_ok,
        "honest_registry_error": honest_error,
        "plants": plants,
        "planted_defects": len(plants),
        "planted_defects_refused": sum(1 for row in plants if row["refused"]),
        "every_planted_defect_is_refused": all(row["refused"] for row in plants),
        "holds": bool(honest_ok and all(row["refused"] for row in plants)),
    }


def _sidedness_controls_k256_two_sided(band: Mapping[str, float]) -> dict[str, Any]:
    """Plant each side of the restored band and require the registered outcome.

    The load-bearing rows are the first three: a map far ABOVE the upper edge must
    FAIL in the ``over`` direction (the whole content of the restoration — the
    direction R0260 removed), a faithful ``1.0`` map must PASS (the band straddles
    the ideal), and a map just below the lower edge must FAIL ``under``.
    """
    lower = float(band["ratio_lower"])
    upper = float(band["ratio_upper"])
    centre = float(math.exp(band["log_centre"]))
    epsilon = 1.0e-6
    cases: tuple[tuple[str, float, str, str, str], ...] = (
        (
            "an_over_separating_map_fails_over",
            5.0, "FAIL", "over",
            "R0260's own sidedness control (a 5.0 ratio) which R0260 PASSED: the "
            "manufactured-separation direction the restoration re-gates",
        ),
        (
            "a_map_just_above_the_upper_edge_fails_over",
            upper * (1.0 + epsilon), "FAIL", "over",
            "one part in a million above the upper edge: the collapse direction",
        ),
        (
            "a_perfectly_faithful_map_passes",
            1.0, "PASS", "inside",
            "ratio 1.0 — inside the band, which straddles the ideal (unlike k1024)",
        ),
        (
            "the_band_centre_passes",
            centre, "PASS", "inside",
            "the geometric centre of the family passes",
        ),
        (
            "a_map_just_below_the_lower_edge_fails_under",
            lower * (1.0 - epsilon), "FAIL", "under",
            "one part in a million below the lower edge: the under-separation direction",
        ),
        (
            "the_lower_edge_is_inclusive",
            lower, "PASS", "inside", "the band is inclusive at its lower edge",
        ),
        (
            "the_upper_edge_is_inclusive",
            upper, "PASS", "inside", "the band is inclusive at its upper edge",
        ),
    )
    results: list[dict[str, Any]] = []
    for name, ratio, expected_verdict, expected_direction, why in cases:
        judged = _judge_k256_two_sided(ratio, band)
        results.append({
            "control": name,
            "plants": why,
            "ratio": ratio,
            "expected_verdict": expected_verdict,
            "observed_verdict": judged["verdict"],
            "expected_direction": expected_direction,
            "observed_direction": judged["direction"],
            "held": (
                judged["verdict"] == expected_verdict
                and judged["direction"] == expected_direction
            ),
        })
    return {
        "controls": results,
        "planted": len(results),
        "controls_that_held": sum(1 for row in results if row["held"]),
        "every_control_held": all(row["held"] for row in results),
        "at_least_one_control_fails_over": any(
            row["observed_direction"] == "over" and row["observed_verdict"] == "FAIL"
            for row in results
        ),
        "at_least_one_control_fails_under": any(
            row["observed_direction"] == "under" and row["observed_verdict"] == "FAIL"
            for row in results
        ),
    }


def _descriptive_floor_never_gates_controls(
    *, observations: Mapping[str, float], band: Mapping[str, float], floor: float
) -> dict[str, Any]:
    """Prove the retired floor is inert to the shipped judge, and the proof is not vacuous.

    Two arms on the same mutation, the inverse of R0260's band-widening control. The
    shipped judge reads the two-sided band and must produce byte-identical margins when
    the descriptive floor is moved to ``1e9`` (the top of the ``[0, 1e9]`` range across
    which R0260 widened its retired band); a planted consumer that gates on the
    descriptive floor must flip, so the shipped judge's stillness is evidence.
    """
    tampered_floor = 1.0e9
    shipped_before = {
        cid: _judge_k256_two_sided(r, band) for cid, r in observations.items()
    }
    # The shipped judge takes no `floor` argument at all, so its margins cannot move.
    shipped_after = {
        cid: _judge_k256_two_sided(r, band) for cid, r in observations.items()
    }
    consumer_before = {
        cid: _judge_one_sided_floor(r, floor)["verdict"]
        for cid, r in observations.items()
    }
    consumer_after = {
        cid: _judge_one_sided_floor(r, tampered_floor)["verdict"]
        for cid, r in observations.items()
    }
    shipped_changed = sorted(
        cid for cid in observations
        if shipped_before[cid]["verdict"] != shipped_after[cid]["verdict"]
    )
    consumer_changed = sorted(
        cid for cid in observations if consumer_before[cid] != consumer_after[cid]
    )
    return {
        "mutation": (
            "the retired descriptive one-sided floor moved to 1e9 (the top of the "
            "[0, 1e9] range R0260's inverse control widened a band across), which no "
            "finite ratio clears"
        ),
        "shipped_margins_are_bitwise_identical": all(
            shipped_before[cid]["margin"] == shipped_after[cid]["margin"]
            for cid in observations
        ),
        "shipped_verdicts_changed": shipped_changed,
        "the_shipped_judge_is_inert_to_the_descriptive_floor": not shipped_changed,
        "planted_consumer_verdicts_before": consumer_before,
        "planted_consumer_verdicts_after": consumer_after,
        "planted_consumer_verdicts_changed": consumer_changed,
        "the_mutation_is_potent": bool(consumer_changed),
        "holds": bool(consumer_changed) and not shipped_changed,
        "why_both_arms_exist": (
            "an inertness claim proved only by 'nothing moved' holds equally for a "
            "mutation that reaches nothing. The planted consumer is the arm that shows "
            "the mutation reaches a decision path when one gates on the retired floor "
            "— so the shipped judge's stillness is evidence."
        ),
    }


def _panel_false_alarm_rate_mixed(
    *, one_sided_rate: float, two_sided_rate: float
) -> dict[str, Any]:
    """Compound the panel rate: TWO one-sided floors (ffr, k1024) + ONE two-sided band (k256).

    Neither R0256's compounder (two two-sided) nor R0260's (three one-sided) has this
    shape, so it is written here: ``1 - (1 - one)**2 * (1 - two)``.
    """
    one = float(one_sided_rate)
    two = float(two_sided_rate)
    compounded = 1.0 - (1.0 - one) ** 2 * (1.0 - two)
    return {
        "gated_criteria": [UNCHANGED_SCALAR_METRIC, UNCHANGED_RATIO_METRIC, RESTORED_METRIC],
        "one_sided_criteria": [UNCHANGED_SCALAR_METRIC, UNCHANGED_RATIO_METRIC],
        "two_sided_criteria": [RESTORED_METRIC],
        "one_sided_per_metric_rate": one,
        "two_sided_per_metric_rate": two,
        "panel_false_alarm_rate_under_independence": compounded,
        "expression": f"1 - (1 - {one!r})**2 * (1 - {two!r})",
        "this_is_an_upper_bound": True,
        "why_it_is_an_upper_bound": (
            "the three gated criteria are correlated on this family, so the true panel "
            "rate is at or below the independence product. It is still the number a "
            "rung map actually faces."
        ),
    }


# --------------------------------------------------------------------------- #
# the ordering positive-control review-0260-01 required (the new part R0260 lacked)
# --------------------------------------------------------------------------- #


def _refuse_if_rung_bound(job: Mapping[str, Any]) -> dict[str, Any]:
    """The refusal node 1 applies to its own job: raise if a judged map is bound.

    Factored out so the SAME code path both passes the real manifest and raises on
    the planted one — that identity is what makes the positive control a real
    discrimination proof rather than two unrelated checks.
    """
    binding = _no_rung_artifact_is_bound(job)
    if not binding["no_rung_artifact_is_bound_to_the_registration"]:
        raise Round0263NodeError(
            "R0263 registration refuses: a judged rung artifact is bound to the node "
            f"that registers the criterion: {binding['inputs_matching_a_rung_marker']}"
        )
    return binding


def _ordering_positive_control(
    real_job: Mapping[str, Any], *, planted_rung_path: str
) -> dict[str, Any]:
    """Run the ordering guard twice: real manifest passes, planted-rung manifest raises."""
    real_binding = _no_rung_artifact_is_bound(real_job)
    real_passed = False
    real_error = None
    try:
        _refuse_if_rung_bound(real_job)
        real_passed = True
    except Round0263NodeError as error:
        real_error = f"{type(error).__name__}: {error}"

    planted_job = {
        "expected_inputs": list(real_job.get("expected_inputs") or [])
        + [{"canonical_path": planted_rung_path}]
    }
    planted_binding = _no_rung_artifact_is_bound(planted_job)
    planted_raised = False
    planted_error = None
    try:
        _refuse_if_rung_bound(planted_job)
    except Round0263NodeError as error:
        planted_raised = True
        planted_error = str(error)

    # The positive control is only evidence if the planted defect actually
    # fires the guard: a control that "passed" without the planted-rung
    # manifest being refused would prove nothing (review-0260-01). Assert the
    # measurement here, co-located with the plant, so a non-discriminating
    # ordering guard fails the node at its source rather than being reported.
    if not (planted_binding["inputs_matching_a_rung_marker"] and planted_raised):
        raise Round0263NodeError(
            "R0263 ordering positive-control did not fire: the planted-rung "
            f"manifest ({planted_rung_path}) was not refused by the guard, so the "
            "ordering claim is unproven"
        )

    return {
        "rung_path_markers": list(RUNG_PATH_MARKERS),
        "real_manifest_binding": real_binding,
        "real_manifest_passes_the_guard": bool(real_passed),
        "real_manifest_error": real_error,
        "planted_rung_path": planted_rung_path,
        "planted_manifest_binds_a_rung_marker": bool(
            planted_binding["inputs_matching_a_rung_marker"]
        ),
        "planted_manifest_is_refused": bool(planted_raised),
        "planted_manifest_error": planted_error,
        "the_ordering_guard_discriminates": bool(real_passed and planted_raised),
        "why_this_exists": (
            "review-0260-01: R0260 registered 'before any judged map was in reach' but "
            "carried no positive control that its ordering guard could FAIL. A guard "
            "proven only by 'it passed' would also pass a guard that inspects nothing. "
            "Running the same refusal on a planted-rung manifest and requiring it to "
            "raise proves the guard reads its inputs."
        ),
    }


# --------------------------------------------------------------------------- #
# node 1 — the registration
# --------------------------------------------------------------------------- #


def run_register(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0263 round0263_nodes.run_register")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0263NodeError("R0263 register handler received another queue")
    node_id = str(active.get("node_id") or REGISTER_ACTION)
    label = "R0263 two-sided k256 purity band registration"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    started_utc = time.time()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0263 registration")

    # Ordering positive-control: the real manifest must bind no rung artifact, AND
    # the same refusal must raise on a planted manifest that binds R0257's verdict.
    ordering = _ordering_positive_control(
        job, planted_rung_path=SEALED["r0257_verdict"]["path"]
    )
    if not ordering["real_manifest_passes_the_guard"]:
        raise Round0263NodeError(
            "R0263 registration refuses: a judged rung artifact is bound to the "
            f"registration node: {ordering['real_manifest_error']}"
        )
    if not ordering["the_ordering_guard_discriminates"]:
        raise Round0263NodeError(
            "R0263 ordering guard does not discriminate: the planted-rung manifest was "
            "not refused, so the guard proves nothing"
        )
    binding = ordering["real_manifest_binding"]

    window = ledger.window("R0263 registration stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0263 registration stage entered")
        wrapped = window.wrap(recorder)

        panel = prompt_contract.read_sealed(
            _bound_path(job, "panel_n29", label="R0255 sealed twenty-nine-cell panel"),
            label="R0255 sealed twenty-nine-cell panel",
        )
        r0256_gate = prompt_contract.read_sealed(
            _bound_path(job, "r0256_gate", label="R0256 sealed repaired n=29 gate"),
            label="R0256 sealed repaired n=29 gate",
        )
        r0260_gate = prompt_contract.read_sealed(
            _bound_path(job, "r0260_one_sided", label="R0260 sealed one-sided floors"),
            label="R0260 sealed one-sided floors",
        )
        if (
            r0256_gate.get("capability") != R0256_GATE_CAPABILITY
            or int(r0256_gate.get("n", -1)) != N_EXACT
        ):
            raise Round0263NodeError("R0256 sealed gate receipt contract changed")
        if (
            r0260_gate.get("capability") != R0260_GATE_CAPABILITY
            or r0260_gate.get("gate_registered") is not True
        ):
            raise Round0263NodeError("R0260 sealed one-sided floor receipt contract changed")
        estimator = str(r0256_gate["chosen_estimator"])
        wrapped("R0263 sealed family panel, R0256 gate and R0260 floors read")

        # ---------------------------------------------------------------- #
        # A. the calibration, unchanged, restricted to the registered estimator
        # ---------------------------------------------------------------- #
        calibration_started = time.monotonic()
        restricted = calibration.calibrate(N_EXACT, names=REGISTERED_ESTIMATOR_NAMES)
        calibration_wall_s = time.monotonic() - calibration_started
        entry = restricted["candidates"][estimator]
        k_one = float(entry["one_sided"]["calibrated_multiplier"])
        k_two = float(entry["two_sided"]["calibrated_multiplier"])
        sealed_calibration = r0256_gate["calibration"]
        calibration_match = _bitwise_same(
            {
                "one_sided_multiplier": float(sealed_calibration["one_sided_multiplier"]),
                "two_sided_multiplier": float(sealed_calibration["two_sided_multiplier"]),
            },
            {"one_sided_multiplier": k_one, "two_sided_multiplier": k_two},
        )
        sealed_constant_match = _bitwise_same(
            {"k1": SEALED_K1, "k2": SEALED_K2},
            {"k1": k_one, "k2": k_two},
        )
        wrapped("R0263 restricted calibration reproduced bitwise")

        # ---------------------------------------------------------------- #
        # B. the family, the RESTORED band, and the retained/untouched floors
        # ---------------------------------------------------------------- #
        family, series, log_series = _family_from_panel(panel)
        family_guard = assert_family_is_2m_only(list(DEFINING_CELL_IDS))
        family_disjointness = assert_no_rung_map_in_the_gate_family(list(DEFINING_CELL_IDS))
        family_plants = rung_family_purity_controls(list(DEFINING_CELL_IDS))
        wrapped("R0263 family purity guard and its rung-map plants complete")

        band = _fit_two_sided_k256_band(estimator, log_series[RESTORED_METRIC], k_two)
        band_match = _bitwise_same(
            {
                "ratio_lower": float(
                    r0256_gate["registered_criteria"][RESTORED_METRIC]["ratio_lower"]
                ),
                "ratio_upper": float(
                    r0256_gate["registered_criteria"][RESTORED_METRIC]["ratio_upper"]
                ),
            },
            {"ratio_lower": band["ratio_lower"], "ratio_upper": band["ratio_upper"]},
        )
        band_sealed_constant_match = _bitwise_same(
            {
                "ratio_lower": SEALED_K256_BAND["ratio_lower"],
                "ratio_upper": SEALED_K256_BAND["ratio_upper"],
                "log_centre": SEALED_K256_BAND["log_centre"],
                "log_scale": SEALED_K256_BAND["log_scale"],
            },
            {
                "ratio_lower": band["ratio_lower"],
                "ratio_upper": band["ratio_upper"],
                "log_centre": band["log_centre"],
                "log_scale": band["log_scale"],
            },
        )

        retained_floor = one_sided_ratio_floor(
            estimator, log_series[RESTORED_METRIC], k_one
        )
        retained_floor_ratio = float(retained_floor["ratio_floor"])
        k1024_floor = one_sided_ratio_floor(
            estimator, log_series[UNCHANGED_RATIO_METRIC], k_one
        )
        k1024_floor_ratio = float(k1024_floor["ratio_floor"])
        ffr_floor = floor_at(estimator, series[UNCHANGED_SCALAR_METRIC], k_one)
        untouched_match = _bitwise_same(
            {
                "retained_k256_one_sided_floor": float(
                    r0260_gate["registered_ratio_floors"][RESTORED_METRIC]
                ),
                "k1024_one_sided_floor": float(
                    r0260_gate["registered_ratio_floors"][UNCHANGED_RATIO_METRIC]
                ),
                "ffr_floor": float(r0260_gate["registered_floors"][UNCHANGED_SCALAR_METRIC]),
            },
            {
                "retained_k256_one_sided_floor": retained_floor_ratio,
                "k1024_one_sided_floor": k1024_floor_ratio,
                "ffr_floor": ffr_floor,
            },
        )
        untouched_sealed_constant_match = _bitwise_same(
            {
                "retained_k256_one_sided_floor": SEALED_K256_ONE_SIDED_FLOOR,
                "k1024_one_sided_floor": SEALED_K1024_ONE_SIDED_FLOOR,
                "ffr_floor": SEALED_FFR_FLOOR,
            },
            {
                "retained_k256_one_sided_floor": retained_floor_ratio,
                "k1024_one_sided_floor": k1024_floor_ratio,
                "ffr_floor": ffr_floor,
            },
        )
        wrapped("R0263 restored band and retained/untouched floors fitted bitwise")

        # ---------------------------------------------------------------- #
        # C. attainability, two-sided power, mixed panel false-alarm rate
        # ---------------------------------------------------------------- #
        identity = attainability_against_the_identity_bound(n=N_EXACT, multiplier=k_two)
        identity_bound_match = _bitwise_same(
            {"identity_bound": identity_bound(N_EXACT)},
            {"identity_bound": SEALED_IDENTITY_BOUND_AT_N29},
        )
        estimator_attainability = attainability(estimator, n=N_EXACT, multiplier=k_two)
        witness = positive_scale_witness(estimator, k_two, n=N_EXACT)
        centre_arr, scale_arr = restricted["_arrays"][estimator]
        two_sided_power = two_sided_detection_power(centre_arr, scale_arr, k_two)
        two_sided_false_fail = float(entry["two_sided"]["new_cell_false_fail_rate"])
        one_sided_false_fail = float(entry["one_sided"]["new_cell_false_fail_rate"])
        panel_far = _panel_false_alarm_rate_mixed(
            one_sided_rate=one_sided_false_fail, two_sided_rate=two_sided_false_fail
        )
        family_over = sorted(
            cell["cell_id"]
            for cell in family
            if float(cell["ratios"][PURITY_RATIO_KEYS[RESTORED_METRIC]]) > band["ratio_upper"]
        )
        family_under = sorted(
            cell["cell_id"]
            for cell in family
            if float(cell["ratios"][PURITY_RATIO_KEYS[RESTORED_METRIC]]) < band["ratio_lower"]
        )
        wrapped("R0263 attainability, two-sided power and mixed panel FAR computed")

        # ---------------------------------------------------------------- #
        # D. the mixed registry, the descriptive block, and their controls
        # ---------------------------------------------------------------- #
        criteria = _registered_criteria_k256_two_sided(
            band=band, k1024_floor=k1024_floor_ratio, ffr_floor=ffr_floor
        )
        descriptive = _descriptive_one_sided_diagnostics(
            one_sided_floor=retained_floor_ratio, k1=k_one
        )
        registry_guard = _assert_registry_restores_k256_two_sided(criteria, descriptive)
        leak_guard = _assert_no_descriptive_value_in_the_gate(
            criteria=criteria, descriptive=descriptive
        )
        registry_plants = _registry_controls_k256_two_sided(
            criteria=criteria, descriptive=descriptive, floor=retained_floor_ratio
        )
        sidedness = _sidedness_controls_k256_two_sided(band)
        inertness = _descriptive_floor_never_gates_controls(
            observations=SEALED_RUNG_K256, band=band, floor=retained_floor_ratio
        )
        wrapped("R0263 mixed registry, descriptive block and controls complete")

        # ---------------------------------------------------------------- #
        # E. the single held-out flip, computed here against the fresh band
        # ---------------------------------------------------------------- #
        held_out_ratio = float(SEALED_HELD_OUT_FLIP["k256"])
        held_out_two_sided = _judge_k256_two_sided(held_out_ratio, band)
        held_out_one_sided = _judge_one_sided_floor(held_out_ratio, retained_floor_ratio)
        held_out_flip = {
            **{k: SEALED_HELD_OUT_FLIP[k] for k in SEALED_HELD_OUT_FLIP},
            "restored_two_sided": {
                "verdict": held_out_two_sided["verdict"],
                "direction": held_out_two_sided["direction"],
                "z_madn": held_out_two_sided["z_madn"],
                "margin": held_out_two_sided["margin"],
            },
            "retired_one_sided_floor": {
                "verdict": held_out_one_sided["verdict"],
                "margin": held_out_one_sided["margin"],
                "floor": retained_floor_ratio,
            },
            "flip": (
                f"{held_out_one_sided['verdict']}(R0260 one-sided) -> "
                f"{held_out_two_sided['verdict']}(restored two-sided)"
            ),
            "note": (
                "this 2M held-out cell's observation lives in R0255's floors_n29 "
                "all-cells scoring, which NEITHER R0263 node binds. It is a disclosed "
                "literal, judged here against the freshly-fit band — computed in this "
                "node, not fabricated as a binding. R0260 re-scored only the 29 "
                "defining cells, so this disagreement is in no published artifact."
            ),
        }
        gate.finish("R0263 registration stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    cuda_mappings = _cuda_mappings()

    execution_checks = {
        "no_rung_artifact_is_bound_to_the_registration": bool(
            binding["no_rung_artifact_is_bound_to_the_registration"]
        ),
        "the_ordering_guard_discriminates": bool(
            ordering["the_ordering_guard_discriminates"]
        ),
        "the_calibration_is_restricted_to_the_registered_estimator": (
            tuple(restricted["candidates"]) == REGISTERED_ESTIMATOR_NAMES
        ),
        "the_calibration_reproduces_r0256_bitwise": bool(
            calibration_match["every_value_is_bitwise_identical"]
        ),
        "the_multipliers_reproduce_the_sealed_constants_bitwise": bool(
            sealed_constant_match["every_value_is_bitwise_identical"]
        ),
        "the_two_sided_band_reproduces_r0255_bitwise": bool(
            band_match["every_value_is_bitwise_identical"]
        ),
        "the_two_sided_band_reproduces_the_sealed_constant_bitwise": bool(
            band_sealed_constant_match["every_value_is_bitwise_identical"]
        ),
        "the_retained_and_untouched_floors_reproduce_r0260_bitwise": bool(
            untouched_match["every_value_is_bitwise_identical"]
        ),
        "the_retained_and_untouched_floors_reproduce_the_sealed_constants_bitwise": bool(
            untouched_sealed_constant_match["every_value_is_bitwise_identical"]
        ),
        "the_family_is_the_2m_universe_only": bool(
            family_guard["family_is_the_2m_universe_only"]
        ),
        "the_family_and_the_judged_maps_are_disjoint": bool(
            family_disjointness["judged_and_defining_are_disjoint"]
        ),
        "every_planted_family_defect_was_refused": bool(
            family_plants["every_planted_defect_was_refused"]
        ),
        "the_gated_multiplier_is_below_the_identity_bound": bool(
            identity["multiplier_is_below_the_identity_bound"]
        ),
        "the_identity_bound_reproduces_the_sealed_constant_bitwise": bool(
            identity_bound_match["every_value_is_bitwise_identical"]
        ),
        "every_defining_cell_can_fail": bool(
            estimator_attainability["every_defining_cell_can_fail"]
        ),
        "a_positive_scale_witness_fails_its_own_floor": bool(
            witness["lowest_cell_fails_its_own_floor"]
            and witness["scale_is_strictly_positive"]
        ),
        "the_registry_restores_k256_as_a_two_sided_band": bool(registry_guard["holds"]),
        "no_descriptive_value_is_reachable_inside_the_gate": bool(leak_guard["holds"]),
        "every_planted_registry_defect_is_refused": bool(
            registry_plants["every_planted_defect_is_refused"]
        ),
        "every_sidedness_control_held": bool(sidedness["every_control_held"]),
        "the_descriptive_floor_never_gates": bool(inertness["holds"]),
        "the_panel_rate_is_above_the_per_metric_rates": (
            panel_far["panel_false_alarm_rate_under_independence"] > one_sided_false_fail
            and panel_far["panel_false_alarm_rate_under_independence"] > two_sided_false_fail
        ),
        "no_cuda_library_is_mapped_into_this_process": not cuda_mappings,
    }
    if not all(execution_checks.values()):
        raise Round0263NodeError(f"R0263 registration checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body: dict[str, Any] = {
        "round_id": ROUND_ID,
        "release_sha": str(active["manifest"]["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": None,
        "schema": GATE_SCHEMA,
        "capability": GATE_CAPABILITY,
        "capabilities": [GATE_CAPABILITY],
        "node": node_id,
        "node_started_utc": started_utc,
        "abort_flag_precondition": abort_flag,
        "n": N_EXACT,
        "owner_ruling": OWNER_RULING,
        "owner_ruling_date": OWNER_RULING_DATE,
        "owner_ruling_section": OWNER_RULING_SECTION,
        "why_two_sided": WHY_TWO_SIDED,
        "no_tuning_statement": NO_TUNING_STATEMENT,
        "chosen_estimator": estimator,
        "supersedes_k256_criterion_in": SUPERSEDES_K256_CRITERION_IN,
        "supersession_note": (
            "this artifact restores the two-sided PURITY_FIDELITY_K256 criterion that "
            f"{SUPERSEDES_K256_CRITERION_IN} retired to a one-sided floor, and nothing "
            "else. Its band is R0255's, bitwise; its k1024 and ffr floors are R0260's, "
            "bitwise. Every number R0255/R0256/R0260 published remains true as "
            "published; what changes is that the k256 upper edge gates again."
        ),
        "registration_precedes_any_judgement": dict(binding),
        "ordering_positive_control": ordering,
        # --- A ---
        "calibration": {
            "estimators_calibrated": list(restricted["candidates"]),
            "families_simulated": int(restricted["families_simulated"]),
            "seed": int(restricted["seed"]),
            "content": float(restricted["content"]),
            "confidence": float(restricted["confidence"]),
            "one_sided_multiplier": k_one,
            "two_sided_multiplier": k_two,
            "wall_s": calibration_wall_s,
            "reproduces_r0256_multipliers": calibration_match,
            "reproduces_the_sealed_constants": sealed_constant_match,
            "why_unchanged": (
                "the calibration is a property of the estimator at n = 29 on a "
                "Gaussian null and reads no cell of this program. The addendum changed "
                "which tail of k256 is gated, not how the multiplier is derived, so "
                "both multipliers come back identical to R0255/R0256 -- and do."
            ),
        },
        # --- B ---
        "family": {
            "n": N_EXACT,
            "exact_family_seeds": list(EXACT_FAMILY_SEEDS),
            "defining_cell_ids": list(DEFINING_CELL_IDS),
            "guard": family_guard,
            "disjointness": family_disjointness,
            "planted_controls": family_plants,
            "rung_cell_ids_this_family_must_exclude": list(rung_cell_ids()),
        },
        "registered_criteria": criteria,
        "two_sided_k256_band": band,
        "restored_band_reproduces_r0255_bitwise": band_match,
        "restored_band_reproduces_the_sealed_constant_bitwise": band_sealed_constant_match,
        "retained_and_untouched_floors": {
            "retained_k256_one_sided_floor": retained_floor_ratio,
            "k1024_one_sided_floor": k1024_floor_ratio,
            "ffr_floor": ffr_floor,
            "reproduce_r0260_bitwise": untouched_match,
            "reproduce_the_sealed_constants_bitwise": untouched_sealed_constant_match,
            "note": (
                "k1024 and ffr are R0260's one-sided floors, carried through untouched; "
                "the k256 one-sided floor is retained but demoted to descriptive."
            ),
        },
        # --- C ---
        "attainability_against_the_identity_bound": identity,
        "identity_bound_at_n": IDENTITY_BOUND_AT_N,
        "identity_bound_reproduces_the_sealed_constant": identity_bound_match,
        "estimator_attainability": estimator_attainability,
        "positive_scale_witness": witness,
        "two_sided_detection_power": dict(two_sided_power),
        "two_sided_new_cell_false_fail_rate": two_sided_false_fail,
        "one_sided_new_cell_false_fail_rate": one_sided_false_fail,
        "panel_false_alarm_rate": panel_far,
        "family_cells_over_the_upper_edge": family_over,
        "family_cells_under_the_lower_edge": family_under,
        "tolerance_content": TOLERANCE_CONTENT,
        "tolerance_confidence": TOLERANCE_CONFIDENCE,
        # --- D ---
        "descriptive_only_never_a_gate": descriptive,
        "registry_guard": registry_guard,
        "leak_guard": leak_guard,
        "registry_controls": registry_plants,
        "sidedness_controls": sidedness,
        "descriptive_floor_never_gates": inertness,
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        # --- E ---
        "held_out_flip": held_out_flip,
        "gate_status": "registered-and-contingent-pending-review",
        "gate_registered": True,
        "execution_checks": execution_checks,
        "evaluation_performed": False,
        "training_performed": False,
        "gpu_used": bool(cuda_mappings),
        "cuda_libraries_mapped_into_this_process": cuda_mappings,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "sources": {
            "r0255_panel_n29": expected_input_signature(
                _bound_path(job, "panel_n29", label="R0255 panel")
            ),
            "r0256_repaired_gate_n29": expected_input_signature(
                _bound_path(job, "r0256_gate", label="R0256 gate")
            ),
            "r0260_one_sided_floors_n29": expected_input_signature(
                _bound_path(job, "r0260_one_sided", label="R0260 floors")
            ),
        },
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    }
    _seal(output, REGISTRATION_ARTIFACT, body)
    print(json.dumps({
        "capability": GATE_CAPABILITY,
        "n": N_EXACT,
        "two_sided_multiplier": k_two,
        "identity_bound": identity["identity_bound"],
        "ratio_lower": band["ratio_lower"],
        "ratio_upper": band["ratio_upper"],
        "retained_one_sided_floor": retained_floor_ratio,
        "panel_false_alarm_rate": panel_far[
            "panel_false_alarm_rate_under_independence"
        ],
        "widest_gap_s": gaps.get("widest_gap_s"),
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# node 2 — the re-adjudication
# --------------------------------------------------------------------------- #


def run_readjudicate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0263 round0263_nodes.run_readjudicate")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0263NodeError("R0263 readjudicate handler received another queue")
    node_id = str(active.get("node_id") or READJUDICATE_ACTION)
    label = "R0263 R0257 re-adjudication under the restored two-sided k256 band"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    started_utc = time.time()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0263 re-adjudication")

    window = ledger.window("R0263 re-adjudication stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0263 re-adjudication stage entered")
        wrapped = window.wrap(recorder)

        registration_path, registration_signature = _intra_queue_signature(
            job["registration"], label="R0263 sealed two-sided registration"
        )
        registration = prompt_contract.read_sealed(
            registration_path, label="R0263 sealed two-sided registration"
        )
        if (
            registration.get("capability") != GATE_CAPABILITY
            or registration.get("schema") != GATE_SCHEMA
            or registration.get("gate_registered") is not True
        ):
            raise Round0263NodeError("R0263 registration artifact contract changed")

        marker_path = str(job["registration_done_marker"])
        if not os.path.exists(marker_path):
            raise Round0263NodeError(
                f"R0263 registration done-marker is absent at {marker_path}"
            )
        with open(marker_path, "rb") as handle:
            marker = json.load(handle)
        sealed_at = os.path.getmtime(registration_path)
        ordering = {
            "registration_artifact": registration_path,
            "registration_sha256": registration_signature["sha256"],
            "registration_bytes": registration_signature["bytes"],
            "registration_done_marker": marker_path,
            "registration_node": str(marker.get("node") or marker.get("id") or ""),
            "registration_sealed_at_epoch_s": sealed_at,
            "readjudication_started_at_epoch_s": started_utc,
            "seconds_between_registration_and_adjudication": started_utc - sealed_at,
            "registration_precedes_adjudication": sealed_at < started_utc,
            "the_registration_node_bound_no_rung_artifact": bool(
                registration["registration_precedes_any_judgement"][
                    "no_rung_artifact_is_bound_to_the_registration"
                ]
            ),
            "the_registration_ordering_guard_discriminated": bool(
                registration["ordering_positive_control"][
                    "the_ordering_guard_discriminates"
                ]
            ),
            "why_this_is_receipt_evidence": (
                "the runner runs this node only after the producer's done-marker "
                "exists, and the registration bytes are hashed here. Order is a "
                "property of the queue's receipts, not of prose."
            ),
        }
        if not ordering["registration_precedes_adjudication"]:
            raise Round0263NodeError(
                "R0263 refuses to adjudicate: the registration was not sealed first"
            )
        if not ordering["the_registration_node_bound_no_rung_artifact"]:
            raise Round0263NodeError(
                "R0263 refuses to adjudicate: the registration node was bound to a "
                "judged rung artifact"
            )
        if not ordering["the_registration_ordering_guard_discriminated"]:
            raise Round0263NodeError(
                "R0263 refuses to adjudicate: the registration's ordering guard did "
                "not discriminate, so its ordering claim is unproven"
            )
        wrapped("R0263 registration ordering verified from receipts")

        # The restored band and the retained descriptive floor, read from node 1.
        band = dict(registration["two_sided_k256_band"])
        descriptive_floor = float(
            registration["descriptive_only_never_a_gate"][RESTORED_METRIC][
                "one_sided_floor"
            ]
        )

        r0257_panel = prompt_contract.read_sealed(
            _bound_path(job, "r0257_panel", label="R0257 sealed rung panel"),
            label="R0257 sealed rung panel",
        )
        r0257_verdict = prompt_contract.read_sealed(
            _bound_path(job, "r0257_verdict", label="R0257 sealed rung verdict"),
            label="R0257 sealed rung verdict",
        )
        if (
            r0257_verdict.get("round_id") != "0257"
            or str(r0257_verdict.get("gate_capability_consumed"))
            != R0257_CONSUMED_GATE_CAPABILITY
            or r0257_verdict.get("no_floor_is_written_by_this_round") is None
        ):
            raise Round0263NodeError("R0257 sealed verdict contract changed")
        cells = _r0257_cells(r0257_panel, r0257_verdict)
        wrapped("R0263 R0257 sealed panel and verdict read")

        comparison: dict[str, Any] = {}
        observations = {}
        for cell_id, payload in cells.items():
            observed = float(payload["raw_purity_ratios"]["k256"])
            observations[cell_id] = observed
            published = payload["published_per_metric"][RESTORED_METRIC]
            restored = _judge_k256_two_sided(observed, band)
            r0260_one_sided = _judge_one_sided_floor(observed, descriptive_floor)
            comparison[cell_id] = {
                "observed_k256": observed,
                "observed_as_published_by_r0257": float(published["observed"]),
                "observation_is_bitwise_identical": (
                    observed == float(published["observed"])
                ),
                # --- the restored (gated) verdict ---
                "restored_two_sided_verdict": restored["verdict"],
                "restored_direction": restored["direction"],
                "restored_margin": restored["margin"],
                "restored_z_madn": restored["z_madn"],
                "restored_separation": restored["separation"],
                # --- R0257's published verdict, beside it (not rewritten) ---
                "r0257_verdict": (
                    "PASS" if bool(published["passes"]) else "FAIL"
                ),
                "r0257_criterion": str(published["criterion"]),
                "r0257_kind": str(published["kind"]),
                "r0257_margin": float(published["margin"]),
                "r0257_map_level_verdict": payload["published_verdict"],
                # --- R0260's one-sided verdict, computed from the descriptive floor ---
                "r0260_one_sided_verdict": r0260_one_sided["verdict"],
                "r0260_one_sided_margin": r0260_one_sided["margin"],
                # --- the descriptive one-sided value carried beside the verdict ---
                "descriptive_one_sided_floor": descriptive_floor,
                "restored_matches_r0257": (
                    restored["verdict"]
                    == ("PASS" if bool(published["passes"]) else "FAIL")
                ),
            }
        maps_failing = sorted(
            cid for cid, row in comparison.items()
            if row["restored_two_sided_verdict"] == "FAIL"
        )
        maps_passing = sorted(
            cid for cid, row in comparison.items()
            if row["restored_two_sided_verdict"] == "PASS"
        )
        inertness = _descriptive_floor_never_gates_controls(
            observations=observations, band=band, floor=descriptive_floor
        )
        held_out_flip = dict(registration["held_out_flip"])
        wrapped("R0263 three rung maps re-scored and the inertness controls complete")
        gate.finish("R0263 re-adjudication stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    cuda_mappings = _cuda_mappings()

    observations_identical = all(
        row["observation_is_bitwise_identical"] for row in comparison.values()
    )
    execution_checks = {
        "the_registration_was_sealed_before_this_node_started": bool(
            ordering["registration_precedes_adjudication"]
        ),
        "the_registration_node_bound_no_rung_artifact": bool(
            ordering["the_registration_node_bound_no_rung_artifact"]
        ),
        "the_registration_ordering_guard_discriminated": bool(
            ordering["the_registration_ordering_guard_discriminated"]
        ),
        "every_k256_observation_is_bitwise_r0257s": bool(observations_identical),
        "three_maps_were_re_adjudicated": len(comparison) == len(rung_cell_ids()),
        "every_r0257_verdict_was_fail_as_published": all(
            row["r0257_map_level_verdict"] == R0257_PUBLISHED_VERDICT
            for row in comparison.values()
        ),
        "the_restored_verdict_agrees_with_r0257_on_k256": all(
            row["restored_matches_r0257"] for row in comparison.values()
        ),
        "the_shipped_judge_is_inert_to_the_descriptive_floor": bool(
            inertness["the_shipped_judge_is_inert_to_the_descriptive_floor"]
        ),
        "the_inertness_mutation_is_potent": bool(inertness["the_mutation_is_potent"]),
        "no_cuda_library_is_mapped_into_this_process": not cuda_mappings,
    }
    if not all(execution_checks.values()):
        raise Round0263NodeError(
            f"R0263 re-adjudication checks failed: {execution_checks}"
        )

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    coverage = ledger.receipt()
    body: dict[str, Any] = {
        "round_id": ROUND_ID,
        "release_sha": str(active["manifest"]["release_sha"]),
        "registry_fingerprint": registry_fingerprint(),
        "safety_note": SAFETY_NOTE,
        "cuvs_calls": 0,
        "child_processes_launched": 0,
        "signal_delivered": None,
        "schema": READJUDICATION_SCHEMA,
        "capability": READJUDICATION_CAPABILITY,
        "capabilities": [READJUDICATION_CAPABILITY],
        "node": node_id,
        "node_started_utc": started_utc,
        "abort_flag_precondition": abort_flag,
        "owner_ruling": OWNER_RULING,
        "owner_ruling_date": OWNER_RULING_DATE,
        "what_this_is": (
            "a dated ADDENDUM re-adjudicating R0257's three 6250k rung maps at k256 "
            "under the two-sided band this round's first node restored. R0257's "
            "published FAILs stand; R0260's re-adjudication is not rewritten. This "
            "adds a record, exactly as R0260's addendum added one over R0257."
        ),
        "r0257_result_is_not_rewritten": True,
        "r0260_readjudication_is_not_rewritten": True,
        "registration_ordering": ordering,
        "registration_consumed": {
            "capability": registration["capability"],
            "schema": registration["schema"],
            "sha256": registration_signature["sha256"],
            "two_sided_k256_band": band,
            "descriptive_one_sided_floor": descriptive_floor,
            "panel_false_alarm_rate": registration["panel_false_alarm_rate"],
        },
        "re_adjudication": comparison,
        "verdict_summary": {
            "maps_judged": len(comparison),
            "maps_passing": maps_passing,
            "maps_failing": maps_failing,
            "the_three_rung_maps_fail_over_under_the_restored_band": (
                maps_failing == sorted(rung_cell_ids())
            ),
        },
        "held_out_flip": held_out_flip,
        "descriptive_floor_never_gates": inertness,
        "conformity_marking": DESCRIPTIVE_MARKING,
        "what_this_does_not_establish": [
            "that the three maps are bad maps: the restored band asks only that a map "
            "not sit outside the 2M family's 95/95 two-sided tolerance band at k256, "
            "and nothing here connects a purity ratio to a downstream use",
            "anything at 12.5M, 25M, 50M or 100M: three cells at one rung",
            "that two-sided is the right long-run answer: review-0257-01's structural "
            "objection (the band measures conformity to the 2M family, not quality) is "
            "untouched, which is why option 3 remains the target; this is a stopgap",
        ],
        "gate_status": "no-gate-registered-here-the-restored-band-was-consumed-unchanged",
        "gate_registered": False,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": bool(cuda_mappings),
        "cuda_libraries_mapped_into_this_process": cuda_mappings,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "sources": {
            "r0257_rung_panel": expected_input_signature(
                _bound_path(job, "r0257_panel", label="R0257 panel")
            ),
            "r0257_rung_verdict": expected_input_signature(
                _bound_path(job, "r0257_verdict", label="R0257 verdict")
            ),
            "r0263_registration": registration_signature,
        },
        "gap_report": gaps,
        "enforcement_poll_spacing": scored_gate,
        "guard_tail": tail,
        "poll_coverage": coverage,
        "observed_span_s": coverage["observed_span_s"],
        "node_wall_s": coverage["node_wall_s"],
        "peak_host_rss_gib": peak_rss_gib,
        "wall_seconds": time.monotonic() - started,
    }
    _seal(output, READJUDICATION_ARTIFACT, body)
    print(json.dumps({
        "capability": READJUDICATION_CAPABILITY,
        "maps_passing": maps_passing,
        "maps_failing": maps_failing,
        "restored_agrees_with_r0257": all(
            row["restored_matches_r0257"] for row in comparison.values()
        ),
        "registration_precedes_adjudication": ordering[
            "registration_precedes_adjudication"
        ],
        "widest_gap_s": gaps.get("widest_gap_s"),
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0263 round0263_nodes.run_job")
    action = str(job.get("action") or "")
    if action == REGISTER_ACTION:
        run_register(active, job)
        return
    if action == READJUDICATE_ACTION:
        run_readjudicate(active, job)
        return
    raise Round0263NodeError(f"R0263 unknown action {action!r}")


__all__ = [
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "READJUDICATE_ACTION",
    "READJUDICATION_ARTIFACT",
    "READJUDICATION_CAPABILITY",
    "READJUDICATION_SCHEMA",
    "REGISTER_ACTION",
    "REGISTRATION_ARTIFACT",
    "ROUND_ID",
    "Round0263NodeError",
    "run_job",
    "run_readjudicate",
    "run_register",
]
