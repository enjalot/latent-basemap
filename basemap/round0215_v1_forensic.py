"""Frozen contract for the v1 150M map forensic.

The MiniLM-100M v2 plan rests on a diagnosis: the v1 150M map's visible
concentration-clumps and inter-cluster filaments are the signature of a
low-recall quantized graph (Modal-era PQ48x8, recall@15 ~ 0.257) plus
uncontrolled duplicate mass. That diagnosis has never been measured. R0215
measures it, so the rebuild rests on evidence rather than on the map looking
wrong.

**Selector thresholds are fixed here, before any metric is computed**, from a
read-only structure probe over a stride-10 subsample. The probe found the two
structures separate cleanly: the densest bin holds 23,557 points against a
median occupied bin of 2, and while the strongest inter-clump segment runs at
100% occupancy and 31x background, 27 of 66 inter-clump segments sit at or
below background. Elevated connectors are therefore a distinguishable subset,
not an artifact of "everything between clumps looks dense".

Three populations are compared, and the third is what makes the result mean
anything:

* **clump** — bins in a connected component of the >= p99.9 density mask.
* **filament** — bins inside a corridor between two clump centroids whose
  median density is at least FILAMENT_BACKGROUND_RATIO x background, excluding
  any bin already classified as clump.
* **field** — bins that are neither, **density-matched to the filament bins**.
  Without this control, "filament points have bad edges" could be nothing more
  than a restatement of "low-density points have bad edges".
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


ROUND_ID = "0215"
CAPABILITY = "minilm-v1-150m-map-forensic-v1"
FORENSIC_SCHEMA = "round0215-minilm-v1-150m-forensic-v1"

ROWS = 150_000_000
DIMENSION = 384
GRAPH_K = 15

# --- selector, fixed from the pre-registration probe -----------------------
HEATMAP_BINS = 1536
COORD_TRIM_PERCENTILE = (0.05, 99.95)
CLUMP_DENSITY_PERCENTILE = 99.9
FILAMENT_BACKGROUND_RATIO = 3.0
FILAMENT_CORRIDOR_BINS = 2
FILAMENT_SEGMENT_OCCUPANCY = 0.95
TOP_CLUMPS_FOR_SEGMENTS = 12
SEGMENT_INTERIOR = (0.12, 0.88)
POPULATIONS = ("clump", "filament", "field")
SAMPLE_ROWS_PER_POPULATION = 2_000
SAMPLE_SEED = 215

# --- pre-registered predictions --------------------------------------------
#: The plan's stated predictions, written down before measurement so the
#: forensic can disconfirm them. A miss is a publishable result: it would mean
#: the rebuild premise is wrong and the v1 damage has another cause.
PREDICTIONS = {
    "filament_edge_precision_below_field": (
        "mean edge precision on filament rows is lower than on the "
        "density-matched field control"
    ),
    "clump_duplicate_membership_above_field": (
        "duplicate-family membership rate in clump cores exceeds the field "
        "control"
    ),
}
#: A difference smaller than this is reported as "no separation", not as support.
MINIMUM_MEANINGFUL_PRECISION_GAP = 0.02
MINIMUM_MEANINGFUL_RATE_GAP = 0.02

#: Exact neighbours are computed over the int8 residency corpus that R0033/R0034
#: actually bound, dequantized to fp32 — not over the original fp16 shards. That
#: is the space the canonical graph indexes, and int8 round-trip error is orders
#: of magnitude below the PQ48x8 error being measured. Stated so a reviewer can
#: weigh it rather than discover it.
EXACT_REFERENCE = "exact fp32 cosine over the dequantized R0025 int8 150M corpus"


class Round0215Error(RuntimeError):
    """The registered v1 forensic contract changed."""


def classify_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    """Fail closed unless all three populations are populated and disjoint."""
    missing = [p for p in POPULATIONS if int(counts.get(p, 0)) <= 0]
    if missing:
        raise Round0215Error(
            f"R0215 populations {missing} are empty; the selector did not "
            "separate the structures it was calibrated on"
        )
    return {population: int(counts[population]) for population in POPULATIONS}


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Round0215Error("R0215 cannot average an empty population")
    return sum(float(v) for v in values) / len(values)


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((float(v) - mean) ** 2 for v in values) / (len(values) - 1))


def population_stats(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        raise Round0215Error("R0215 population is empty")

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return ordered[idx]

    return {
        "n": n,
        "mean": _mean(ordered),
        "sd": _sd(ordered),
        "p10": pct(10),
        "median": pct(50),
        "p90": pct(90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def verdict(
    *,
    edge_precision: Mapping[str, Mapping[str, Any]],
    duplicate_rate: Mapping[str, float],
) -> dict[str, Any]:
    """Adjudicate both predictions, and say plainly when neither separates."""
    for table, label in ((edge_precision, "edge precision"), (duplicate_rate, "duplicate rate")):
        if set(table) != set(POPULATIONS):
            raise Round0215Error(f"R0215 {label} is missing a population")
    precision_gap = (
        float(edge_precision["field"]["mean"]) - float(edge_precision["filament"]["mean"])
    )
    duplicate_gap = float(duplicate_rate["clump"]) - float(duplicate_rate["field"])
    filament_supported = precision_gap >= MINIMUM_MEANINGFUL_PRECISION_GAP
    clump_supported = duplicate_gap >= MINIMUM_MEANINGFUL_RATE_GAP
    return {
        "predictions": dict(PREDICTIONS),
        "filament_edge_precision_below_field": {
            "field_mean": float(edge_precision["field"]["mean"]),
            "filament_mean": float(edge_precision["filament"]["mean"]),
            "gap": precision_gap,
            "minimum_meaningful_gap": MINIMUM_MEANINGFUL_PRECISION_GAP,
            "supported": filament_supported,
        },
        "clump_duplicate_membership_above_field": {
            "clump_rate": float(duplicate_rate["clump"]),
            "field_rate": float(duplicate_rate["field"]),
            "gap": duplicate_gap,
            "minimum_meaningful_gap": MINIMUM_MEANINGFUL_RATE_GAP,
            "supported": clump_supported,
        },
        "both_predictions_supported": bool(filament_supported and clump_supported),
        "rebuild_premise": (
            "supported" if (filament_supported or clump_supported)
            else "NOT supported by either prediction; the v1 damage needs "
                 "another explanation before the rebuild cites this diagnosis"
        ),
        "exact_reference": EXACT_REFERENCE,
    }


__all__ = [
    "CAPABILITY",
    "CLUMP_DENSITY_PERCENTILE",
    "COORD_TRIM_PERCENTILE",
    "DIMENSION",
    "EXACT_REFERENCE",
    "FILAMENT_BACKGROUND_RATIO",
    "FILAMENT_CORRIDOR_BINS",
    "FILAMENT_SEGMENT_OCCUPANCY",
    "FORENSIC_SCHEMA",
    "GRAPH_K",
    "HEATMAP_BINS",
    "MINIMUM_MEANINGFUL_PRECISION_GAP",
    "MINIMUM_MEANINGFUL_RATE_GAP",
    "POPULATIONS",
    "PREDICTIONS",
    "ROUND_ID",
    "ROWS",
    "Round0215Error",
    "SAMPLE_ROWS_PER_POPULATION",
    "SAMPLE_SEED",
    "SEGMENT_INTERIOR",
    "TOP_CLUMPS_FOR_SEGMENTS",
    "classify_summary",
    "population_stats",
    "verdict",
]
