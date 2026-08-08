"""Binding MiniLM mixed-2M quality-gate registration for R0219.

The R0161/R0193 method, unchanged: **family mean minus two sample standard
deviations (ddof=1)**, higher-is-better, computed over the cells of one
commensurate seed family on one universe. Nothing is trained, nothing is scored,
and no floor from another universe is touched.

What *is* new here is the metric set, and the reason is measured rather than
stylistic.

**The gate covers FFR and the two purity fidelities. It does not cover
`density_v2` or `heldout_recall_at_10`.** R0214 measured a two-cell spread under
an identical treatment and found FFR differed by `0.08%` and purity by `< 0.7%`,
while `density_v2` moved `10.30%` and held-out recall@10 moved `47.06%` on a base
of `0.0034`. A mean - 2 sigma floor built on those two quantities would be
calibrating against its own sampling noise: the floor would be enormous relative
to the signal, it would pass anything, and a later regression in the metrics that
*do* separate treatments would slip under it. So `density_v2` stays what it is
everywhere else in this program — **diagnostic-only, transcribed** — and
`heldout_recall_at_10` is not in R0218's panel at all.

The other honesty constraint is `n`. Four cells is enough to compute a sample
standard deviation and not enough to pretend it is precise. The receipt states
`n = 4` in the payload, names the estimator, and carries the per-metric spread so
a reader can see how thin the estimate is instead of inferring it.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from .round0218_minilm_2m_panel import (
    CAPABILITY as PANEL_CAPABILITY,
    CORPUS_SLUGS,
    DIAGNOSTIC_METRICS,
    EVALUATION_SCHEMA as PANEL_SCHEMA,
    PANEL_METRICS,
    ROWS,
    SEEDS,
)


ROUND_ID = "0219"
CAPABILITY = "minilm-mixed-2m-quality-gates-v1"
GATE_SCHEMA = "round0219-minilm-mixed-2m-quality-gates-v1"
FORMULA = "family mean - 2 * sample standard deviation (ddof=1)"
MULTIPLIER = 2.0
SD_DDOF = 1

#: The registered gate family. FFR and the two purity fidelities only.
GATE_METRICS: tuple[str, ...] = (
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)

#: Deliberately excluded, with the measurement that justifies each exclusion.
EXCLUDED_METRICS: dict[str, str] = {
    "density_v2": (
        "R0214 measured a 10.30% two-cell spread on an identical treatment "
        "against 0.08% for FFR; a mean - 2 sigma floor here would calibrate "
        "against its own noise. Reported as diagnostic-only, transcribed."
    ),
    "heldout_recall_at_10": (
        "R0214 measured a 47.06% two-cell spread on a base of 0.0034, and R0218's "
        "panel does not compute it on this universe at all."
    ),
}

#: Per-corpus FFR is reported alongside the gates and is explicitly **not**
#: registered as a floor: four cells x four corpora is a thinner estimate than
#: the pooled metric, and no rung has yet been run for it to gate.
CORPUS_SLICE_ROLE = "descriptive per-corpus spread; not a registered floor"

PRECISION_NOTE = (
    "n = 4 cells. The sample standard deviation is a four-point estimate with "
    "3 degrees of freedom; it is reported because the family was designed to "
    "supply it, not because four points make it precise."
)


class Round0219Error(RuntimeError):
    """The R0219 gate registration changed after preregistration."""


def _gate_cell(metric: str, values: list[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS) or any(
        not math.isfinite(value) or not 0.0 < value <= 1.0 for value in values
    ):
        raise Round0219Error(f"R0219 gate metric {metric} is invalid")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    floor = mean - MULTIPLIER * sample_sd
    if not math.isfinite(floor):
        raise Round0219Error(f"R0219 gate floor {metric} is invalid")
    spread = max(values) - min(values)
    return {
        "direction": "higher-is-better",
        "seed_order": list(SEEDS),
        "values": values,
        "mean": mean,
        "sample_sd_ddof1": sample_sd,
        "multiplier": MULTIPLIER,
        "floor": floor,
        "observed_spread": spread,
        "relative_spread_of_mean": (spread / mean) if mean else None,
        "floor_is_vacuous": floor <= 0.0,
    }


def register_minilm_gates(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Turn R0218's four scored cells into the registered mean - 2 sigma family."""
    if set(GATE_METRICS) & set(EXCLUDED_METRICS):
        raise Round0219Error("R0219 gate set overlaps the excluded metrics")
    if not set(GATE_METRICS).issubset(set(PANEL_METRICS)):
        raise Round0219Error("R0219 gate set is not a subset of the R0218 panel")
    if (
        evidence.get("schema") != PANEL_SCHEMA
        or evidence.get("round_id") != "0218"
        or evidence.get("capability") != PANEL_CAPABILITY
        or evidence.get("outcome") != "minilm-mixed-2m-four-seed-panel-complete"
        or evidence.get("seeds") != list(SEEDS)
        or int(evidence.get("n", -1)) != len(SEEDS)
        or evidence.get("gate_registered") is not False
        or evidence.get("gate_registerable_here") is not False
        or evidence.get("gate_registration_deferred_to_reviewed_cpu_round") is not True
        or evidence.get("training_performed") is not False
        or evidence.get("metrics") != list(PANEL_METRICS)
    ):
        raise Round0219Error("R0219 panel-family premise changed")
    checks = evidence.get("execution_checks") or {}
    if not checks or not all(bool(value) for value in checks.values()):
        raise Round0219Error("R0219 refuses a panel whose execution checks did not pass")

    cells = evidence.get("panel_metric_cells") or {}
    if set(cells) != {str(seed) for seed in SEEDS}:
        raise Round0219Error("R0219 gate family is incomplete")
    for seed in SEEDS:
        if set(cells[str(seed)]) != set(PANEL_METRICS):
            raise Round0219Error(f"R0219 seed-{seed} metric coverage changed")

    gates = {
        metric: _gate_cell(
            metric, [float(cells[str(seed)][metric]) for seed in SEEDS]
        )
        for metric in GATE_METRICS
    }

    diagnostics: dict[str, Any] = {}
    for metric in DIAGNOSTIC_METRICS:
        values = [float(cells[str(seed)][metric]) for seed in SEEDS]
        if any(not math.isfinite(value) for value in values):
            raise Round0219Error(f"R0219 diagnostic {metric} is not finite")
        mean = statistics.fmean(values)
        diagnostics[metric] = {
            "seed_order": list(SEEDS),
            "values": values,
            "mean": mean,
            "sample_sd_ddof1": statistics.stdev(values),
            "observed_spread": max(values) - min(values),
            "relative_spread_of_mean": (
                (max(values) - min(values)) / mean if mean else None
            ),
            "registered_as_floor": False,
            "role": "diagnostic-only, transcribed",
            "exclusion_reason": EXCLUDED_METRICS[metric],
        }

    corpus_cells = evidence.get("corpus_ffr_cells") or {}
    if set(corpus_cells) != {str(seed) for seed in SEEDS}:
        raise Round0219Error("R0219 per-corpus FFR cells are incomplete")
    corpus_slices: dict[str, Any] = {}
    for slug in CORPUS_SLUGS:
        values = [float(corpus_cells[str(seed)][slug]["ffr"]) for seed in SEEDS]
        if any(not math.isfinite(value) for value in values):
            raise Round0219Error(f"R0219 corpus {slug} FFR is not finite")
        corpus_slices[slug] = {
            "seed_order": list(SEEDS),
            "values": values,
            "mean": statistics.fmean(values),
            "sample_sd_ddof1": statistics.stdev(values),
            "anchors_seed42": int(corpus_cells[str(SEEDS[0])][slug]["anchors"]),
            "registered_as_floor": False,
            "role": CORPUS_SLICE_ROLE,
        }

    return {
        "schema": GATE_SCHEMA,
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "registered": True,
        "population": (
            f"R0216 queue-correction-3 mixed MiniLM {ROWS}-row substrate "
            "(fineweb-edu 40% / RedPajama 25% / pile 25% / starcoderdata code 10%)"
        ),
        "rows": ROWS,
        "seed_family": list(SEEDS),
        "n": len(SEEDS),
        "formula": FORMULA,
        "sample_standard_deviation_ddof": SD_DDOF,
        "multiplier": MULTIPLIER,
        "precision_note": PRECISION_NOTE,
        "gate_metrics": list(GATE_METRICS),
        "gates": gates,
        "excluded_metrics": dict(EXCLUDED_METRICS),
        "diagnostic_metrics": diagnostics,
        "density_v2_role": "diagnostic-only, transcribed",
        "per_corpus_ffr": corpus_slices,
        "per_corpus_role": CORPUS_SLICE_ROLE,
        "applies_to": (
            "maps over this exact universe only; never a cross-universe floor"
        ),
        "r0161_prompted_floors_unchanged": True,
        "r0193_mixed_english_floors_unchanged": True,
        "raw_universe_floors_unchanged": True,
        "training_performed": False,
        "evaluation_performed": False,
    }


__all__ = [
    "CAPABILITY",
    "CORPUS_SLICE_ROLE",
    "EXCLUDED_METRICS",
    "FORMULA",
    "GATE_METRICS",
    "GATE_SCHEMA",
    "MULTIPLIER",
    "PANEL_CAPABILITY",
    "PANEL_SCHEMA",
    "PRECISION_NOTE",
    "ROUND_ID",
    "SD_DDOF",
    "SEEDS",
    "Round0219Error",
    "register_minilm_gates",
]
