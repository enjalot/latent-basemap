"""Binding MiniLM mixed-2M quality-gate registration at **n = 8** (R0222).

This module exists to correct two defects that independent Review 0219 found in
R0219's gate, and it states both corrections in code rather than in prose.

**Defect 1 — n = 4 cannot support a mean - 2 sigma gate.** For any sample of
size `n`, `max_i |x_i - xbar| / s <= (n - 1) / sqrt(n)`. At `n = 4` that bound is
exactly `1.5`, so no defining cell can fall below its own `mean - 2 sigma` floor,
for any four numbers whatsoever. "4/4 cells pass" was a theorem. At `n = 8` the
bound is `2.4749`, so a defining cell *can* fail, and the pass count becomes
information. `plan-minilm-100m-v2.md` now requires `n >= 8` and `n` stated beside
every floor; `N_REQUIRED` and `CELLS_CLEARING_IS_INFORMATIVE_AT` encode that.

**Defect 2 — the gated metric set was wrongly narrowed.** R0219 gated FFR and the
two purity fidelities only, excluding `density_v2`, and justified the exclusion
with this sentence:

    "`density_v2` therefore stays exactly what it is everywhere else in this
    program - reported, transcribed, never gated"

**That sentence is false, and R0222 retracts it.** `density_v2` is a registered,
accepted, *released* floor in both precedents R0219 named as its method:

* R0161 `jina-prompted-universe-quality-gates-v1` - `density_v2` floor
  `0.19134355783912885` (review-0161, accepted)
* R0193 `jina-mixed-english-2m-quality-gates-v1` - `density_v2` floor
  `0.18616941334799972` (review-0193, accepted)

Both artifacts carry the same six gate keys, `ACCEPTED_SIX_METRIC_SET` below.
Those two floor values are **read out of the sealed artifacts at run time**, not
trusted from this docstring; `assert_density_v2_is_gated_in_precedent` is the
check, and it is what makes the retraction evidence rather than assertion.

R0219's narrowing rested on R0214 — measured on *jina*, at `n = 2`. On MiniLM the
ordering reverses: `density_v2`'s `2 sigma` band is `2.05%` of its mean against
FFR's `4.71%`. So this round gates **every metric of the accepted six-metric set
that R0218's panel computes on this universe**, and the structural assertion at
the top of `register_minilm_gates_n8` makes it impossible to gate fewer:

    GATE_METRICS == ACCEPTED_SIX_METRIC_SET & R0218 PANEL_METRICS

`EXCLUDED_BY_JUDGEMENT` is empty and is asserted empty. Two of the six —
`heldout_recall_at_10` and `projection_ffr` — are **not computed by the R0218
panel on this universe at all**; both need a held-out projection set that this
universe has never had. They are recorded as *unavailable*, which is a fact about
the panel, not a judgement about the metric, and the round says so plainly rather
than describing a four-metric gate as a six-metric one.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .round0218_minilm_2m_panel import (
    CAPABILITY as PANEL_CAPABILITY,
    CORPUS_SLUGS,
    DIAGNOSTIC_METRICS as R0218_DIAGNOSTIC_METRICS,
    EVALUATION_SCHEMA as PANEL_SCHEMA,
    MAX_ABS_CORRELATION,
    MAX_RATIO_METRIC,
    MIN_RATIO_METRIC,
    PANEL_METRICS,
    ROWS,
    SEEDS as R0218_SEEDS,
)
from .round0221_minilm_2m_seed_extension import (
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    SEEDS as R0221_SEEDS,
)


ROUND_ID = "0222"
CAPABILITY = "minilm-mixed-2m-quality-gates-n8-v1"
PANEL_EXTENSION_CAPABILITY = "minilm-mixed-2m-seed-extension-panel-v1"
GATE_SCHEMA = "round0222-minilm-mixed-2m-quality-gates-n8-v1"

#: The estimator. Unchanged from R0161/R0193/R0219 — only `n` and the metric set
#: move, and both moves are the point of the round.
FORMULA = "family mean - 2 * sample standard deviation (ddof=1)"
MULTIPLIER = 2.0
SD_DDOF = 1

#: `plan-minilm-100m-v2.md`, rule earned in Phase 1: gates come from >= 8 seeds.
N_REQUIRED = 8
CELLS_CLEARING_IS_INFORMATIVE_AT = 8

#: The six gate keys carried by **both** sealed precedent artifacts.
ACCEPTED_SIX_METRIC_SET: tuple[str, ...] = (
    "density_v2",
    "ffr",
    "heldout_recall_at_10",
    "projection_ffr",
    "purity_fidelity_k1024",
    "purity_fidelity_k256",
)

#: The precedent artifacts whose `density_v2` floors refute R0219's claim. Paths
#: only — every number is read from the sealed bytes at run time.
PRECEDENT_GATE_ARTIFACTS: dict[str, str] = {
    "0161": (
        "/data/latent-basemap/runs/round-0161/queue/artifacts/"
        "jina-prompted-universe-quality-gates-v1/prompted-quality-gates.json"
    ),
    "0193": (
        "/data/latent-basemap/runs/round-0193/queue/artifacts/"
        "jina-mixed-english-2m-quality-gates-v1/mixed-quality-gates.json"
    ),
}
PRECEDENT_CAPABILITIES: dict[str, str] = {
    "0161": "jina-prompted-universe-quality-gates-v1",
    "0193": "jina-mixed-english-2m-quality-gates-v1",
}

#: The gated set: every accepted metric this universe's panel actually computes.
GATE_METRICS: tuple[str, ...] = tuple(
    metric for metric in PANEL_METRICS if metric in ACCEPTED_SIX_METRIC_SET
)

#: Not computed by the R0218 panel on this universe. A property of the panel.
UNAVAILABLE_METRICS: dict[str, str] = {
    "heldout_recall_at_10": (
        "R0218's panel does not compute it on this universe: there is no "
        "registered held-out projection set for the R0216 queue-correction-3 "
        "substrate. Unavailable, not excluded."
    ),
    "projection_ffr": (
        "Same cause: projection fidelity needs a held-out query set that this "
        "universe has never had. Unavailable, not excluded."
    ),
}

#: Deliberately empty, and asserted empty. No metric this panel computes is
#: withheld from the gate. This is the correction to R0219.
EXCLUDED_BY_JUDGEMENT: dict[str, str] = {}

#: The sentence R0222 retracts, verbatim, so the retraction is greppable.
RETRACTED_CLAIM = (
    "density_v2 therefore stays exactly what it is everywhere else in this "
    "program - reported, transcribed, never gated"
)
RETRACTION = (
    "FALSE. R0161 and R0193 both seal a density_v2 floor under an accepted "
    "review, and both artifacts carry the same six gate keys. R0219 narrowed an "
    "accepted six-metric set to three on jina-derived n=2 evidence (R0214) and "
    "described the narrowing as program practice. R0222 gates every metric of "
    "the accepted set that this universe's panel computes, density_v2 included, "
    "and reads both precedent floors out of their sealed artifacts to prove the "
    "retraction rather than assert it."
)

#: Per-corpus FFR stays descriptive. Eight cells at 445-1,637 anchors per slice
#: is still a thinner estimate than the pooled metric and no rung has been run
#: for a per-corpus floor to guard.
CORPUS_SLICE_ROLE = "descriptive per-corpus spread; not a registered floor"

PRECISION_NOTE = (
    "n = 8 cells (seeds 42-49). At n = 8 the identity max|x - xbar|/s <= "
    "(n-1)/sqrt(n) = 2.4749 exceeds the multiplier 2.0, so a defining cell CAN "
    "fall below its own floor and the cells-clearing count is informative. At "
    "n = 4 the bound is 1.5 and the same count is a theorem."
)


class Round0222Error(RuntimeError):
    """The R0222 n=8 gate registration changed after preregistration."""


def identity_bound(n: int) -> float:
    """`(n-1)/sqrt(n)`: the largest |z| any single sample point can reach."""
    if int(n) < 2:
        raise Round0222Error("R0222 identity bound needs at least two cells")
    return (int(n) - 1) / math.sqrt(int(n))


def _admissible(metric: str, value: float) -> bool:
    if not math.isfinite(value):
        return False
    if metric in R0218_DIAGNOSTIC_METRICS:
        # density_v2 is a Pearson correlation: bounded by +/-1, may be negative.
        return abs(value) <= MAX_ABS_CORRELATION
    return MIN_RATIO_METRIC < value <= MAX_RATIO_METRIC


def gate_cell(
    metric: str, values: Sequence[float], seeds: Sequence[int]
) -> dict[str, Any]:
    """One floor, plus everything a reader needs to judge how thin it is."""
    values = [float(value) for value in values]
    if len(values) != len(seeds) or len(values) < 2:
        raise Round0222Error(f"R0222 gate metric {metric} has the wrong cell count")
    if not all(_admissible(metric, value) for value in values):
        raise Round0222Error(f"R0222 gate metric {metric} is invalid: {values}")
    n = len(values)
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    floor = mean - MULTIPLIER * sample_sd
    if not math.isfinite(floor):
        raise Round0222Error(f"R0222 gate floor {metric} is invalid")
    spread = max(values) - min(values)
    clearing = [int(seed) for seed, value in zip(seeds, values) if value >= floor]
    failing = [int(seed) for seed, value in zip(seeds, values) if value < floor]
    max_abs_z = max(abs(value - mean) for value in values) / sample_sd if sample_sd else 0.0
    return {
        "direction": "higher-is-better",
        "n": n,
        "seed_order": [int(seed) for seed in seeds],
        "values": values,
        "mean": mean,
        "sample_sd_ddof1": sample_sd,
        "multiplier": MULTIPLIER,
        "floor": floor,
        "observed_spread": spread,
        "relative_spread_of_mean": (spread / mean) if mean else None,
        "two_sigma_band_over_mean": (
            (MULTIPLIER * sample_sd / mean) if mean else None
        ),
        "cells_clearing_floor": len(clearing),
        "cells_total": n,
        "seeds_clearing_floor": clearing,
        "seeds_below_floor": failing,
        "max_abs_z": max_abs_z,
        "identity_bound_on_max_abs_z": identity_bound(n),
        "cells_clearing_is_informative": identity_bound(n) > MULTIPLIER,
        "floor_is_vacuous": floor <= 0.0,
    }


def jackknife(metric: str, values: Sequence[float], seeds: Sequence[int]) -> dict[str, Any]:
    """Leave-one-out floors. How far the floor moves when one cell is dropped."""
    values = [float(value) for value in values]
    seeds = [int(seed) for seed in seeds]
    if len(values) != len(seeds) or len(values) < 3:
        raise Round0222Error(f"R0222 jackknife for {metric} needs at least three cells")
    loo: dict[str, float] = {}
    for index, seed in enumerate(seeds):
        rest = values[:index] + values[index + 1 :]
        loo[str(seed)] = statistics.fmean(rest) - MULTIPLIER * statistics.stdev(rest)
    full = statistics.fmean(values) - MULTIPLIER * statistics.stdev(values)
    span = max(loo.values()) - min(loo.values())
    largest_shift = max(abs(value - full) for value in loo.values())
    band = MULTIPLIER * statistics.stdev(values)
    return {
        "n": len(values),
        "leave_one_out_n": len(values) - 1,
        "floor": full,
        "leave_one_out_floors": loo,
        "loo_range": span,
        "loo_relative_range_of_floor": (span / full) if full else None,
        "largest_single_cell_shift": largest_shift,
        "two_sigma_band": band,
        "largest_shift_exceeds_two_sigma_band": largest_shift > band,
    }


def assert_density_v2_is_gated_in_precedent(
    precedents: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Read R0161's and R0193's sealed gate artifacts and refute R0219's claim.

    Fails closed if either precedent turns out *not* to gate `density_v2` — in
    which case R0219's sentence would have been true and this round's premise
    wrong, which is exactly the direction a check should be able to fail in.
    """
    if set(precedents) != set(PRECEDENT_GATE_ARTIFACTS):
        raise Round0222Error("R0222 requires both precedent gate artifacts")
    evidence: dict[str, Any] = {}
    for round_id, artifact in sorted(precedents.items()):
        gates = artifact.get("gates") or {}
        if artifact.get("capability") != PRECEDENT_CAPABILITIES[round_id]:
            raise Round0222Error(f"R{round_id} gate artifact identity changed")
        if tuple(sorted(gates)) != tuple(sorted(ACCEPTED_SIX_METRIC_SET)):
            raise Round0222Error(
                f"R{round_id} gate keys are {sorted(gates)}, not the accepted "
                f"six-metric set {sorted(ACCEPTED_SIX_METRIC_SET)}"
            )
        if artifact.get("formula") != FORMULA:
            raise Round0222Error(f"R{round_id} does not use the registered estimator")
        density = gates.get("density_v2") or {}
        floor = float(density.get("floor", float("nan")))
        if not math.isfinite(floor):
            raise Round0222Error(f"R{round_id} density_v2 floor is not a number")
        evidence[round_id] = {
            "capability": PRECEDENT_CAPABILITIES[round_id],
            "gate_keys": sorted(gates),
            "density_v2_floor": floor,
            "density_v2_mean": float(density.get("mean", float("nan"))),
            "density_v2_sample_sd_ddof1": float(
                density.get("sample_sd_ddof1", float("nan"))
            ),
            "n": int(artifact.get("n", -1)),
            "seed_family": list(artifact.get("seed_family") or []),
            "density_v2_is_a_registered_floor": True,
        }
    return {
        "retracted_claim": RETRACTED_CLAIM,
        "retraction": RETRACTION,
        "retracted_from": "round-0219-2026-08-08.md and "
        "basemap/round0219_minilm_2m_gate_registration.py lines 18-20",
        "precedents": evidence,
        "density_v2_gated_in_both_precedents": True,
    }


def _cell_values(
    cells: Mapping[str, Mapping[str, float]], metric: str, seeds: Sequence[int]
) -> list[float]:
    return [float(cells[str(seed)][metric]) for seed in seeds]


def register_minilm_gates_n8(
    *,
    pooled_cells: Mapping[str, Mapping[str, float]],
    corpus_cells: Mapping[str, Mapping[str, Mapping[str, float]]],
    precedents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Register the n=8 gate, the n=4 comparison, and the jackknife."""
    # --- structural assertions, before any number is read -------------------
    available = tuple(
        metric for metric in PANEL_METRICS if metric in ACCEPTED_SIX_METRIC_SET
    )
    if tuple(GATE_METRICS) != available:
        raise Round0222Error(
            "R0222 gate set must be every accepted metric this panel computes"
        )
    if EXCLUDED_BY_JUDGEMENT:
        raise Round0222Error(
            "R0222 excludes no computed metric by judgement; that was R0219's defect"
        )
    if "density_v2" not in GATE_METRICS:
        raise Round0222Error("R0222 gate set must include density_v2")
    if set(UNAVAILABLE_METRICS) != set(ACCEPTED_SIX_METRIC_SET) - set(PANEL_METRICS):
        raise Round0222Error(
            "R0222 unavailable set must be exactly the accepted metrics this "
            "panel does not compute"
        )
    if set(GATE_METRICS) & set(UNAVAILABLE_METRICS):
        raise Round0222Error("R0222 gate set overlaps the unavailable metrics")

    if set(pooled_cells) != {str(seed) for seed in POOLED_SEEDS}:
        raise Round0222Error(
            f"R0222 pooled family must be exactly seeds {list(POOLED_SEEDS)}"
        )
    if len(POOLED_SEEDS) != N_REQUIRED:
        raise Round0222Error("R0222 requires exactly eight pooled cells")
    for seed in POOLED_SEEDS:
        if set(pooled_cells[str(seed)]) != set(PANEL_METRICS):
            raise Round0222Error(f"R0222 seed-{seed} metric coverage changed")
    if set(corpus_cells) != {str(seed) for seed in POOLED_SEEDS}:
        raise Round0222Error("R0222 per-corpus FFR cells are incomplete")

    retraction = assert_density_v2_is_gated_in_precedent(precedents)

    gates = {
        metric: gate_cell(
            metric, _cell_values(pooled_cells, metric, POOLED_SEEDS), POOLED_SEEDS
        )
        for metric in GATE_METRICS
    }
    n4 = {
        metric: gate_cell(
            metric, _cell_values(pooled_cells, metric, R0218_SEEDS), R0218_SEEDS
        )
        for metric in GATE_METRICS
    }
    comparison = {
        metric: {
            "n4_seeds": list(R0218_SEEDS),
            "n4_floor": n4[metric]["floor"],
            "n4_mean": n4[metric]["mean"],
            "n4_sample_sd_ddof1": n4[metric]["sample_sd_ddof1"],
            "n4_cells_clearing": n4[metric]["cells_clearing_floor"],
            "n4_cells_clearing_is_informative": n4[metric][
                "cells_clearing_is_informative"
            ],
            "n8_seeds": list(POOLED_SEEDS),
            "n8_floor": gates[metric]["floor"],
            "n8_mean": gates[metric]["mean"],
            "n8_sample_sd_ddof1": gates[metric]["sample_sd_ddof1"],
            "n8_cells_clearing": gates[metric]["cells_clearing_floor"],
            "n8_cells_clearing_is_informative": gates[metric][
                "cells_clearing_is_informative"
            ],
            "floor_shift_n4_to_n8": gates[metric]["floor"] - n4[metric]["floor"],
            "floor_shift_relative_to_n4": (
                (gates[metric]["floor"] - n4[metric]["floor"]) / n4[metric]["floor"]
                if n4[metric]["floor"]
                else None
            ),
        }
        for metric in GATE_METRICS
    }
    jackknives = {
        "n8": {
            metric: jackknife(
                metric, _cell_values(pooled_cells, metric, POOLED_SEEDS), POOLED_SEEDS
            )
            for metric in GATE_METRICS
        },
        "n4": {
            metric: jackknife(
                metric, _cell_values(pooled_cells, metric, R0218_SEEDS), R0218_SEEDS
            )
            for metric in GATE_METRICS
        },
    }

    corpus_slices: dict[str, Any] = {}
    for slug in CORPUS_SLUGS:
        values = [
            float(corpus_cells[str(seed)][slug]["ffr"]) for seed in POOLED_SEEDS
        ]
        if any(not math.isfinite(value) for value in values):
            raise Round0222Error(f"R0222 corpus {slug} FFR is not finite")
        corpus_slices[slug] = {
            "n": len(POOLED_SEEDS),
            "seed_order": list(POOLED_SEEDS),
            "values": values,
            "mean": statistics.fmean(values),
            "sample_sd_ddof1": statistics.stdev(values),
            "anchors": int(corpus_cells[str(POOLED_SEEDS[0])][slug]["anchors"]),
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
        "seed_family": list(POOLED_SEEDS),
        "n": len(POOLED_SEEDS),
        "n_required_by_plan": N_REQUIRED,
        "source_rounds": {
            "0217": list(R0218_SEEDS),
            "0221": list(R0221_SEEDS),
        },
        "family_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "formula": FORMULA,
        "sample_standard_deviation_ddof": SD_DDOF,
        "multiplier": MULTIPLIER,
        "precision_note": PRECISION_NOTE,
        "identity_bound_at_n4": identity_bound(len(R0218_SEEDS)),
        "identity_bound_at_n8": identity_bound(len(POOLED_SEEDS)),
        "accepted_six_metric_set": list(ACCEPTED_SIX_METRIC_SET),
        "panel_metrics_available": list(PANEL_METRICS),
        "gate_metrics": list(GATE_METRICS),
        "gates": gates,
        "n4_gates_for_comparison": n4,
        "n4_vs_n8": comparison,
        "jackknife": jackknives,
        "unavailable_metrics": dict(UNAVAILABLE_METRICS),
        "excluded_by_judgement": dict(EXCLUDED_BY_JUDGEMENT),
        "density_v2_role": "registered floor (R0219's exclusion is retracted)",
        "r0219_retraction": retraction,
        "per_corpus_ffr": corpus_slices,
        "per_corpus_role": CORPUS_SLICE_ROLE,
        "applies_to": (
            "maps over this exact universe only; never a cross-universe floor"
        ),
        "supersedes_capability": "minilm-mixed-2m-quality-gates-v1",
        "r0161_prompted_floors_unchanged": True,
        "r0193_mixed_english_floors_unchanged": True,
        "raw_universe_floors_unchanged": True,
        "training_performed": False,
    }


__all__ = [
    "ACCEPTED_SIX_METRIC_SET",
    "CAPABILITY",
    "CELLS_CLEARING_IS_INFORMATIVE_AT",
    "CORPUS_SLICE_ROLE",
    "CORPUS_SLUGS",
    "EXCLUDED_BY_JUDGEMENT",
    "FORMULA",
    "GATE_METRICS",
    "GATE_SCHEMA",
    "MULTIPLIER",
    "N_REQUIRED",
    "PANEL_CAPABILITY",
    "PANEL_EXTENSION_CAPABILITY",
    "PANEL_METRICS",
    "PANEL_SCHEMA",
    "POOLED_SEEDS",
    "PRECEDENT_CAPABILITIES",
    "PRECEDENT_GATE_ARTIFACTS",
    "PRECISION_NOTE",
    "R0217_SEED_INVARIANT_SHA256",
    "R0218_SEEDS",
    "R0221_SEEDS",
    "RETRACTED_CLAIM",
    "RETRACTION",
    "ROUND_ID",
    "ROWS",
    "Round0222Error",
    "SD_DDOF",
    "UNAVAILABLE_METRICS",
    "assert_density_v2_is_gated_in_precedent",
    "gate_cell",
    "identity_bound",
    "jackknife",
    "register_minilm_gates_n8",
]
