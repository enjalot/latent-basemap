"""Frozen contract for the R0218 MiniLM 2M four-seed panel.

Phase 1 of `guides/plan-minilm-100m-v2.md`. R0217 produced four commensurate maps
over R0216's sealed, corpus-spanning 2,000,000-row mixed MiniLM substrate and its
exact k15 fuzzy graph — same bytes, same recipe, same horizon, same dose, seed as
the only treatment. R0217 deliberately measured **nothing** about those maps.

This round measures them, and only measures them. It scores all four cells on one
frozen panel and reports, per seed:

* `density_v2` — the panel's exact log-radius correlation (`panel["density"]`),
  the same quantity R0160's `metric_view` calls `density_v2`. **Diagnostic only,
  transcribed**, exactly as the program does everywhere else.
* `ffr` — fixed-fraction recall of the exact high-D top-`k_hit` inside the map's
  top-`k_frac`.
* `purity_fidelity_k256` / `purity_fidelity_k1024` — R0160's symmetric
  `exp(-|log ratio|)` fidelity of the centroid-label purity ratio, so that a map
  which *over*-separates is penalised the same as one that under-separates.

plus **per-corpus FFR slices** over the four corpora of the substrate
(fineweb-edu / RedPajama / pile / starcoderdata code), computed from R0216's
`provenance.npy` by labelling every sampled panel anchor with its corpus. The
plan requires per-corpus slices at every rung and the code slice is new to this
program, so it is worth watching from the first measurement.

Two things this round is not:

1. **It registers no gate.** `GATE_REGISTERABLE_HERE` is `False`. Four scored
   cells are the *population* for a mean - 2 sigma registration; performing that
   registration is R0219's job and it must name the metrics it covers.
2. **It makes no quality claim.** There is no floor to clear here, because no
   floor for this universe exists yet — that is precisely what is missing. The
   registered checks are execution checks: every cell scores, every metric is
   finite, no panel is collapsed, and the map transform over all 2,000,000 rows
   is finite.

The panel machinery is the accepted one, unchanged: `basemap/panel_v2.py`'s
`score_panel` with a single shared high-D reference (R0160's reuse pattern —
map-independent neighbour, radius and label arrays computed once and verified by
content key for every map), the accepted R0113 panel configuration, and
`experiments/score_complete_panel.frozen_centroids` for the purity vocabularies.
Nothing about the evaluator is re-implemented here.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from .round0160_prompted_seed_family import purity_fidelity
from .round0216_minilm_2m_substrate import (
    COMPOSITION,
    DIMENSION,
    ROWS,
)
from .round0217_minilm_2m_seed_family import (
    CAPABILITY_TEMPLATE as MAP_CAPABILITY_TEMPLATE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    GRAPH_SCHEMA,
    GRAPH_SOURCE_ROUND_ID,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    TRAIN_SCHEMA as MAP_TRAIN_SCHEMA,
)


ROUND_ID = "0218"
CAPABILITY = "minilm-mixed-2m-seed-family-panel-v1"
EVALUATION_SCHEMA = "round0218-minilm-mixed-2m-seed-family-panel-v1"
REFERENCE_NOTE = (
    "one shared high-D reference for all four maps; map-independent neighbour, "
    "radius and centroid-label arrays computed once and re-verified by content "
    "key inside every score_panel call"
)

#: Purity vocabularies. The accepted panel granularities, unchanged since R0027.
CENTROID_KS: tuple[int, ...] = (256, 1024)
CENTROID_SEED = 0
CENTROID_ITERS = 25

#: Per-seed decision metrics. `density_v2` is carried as a **diagnostic**: it is
#: reported and transcribed, never gated. R0214 measured a two-cell spread on an
#: identical treatment where FFR moved 0.08% and purity < 0.7% while density_v2
#: moved 10.30%, so a gate that included it would be measuring its own noise.
PANEL_METRICS: tuple[str, ...] = (
    "density_v2",
    "ffr",
    "purity_fidelity_k256",
    "purity_fidelity_k1024",
)
DIAGNOSTIC_METRICS: tuple[str, ...] = ("density_v2",)

#: The substrate's corpus blocks, in `provenance.npy` corpus-id order. The ids
#: are positions in R0216's registered COMPOSITION tuple, so this table cannot
#: drift from the assembler without the import failing.
CORPUS_SLUGS: tuple[str, ...] = ("fineweb", "redpajama", "pile", "code")
CORPORA: tuple[tuple[int, str, str, int], ...] = tuple(
    (index, CORPUS_SLUGS[index], name, rows)
    for index, (name, rows) in enumerate(COMPOSITION)
)
CORPUS_ROWS: dict[str, int] = {slug: rows for _i, slug, _n, rows in CORPORA}

#: The 3.07 GB substrate is served from a memmap and the map coordinates are
#: 2M x 2 float32. Nothing here needs to be resident.
HOST_RSS_LIMIT_GIB = 32.0

#: This round registers no gate. R0219 does, from these four cells.
GATE_REGISTERABLE_HERE = False

#: Registered numeric admissibility. FFR and purity fidelity are bounded ratios;
#: density_v2 is a Pearson correlation, so it may legitimately be negative and is
#: bounded by +/- 1 instead of being required positive.
MIN_RATIO_METRIC = 0.0
MAX_RATIO_METRIC = 1.0
MAX_ABS_CORRELATION = 1.0


class Round0218Error(RuntimeError):
    """The registered MiniLM 2M panel contract changed."""


def map_capability(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise Round0218Error(f"R0218 seed {seed!r} is not a registered cell")
    return MAP_CAPABILITY_TEMPLATE.format(seed=int(seed))


def panel_metric_view(panel: Mapping[str, Any]) -> dict[str, float]:
    """The four per-seed numbers, derived from one scored panel payload."""
    purity = panel.get("purity")
    if not isinstance(purity, Mapping) or {"k256", "k1024"} - set(purity):
        raise Round0218Error("R0218 panel cell lacks both purity granularities")
    if any(purity[key] is None for key in ("k256", "k1024")):
        raise Round0218Error(
            "R0218 purity ratio is undefined: the high-D label agreement "
            "denominator was zero"
        )
    values = {
        "density_v2": float(panel["density"]),
        "ffr": float(panel["ffr"]),
        "purity_fidelity_k256": purity_fidelity(purity["k256"]),
        "purity_fidelity_k1024": purity_fidelity(purity["k1024"]),
    }
    if set(values) != set(PANEL_METRICS):
        raise Round0218Error("R0218 panel metric vector changed")
    for name, value in values.items():
        if not math.isfinite(value):
            raise Round0218Error(f"R0218 panel metric {name} is not finite")
        if name in DIAGNOSTIC_METRICS:
            if abs(value) > MAX_ABS_CORRELATION:
                raise Round0218Error(
                    f"R0218 diagnostic metric {name}={value!r} is not a correlation"
                )
        elif not MIN_RATIO_METRIC < value <= MAX_RATIO_METRIC:
            raise Round0218Error(
                f"R0218 panel metric {name}={value!r} is outside (0, 1]"
            )
    return values


def corpus_ffr_view(panel: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """The per-corpus FFR slices, one cell per substrate corpus."""
    groups = panel.get("ffr_by_group")
    if not isinstance(groups, Mapping) or set(groups) != set(CORPUS_SLUGS):
        raise Round0218Error(
            f"R0218 per-corpus FFR slices must be exactly {sorted(CORPUS_SLUGS)}"
        )
    output: dict[str, dict[str, float]] = {}
    for slug in CORPUS_SLUGS:
        cell = groups[slug]
        anchors = int(cell.get("anchors", 0))
        value = float(cell.get("ffr", float("nan")))
        if anchors <= 0:
            raise Round0218Error(f"R0218 corpus {slug} FFR slice has no anchors")
        if not math.isfinite(value) or not (
            MIN_RATIO_METRIC <= value <= MAX_RATIO_METRIC
        ):
            raise Round0218Error(f"R0218 corpus {slug} FFR {value!r} is invalid")
        output[slug] = {"anchors": anchors, "ffr": value}
    return output


def panel_execution_ok(panel: Mapping[str, Any]) -> bool:
    """The accepted finite/non-collapsed panel guard (R0166's shape)."""
    guards = panel.get("guards") or {}
    return bool(
        guards.get("coords_finite") is True
        and guards.get("coords_collapsed") is False
        and guards.get("emb_finite") is True
        and guards.get("emb_zero_rows") == 0
    )


def descriptive_summaries(
    cells: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, Any]]:
    """Family mean and sample sd per metric. **Descriptive, not a gate.**

    With four cells the sample standard deviation is a four-point estimate and
    this round says so in the payload rather than implying more precision than
    the design carries.
    """
    summaries: dict[str, dict[str, Any]] = {}
    for metric in PANEL_METRICS:
        values = [float(cells[str(seed)][metric]) for seed in SEEDS]
        if any(not math.isfinite(value) for value in values):
            raise Round0218Error(f"R0218 summary metric {metric} is not finite")
        summaries[metric] = {
            "seed_order": list(SEEDS),
            "values": values,
            "mean": statistics.fmean(values),
            "sample_sd_ddof1": statistics.stdev(values),
            "n": len(SEEDS),
            "role": (
                "diagnostic-only, transcribed"
                if metric in DIAGNOSTIC_METRICS
                else "gate-eligible population, registered in a separate round"
            ),
        }
    return summaries


def build_family_panel_evidence(
    cells: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind the four scored cells. Registers no gate and makes no quality claim."""
    if {int(seed) for seed in cells} != set(SEEDS):
        raise Round0218Error(
            f"R0218 panel family must be exactly seeds {list(SEEDS)}"
        )
    panel_cells: dict[str, dict[str, float]] = {}
    corpus_cells: dict[str, dict[str, dict[str, float]]] = {}
    output: dict[str, Any] = {}
    for seed in SEEDS:
        cell = cells[seed]
        metrics = cell.get("panel_metrics")
        slices = cell.get("corpus_ffr")
        if (
            int(cell.get("seed", -1)) != seed
            or str(cell.get("capability") or "") != map_capability(seed)
            or not isinstance(metrics, Mapping)
            or not isinstance(slices, Mapping)
        ):
            raise Round0218Error(f"R0218 seed-{seed} panel cell identity changed")
        if set(metrics) != set(PANEL_METRICS) or set(slices) != set(CORPUS_SLUGS):
            raise Round0218Error(f"R0218 seed-{seed} metric coverage changed")
        panel_cells[str(seed)] = {
            key: float(value) for key, value in metrics.items()
        }
        corpus_cells[str(seed)] = {
            slug: {
                "anchors": int(slices[slug]["anchors"]),
                "ffr": float(slices[slug]["ffr"]),
            }
            for slug in CORPUS_SLUGS
        }
        output[str(seed)] = dict(cell)
    return {
        "schema": EVALUATION_SCHEMA,
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "outcome": "minilm-mixed-2m-four-seed-panel-complete",
        "population": {
            "rows": ROWS,
            "dimension": DIMENSION,
            "substrate_capability": GRAPH_CAPABILITY,
            "substrate_round_id": GRAPH_SOURCE_ROUND_ID,
            "graph_k": GRAPH_K,
            "sealed_directed_edges": SEALED_DIRECTED_EDGES,
            "composition": {slug: CORPUS_ROWS[slug] for slug in CORPUS_SLUGS},
        },
        "seeds": list(SEEDS),
        "n": len(SEEDS),
        "map_capabilities": {str(seed): map_capability(seed) for seed in SEEDS},
        "map_train_schema": MAP_TRAIN_SCHEMA,
        "map_graph_schema": GRAPH_SCHEMA,
        "metrics": list(PANEL_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "corpus_slices": list(CORPUS_SLUGS),
        "panel_metric_cells": panel_cells,
        "corpus_ffr_cells": corpus_cells,
        "descriptive_summaries": descriptive_summaries(panel_cells),
        "cells": output,
        "reference_reuse": REFERENCE_NOTE,
        "density_v2_role": "diagnostic-only, transcribed",
        "gate_registered": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_registration_deferred_to_reviewed_cpu_round": True,
        "map_quality_claim_available": False,
        "training_performed": False,
    }


__all__ = [
    "CAPABILITY",
    "CENTROID_ITERS",
    "CENTROID_KS",
    "CENTROID_SEED",
    "CORPORA",
    "CORPUS_ROWS",
    "CORPUS_SLUGS",
    "DIAGNOSTIC_METRICS",
    "DIMENSION",
    "EVALUATION_SCHEMA",
    "GATE_REGISTERABLE_HERE",
    "GRAPH_CAPABILITY",
    "GRAPH_K",
    "GRAPH_SCHEMA",
    "GRAPH_SOURCE_ROUND_ID",
    "HOST_RSS_LIMIT_GIB",
    "MAP_CAPABILITY_TEMPLATE",
    "MAP_TRAIN_SCHEMA",
    "PANEL_METRICS",
    "REFERENCE_NOTE",
    "ROUND_ID",
    "ROWS",
    "Round0218Error",
    "SEALED_DIRECTED_EDGES",
    "SEEDS",
    "build_family_panel_evidence",
    "corpus_ffr_view",
    "descriptive_summaries",
    "map_capability",
    "panel_execution_ok",
    "panel_metric_view",
    "purity_fidelity",
]
