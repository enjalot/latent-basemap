"""Frozen contract for R0230's thirteen-cell panel pooling.

R0230 trains five new cells (seeds 50-54) and then scores them on the **same
frozen panel** R0218 built and R0222 reused: the accepted R0113 `panel_config()`,
R0218's published purity centroid arrays loaded from disk rather than re-fitted,
and — the load-bearing part — R0218's **published high-D reference bytes**.

If that reference is not byte-identical, the thirteen cells are not poolable and
the round must stop and say so rather than pool them. Byte-identity is checked
five ways, all registered here as constants read from R0218's and R0222's sealed
artifacts:

* the reference **file signature** (`80,395,632` B, sha256 `b26319f9...f335b7`);
* its content-addressed **key** `a0fd56fc...815006` and `content_sha256`
  `dcf1f77e...8036e74`;
* the key **re-derived** from this substrate, these anchors, this config and these
  centroids;
* the **anchor ids** and their per-corpus counts;
* the **`hi_D_agreement` numerators**, which are the map-independent side of every
  purity ratio and must be literally `0.3828` (`k256`) and `0.2385` (`k1024`) in
  every one of the thirteen cells — R0218 published those four times and R0222
  four more.

This round registers **no floor**. It produces the pooled thirteen-cell table —
four metrics plus the **raw, unfolded** purity ratios and the per-corpus FFR
slices — and R0231 fits floors to it on a robust scale. `density_v2` is carried
here **descriptive-only**: one anchor of `4,000` (index `2846`, substrate row
`1449227`, `1,377` duplicate coordinates and `r_hd == 0`) supplies about
two-thirds of its value, and with that anchor dropped the cell ranking is
uncorrelated with the as-defined metric at Pearson `-0.047` (Review 0225).

The raw ratio, not the fold, is what R0231 gates. `panel_v2` reports
`purity[kNNN]` as the ratio itself, already rounded to four decimals inside the
scorer, and `purity_fidelity` folds it to `exp(-|log r|)`. The fold reflects about
`r = 1.0` while the family centres above it, manufacturing `|z|`, and it destroys
the **direction** of the deviation so over-separation and under-separation become
the same failure. Both are carried; only the unfolded one is gate material.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .round0218_minilm_2m_panel import (
    CAPABILITY as PANEL_CAPABILITY,
    CENTROID_KS,
    CORPUS_SLUGS,
    DIAGNOSTIC_METRICS,
    DIMENSION,
    EVALUATION_SCHEMA as PANEL_SCHEMA,
    HOST_RSS_LIMIT_GIB,
    MAX_ABS_CORRELATION,
    MAX_RATIO_METRIC,
    MIN_RATIO_METRIC,
    PANEL_METRICS,
    ROWS,
    SEALED_DIRECTED_EDGES,
    SEEDS as R0218_SEEDS,
    corpus_ffr_view,
    panel_execution_ok,
    panel_metric_view,
)
from .round0230_minilm_2m_seed_extension_n13 import (
    IDENTITY_BOUND_AT_N13,
    N_TARGET,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    R0221_SEEDS,
    Round0230Error,
    SEEDS as R0230_SEEDS,
)


ROUND_ID = "0230"
PANEL_CAPABILITY_N13 = "minilm-mixed-2m-seed-family-panel-n13-v1"
PANEL_SCHEMA_N13 = "round0230-minilm-mixed-2m-seed-family-panel-n13-v1"

#: R0218's published reference, as sealed in `seed-family-panel.json` and
#: re-verified by R0222 and R0223. Registered constants: if the bytes on disk
#: differ from these, the thirteen cells are not poolable and the node aborts.
REFERENCE_BYTES = 80_395_632
REFERENCE_SHA256 = (
    "b26319f9448ae4f395f6eea4765156c84b75a31a25980f4f30f581790af335b7"
)
REFERENCE_KEY = "a0fd56fc47afcd5b702cab7aba041dcf95efeb5059ff5fee2351ee34aa815006"
REFERENCE_CONTENT_SHA256 = (
    "dcf1f77ed266d902eccee0550bd5056c3fa93928cfbe1347c2f0d6a708036e74"
)
REFERENCE_SOURCE_ROUND = "0218"

#: The map-independent side of every purity ratio. R0218 published these for
#: seeds 42-45 and R0222 reproduced them for 46-49; every R0230 cell must match.
HI_D_AGREEMENT = {"k256": 0.3828, "k1024": 0.2385}

#: R0218's sealed anchor slice, by corpus.
ANCHOR_CORPUS_COUNTS = {
    "code": 445,
    "fineweb": 1637,
    "pile": 993,
    "redpajama": 925,
}
ANCHORS = sum(ANCHOR_CORPUS_COUNTS.values())

REFERENCE_MISMATCH_MESSAGE = (
    "R0230 high-D reference is not byte-identical to R0218's. The thirteen cells "
    "are NOT poolable and no n=13 family may be built from them. STOP and report."
)

#: `purity_fidelity_kNNN` -> the key of its raw, unfolded ratio.
PURITY_RATIO_KEYS: dict[str, str] = {
    "purity_fidelity_k256": "k256",
    "purity_fidelity_k1024": "k1024",
}

#: Carried, reported, never used to fail a map — until the anchor set excludes
#: degenerate `r_hd == 0` rows and the metric is re-registered from a fresh
#: family. This is a same-universe measured exclusion (Review 0225), not a
#: judgement imported from another universe.
DENSITY_V2_STATUS = (
    "DESCRIPTIVE ONLY (review-0225-01, binding in plan-minilm-100m-v2.md): anchor "
    "index 2846 / substrate row 1449227 has 1,377 duplicate coordinates and "
    "r_hd == 0, supplying roughly two-thirds of the metric's value; with that "
    "anchor dropped the family mean moves 0.4421 -> 0.1544, 2*sd/mean moves "
    "1.93% -> 26.03%, and the cell ranking is uncorrelated with the as-defined "
    "metric (Pearson -0.047). Reported here; never used to pass or fail a map."
)

GATE_REGISTERABLE_HERE = False

#: Where the eight already-scored cells come from. R0230 does not rescore them:
#: it reads them from bytes an accepted review already bound.
POOLED_CELL_SOURCES = {
    "42": "0218",
    "43": "0218",
    "44": "0218",
    "45": "0218",
    "46": "0222",
    "47": "0222",
    "48": "0222",
    "49": "0222",
    "50": "0230",
    "51": "0230",
    "52": "0230",
    "53": "0230",
    "54": "0230",
}


class Round0230PanelError(RuntimeError):
    """The registered R0230 n=13 panel-pooling contract changed."""


def assert_reference_identity(
    *,
    file_signature: Mapping[str, Any],
    key: str,
    content_sha256: str,
    rederived_key: str,
    anchor_corpus_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Byte-identity of R0218's reference, or a stop. Never a workaround."""
    problems: list[str] = []
    if int(file_signature.get("bytes", -1)) != REFERENCE_BYTES:
        problems.append(
            f"file size {file_signature.get('bytes')!r} != {REFERENCE_BYTES}"
        )
    if str(file_signature.get("sha256") or "") != REFERENCE_SHA256:
        problems.append("file sha256 drift")
    if str(key) != REFERENCE_KEY:
        problems.append("content key drift")
    if str(content_sha256) != REFERENCE_CONTENT_SHA256:
        problems.append("content sha256 drift")
    if str(rederived_key) != REFERENCE_KEY:
        problems.append("re-derived key drift")
    if dict(anchor_corpus_counts) != dict(ANCHOR_CORPUS_COUNTS):
        problems.append(
            f"anchor corpus counts {dict(anchor_corpus_counts)!r} != "
            f"{dict(ANCHOR_CORPUS_COUNTS)}"
        )
    if problems:
        raise Round0230PanelError(f"{REFERENCE_MISMATCH_MESSAGE} {'; '.join(problems)}")
    return {
        "reference_byte_identical_to_r0218": True,
        "reference_source_round": REFERENCE_SOURCE_ROUND,
        "file_bytes": REFERENCE_BYTES,
        "file_sha256": REFERENCE_SHA256,
        "key": REFERENCE_KEY,
        "content_sha256": REFERENCE_CONTENT_SHA256,
        "rederived_key_matches": True,
        "anchor_corpus_counts": dict(ANCHOR_CORPUS_COUNTS),
        "anchors": ANCHORS,
    }


def assert_hi_d_agreement(seed: int, numerators: Mapping[str, Any]) -> dict[str, float]:
    """The map-independent numerators must be R0218's, literally."""
    observed: dict[str, float] = {}
    for key, expected in HI_D_AGREEMENT.items():
        cell = numerators.get(key)
        if not isinstance(cell, Mapping):
            raise Round0230PanelError(
                f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed} has no {key} numerator"
            )
        value = float(cell.get("hi_D_agreement", float("nan")))
        if value != float(expected):
            raise Round0230PanelError(
                f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed} hi-D agreement {key} "
                f"is {value!r}, R0218 published {expected!r}"
            )
        observed[key] = value
    return observed


def raw_purity_ratios(panel: Mapping[str, Any]) -> dict[str, float]:
    """The unfolded ratios, exactly as `panel_v2` reports them (4 decimals).

    `panel_v2.score_panel` rounds `map_purity / hi_D_agreement` to four decimals
    inside the scorer, so this is the only precision the ratio has ever had on
    this program — R0223's sealed `exact_family_purity_ratios` carry the same
    quantisation. The consequence for a band fitted to them (`+/- 5e-5` in `r`)
    is stated in the receipt rather than left for a reviewer to find.
    """
    purity = panel.get("purity")
    if not isinstance(purity, Mapping) or {"k256", "k1024"} - set(purity):
        raise Round0230PanelError("R0230 panel cell lacks both purity granularities")
    ratios: dict[str, float] = {}
    for key in ("k256", "k1024"):
        value = purity[key]
        if value is None:
            raise Round0230PanelError(
                f"R0230 purity ratio {key} is undefined: the high-D label "
                "agreement denominator was zero"
            )
        ratio = float(value)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise Round0230PanelError(
                f"R0230 purity ratio {key}={ratio!r} is not a positive finite ratio"
            )
        ratios[key] = ratio
    return ratios


def _admissible(metric: str, value: float) -> bool:
    if not math.isfinite(value):
        return False
    if metric in DIAGNOSTIC_METRICS:
        return abs(value) <= MAX_ABS_CORRELATION
    return MIN_RATIO_METRIC < value <= MAX_RATIO_METRIC


def pool_thirteen_cells(
    *,
    cells: Mapping[str, Mapping[str, float]],
    ratios: Mapping[str, Mapping[str, float]],
    corpus: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the thirteen-cell family. Registers no floor and makes no gate claim."""
    want = {str(seed) for seed in POOLED_SEEDS}
    if set(cells) != want:
        raise Round0230PanelError(
            f"R0230 pooled family must be exactly seeds {list(POOLED_SEEDS)}"
        )
    if set(ratios) != want or set(corpus) != want:
        raise Round0230PanelError("R0230 pooled ratio or corpus coverage is incomplete")
    if dict(sources) != dict(POOLED_CELL_SOURCES):
        raise Round0230PanelError("R0230 pooled cell provenance table changed")
    for seed in POOLED_SEEDS:
        key = str(seed)
        if set(cells[key]) != set(PANEL_METRICS):
            raise Round0230PanelError(f"R0230 seed-{seed} metric coverage changed")
        for metric, value in cells[key].items():
            if not _admissible(metric, float(value)):
                raise Round0230PanelError(
                    f"R0230 seed-{seed} {metric}={value!r} is inadmissible"
                )
        if set(ratios[key]) != {"k256", "k1024"}:
            raise Round0230PanelError(f"R0230 seed-{seed} raw ratios are incomplete")
        if set(corpus[key]) != set(CORPUS_SLUGS):
            raise Round0230PanelError(f"R0230 seed-{seed} corpus slices are incomplete")
    return {
        "n": len(POOLED_SEEDS),
        "seed_order": list(POOLED_SEEDS),
        "source_rounds": {
            "0217": list(R0218_SEEDS),
            "0221": list(R0221_SEEDS),
            "0230": list(R0230_SEEDS),
        },
        "scored_in_round_by_seed": dict(POOLED_CELL_SOURCES),
        "metrics": list(PANEL_METRICS),
        "panel_metric_cells": {
            str(seed): {
                metric: float(cells[str(seed)][metric]) for metric in PANEL_METRICS
            }
            for seed in POOLED_SEEDS
        },
        "raw_purity_ratios": {
            str(seed): {
                key: float(ratios[str(seed)][key]) for key in ("k256", "k1024")
            }
            for seed in POOLED_SEEDS
        },
        "corpus_ffr_cells": {
            str(seed): {
                slug: {
                    "anchors": int(corpus[str(seed)][slug]["anchors"]),
                    "ffr": float(corpus[str(seed)][slug]["ffr"]),
                }
                for slug in CORPUS_SLUGS
            }
            for seed in POOLED_SEEDS
        },
        "family_seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "identity_bound_at_n": IDENTITY_BOUND_AT_N13,
        "identity_bound_note": (
            "max|x - xbar| / s <= (n-1)/sqrt(n) = "
            f"{IDENTITY_BOUND_AT_N13!r} at n = {N_TARGET}. Every mean - k*s "
            "family R0231 reports uses a k below this bound, so a defining cell "
            "CAN fail — which was arithmetically impossible at n = 4 (bound 1.5) "
            "and at n = 8 under k = 3.187 (bound 2.4749)."
        ),
        "purity_ratio_quantisation": (
            "panel_v2 rounds each purity ratio to 4 decimals inside the scorer, so "
            "any band fitted to these ratios inherits a +/- 5e-5 quantisation in r"
        ),
        "density_v2_status": DENSITY_V2_STATUS,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    }


def descriptive_family_summary(
    values_by_metric: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    """Mean/sd per metric. **Descriptive.** R0231 fits the floors, not this round."""
    import statistics

    summary: dict[str, Any] = {}
    for metric, values in values_by_metric.items():
        numbers = [float(value) for value in values]
        if len(numbers) < 3 or any(not math.isfinite(value) for value in numbers):
            raise Round0230PanelError(f"R0230 summary for {metric} is unusable")
        summary[metric] = {
            "n": len(numbers),
            "mean": statistics.fmean(numbers),
            "sample_sd_ddof1": statistics.stdev(numbers),
            "median": statistics.median(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "role": (
                "descriptive-only, never a floor"
                if metric in DIAGNOSTIC_METRICS
                else "population for R0231's robust floor registration"
            ),
        }
    return summary


__all__ = [
    "ANCHORS",
    "ANCHOR_CORPUS_COUNTS",
    "CENTROID_KS",
    "CORPUS_SLUGS",
    "DENSITY_V2_STATUS",
    "DIAGNOSTIC_METRICS",
    "DIMENSION",
    "GATE_REGISTERABLE_HERE",
    "HI_D_AGREEMENT",
    "HOST_RSS_LIMIT_GIB",
    "PANEL_CAPABILITY",
    "PANEL_CAPABILITY_N13",
    "PANEL_METRICS",
    "PANEL_SCHEMA",
    "PANEL_SCHEMA_N13",
    "POOLED_CELL_SOURCES",
    "POOLED_SEEDS",
    "PURITY_RATIO_KEYS",
    "REFERENCE_BYTES",
    "REFERENCE_CONTENT_SHA256",
    "REFERENCE_KEY",
    "REFERENCE_MISMATCH_MESSAGE",
    "REFERENCE_SHA256",
    "REFERENCE_SOURCE_ROUND",
    "ROUND_ID",
    "ROWS",
    "R0217_SEED_INVARIANT_SHA256",
    "R0218_SEEDS",
    "R0221_SEEDS",
    "R0230_SEEDS",
    "Round0230Error",
    "Round0230PanelError",
    "SEALED_DIRECTED_EDGES",
    "assert_hi_d_agreement",
    "assert_reference_identity",
    "corpus_ffr_view",
    "descriptive_family_summary",
    "panel_execution_ok",
    "panel_metric_view",
    "pool_thirteen_cells",
    "raw_purity_ratios",
]
