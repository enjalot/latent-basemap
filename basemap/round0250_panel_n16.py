"""Frozen contract for R0250's sixteen-cell panel pooling.

R0250 trains three new cells (seeds `55/56/57`) and scores them on the **same
frozen panel** R0218 built, R0222 reused and R0230 pooled thirteen cells on: the
accepted R0113 `panel_config()`, R0218's published purity centroid arrays loaded
from disk rather than re-fitted, and R0218's **published high-D reference bytes**.

If that reference is not byte-identical, the sixteen cells are not poolable and the
round must STOP and say so rather than pool them. Byte-identity is checked on the
same five components R0230 checked, and the constants are imported from R0230's
contract rather than restated here:

* the reference **file signature** (`80,395,632` B, sha256 `b26319f9...f335b7`);
* its content-addressed **key** `a0fd56fc...815006` and `content_sha256`;
* the key **re-derived** from this substrate, these anchors, this config and these
  centroids;
* the **anchor ids** and their per-corpus counts
  (`fineweb 1,637 / pile 993 / redpajama 925 / code 445`);
* the **`hi_D_agreement` numerators**, which are the map-independent side of every
  purity ratio and must be literally `0.3828` (`k256`) and `0.2385` (`k1024`) in
  every one of the sixteen cells.

**The thirteen existing cells are not rescored.** They are read from R0230's sealed
`seed-family-panel-n13.json`, which already carries `panel_metric_cells`,
`raw_purity_ratios` and `corpus_ffr_cells` for seeds `42-54` and whose own
`high_d_reference_key` must equal the one loaded here. R0230 in turn read `42-45`
from R0218, `46-49` from R0222 and all eight raw ratios from R0223, so the lineage
of every prior cell is a chain of sealed bytes rather than a recomputation.

This round's panel registers **no floor**. `round0250_gate_n16` fits them.
`density_v2` is carried **descriptive-only**, unchanged, for the reason R0225
measured and `plan-minilm-100m-v2.md` made binding.
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
from .round0230_minilm_2m_panel_n13 import (
    ANCHOR_CORPUS_COUNTS,
    ANCHORS,
    DENSITY_V2_STATUS,
    HI_D_AGREEMENT,
    PANEL_CAPABILITY_N13,
    PANEL_SCHEMA_N13,
    POOLED_CELL_SOURCES as R0230_CELL_SOURCES,
    PURITY_RATIO_KEYS,
    REFERENCE_BYTES,
    REFERENCE_CONTENT_SHA256,
    REFERENCE_KEY,
    REFERENCE_SHA256,
    REFERENCE_SOURCE_ROUND,
    R0221_SEEDS,
    R0230_SEEDS,
    assert_hi_d_agreement as r0230_assert_hi_d_agreement,
    raw_purity_ratios,
)
from .round0250_seed_extension_n16 import (
    IDENTITY_BOUND_AT_N16,
    N_TARGET,
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    SEEDS as R0250_SEEDS,
    STANDING_MINIMUM_N,
)


ROUND_ID = "0250"
PANEL_CAPABILITY_N16 = "minilm-mixed-2m-seed-family-panel-n16-v1"
PANEL_SCHEMA_N16 = "round0250-minilm-mixed-2m-seed-family-panel-n16-v1"

REFERENCE_MISMATCH_MESSAGE = (
    "R0250 high-D reference is not byte-identical to R0218's. The sixteen cells "
    "are NOT poolable and no n=16 gate may be built from them. STOP and report."
)

GATE_REGISTERABLE_HERE = False

#: Where each of the sixteen cells was scored. R0250 rescores nothing that
#: already exists: the thirteen come from R0230's sealed pooled table, whose own
#: provenance map is imported rather than retyped.
POOLED_CELL_SOURCES: dict[str, str] = {
    **{key: value for key, value in R0230_CELL_SOURCES.items()},
    **{str(seed): "0250" for seed in R0250_SEEDS},
}


class Round0250PanelError(RuntimeError):
    """The registered R0250 n=16 panel-pooling contract changed."""


def assert_reference_identity(
    *,
    file_signature: Mapping[str, Any],
    key: str,
    content_sha256: str,
    rederived_key: str,
    anchor_corpus_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Byte-identity of R0218's reference, or a stop. Never a workaround.

    The five comparisons are R0230's registered constants, imported. Only the
    message and the receipt's `n` differ, because the consequence of a mismatch is
    now that sixteen cells rather than thirteen are unpoolable.
    """
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
        raise Round0250PanelError(f"{REFERENCE_MISMATCH_MESSAGE} {'; '.join(problems)}")
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
        "n_pooled": N_TARGET,
    }


def assert_hi_d_agreement(seed: int, numerators: Mapping[str, Any]) -> dict[str, float]:
    """R0230's registered check, called rather than re-typed.

    The comparison values `0.3828` / `0.2385` live in R0230's `HI_D_AGREEMENT` and
    the raising comparison lives in its `assert_hi_d_agreement`; R0250 re-raises as
    its own error type so a failure names the sixteen-cell consequence.
    """
    try:
        return r0230_assert_hi_d_agreement(seed, numerators)
    except Exception as error:  # noqa: BLE001 - re-typed with the n=16 consequence
        raise Round0250PanelError(
            f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed}: {error}"
        ) from error


def _admissible(metric: str, value: float) -> bool:
    if not math.isfinite(value):
        return False
    if metric in DIAGNOSTIC_METRICS:
        return abs(value) <= MAX_ABS_CORRELATION
    return MIN_RATIO_METRIC < value <= MAX_RATIO_METRIC


def pool_sixteen_cells(
    *,
    cells: Mapping[str, Mapping[str, float]],
    ratios: Mapping[str, Mapping[str, float]],
    corpus: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the sixteen-cell family. Registers no floor and makes no gate claim."""
    want = {str(seed) for seed in POOLED_SEEDS}
    if set(cells) != want:
        raise Round0250PanelError(
            f"R0250 pooled family must be exactly seeds {list(POOLED_SEEDS)}"
        )
    if set(ratios) != want or set(corpus) != want:
        raise Round0250PanelError("R0250 pooled ratio or corpus coverage is incomplete")
    if dict(sources) != dict(POOLED_CELL_SOURCES):
        raise Round0250PanelError("R0250 pooled cell provenance table changed")
    for seed in POOLED_SEEDS:
        key = str(seed)
        if set(cells[key]) != set(PANEL_METRICS):
            raise Round0250PanelError(f"R0250 seed-{seed} metric coverage changed")
        for metric, value in cells[key].items():
            if not _admissible(metric, float(value)):
                raise Round0250PanelError(
                    f"R0250 seed-{seed} {metric}={value!r} is inadmissible"
                )
        if set(ratios[key]) != {"k256", "k1024"}:
            raise Round0250PanelError(f"R0250 seed-{seed} raw ratios are incomplete")
        if set(corpus[key]) != set(CORPUS_SLUGS):
            raise Round0250PanelError(f"R0250 seed-{seed} corpus slices are incomplete")
    return {
        "n": len(POOLED_SEEDS),
        "seed_order": list(POOLED_SEEDS),
        "source_rounds": {
            "0217": list(R0218_SEEDS),
            "0221": list(R0221_SEEDS),
            "0230": list(R0230_SEEDS),
            "0250": list(R0250_SEEDS),
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
        "standing_minimum_n": STANDING_MINIMUM_N,
        "reaches_the_standing_minimum": len(POOLED_SEEDS) >= STANDING_MINIMUM_N,
        "identity_bound_at_n": IDENTITY_BOUND_AT_N16,
        "identity_bound_note": (
            "max|x - xbar| / s <= (n-1)/sqrt(n) = "
            f"{IDENTITY_BOUND_AT_N16!r} at n = {N_TARGET}. It is the yardstick for "
            "the variance families reported beside the registered robust one; for "
            "median - k*MAD_n the rank-slack bound is +inf and a defining cell can "
            "fail at any multiplier."
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
    """Mean/sd per metric. **Descriptive.** `round0250_gate_n16` fits the floors."""
    import statistics

    summary: dict[str, Any] = {}
    for metric, values in values_by_metric.items():
        numbers = [float(value) for value in values]
        if len(numbers) < 3 or any(not math.isfinite(value) for value in numbers):
            raise Round0250PanelError(f"R0250 summary for {metric} is unusable")
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
                else "population for the n=16 calibrated floor registration"
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
    "PANEL_CAPABILITY_N16",
    "PANEL_METRICS",
    "PANEL_SCHEMA_N13",
    "PANEL_SCHEMA_N16",
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
    "R0250_SEEDS",
    "Round0250PanelError",
    "SEALED_DIRECTED_EDGES",
    "STANDING_MINIMUM_N",
    "assert_hi_d_agreement",
    "assert_reference_identity",
    "corpus_ffr_view",
    "descriptive_family_summary",
    "panel_execution_ok",
    "panel_metric_view",
    "pool_sixteen_cells",
    "raw_purity_ratios",
]
