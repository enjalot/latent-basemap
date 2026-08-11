"""Frozen contract for R0255's twenty-nine-cell panel pooling.

R0255 trains thirteen new cells (seeds `58-70`) and scores them on the **same frozen
panel** R0218 built, R0222 reused, R0230 pooled thirteen cells on and R0250 pooled
sixteen on: the accepted R0113 `panel_config()`, R0218's published purity centroid
arrays loaded from disk rather than re-fitted, and R0218's **published high-D
reference bytes**.

If that reference is not byte-identical, the twenty-nine cells are not poolable and
the round must STOP and say so rather than pool them. The five components checked are
R0230's and R0250's, imported rather than restated.

**The sixteen existing cells are not rescored.** They are read from R0250's sealed
`seed-family-panel-n16.json`, which carries `panel_metric_cells`, `raw_purity_ratios`
and `corpus_ffr_cells` for seeds `42-57` and whose own `high_d_reference_key` must
equal the one loaded here. R0250 in turn read thirteen from R0230, which read from
R0218/R0222/R0223, so the lineage of every prior cell is a chain of sealed bytes.

**The replay control is scored and NOT pooled.** Seed 42 is already a family cell;
its replay exists to test the *treatment*, not to add evidence. It is carried in its
own top-level key, it is excluded from the pooled family by construction, and
`round0255_treatment.assert_family_is_2m_only` refuses a family that contains it --
with a positive control that plants exactly that.

This round's panel registers **no floor**. `round0255_gate_n29` fits them.
`density_v2` is carried **descriptive-only**, unchanged.
"""
from __future__ import annotations

import math
import statistics
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
from .round0250_panel_n16 import (
    PANEL_CAPABILITY_N16,
    PANEL_SCHEMA_N16,
    POOLED_CELL_SOURCES as R0250_CELL_SOURCES,
)
from .round0255_seed_extension_n29 import (
    IDENTITY_BOUND_AT_N29,
    N_TARGET,
    OWNER_RULING_N,
    POOLED_SEEDS,
    REPLAY_CONTROL_CAPABILITY,
    REPLAY_CONTROL_SEED,
    R0217_SEED_INVARIANT_SHA256,
    R0250_POOLED_SEEDS,
    R0250_SEEDS,
    SEEDS as R0255_SEEDS,
    STANDING_MINIMUM_N,
)


ROUND_ID = "0255"
PANEL_CAPABILITY_N29 = "minilm-mixed-2m-seed-family-panel-n29-v1"
PANEL_SCHEMA_N29 = "round0255-minilm-mixed-2m-seed-family-panel-n29-v1"

REFERENCE_MISMATCH_MESSAGE = (
    "R0255 high-D reference is not byte-identical to R0218's. The twenty-nine cells "
    "are NOT poolable and no n=29 gate may be built from them. STOP and report."
)

GATE_REGISTERABLE_HERE = False

#: Where each of the twenty-nine cells was scored. R0255 rescores nothing that
#: already exists.
POOLED_CELL_SOURCES: dict[str, str] = {
    **{key: value for key, value in R0250_CELL_SOURCES.items()},
    **{str(seed): "0255" for seed in R0255_SEEDS},
}


class Round0255PanelError(RuntimeError):
    """The registered R0255 n=29 panel-pooling contract changed."""


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
        raise Round0255PanelError(f"{REFERENCE_MISMATCH_MESSAGE} {'; '.join(problems)}")
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
    """R0230's registered check, called rather than re-typed."""
    try:
        return r0230_assert_hi_d_agreement(seed, numerators)
    except Exception as error:  # noqa: BLE001 - re-typed with the n=29 consequence
        raise Round0255PanelError(
            f"{REFERENCE_MISMATCH_MESSAGE} seed-{seed}: {error}"
        ) from error


def _admissible(metric: str, value: float) -> bool:
    if not math.isfinite(value):
        return False
    if metric in DIAGNOSTIC_METRICS:
        return abs(value) <= MAX_ABS_CORRELATION
    return MIN_RATIO_METRIC < value <= MAX_RATIO_METRIC


def pool_twenty_nine_cells(
    *,
    cells: Mapping[str, Mapping[str, float]],
    ratios: Mapping[str, Mapping[str, float]],
    corpus: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the twenty-nine-cell family. Registers no floor and makes no gate claim."""
    want = {str(seed) for seed in POOLED_SEEDS}
    if set(cells) != want:
        raise Round0255PanelError(
            f"R0255 pooled family must be exactly seeds {list(POOLED_SEEDS)}"
        )
    if set(ratios) != want or set(corpus) != want:
        raise Round0255PanelError("R0255 pooled ratio or corpus coverage is incomplete")
    if dict(sources) != dict(POOLED_CELL_SOURCES):
        raise Round0255PanelError("R0255 pooled cell provenance table changed")
    if REPLAY_CONTROL_CAPABILITY in set(cells):
        raise Round0255PanelError(
            "R0255 replay control appears in the pooled family; it is a control"
        )
    for seed in POOLED_SEEDS:
        key = str(seed)
        if set(cells[key]) != set(PANEL_METRICS):
            raise Round0255PanelError(f"R0255 seed-{seed} metric coverage changed")
        for metric, value in cells[key].items():
            if not _admissible(metric, float(value)):
                raise Round0255PanelError(
                    f"R0255 seed-{seed} {metric}={value!r} is inadmissible"
                )
        if set(ratios[key]) != {"k256", "k1024"}:
            raise Round0255PanelError(f"R0255 seed-{seed} raw ratios are incomplete")
        if set(corpus[key]) != set(CORPUS_SLUGS):
            raise Round0255PanelError(f"R0255 seed-{seed} corpus slices are incomplete")
    return {
        "n": len(POOLED_SEEDS),
        "owner_ruling_n": OWNER_RULING_N,
        "reaches_the_owner_ruling_n": len(POOLED_SEEDS) == OWNER_RULING_N,
        "seed_order": list(POOLED_SEEDS),
        "source_rounds": {
            "0217": list(R0218_SEEDS),
            "0221": list(R0221_SEEDS),
            "0230": list(R0230_SEEDS),
            "0250": list(R0250_SEEDS),
            "0255": list(R0255_SEEDS),
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
        "identity_bound_at_n": IDENTITY_BOUND_AT_N29,
        "identity_bound_note": (
            "max|x - xbar| / s <= (n-1)/sqrt(n) = "
            f"{IDENTITY_BOUND_AT_N29!r} at n = {N_TARGET} (= 28/sqrt(29)). It is the "
            "yardstick for the variance families reported beside the registered "
            "robust one; for median - k*MAD_n the rank-slack bound is +inf and a "
            "defining cell can fail at any multiplier."
        ),
        "the_family_is_2m_only": (
            "every pooled cell is a 2M exact-graph map trained under R0217's "
            "treatment. No rung map, no held-out cell and no control cell is in it: "
            "a gate whose family grows to include the maps it judges is not a test."
        ),
        "replay_control_is_not_a_family_cell": True,
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
    """Mean/sd per metric. **Descriptive.** `round0255_gate_n29` fits the floors."""
    summary: dict[str, Any] = {}
    for metric, values in values_by_metric.items():
        numbers = [float(value) for value in values]
        if len(numbers) < 3 or any(not math.isfinite(value) for value in numbers):
            raise Round0255PanelError(f"R0255 summary for {metric} is unusable")
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
                else "population for the n=29 calibrated floor registration"
            ),
        }
    return summary


def replay_control_comparison(
    *,
    observed: Mapping[str, float],
    sealed_r0218: Mapping[str, float],
    observed_ratios: Mapping[str, float],
    sealed_ratios: Mapping[str, float],
    tolerance: float,
) -> dict[str, Any]:
    """Compare the retrained seed-42 map to R0218's sealed seed-42 panel values.

    The tolerance is one `panel_v2` rounding quantum, fixed before the run, and is
    the same `1e-4` R0251 used for its scorer-side control. This is the *train*-side
    half: R0251 proved the scorer reproduces on an archived checkpoint; this proves
    the checkpoint a fresh train produces scores the same.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(set(sealed_r0218)):
        target = float(sealed_r0218[name])
        value = float(observed[name])
        rows.append({
            "kind": "panel metric",
            "name": name,
            "r0218_sealed": target,
            "r0255_replay": value,
            "delta": abs(value - target),
            "within_one_quantum": abs(value - target) <= tolerance,
            "exactly_equal": value == target,
        })
    for name in sorted(set(sealed_ratios)):
        target = float(sealed_ratios[name])
        value = float(observed_ratios[name])
        rows.append({
            "kind": "raw unfolded ratio",
            "name": name,
            "r0218_sealed": target,
            "r0255_replay": value,
            "delta": abs(value - target),
            "within_one_quantum": abs(value - target) <= tolerance,
            "exactly_equal": value == target,
        })
    return {
        "control": "seed-42 replay",
        "capability": REPLAY_CONTROL_CAPABILITY,
        "seed": REPLAY_CONTROL_SEED,
        "is_a_family_cell": False,
        "tolerance": float(tolerance),
        "rows": rows,
        "values_compared": len(rows),
        "values_exactly_equal": sum(1 for row in rows if row["exactly_equal"]),
        "values_within_one_quantum": sum(
            1 for row in rows if row["within_one_quantum"]
        ),
        "the_train_side_treatment_reproduces": all(
            row["within_one_quantum"] for row in rows
        ),
        "what_it_settles": (
            "R0251 re-scored an ARCHIVED checkpoint and proved the scorer stable. "
            "This retrains R0217's own cell on THIS release and scores the fresh "
            "checkpoint, so it tests the training path R0251's control could not "
            "reach. Together with the sealed source-closure diff they are the two "
            "pieces of evidence review-0251-01 named."
        ),
        "what_it_does_not_settle": (
            "one cell, one seed. It does not prove every seed reproduces, and a "
            "training path is not required to be bit-reproducible across releases "
            "-- the checkpoint-digest comparison is reported for what it is and the "
            "panel-value comparison at one quantum is the criterion."
        ),
    }


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
    "PANEL_CAPABILITY_N29",
    "PANEL_METRICS",
    "PANEL_SCHEMA_N13",
    "PANEL_SCHEMA_N16",
    "PANEL_SCHEMA_N29",
    "POOLED_CELL_SOURCES",
    "POOLED_SEEDS",
    "PURITY_RATIO_KEYS",
    "REFERENCE_BYTES",
    "REFERENCE_CONTENT_SHA256",
    "REFERENCE_KEY",
    "REFERENCE_MISMATCH_MESSAGE",
    "REFERENCE_SHA256",
    "REFERENCE_SOURCE_ROUND",
    "REPLAY_CONTROL_CAPABILITY",
    "REPLAY_CONTROL_SEED",
    "ROUND_ID",
    "ROWS",
    "R0217_SEED_INVARIANT_SHA256",
    "R0218_SEEDS",
    "R0221_SEEDS",
    "R0230_SEEDS",
    "R0250_POOLED_SEEDS",
    "R0250_SEEDS",
    "R0255_SEEDS",
    "Round0255PanelError",
    "SEALED_DIRECTED_EDGES",
    "STANDING_MINIMUM_N",
    "assert_hi_d_agreement",
    "assert_reference_identity",
    "corpus_ffr_view",
    "descriptive_family_summary",
    "panel_execution_ok",
    "panel_metric_view",
    "pool_twenty_nine_cells",
    "raw_purity_ratios",
    "replay_control_comparison",
]
