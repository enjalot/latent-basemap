"""Execute R0264 (DRAFT) — register the collapse floor, the fail-closed fog ceiling,
and density_v3 as descriptive-only.

**This module is a DRAFT node package.** The round it belongs to
(`latent-labs/basemap-100m/round-0264-2026-08-13.md`) is `status: draft` and waits
on P1's dose x N decomposition before its N-adjustment rule is filled and it is
issued. Nothing here has been run; the code is authored to be reviewable and to
run unchanged once the round is issued.

Two CPU nodes, in that order, and the order is a discipline (register the criteria
before compiling the record that separates the modes under them).

**Node 1, ``register_collapse_and_fog_metrics_n29``.** Reproduces the Gaussian-null
calibration bitwise — R0234's published n = 13 ``median_minus_k_madn`` multiplier
(the validation gate) and R0255's sealed n = 29 ``k1`` — both via the program's own
``round0234_calibration`` AND cross-checked against the sealed R0234 / R0255 JSONs.
Fits the PROVISIONAL collapse floor and fog ceiling on the HEALTHY family (the seven
core ``min_dist = 0.00`` umap arms of ``experiments/metrics/REPORT.md`` §4/§6, n = 7)
at the healthy-family multiplier ``k7``; the legacy n = 29 2M family is used ONLY as
the collapsed fingerprint and never as a band source. Registers COLLAPSE as a
one-sided lower floor, FOG as a one-sided upper ceiling that FAILS CLOSED on
degeneracy (``degenerate: true`` or ``resolution_levels < 1`` => NOT_MEASURABLE,
never a pass), and ``density_v3`` as ``descriptive_only_never_a_gate``. Publishes
attainability, one-sided detection power and the panel false-alarm rate, and runs
every guard's positive control — most importantly the FOG FAIL-CLOSED control: a
planted degenerate map (peak bin below ``1/threshold``, ``resolution_levels < 1``,
like the sandbox arm ``umap-md035-x2``) is fed to the fog judge and must be REFUSED,
and the same map is shown to have PASSED a naive judge that skips the guard, so the
guard is proven to change the outcome. Seals the capability artifact and done-marker.

**Node 2, ``compile_collapse_fog_separation_record``.** Depends on node 1; refuses to
run unless node 1 sealed first (the producer's own done-marker). Reads the registered
floor and ceiling by hash and applies them to the legacy fingerprint (29 collapsed
cells -> all FAIL the collapse floor), the healthy family (7 core arms -> PASS both)
and the cuML reference (PASS both), and re-asserts the fog fail-closed guard on the
degenerate ``umap-md035-x2`` (-> NOT_MEASURABLE). Publishes the 5.5x separation as a
dated companion, registering nothing new.

Nothing here trains anything, touches a floor R0255/R0256/R0260 registered, or gates
on ``density_v3`` or the fog companion diagnostics. The metric repository
(``/home/enjalot/code/latent-basemap/experiments/metrics``) is NOT importable from the
run checkout, so this round seals its OWN copies of the constants and mirrors the
metric logic (``map_fog``, the robust ``median +/- k*MAD_n`` floor/ceiling) here,
reproducing every sealed number bitwise against the artifact that first published it.
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

from basemap.artifact_identity import (
    expected_input_signature,
    matches_cited_display_value,
)
from basemap.output_safety import create_fresh_directory
from basemap import round0113_prompt_contrast as prompt_contract
from basemap import round0234_calibration as calibration
from basemap.round0234_calibrated_floors import (
    TOLERANCE_CONFIDENCE,
    TOLERANCE_CONTENT,
    identity_bound,
)
from basemap.round0247_registry import registry_fingerprint
from basemap.round0251_trainer_setup import PollRecorder
from basemap.round0252_stoppability import gap_report
from basemap.round0253_coverage import CoverageLedger
from basemap.round0253_stop_hooks import install_stop_hooks
from basemap.round0255_gate_n29 import N_EXACT
from basemap.round0256_gate_repair import REGISTERED_ESTIMATOR_NAMES

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
# The bitwise comparator, the intra-queue signer and the GPU-mapping probe are
# R0260's, imported rather than re-typed. This round adds only the collapse/fog
# metric mirror, its two judges, the mixed registry and the fail-closed controls.
from experiments.round0260_nodes import (
    _bitwise_same,
    _cuda_mappings,
    _intra_queue_signature,
)


REGISTER_ACTION = "register_collapse_and_fog_metrics_n29"
SEPARATION_ACTION = "compile_collapse_fog_separation_record"

ROUND_ID = "0264"

GATE_CAPABILITY = "minilm-collapse-floor-fog-ceiling-provisional-n29-v1"
GATE_SCHEMA = "round0264-minilm-collapse-floor-fog-ceiling-provisional-n29-v1"
REGISTRATION_ARTIFACT = "minilm-collapse-floor-fog-ceiling-provisional-n29.json"

SEPARATION_CAPABILITY = "minilm-collapse-fog-legacy-vs-healthy-separation-v1"
SEPARATION_SCHEMA = "round0264-minilm-collapse-fog-legacy-vs-healthy-separation-v1"
SEPARATION_ARTIFACT = "collapse-fog-legacy-vs-healthy-separation.json"

#: The estimator the owner ruled and the whole program calibrates: `median - k*MAD_n`.
ESTIMATOR = REGISTERED_ESTIMATOR_NAMES[0]  # "median_minus_k_madn"

#: The metric names this round governs. COLLAPSE / FOG gate a map; density_v3 does not.
COLLAPSE_METRIC = "collapse"
FOG_METRIC = "fog"
DENSITY_V3_METRIC = "density_v3"
GATED_METRICS = (COLLAPSE_METRIC, FOG_METRIC)
DESCRIPTIVE_METRICS = (DENSITY_V3_METRIC,)

KIND_ONE_SIDED_LOWER_FLOOR = "one_sided_lower_floor"
KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED = "one_sided_upper_ceiling_fail_closed"
DESCRIPTIVE_MARKING = "descriptive_only_never_a_gate"

VERDICT_NOT_MEASURABLE = "NOT_MEASURABLE"

#: The Gaussian-null calibration convention (R0234/R0255): 4M families, seed 20260809,
#: 95/95 one-sided tolerance bound, closed-form quantile. The healthy-family multiplier
#: is calibrated at n = 7 on 1,000,000 families (REPORT.md §6).
CALIBRATION_FAMILIES = 4_000_000
HEALTHY_FAMILIES = 1_000_000
MADN_CONSISTENCY = 1.4826


# --------------------------------------------------------------------------- #
# Sealed constants — R0264's OWN copy, each reproduced this round against the
# artifact that first published it. Provenance is in SEALED below.
# --------------------------------------------------------------------------- #

#: R0234's Gaussian-null one-sided `median_minus_k_madn` multiplier at n = 13. The
#: validation gate: the calibrator must reproduce this before n = 29 is trusted.
#: /data/latent-basemap/runs/round-0234/.../minilm-calibrated-robust-floors-n13.json
#:   calibration.n13.candidates.median_minus_k_madn.one_sided.calibrated_multiplier
SEALED_VALIDATION_GATE_N13 = 3.7363661743730416

#: R0255's sealed same multiplier at n = 29 — the multiplier `k1` these floors use.
#: /data/latent-basemap/runs/round-0255/.../minilm-calibrated-madn-floors-n29.json
#:   calibration.n29.candidates.median_minus_k_madn.one_sided.calibrated_multiplier
SEALED_K1_N29 = 2.6934393367626024

#: The healthy-family multiplier at n = 7 (REPORT.md §6, 1,000,000 null families),
#: to the precision the report publishes. The node re-derives it to full precision;
#: this literal is the published anchor, not a bitwise target (see WHY_K7_IS_SOFT).
SEALED_K7_HEALTHY_ROUNDED = 6.05513

#: (n-1)/sqrt(n): the identity bound. At n = 29 the calibration multiplier `k1` sits
#: strictly below it (5.199), which is the calibrator-soundness check. At n = 7 the
#: provisional `k7` EXCEEDS it (6.055 > 2.268) — a known consequence of fitting a
#: 95/95 bound to a seven-cell non-seed family, and exactly why the bands are
#: provisional. That is reported, not gated.
SEALED_IDENTITY_BOUND_AT_N29 = 5.199469468957452
SEALED_IDENTITY_BOUND_AT_N7 = 2.2677868380553634

#: The seven core `min_dist = 0.00` umap arms — the HEALTHY family the bands are
#: fitted to (REPORT.md §4). Full-precision collapse / fog, from
#: experiments/metrics/results/collapse-fog-measurements.json (sandbox arms).
HEALTHY_FAMILY: tuple[dict[str, Any], ...] = (
    {"arm": "umap-md000-x2", "rung": "2m", "n_rows": 2_000_000,
     "coordinates": "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2/coordinates.npy",
     "collapse": 1.1379113513907475, "fog": 0.5013022532556332,
     "peak_bin_count": 1435, "resolution_levels": 14},
    {"arm": "umap-md000-x2-seed43", "rung": "2m", "n_rows": 2_000_000,
     "coordinates": "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x2-seed43/coordinates.npy",
     "collapse": 1.2316891541342236, "fog": 0.5122505,
     "peak_bin_count": 1435, "resolution_levels": 14},
    {"arm": "umap-md000-x4", "rung": "2m", "n_rows": 2_000_000,
     "coordinates": "/data/latent-basemap/sandbox/2m-knobs/umap-md000-x4/coordinates.npy",
     "collapse": 1.183425933373684, "fog": 0.4587375,
     "peak_bin_count": 1432, "resolution_levels": 14},
    {"arm": "umap-mind0", "rung": "2m", "n_rows": 2_000_000,
     "coordinates": "/data/latent-basemap/sandbox/2m-knobs/umap-mind0/coordinates.npy",
     "collapse": 1.1928828312336366, "fog": 0.5396465,
     "peak_bin_count": 1435, "resolution_levels": 14},
    {"arm": "umap-md000-x2", "rung": "6250k", "n_rows": 6_250_000,
     "coordinates": "/data/latent-basemap/sandbox/6250k-knobs/umap-md000-x2/coordinates.npy",
     "collapse": 1.0551140155156606, "fog": 0.45087838397960256,
     "peak_bin_count": 4544, "resolution_levels": 45},
    {"arm": "umap-md000-x4", "rung": "6250k", "n_rows": 6_250_000,
     "coordinates": "/data/latent-basemap/sandbox/6250k-knobs/umap-md000-x4/coordinates.npy",
     "collapse": 1.047621099372249, "fog": 0.4456152,
     "peak_bin_count": 4541, "resolution_levels": 45},
    {"arm": "umap-mind0", "rung": "6250k", "n_rows": 6_250_000,
     "coordinates": "/data/latent-basemap/sandbox/6250k-knobs/umap-mind0/coordinates.npy",
     "collapse": 1.1196409023913483, "fog": 0.4791512,
     "peak_bin_count": 4545, "resolution_levels": 45},
)

#: The cuML 1M reference — the external aspiration point. Passes both bands; NOT in
#: the fit (a different recipe and N). REPORT.md §4.
CUML_REFERENCE = {
    "arm": "cuml-1m-reference", "rung": "1m", "n_rows": 1_000_000,
    "coordinates": "/data/latent-basemap/sandbox/cuml-1m/cuml-xy.npy",
    "collapse": 1.6723728930654158, "fog": 0.03974744250970238,
    "peak_bin_count": 198, "resolution_levels": 1, "degenerate": False,
}

#: The degeneracy case a fail-closed fog gate MUST reject: fog reads 0.0000 with peak
#: bin 99 (resolution_levels 0) while it is one of the haziest maps in the sandbox.
SEALED_MD035_DEGENERATE = {
    "arm": "umap-md035-x2", "rung": "2m", "n_rows": 2_000_000,
    "coordinates": "/data/latent-basemap/sandbox/2m-knobs/umap-md035-x2/coordinates.npy",
    "collapse": 2.633976001529383, "fog": 0.0,
    "peak_bin_count": 99, "resolution_levels": 0, "degenerate": True,
}

#: The legacy n = 29 2M seed family, used ONLY as the collapsed fingerprint. Summary
#: statistics (median / MAD_n / min / max) from the 29 cells of the measurements JSON.
#: Its MAX collapse (0.190434) sits far below the healthy MIN (1.047621): the 5.5x
#: separation the floor sits inside. Its fog is LOWER than the healthy family's, which
#: is precisely why the legacy family cannot source a fog ceiling.
LEGACY_FINGERPRINT = {
    "n": 29,
    "collapse": {"median": 0.1660253507577953, "madn": 0.019706756482657058,
                 "min": 0.1355686, "max": 0.1904337},
    "fog": {"median": 0.089990, "madn": 0.012089, "min": 0.067180, "max": 0.158462},
}

#: The provisional bands, to the precision REPORT.md §6 publishes. The node re-fits
#: them from HEALTHY_FAMILY at the re-derived `k7` and requires they round to these.
SEALED_COLLAPSE_FLOOR = 0.644414
SEALED_FOG_CEILING = 0.732966

WHY_K7_IS_SOFT = (
    "k7 is calibrated on 1,000,000 Gaussian-null families by the same closed-form "
    "quantile as k1; it reproduces bit-for-bit given the seed, family count and method, "
    "but REPORT.md publishes it only to 6.05513, so this round requires the floor and "
    "ceiling to reproduce the published bands to 4 decimals rather than bitwise to a "
    "rounded literal. The LOAD-BEARING check is the separation: the legacy fingerprint's "
    "collapse max (0.190434) sits below the floor and the healthy family's collapse min "
    "(1.047621) sits above it, a gap of 5.5x that no last-digit of k7 can move."
)

PROVISIONAL_STATUS = (
    "PROVISIONAL reference bands. The n = 7 core family is not a healthy SEED family "
    "(it varies seed, dose and rung, not seed alone), so the 95/95 tolerance machinery "
    "is applied for consistency with the program's convention, not because the sample "
    "earns it. A registrable healthy floor still needs a real one-recipe seed family "
    "(n >= 13). The collapse statistic separates the two modes cleanly and is ready to "
    "gate; the fog ceiling is a ceiling on regression, not a quality bar."
)

N_RULE = "sqrt_N_normalized_floor_is_N_invariant_residual_saturates"
N_RULE_NOTE = (
    "N-adjustment rule, filled from P1's dose x N decomposition (frozen analysis-v2, "
    "commits a9651cf -> db0113a; 7 same-N-validated points, x2 at 2M/6.25M/12.5M/25M + "
    "x4 at 2M/6.25M/12.5M). The collapse statistic is sqrt(N)-normalized (removes 2D "
    "packing; REPORT.md §4). MEASURED: the residual drift SATURATES rather than "
    "declining -- the fneg treatment's normalized collapse LEVELS at an asymptote "
    "(saturating fit lambda=37.08 => leveled by 6.25M; y_inf 0.965 [0.930,0.985] x2, "
    "0.991 [0.961,1.016] x4, both band-lowers above the 0.9 floor). THEREFORE the "
    "registered floor applies to the sqrt(N)-normalized statistic N-INVARIANTLY for "
    "N >= 2M: no N-dependent floor relaxation is registered -- a map clearing the floor "
    "at its rung earns no scale credit, and a map failing at its rung is not excused by "
    "N. This is measurement on the fneg-treatment family, not an assumption; it is "
    "revisited if a future treatment shows a declining (non-saturating) residual. The "
    "band-level rule is deliberately conservative: it does not widen the floor with N."
)
# Retained alias so any downstream reference to the old name resolves.
N_RULE_PENDING = N_RULE

SEALED: dict[str, Any] = {
    "r0234_n13_calibration": {
        "path": (
            "/data/latent-basemap/runs/round-0234/queue/artifacts/"
            "minilm-mixed-2m-calibrated-robust-floors-n13-v1/"
            "minilm-calibrated-robust-floors-n13.json"
        ),
        "key": "calibration.n13.candidates.median_minus_k_madn.one_sided.calibrated_multiplier",
        "supplies": "the sealed n=13 validation-gate multiplier (3.7363661743730416)",
    },
    "r0255_n29_calibration": {
        "path": (
            "/data/latent-basemap/runs/round-0255/queue/artifacts/"
            "minilm-mixed-2m-calibrated-madn-floors-n29-v1/"
            "minilm-calibrated-madn-floors-n29.json"
        ),
        "key": "calibration.n29.candidates.median_minus_k_madn.one_sided.calibrated_multiplier",
        "supplies": "the sealed n=29 multiplier k1 (2.6934393367626024)",
    },
    "collapse_fog_calibration": {
        "path": "/home/enjalot/code/latent-basemap/experiments/metrics/results/collapse-fog-calibration.json",
        "supplies": "the collapse/fog program's own record of the bit-exact k1 / n=13 reproduction (optional cross-check)",
    },
    "collapse_fog_measurements": {
        "path": "/data/latent-basemap/runs/round-0264/inputs/collapse-fog-measurements.json",
        "supplies": "the healthy family, legacy fingerprint, cuML reference and md035 measurements (family source + MANDATORY cross-check); R0264's own sealed /data copy of the collapse/fog program's measurement record",
    },
}


class Round0264NodeError(RuntimeError):
    """The R0264 node contract changed."""


# --------------------------------------------------------------------------- #
# Local mirror of the sealed metric logic (collapse_fog.py). numpy-only; no scipy
# is needed here because node 1 measures no coordinates — the family statistics are
# sealed constants and the only live measurement is the planted degenerate fog map.
# --------------------------------------------------------------------------- #

CHUNK_ROWS = 2_000_000
EXTENT_SAMPLE_ROWS = 2_000_000


def _madn(values) -> float:
    """Scaled median absolute deviation, consistency constant 1.4826 (collapse_fog.madn)."""
    v = np.asarray(values, dtype=np.float64)
    centre = np.median(v)
    return float(MADN_CONSISTENCY * np.median(np.abs(v - centre)))


def _robust_floor(values, k: float) -> float:
    """One-sided lower floor: median - k * MAD_n (collapse_fog.robust_floor)."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.median(v) - float(k) * _madn(v))


def _robust_ceiling(values, k: float) -> float:
    """One-sided upper ceiling: median + k * MAD_n (collapse_fog.robust_ceiling)."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.median(v) + float(k) * _madn(v))


def _robust_extent(arr, extent_pct: tuple[float, float], sample_rows: int) -> list[float]:
    """Percentile extent with 2% pad — byte-for-byte map_renders.robust_extent."""
    n = int(arr.shape[0])
    if n > sample_rows:
        idx = np.linspace(0, n - 1, sample_rows).astype(np.int64)
        pts = np.asarray(arr[idx], dtype=np.float32)
    else:
        pts = np.asarray(arr, dtype=np.float32)
    lo, hi = extent_pct
    x0, x1 = np.percentile(pts[:, 0], [lo, hi])
    y0, y1 = np.percentile(pts[:, 1], [lo, hi])
    pad_x = 0.02 * (x1 - x0) or 1.0
    pad_y = 0.02 * (y1 - y0) or 1.0
    return [float(x0 - pad_x), float(x1 + pad_x), float(y0 - pad_y), float(y1 + pad_y)]


def _binned_counts(arr, extent: list[float], bins: int) -> np.ndarray:
    """Chunked 2D histogram; off-extent points clip into edge bins (map_renders.binned_counts)."""
    x0, x1, y0, y1 = extent
    edges_x = np.linspace(x0, x1, bins + 1)
    edges_y = np.linspace(y0, y1, bins + 1)
    counts = np.zeros((bins, bins), dtype=np.int64)
    for i in range(0, int(arr.shape[0]), CHUNK_ROWS):
        chunk = np.asarray(arr[i:i + CHUNK_ROWS], dtype=np.float32)
        cx = np.clip(chunk[:, 0], x0, x1)
        cy = np.clip(chunk[:, 1], y0, y1)
        h, _, _ = np.histogram2d(cx, cy, bins=[edges_x, edges_y])
        counts += h.astype(np.int64)
    return counts


def _map_fog(
    xy,
    bins: int = 1024,
    extent_pct: tuple[float, float] = (0.1, 99.9),
    threshold: float = 0.01,
) -> dict[str, Any]:
    """Diffuse-haze statistic — mirror of collapse_fog.map_fog.

    Returns ``fog``/``low_density_mass_fraction`` plus the two validity keys the
    fail-closed gate consults: ``resolution_levels = floor(threshold * peak)`` and
    ``degenerate = (resolution_levels < 1)``. A map whose peak bin holds fewer than
    ``1 / threshold`` points reports fog identically 0.0 regardless of haze.
    """
    arr = np.asarray(xy, dtype=np.float32) if not hasattr(xy, "shape") else xy
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (N, 2) coordinates, got shape {arr.shape}")
    n_rows = int(arr.shape[0])
    if n_rows == 0:
        raise ValueError("empty coordinate array")
    extent = _robust_extent(arr, extent_pct, EXTENT_SAMPLE_ROWS)
    counts = _binned_counts(arr, extent, bins)
    total = int(counts.sum())
    occupied = counts > 0
    peak = int(counts.max())
    cutoff_raw = peak * threshold
    cutoff = max(cutoff_raw, 1.0)
    low = occupied & (counts < cutoff)
    fog = float(counts[low].sum() / total) if total else float("nan")
    resolution_levels = int(math.floor(cutoff_raw))
    return {
        "metric": "fog",
        "statistic": "low_density_mass_fraction",
        "n_rows": n_rows,
        "bins": int(bins),
        "threshold": float(threshold),
        "total_points_binned": total,
        "peak_bin_count": peak,
        "low_bin_cutoff": float(cutoff),
        "resolution_levels": resolution_levels,
        "degenerate": bool(resolution_levels < 1),
        "occupied_bins": int(occupied.sum()),
        "low_density_mass_fraction": fog,
        "fog": fog,
    }


# --------------------------------------------------------------------------- #
# The two judges. COLLAPSE is a one-sided lower floor. FOG FAILS CLOSED.
# --------------------------------------------------------------------------- #


def _judge_collapse(value: float, floor: float) -> dict[str, Any]:
    """One-sided lower floor on r10/R*sqrt(N): too small means bead collapse."""
    v = float(value)
    if not math.isfinite(v):
        raise Round0264NodeError("R0264 refuses a nonfinite collapse statistic")
    passes = v >= float(floor)
    return {
        "metric": COLLAPSE_METRIC,
        "verdict": "PASS" if passes else "FAIL",
        "passes": passes,
        "measurable": True,
        "value": v,
        "floor": float(floor),
        "margin": v - float(floor),
        "gate_direction": KIND_ONE_SIDED_LOWER_FLOOR,
    }


def _judge_fog(fog_result: Mapping[str, Any], ceiling: float) -> dict[str, Any]:
    """One-sided upper ceiling on fog — FAILS CLOSED on degeneracy.

    The load-bearing line: BEFORE the value is compared to the ceiling, a
    ``degenerate`` measurement (or ``resolution_levels < 1``) is refused as
    NOT_MEASURABLE. A not-measurable fog is NEVER a pass — which is what stops
    md035's fog of 0.0000 (peak bin 99) from clearing the ceiling spuriously.
    """
    resolution_levels = int(fog_result.get("resolution_levels", -1))
    degenerate = bool(fog_result.get("degenerate")) or resolution_levels < 1
    if degenerate:
        return {
            "metric": FOG_METRIC,
            "verdict": VERDICT_NOT_MEASURABLE,
            "passes": False,
            "measurable": False,
            "value": fog_result.get("fog"),
            "ceiling": float(ceiling),
            "resolution_levels": resolution_levels,
            "peak_bin_count": fog_result.get("peak_bin_count"),
            "degenerate": True,
            "gate_direction": KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED,
            "why_not_measurable": (
                "peak bin below 1/threshold: no usable low-density level exists, so fog "
                "is identically 0.0 regardless of how hazy the map is. A not-measurable "
                "fog is never a pass"
            ),
        }
    v = float(fog_result["fog"])
    passes = v <= float(ceiling)
    return {
        "metric": FOG_METRIC,
        "verdict": "PASS" if passes else "FAIL",
        "passes": passes,
        "measurable": True,
        "value": v,
        "ceiling": float(ceiling),
        "resolution_levels": resolution_levels,
        "peak_bin_count": fog_result.get("peak_bin_count"),
        "degenerate": False,
        "margin": float(ceiling) - v,
        "gate_direction": KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED,
    }


def _judge_fog_naive(fog_result: Mapping[str, Any], ceiling: float) -> dict[str, Any]:
    """The GUARD-LESS judge — value only, degeneracy ignored. Used ONLY inside the
    fail-closed positive control to prove the guard changes the outcome; it is never
    a shipped criterion."""
    v = float(fog_result["fog"])
    passes = v <= float(ceiling)
    return {"verdict": "PASS" if passes else "FAIL", "passes": passes, "value": v}


# --------------------------------------------------------------------------- #
# The calibration, reproduced bitwise
# --------------------------------------------------------------------------- #


def _read_sealed_multiplier(path: str, n_key: str) -> float:
    """Read a sealed calibrated one-sided MAD_n multiplier — plain JSON, read-only.

    Mirrors null_calibration.sealed_multiplier: these calibration artifacts are plain
    JSON with a top-level ``calibration`` key, not seal-enveloped.
    """
    with open(path, "r", encoding="utf-8") as handle:
        d = json.load(handle)
    return float(
        d["calibration"][n_key]["candidates"]["median_minus_k_madn"]["one_sided"][
            "calibrated_multiplier"
        ]
    )


def _calibrate_one_sided(n: int, families: int) -> float:
    """The program's Gaussian-null one-sided multiplier at `n`, via round0234_calibration.

    Same estimator, seed and closed-form quantile the metric program's own
    null_calibration uses; the chunk size is irrelevant because the Generator draws
    the identical concatenated stream regardless of how it is batched.
    """
    result = calibration.calibrate(
        n, names=REGISTERED_ESTIMATOR_NAMES, families=families, two_sided=False
    )
    return float(result["candidates"][ESTIMATOR]["one_sided"]["calibrated_multiplier"])


def _calibration_receipt(job: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the validation gate (n=13) and k1 (n=29) bitwise, three ways."""
    started = time.monotonic()
    computed_k13 = _calibrate_one_sided(13, CALIBRATION_FAMILIES)
    computed_k1 = _calibrate_one_sided(N_EXACT, CALIBRATION_FAMILIES)
    computed_k7 = _calibrate_one_sided(7, HEALTHY_FAMILIES)
    wall_s = time.monotonic() - started

    sealed_k13 = _read_sealed_multiplier(
        _bound_path(job, "r0234_n13", label="R0234 sealed n=13 calibration"), "n13"
    )
    sealed_k1 = _read_sealed_multiplier(
        _bound_path(job, "r0255_n29", label="R0255 sealed n=29 calibration"), "n29"
    )

    gate_match = _bitwise_same(
        {"n13_multiplier": sealed_k13, "n13_constant": SEALED_VALIDATION_GATE_N13},
        {"n13_multiplier": computed_k13, "n13_constant": computed_k13},
    )
    k1_match = _bitwise_same(
        {"n29_multiplier": sealed_k1, "n29_constant": SEALED_K1_N29},
        {"n29_multiplier": computed_k1, "n29_constant": computed_k1},
    )
    return {
        "estimator": ESTIMATOR,
        "families": CALIBRATION_FAMILIES,
        "healthy_families": HEALTHY_FAMILIES,
        "wall_s": wall_s,
        "validation_gate_n13": {
            "sealed_value": sealed_k13,
            "sealed_constant": SEALED_VALIDATION_GATE_N13,
            "reproduced_value": computed_k13,
            "reproduces_bitwise": gate_match["every_value_is_bitwise_identical"],
            "detail": gate_match,
        },
        "k1_n29": {
            "sealed_value": sealed_k1,
            "sealed_constant": SEALED_K1_N29,
            "reproduced_value": computed_k1,
            "reproduces_bitwise": k1_match["every_value_is_bitwise_identical"],
            "cross_check_source": SEALED["r0255_n29_calibration"]["path"],
            "detail": k1_match,
        },
        "k7_healthy": {
            "reproduced_value": computed_k7,
            "published_anchor": SEALED_K7_HEALTHY_ROUNDED,
            "rounds_to_the_published_anchor": matches_cited_display_value(
                computed_k7, SEALED_K7_HEALTHY_ROUNDED, decimals=5
            ),
            "why_soft": WHY_K7_IS_SOFT,
        },
        "identity_bound_at_n29": identity_bound(N_EXACT),
        "k1_below_identity_bound_at_n29": computed_k1 < identity_bound(N_EXACT),
        "identity_bound_at_n7": identity_bound(7),
        "k7_exceeds_identity_bound_at_n7": computed_k7 > identity_bound(7),
        "why_k7_exceeds_the_bound_is_not_an_error": (
            "at n = 7 a 95/95 one-sided tolerance multiplier is large; the band is "
            "provisional for exactly this reason. The calibrator-soundness check is "
            "k1 < identity_bound(29), which holds."
        ),
    }


# --------------------------------------------------------------------------- #
# The provisional bands, fitted to the HEALTHY family
# --------------------------------------------------------------------------- #


def _fit_provisional_bands(k7: float) -> dict[str, Any]:
    """Fit the collapse floor and fog ceiling on the seven core umap arms."""
    collapse_values = [float(arm["collapse"]) for arm in HEALTHY_FAMILY]
    fog_values = [float(arm["fog"]) for arm in HEALTHY_FAMILY]
    collapse_floor = _robust_floor(collapse_values, k7)
    fog_ceiling = _robust_ceiling(fog_values, k7)
    return {
        "family_size": len(HEALTHY_FAMILY),
        "family_arms": [f"{a['arm']}@{a['rung']}" for a in HEALTHY_FAMILY],
        "multiplier_k7": float(k7),
        "collapse": {
            "direction": KIND_ONE_SIDED_LOWER_FLOOR,
            "median": float(np.median(collapse_values)),
            "madn": _madn(collapse_values),
            "min": float(min(collapse_values)),
            "max": float(max(collapse_values)),
            "floor": collapse_floor,
        },
        "fog": {
            "direction": KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED,
            "median": float(np.median(fog_values)),
            "madn": _madn(fog_values),
            "min": float(min(fog_values)),
            "max": float(max(fog_values)),
            "ceiling": fog_ceiling,
        },
    }


def _separation_record(*, collapse_floor: float) -> dict[str, Any]:
    """The clean gap the floor sits inside: legacy fingerprint vs healthy family."""
    legacy_max = float(LEGACY_FINGERPRINT["collapse"]["max"])
    healthy_min = float(min(arm["collapse"] for arm in HEALTHY_FAMILY))
    return {
        "legacy_collapse_max": legacy_max,
        "collapse_floor": float(collapse_floor),
        "healthy_collapse_min": healthy_min,
        "floor_is_above_the_legacy_mode": collapse_floor > legacy_max,
        "floor_is_below_the_healthy_mode": collapse_floor < healthy_min,
        "separation_factor": healthy_min / legacy_max,
        "the_two_modes_do_not_overlap": legacy_max < healthy_min,
        "why_the_legacy_family_is_only_a_fingerprint": (
            "the legacy family is 29 replicates of one collapsed legacy_lp recipe; a "
            "floor fitted to it certifies a new map is AS collapsed as the collapsed "
            "ones, and its fog (median 0.090) is LOWER than the healthy family's "
            "(0.479), so a fog ceiling fitted to it would reject every healthy map. It "
            "is used only as the collapsed mode's fingerprint, never as a band source."
        ),
    }


# --------------------------------------------------------------------------- #
# The mixed registry: two gated criteria, density_v3 descriptive-only
# --------------------------------------------------------------------------- #


def _registered_criteria_collapse_fog(
    *, collapse_floor: float, fog_ceiling: float
) -> dict[str, Any]:
    return {
        COLLAPSE_METRIC: {
            "kind": KIND_ONE_SIDED_LOWER_FLOOR,
            "floor": float(collapse_floor),
            "statistic": "r10_over_radius_times_sqrt_n",
            "gates_a_map": True,
            "fails_only": "bead_collapse",
            "applicable_power": "one_sided",
            "read_as": (
                "the N-adjusted r10/R*sqrt(N) must be at or above `floor`; below it the "
                "map has folded into point-like beads, which registered FFR rewards"
            ),
        },
        FOG_METRIC: {
            "kind": KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED,
            "ceiling": float(fog_ceiling),
            "statistic": "low_density_mass_fraction",
            "gates_a_map": True,
            "fails_only": "diffuse_haze",
            "fails_closed_on_degeneracy": True,
            "applicable_power": "one_sided",
            "read_as": (
                "low_density_mass_fraction must be at or below `ceiling`. A degenerate "
                "measurement (resolution_levels < 1) is NOT_MEASURABLE and never a pass"
            ),
        },
    }


def _descriptive_density_v3_diagnostics() -> dict[str, Any]:
    return {
        DESCRIPTIVE_MARKING: True,
        "contributes_to_verdict": False,
        DENSITY_V3_METRIC: {
            "kind": "local_density_agreement_spearman_DESCRIPTIVE_ONLY",
            "statistic": "spearman",
            "gates_a_map": False,
        },
        "what_this_is": (
            "density_v3 (experiments/metrics/density_v3.py): a repaired local-density-"
            "agreement metric, published beside verdicts and never gating a map"
        ),
        "why_it_never_gates": (
            "its ordering INVERTS visual quality — it ranks the collapsed legacy family "
            "highest (rho ~0.20-0.26) and the clean cuML reference lowest (rho 0.054), "
            "an ordering that survives a commensurate-universe control. A high density_v3 "
            "is not a good map, so a floor on it would be a floor on the wrong thing."
        ),
    }


def _assert_registry_gates_collapse_and_fog_only(
    criteria: Mapping[str, Any], descriptive: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse a registry that mis-shapes either gate or that gates a descriptive metric."""
    problems: list[str] = []

    collapse = criteria.get(COLLAPSE_METRIC)
    if collapse is None:
        problems.append(f"{COLLAPSE_METRIC} gates a map and must be registered")
    else:
        if str(collapse.get("kind")) != KIND_ONE_SIDED_LOWER_FLOOR:
            problems.append(f"{COLLAPSE_METRIC} must be a {KIND_ONE_SIDED_LOWER_FLOOR}")
        if "floor" not in collapse:
            problems.append(f"{COLLAPSE_METRIC} must publish a floor")
        if str(collapse.get("applicable_power")) != "one_sided":
            problems.append(f"{COLLAPSE_METRIC} must carry one-sided power")

    fog = criteria.get(FOG_METRIC)
    if fog is None:
        problems.append(f"{FOG_METRIC} gates a map and must be registered")
    else:
        if str(fog.get("kind")) != KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED:
            problems.append(
                f"{FOG_METRIC} must be a {KIND_ONE_SIDED_UPPER_CEILING_FAIL_CLOSED}"
            )
        if "ceiling" not in fog:
            problems.append(f"{FOG_METRIC} must publish a ceiling")
        if fog.get("fails_closed_on_degeneracy") is not True:
            problems.append(
                f"{FOG_METRIC} must declare fails_closed_on_degeneracy: true — a fog gate "
                "that reads the value before the degeneracy flag passes md035 spuriously"
            )

    for metric in DESCRIPTIVE_METRICS:
        if metric in criteria:
            problems.append(f"{metric} is descriptive-only and may not be a criterion")

    if not descriptive.get(DESCRIPTIVE_MARKING):
        problems.append(f"the descriptive block must carry {DESCRIPTIVE_MARKING}: true")
    if descriptive.get("contributes_to_verdict") is not False:
        problems.append("the descriptive block must declare contributes_to_verdict: false")

    if problems:
        raise Round0264NodeError(
            "R0264 registry refuses: " + "; ".join(sorted(set(problems)))
        )
    return {
        "holds": True,
        "gated_criteria_registered": sorted(criteria),
        "collapse_is_a_one_sided_lower_floor": True,
        "fog_is_a_fail_closed_upper_ceiling": True,
        "density_v3_absent_from_the_registry": sorted(DESCRIPTIVE_METRICS),
    }


def _registry_controls_collapse_fog(
    *, criteria: Mapping[str, Any], descriptive: Mapping[str, Any]
) -> dict[str, Any]:
    """Positive controls for the registry-shape guard; every plant must be refused."""
    honest_ok = True
    honest_error = None
    try:
        _assert_registry_gates_collapse_and_fog_only(criteria, descriptive)
    except Round0264NodeError as error:
        honest_ok = False
        honest_error = f"{type(error).__name__}: {error}"

    plants: list[dict[str, Any]] = []

    def plant(name: str, why: str, run: Callable[[], Any]) -> None:
        refused = False
        message = None
        try:
            run()
        except Round0264NodeError as error:
            refused = True
            message = str(error)
        plants.append({
            "plant": name,
            "why_this_is_the_defect": why,
            "refused": refused,
            "error": message,
        })

    plant(
        "density_v3_registered_as_a_gate",
        "density_v3's ordering inverts quality; gating it would fail the clean maps",
        lambda: _assert_registry_gates_collapse_and_fog_only(
            {**dict(criteria), DENSITY_V3_METRIC: {
                "kind": KIND_ONE_SIDED_LOWER_FLOOR, "floor": 0.1,
                "applicable_power": "one_sided",
            }},
            descriptive,
        ),
    )
    plant(
        "fog_without_the_fail_closed_flag",
        "a fog ceiling that reads the value before the degeneracy flag passes md035 "
        "(fog 0.0, peak 99) spuriously",
        lambda: _assert_registry_gates_collapse_and_fog_only(
            {**dict(criteria), FOG_METRIC: {
                # the defect IS the honest fail-closed flag, flipped off — derived from the
                # registered criterion rather than asserted as a literal
                **dict(criteria[FOG_METRIC]),
                "fails_closed_on_degeneracy": not criteria[FOG_METRIC]["fails_closed_on_degeneracy"],
            }},
            descriptive,
        ),
    )
    plant(
        "the_collapse_floor_dropped",
        "the round does half its job — the bead-collapse direction is left ungated",
        lambda: _assert_registry_gates_collapse_and_fog_only(
            {k: v for k, v in criteria.items() if k != COLLAPSE_METRIC},
            descriptive,
        ),
    )
    plant(
        "fog_registered_as_a_lower_floor",
        "wrong direction: haze fails from ABOVE, a lower floor cannot see it",
        lambda: _assert_registry_gates_collapse_and_fog_only(
            {**dict(criteria), FOG_METRIC: {
                **dict(criteria[FOG_METRIC]), "kind": KIND_ONE_SIDED_LOWER_FLOOR,
            }},
            descriptive,
        ),
    )
    plant(
        "the_descriptive_block_left_unmarked",
        "an unmarked density_v3 block is indistinguishable from a registry",
        lambda: _assert_registry_gates_collapse_and_fog_only(
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


# --------------------------------------------------------------------------- #
# The FOG FAIL-CLOSED positive control — the load-bearing guard of this round
# --------------------------------------------------------------------------- #


def _planted_degenerate_map(seed: int = 0) -> np.ndarray:
    """A hazy small map whose peak bin is below 1/threshold, so fog is forced to 0.0.

    N = 90 < 100 guarantees no bin can hold 100 points, so ``resolution_levels`` is 0
    and ``degenerate`` is true no matter how the points fall — the same trap md035
    (peak bin 99) fell into at 2M rows. A tight clump plus a spread halo makes it a
    map that WOULD read hazy if it had the resolution to.
    """
    rng = np.random.default_rng(seed)
    clump = rng.normal(0.0, 0.02, size=(40, 2))
    halo = rng.uniform(-1.0, 1.0, size=(50, 2))
    return np.concatenate([clump, halo]).astype(np.float32)


def _fog_fail_closed_controls(*, fog_ceiling: float) -> dict[str, Any]:
    """Prove the fog gate refuses a degenerate map, and that the refusal changes the outcome.

    Three arms:
      (1) a planted degenerate map, measured through the real ``_map_fog`` binning —
          ``resolution_levels < 1``, ``degenerate`` true, fog 0.0;
      (2) the shipped judge REFUSES it (NOT_MEASURABLE, never a pass) while the naive
          guard-less judge PASSES it (0.0 <= ceiling) — so the guard is proven to
          change the outcome, not merely to be present;
      (3) the sealed md035 case (peak bin 99) is refused too, and an honest
          non-degenerate map is measured and passes, so the guard is not a blanket veto.
    """
    planted_xy = _planted_degenerate_map()
    planted = _map_fog(planted_xy)
    planted_is_degenerate = bool(planted["degenerate"]) and planted["resolution_levels"] < 1
    planted_fog_is_zero = planted["fog"] == 0.0

    shipped = _judge_fog(planted, fog_ceiling)
    naive = _judge_fog_naive(planted, fog_ceiling)
    shipped_refused = (
        shipped["verdict"] == VERDICT_NOT_MEASURABLE and shipped["passes"] is False
    )
    naive_would_pass = bool(naive["passes"])
    guard_changed_the_outcome = shipped_refused and naive_would_pass

    md035_result = {
        "fog": float(SEALED_MD035_DEGENERATE["fog"]),
        "peak_bin_count": int(SEALED_MD035_DEGENERATE["peak_bin_count"]),
        "resolution_levels": int(SEALED_MD035_DEGENERATE["resolution_levels"]),
        "degenerate": bool(SEALED_MD035_DEGENERATE["degenerate"]),
    }
    md035_shipped = _judge_fog(md035_result, fog_ceiling)
    md035_naive = _judge_fog_naive(md035_result, fog_ceiling)
    md035_refused = (
        md035_shipped["verdict"] == VERDICT_NOT_MEASURABLE
        and md035_shipped["passes"] is False
    )
    md035_naive_would_pass = bool(md035_naive["passes"])

    honest_resolution_levels = int(HEALTHY_FAMILY[0]["resolution_levels"])
    honest = {
        "fog": float(HEALTHY_FAMILY[0]["fog"]),
        "peak_bin_count": int(HEALTHY_FAMILY[0]["peak_bin_count"]),
        "resolution_levels": honest_resolution_levels,
        # degeneracy is the measurement, not a literal: _map_fog defines it as
        # resolution_levels < 1, so the honest map's ~14 levels compute to non-degenerate
        "degenerate": honest_resolution_levels < 1,
    }
    honest_judged = _judge_fog(honest, fog_ceiling)
    honest_measurable_and_passes = (
        honest_judged["measurable"] and honest_judged["passes"]
    )

    holds = bool(
        planted_is_degenerate
        and planted_fog_is_zero
        and shipped_refused
        and naive_would_pass
        and guard_changed_the_outcome
        and md035_refused
        and md035_naive_would_pass
        and honest_measurable_and_passes
    )
    return {
        "planted_map": {
            "n_rows": int(planted["n_rows"]),
            "peak_bin_count": int(planted["peak_bin_count"]),
            "resolution_levels": int(planted["resolution_levels"]),
            "degenerate": bool(planted["degenerate"]),
            "fog": planted["fog"],
            "is_degenerate_from_real_binning": planted_is_degenerate,
        },
        "shipped_judge_refuses_the_planted_map": shipped_refused,
        "shipped_verdict": shipped["verdict"],
        "naive_judge_would_have_passed_it": naive_would_pass,
        "the_guard_changed_the_outcome": guard_changed_the_outcome,
        "md035": {
            "shipped_verdict": md035_shipped["verdict"],
            "shipped_refused": md035_refused,
            "naive_would_have_passed": md035_naive_would_pass,
        },
        "an_honest_non_degenerate_map_is_measurable_and_passes": honest_measurable_and_passes,
        "holds": holds,
        "why_this_exists": (
            "a fail-closed guard proven only by 'the degenerate map did not pass' holds "
            "equally for a guard that refuses everything. The naive-judge arm shows the "
            "same map PASSES when the degeneracy check is skipped, so the refusal is a "
            "decision the guard makes, not a coincidence — and the honest arm shows the "
            "guard still lets a real map through."
        ),
    }


# --------------------------------------------------------------------------- #
# The COLLAPSE floor positive control
# --------------------------------------------------------------------------- #


def _collapse_floor_controls(*, collapse_floor: float) -> dict[str, Any]:
    """A planted bead-collapse value and the legacy fingerprint FAIL; healthy + cuML PASS."""
    planted_bead = _judge_collapse(0.0, collapse_floor)
    legacy_max = _judge_collapse(LEGACY_FINGERPRINT["collapse"]["max"], collapse_floor)
    healthy = {
        f"{arm['arm']}@{arm['rung']}": _judge_collapse(arm["collapse"], collapse_floor)
        for arm in HEALTHY_FAMILY
    }
    cuml = _judge_collapse(CUML_REFERENCE["collapse"], collapse_floor)
    planted_bead_fails = planted_bead["verdict"] == "FAIL"
    legacy_fails = legacy_max["verdict"] == "FAIL"
    every_healthy_passes = all(row["verdict"] == "PASS" for row in healthy.values())
    cuml_passes = cuml["verdict"] == "PASS"
    return {
        "planted_bead_value": 0.0,
        "planted_bead_fails_the_floor": planted_bead_fails,
        "legacy_fingerprint_max_fails_the_floor": legacy_fails,
        "every_healthy_arm_passes_the_floor": every_healthy_passes,
        "cuml_reference_passes_the_floor": cuml_passes,
        "healthy_verdicts": {k: v["verdict"] for k, v in healthy.items()},
        "holds": bool(
            planted_bead_fails and legacy_fails and every_healthy_passes and cuml_passes
        ),
        "why_this_exists": (
            "a floor proven only by 'the healthy maps passed' would also be passed by a "
            "floor of -inf. The planted bead and the legacy fingerprint are the failing "
            "inputs a collapse floor exists to catch."
        ),
    }


# --------------------------------------------------------------------------- #
# attainability / power / false-alarm rate (reported, provisional)
# --------------------------------------------------------------------------- #


def _one_sided_report(job: Mapping[str, Any], calibration_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """One-sided false-fail rate and detection power at the calibration n = 29, plus the
    two-criterion panel false-alarm rate under independence."""
    result = calibration.calibrate(
        N_EXACT, names=REGISTERED_ESTIMATOR_NAMES, families=CALIBRATION_FAMILIES,
        two_sided=False,
    )
    entry = result["candidates"][ESTIMATOR]["one_sided"]
    per_metric_rate = float(entry["new_cell_false_fail_rate"])
    panel = 1.0 - (1.0 - per_metric_rate) ** len(GATED_METRICS)
    return {
        "n_for_power": N_EXACT,
        "one_sided_new_cell_false_fail_rate": per_metric_rate,
        "one_sided_detection_power": {k: float(v) for k, v in entry["detection_power"].items()},
        "gated_criteria": list(GATED_METRICS),
        "panel_false_alarm_rate_under_independence": panel,
        "panel_expression": f"1 - (1 - {per_metric_rate!r})**{len(GATED_METRICS)}",
        "this_is_an_upper_bound": True,
        "why_power_uses_n29_not_n7": (
            "detection power and the new-cell false-fail rate are properties of the "
            "calibration at the family size where the multiplier is sealed (n = 29). The "
            "provisional bands borrow the estimator; the n = 7 fit inflates the "
            "multiplier and is reported for the band edges, not as a power claim."
        ),
    }


# --------------------------------------------------------------------------- #
# mandatory cross-check against the metric program's own measurement record
# --------------------------------------------------------------------------- #


def _measurements_cross_check(job: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check the sealed healthy / legacy / md035 constants against R0264's own
    sealed measurement record. MANDATORY: the ``collapse_fog_measurements`` input must be
    bound and present — a missing binding or an absent/altered file RAISES rather than
    skipping, so the returned value is always a computation over the record, never a
    skippable literal. NON-load-bearing for the gate (the sealed constants are what the
    gate reads), but the cross-check itself always executes and reports the arms it
    matched and any mismatch against the record."""
    reference = job.get("collapse_fog_measurements")
    if not isinstance(reference, Mapping):
        raise Round0264NodeError(
            "R0264 measurements cross-check requires the collapse_fog_measurements input; "
            "it is not bound in this job"
        )
    # _bound_path verifies the sealed sha256/bytes when signed and RAISES if the file is
    # absent or altered — the same admission check the separation node uses.
    path = _bound_path(job, "collapse_fog_measurements", label="collapse/fog measurements")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    sandbox = {(a["arm"], a.get("rung")): a for a in data.get("sandbox", [])}
    mismatches: list[str] = []
    arms_matched: list[str] = []
    for arm in HEALTHY_FAMILY:
        label = f"{arm['arm']}@{arm['rung']}"
        row = sandbox.get((arm["arm"], arm["rung"]))
        if row is None or row["collapse"] != arm["collapse"] or row["fog"] != arm["fog"]:
            mismatches.append(label)
        else:
            arms_matched.append(label)
    md035 = sandbox.get((SEALED_MD035_DEGENERATE["arm"], SEALED_MD035_DEGENERATE["rung"]))
    if md035 is None or md035["fog"] != SEALED_MD035_DEGENERATE["fog"] or int(
        md035["peak_bin_count"]
    ) != SEALED_MD035_DEGENERATE["peak_bin_count"]:
        mismatches.append("umap-md035-x2")
    else:
        arms_matched.append("umap-md035-x2")
    return {
        "measurements_path": path,
        "arms_cross_checked": len(HEALTHY_FAMILY) + 1,
        "arms_matched": arms_matched,
        "healthy_and_md035_match_the_sealed_constants": not mismatches,
        "mismatches": mismatches,
    }


# --------------------------------------------------------------------------- #
# node 1 — the registration
# --------------------------------------------------------------------------- #


def run_register(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0264 round0264_nodes.run_register")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0264NodeError("R0264 register handler received another queue")
    node_id = str(active.get("node_id") or REGISTER_ACTION)
    label = "R0264 collapse floor / fail-closed fog ceiling registration (DRAFT)"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    started_utc = time.time()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0264 registration")

    window = ledger.window("R0264 registration stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0264 registration stage entered")
        wrapped = window.wrap(recorder)

        # A. the calibration, reproduced bitwise (validation gate + R0255 cross-check)
        calibration_receipt = _calibration_receipt(job)
        k1 = float(calibration_receipt["k1_n29"]["reproduced_value"])
        k7 = float(calibration_receipt["k7_healthy"]["reproduced_value"])
        wrapped("R0264 calibration reproduced (n=13 gate + n=29 k1 bitwise)")

        # B. the provisional bands, fitted on the HEALTHY family only
        bands = _fit_provisional_bands(k7)
        collapse_floor = float(bands["collapse"]["floor"])
        fog_ceiling = float(bands["fog"]["ceiling"])
        separation = _separation_record(collapse_floor=collapse_floor)
        # Reproduce the published anchors at THEIR precision (6 decimals):
        # SEALED_* are the REPORT §6 values 0.644414 / 0.732966, so the computed
        # floor/ceiling must round to 6 places, not 4 (a round-to-4 computed value
        # can never equal a 6-decimal literal — the R0264 setup-fix bug). Uses the
        # shared round-vs-literal helper so the convention lives in one place.
        bands_reproduce_published = (
            matches_cited_display_value(collapse_floor, SEALED_COLLAPSE_FLOOR, decimals=6)
            and matches_cited_display_value(fog_ceiling, SEALED_FOG_CEILING, decimals=6)
        )
        measurements_cross_check = _measurements_cross_check(job)
        wrapped("R0264 provisional bands fitted on the healthy family")

        # C. attainability / power / panel false-alarm rate (reported, provisional)
        one_sided = _one_sided_report(job, calibration_receipt)
        wrapped("R0264 one-sided power and panel false-alarm rate computed")

        # D. the mixed registry, the descriptive block, and their controls
        criteria = _registered_criteria_collapse_fog(
            collapse_floor=collapse_floor, fog_ceiling=fog_ceiling
        )
        descriptive = _descriptive_density_v3_diagnostics()
        registry_guard = _assert_registry_gates_collapse_and_fog_only(criteria, descriptive)
        registry_plants = _registry_controls_collapse_fog(
            criteria=criteria, descriptive=descriptive
        )
        fog_fail_closed = _fog_fail_closed_controls(fog_ceiling=fog_ceiling)
        collapse_controls = _collapse_floor_controls(collapse_floor=collapse_floor)
        wrapped("R0264 registry, fail-closed fog control and collapse control complete")
        gate.finish("R0264 registration stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    cuda_mappings = _cuda_mappings()

    execution_checks = {
        "the_validation_gate_reproduces_r0234_n13_bitwise": bool(
            calibration_receipt["validation_gate_n13"]["reproduces_bitwise"]
        ),
        "k1_reproduces_r0255_n29_bitwise": bool(
            calibration_receipt["k1_n29"]["reproduces_bitwise"]
        ),
        "k1_is_below_the_identity_bound_at_n29": bool(
            calibration_receipt["k1_below_identity_bound_at_n29"]
        ),
        "the_provisional_bands_reproduce_the_published_values": bool(
            bands_reproduce_published
        ),
        "the_collapse_floor_separates_the_two_modes": bool(
            separation["the_two_modes_do_not_overlap"]
            and separation["floor_is_above_the_legacy_mode"]
            and separation["floor_is_below_the_healthy_mode"]
        ),
        "the_registry_gates_collapse_and_fog_only": bool(registry_guard["holds"]),
        "density_v3_is_absent_from_the_registry": DENSITY_V3_METRIC not in criteria,
        "every_planted_registry_defect_is_refused": bool(
            registry_plants["every_planted_defect_is_refused"]
        ),
        "the_fog_gate_fails_closed_on_a_planted_degenerate_map": bool(
            fog_fail_closed["holds"]
        ),
        "the_fail_closed_guard_changed_the_outcome": bool(
            fog_fail_closed["the_guard_changed_the_outcome"]
        ),
        "the_collapse_floor_control_holds": bool(collapse_controls["holds"]),
        "the_panel_rate_is_above_the_per_metric_rate": (
            one_sided["panel_false_alarm_rate_under_independence"]
            > one_sided["one_sided_new_cell_false_fail_rate"]
        ),
        "no_cuda_library_is_mapped_into_this_process": not cuda_mappings,
    }
    if not all(execution_checks.values()):
        raise Round0264NodeError(f"R0264 registration checks failed: {execution_checks}")

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
        "n_adjustment_rule_resolved_note": (
            "N-adjustment rule filled from P1's dose x N decomposition (GO verdict "
            f"2026-08-14); rule id {N_RULE}. See n_adjustment_rule / n_adjustment_rule_note."
        ),
        "estimator": ESTIMATOR,
        "provisional_status": PROVISIONAL_STATUS,
        # --- A ---
        "calibration": calibration_receipt,
        # --- B ---
        "provisional_bands": bands,
        "bands_reproduce_the_published_values": bands_reproduce_published,
        "separation_record": separation,
        "measurements_cross_check": measurements_cross_check,
        "healthy_family": [dict(arm) for arm in HEALTHY_FAMILY],
        "cuml_reference": dict(CUML_REFERENCE),
        "legacy_fingerprint": dict(LEGACY_FINGERPRINT),
        "the_legacy_family_is_a_fingerprint_only": True,
        # --- C ---
        "one_sided_report": one_sided,
        "identity_bound_at_n29": identity_bound(N_EXACT),
        "identity_bound_at_n7": identity_bound(7),
        "tolerance_content": TOLERANCE_CONTENT,
        "tolerance_confidence": TOLERANCE_CONFIDENCE,
        # --- D ---
        "registered_criteria": criteria,
        "descriptive_only_never_a_gate": descriptive,
        "registry_guard": registry_guard,
        "registry_controls": registry_plants,
        "fog_fail_closed_control": fog_fail_closed,
        "collapse_floor_control": collapse_controls,
        "gated_metrics": list(GATED_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        # --- N-rule placeholder (the one deliberate hole) ---
        "n_adjustment_rule": N_RULE_PENDING,
        "n_adjustment_rule_note": N_RULE_NOTE,
        "gate_status": "registered-draft-pending-p1-decomposition-and-review",
        "gate_registered": True,
        "execution_checks": execution_checks,
        "evaluation_performed": False,
        "training_performed": False,
        "gpu_used": bool(cuda_mappings),
        "cuda_libraries_mapped_into_this_process": cuda_mappings,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "sources": {
            "r0234_n13_calibration": expected_input_signature(
                _bound_path(job, "r0234_n13", label="R0234 n=13 calibration")
            ),
            "r0255_n29_calibration": expected_input_signature(
                _bound_path(job, "r0255_n29", label="R0255 n=29 calibration")
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
        "collapse_floor": collapse_floor,
        "fog_ceiling": fog_ceiling,
        "fog_fails_closed": True,
        "density_v3": DESCRIPTIVE_MARKING,
        "n_adjustment_rule": N_RULE_PENDING,
        "separation_factor": separation["separation_factor"],
        "widest_gap_s": gaps.get("widest_gap_s"),
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


# --------------------------------------------------------------------------- #
# node 2 — the separation record
# --------------------------------------------------------------------------- #


def _read_measurement_families(path: str) -> dict[str, Any]:
    """The legacy 29 cells, the healthy core arms, cuML and md035 from the sealed
    measurements JSON, keyed for the separation record."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    legacy = [
        {"seed": int(c["seed"]), "collapse": float(c["collapse"]), "fog": float(c["fog"]),
         "resolution_levels": int(c["fog_resolution_levels"]),
         "degenerate": bool(c["fog_degenerate"])}
        for c in data.get("family", [])
    ]
    sandbox = {(a["arm"], a.get("rung")): a for a in data.get("sandbox", [])}
    return {"legacy": legacy, "sandbox": sandbox}


def run_separation(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0264 round0264_nodes.run_separation")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0264NodeError("R0264 separation handler received another queue")
    node_id = str(active.get("node_id") or SEPARATION_ACTION)
    label = "R0264 legacy-vs-healthy collapse/fog separation record (DRAFT)"
    ledger = CoverageLedger(node=node_id)
    abort_flag = _start_node(label)
    started = time.monotonic()
    started_utc = time.time()
    output = create_fresh_directory(str(job["outputs"][0]), label="R0264 separation")

    window = ledger.window("R0264 separation stage")
    guard_ctx = _node_guard(label)
    gate = _node_gate(label, training_performed=False)
    with guard_ctx:
        gate.start()
        recorder = PollRecorder(gate=gate, clock=time.monotonic)
        recorder.anchor("R0264 separation stage entered")
        wrapped = window.wrap(recorder)

        registration_path, registration_signature = _intra_queue_signature(
            job["registration"], label="R0264 sealed registration"
        )
        registration = prompt_contract.read_sealed(
            registration_path, label="R0264 sealed registration"
        )
        if (
            registration.get("capability") != GATE_CAPABILITY
            or registration.get("schema") != GATE_SCHEMA
            or registration.get("gate_registered") is not True
        ):
            raise Round0264NodeError("R0264 registration artifact contract changed")

        marker_path = str(job["registration_done_marker"])
        if not os.path.exists(marker_path):
            raise Round0264NodeError(
                f"R0264 registration done-marker is absent at {marker_path}"
            )
        with open(marker_path, "rb") as handle:
            marker = json.load(handle)
        sealed_at = os.path.getmtime(registration_path)
        ordering = {
            "registration_artifact": registration_path,
            "registration_sha256": registration_signature["sha256"],
            "registration_done_marker": marker_path,
            "registration_node": str(marker.get("node") or marker.get("id") or ""),
            "registration_sealed_at_epoch_s": sealed_at,
            "separation_started_at_epoch_s": started_utc,
            "registration_precedes_separation": sealed_at < started_utc,
            "why_this_is_receipt_evidence": (
                "the runner runs this node only after the producer's done-marker exists, "
                "and the registration bytes are hashed here. Order is a property of the "
                "queue's receipts, not of prose."
            ),
        }
        if not ordering["registration_precedes_separation"]:
            raise Round0264NodeError(
                "R0264 refuses to compile the record: the registration was not sealed first"
            )
        wrapped("R0264 registration ordering verified from receipts")

        collapse_floor = float(registration["registered_criteria"][COLLAPSE_METRIC]["floor"])
        fog_ceiling = float(registration["registered_criteria"][FOG_METRIC]["ceiling"])

        families = _read_measurement_families(
            _bound_path(job, "collapse_fog_measurements", label="collapse/fog measurements")
        )

        # Legacy fingerprint: every cell FAILS the collapse floor (the whole point).
        legacy_rows = []
        for cell in families["legacy"]:
            collapse_v = _judge_collapse(cell["collapse"], collapse_floor)
            fog_result = {
                "fog": cell["fog"], "resolution_levels": cell["resolution_levels"],
                "degenerate": cell["degenerate"],
            }
            fog_v = _judge_fog(fog_result, fog_ceiling)
            legacy_rows.append({
                "seed": cell["seed"], "collapse": cell["collapse"],
                "collapse_verdict": collapse_v["verdict"], "fog": cell["fog"],
                "fog_verdict": fog_v["verdict"],
            })
        every_legacy_fails_collapse = all(
            row["collapse_verdict"] == "FAIL" for row in legacy_rows
        )

        # Healthy family + cuML: PASS both.
        healthy_rows = []
        for arm in HEALTHY_FAMILY:
            row = families["sandbox"].get((arm["arm"], arm["rung"]), arm)
            collapse_v = _judge_collapse(row["collapse"], collapse_floor)
            fog_result = {
                "fog": row["fog"],
                "resolution_levels": int(row.get("fog_resolution_levels", arm["resolution_levels"])),
                "degenerate": bool(row.get("fog_degenerate", False)),
            }
            fog_v = _judge_fog(fog_result, fog_ceiling)
            healthy_rows.append({
                "arm": f"{arm['arm']}@{arm['rung']}", "collapse": float(row["collapse"]),
                "collapse_verdict": collapse_v["verdict"], "fog": float(row["fog"]),
                "fog_verdict": fog_v["verdict"],
            })
        every_healthy_passes = all(
            row["collapse_verdict"] == "PASS" and row["fog_verdict"] == "PASS"
            for row in healthy_rows
        )
        cuml_collapse = _judge_collapse(CUML_REFERENCE["collapse"], collapse_floor)
        cuml_fog = _judge_fog(
            {"fog": CUML_REFERENCE["fog"], "resolution_levels": CUML_REFERENCE["resolution_levels"],
             "degenerate": CUML_REFERENCE["degenerate"]},
            fog_ceiling,
        )
        cuml_passes = cuml_collapse["verdict"] == "PASS" and cuml_fog["verdict"] == "PASS"

        # md035: fail-closed → NOT_MEASURABLE, re-asserted here.
        md035_fog = _judge_fog(
            {"fog": SEALED_MD035_DEGENERATE["fog"],
             "resolution_levels": SEALED_MD035_DEGENERATE["resolution_levels"],
             "degenerate": SEALED_MD035_DEGENERATE["degenerate"]},
            fog_ceiling,
        )
        md035_refused = (
            md035_fog["verdict"] == VERDICT_NOT_MEASURABLE and md035_fog["passes"] is False
        )

        separation = _separation_record(collapse_floor=collapse_floor)
        wrapped("R0264 legacy / healthy / cuML / md035 applied under the registered bands")
        gate.finish("R0264 separation stage end")
    window.close()
    tail = _guard_tail_reported(guard_ctx, label=label)
    scored_gate = _score_gate_without_raising(gate, tail, label=label)
    gaps = gap_report(recorder.records, arm=node_id)
    cuda_mappings = _cuda_mappings()

    execution_checks = {
        "the_registration_was_sealed_before_this_node_started": bool(
            ordering["registration_precedes_separation"]
        ),
        "every_legacy_cell_fails_the_collapse_floor": bool(every_legacy_fails_collapse),
        "the_legacy_family_has_29_cells": len(legacy_rows) == LEGACY_FINGERPRINT["n"],
        "every_healthy_arm_passes_both_bands": bool(every_healthy_passes),
        "the_cuml_reference_passes_both_bands": bool(cuml_passes),
        "the_degenerate_md035_map_is_refused_not_passed": bool(md035_refused),
        "the_two_modes_do_not_overlap": bool(separation["the_two_modes_do_not_overlap"]),
        "no_cuda_library_is_mapped_into_this_process": not cuda_mappings,
    }
    if not all(execution_checks.values()):
        raise Round0264NodeError(f"R0264 separation checks failed: {execution_checks}")

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
        "schema": SEPARATION_SCHEMA,
        "capability": SEPARATION_CAPABILITY,
        "capabilities": [SEPARATION_CAPABILITY],
        "node": node_id,
        "node_started_utc": started_utc,
        "abort_flag_precondition": abort_flag,
        "what_this_is": (
            "a dated separation record applying the collapse floor and fail-closed fog "
            "ceiling this round's first node registered to the legacy fingerprint, the "
            "healthy family, cuML and the degenerate md035 arm. It registers nothing new"
        ),
        "registration_ordering": ordering,
        "registration_consumed": {
            "capability": registration["capability"],
            "schema": registration["schema"],
            "sha256": registration_signature["sha256"],
            "collapse_floor": collapse_floor,
            "fog_ceiling": fog_ceiling,
        },
        "legacy_fingerprint_under_the_floor": {
            "cells": legacy_rows,
            "every_cell_fails_the_collapse_floor": every_legacy_fails_collapse,
        },
        "healthy_family_under_the_bands": {
            "arms": healthy_rows,
            "every_arm_passes_both_bands": every_healthy_passes,
        },
        "cuml_reference_under_the_bands": {
            "collapse_verdict": cuml_collapse["verdict"],
            "fog_verdict": cuml_fog["verdict"],
            "passes_both": cuml_passes,
        },
        "md035_under_the_fail_closed_fog_gate": {
            "verdict": md035_fog["verdict"],
            "refused_not_passed": md035_refused,
            "note": "peak bin 99, resolution_levels 0: never a pass",
        },
        "separation_record": separation,
        "gate_status": "no-gate-registered-here-the-registered-bands-were-consumed-unchanged",
        "gate_registered": False,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gpu_used": bool(cuda_mappings),
        "cuda_libraries_mapped_into_this_process": cuda_mappings,
        "production_or_publishing": False,
        "upstream_review_state": dict(job["upstream_review_state"]),
        "sources": {
            "collapse_fog_measurements": expected_input_signature(
                _bound_path(job, "collapse_fog_measurements", label="collapse/fog measurements")
            ),
            "r0264_registration": registration_signature,
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
    _seal(output, SEPARATION_ARTIFACT, body)
    print(json.dumps({
        "capability": SEPARATION_CAPABILITY,
        "every_legacy_cell_fails_collapse": every_legacy_fails_collapse,
        "every_healthy_arm_passes": every_healthy_passes,
        "cuml_passes_both": cuml_passes,
        "md035_refused": md035_refused,
        "separation_factor": separation["separation_factor"],
        "widest_gap_s": gaps.get("widest_gap_s"),
        "observed_span_s": coverage["observed_span_s"],
        "covered_fraction": coverage["covered_fraction"],
    }))


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    install_stop_hooks(label="R0264 round0264_nodes.run_job")
    action = str(job.get("action") or "")
    if action == REGISTER_ACTION:
        run_register(active, job)
        return
    if action == SEPARATION_ACTION:
        run_separation(active, job)
        return
    raise Round0264NodeError(f"R0264 unknown action {action!r}")


__all__ = [
    "GATE_CAPABILITY",
    "GATE_SCHEMA",
    "REGISTER_ACTION",
    "REGISTRATION_ARTIFACT",
    "ROUND_ID",
    "Round0264NodeError",
    "SEPARATION_ACTION",
    "SEPARATION_ARTIFACT",
    "SEPARATION_CAPABILITY",
    "SEPARATION_SCHEMA",
    "run_job",
    "run_register",
    "run_separation",
]
