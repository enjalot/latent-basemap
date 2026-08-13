"""Restored TWO-SIDED ``purity_fidelity_k256`` criterion (Metric Option B).

PROPOSAL ARTIFACT -- NOT A REGISTRATION.  Nothing in this module writes a round
file, a gate artifact or a capability.  It re-implements, on the CPU, the
criterion that R0255 registered and that the owner's 2026-08-12 ruling
(``OWNER-DECISIONS-PENDING.md`` §5) retired from gating, so that a would-be
verdict can be computed for every map with a published ``k256`` value.

What the criterion is
---------------------
``purity_fidelity_k256`` = ``map_agreement / hi_D_agreement`` at ``k = 256``
centroid labels.  Ideal is **1.0**: the map's 2D neighbourhoods agree with the
``k = 256`` cluster labelling exactly as often as the high-D reference's do.

* ratio **> 1** -> the map is PURER than reality: manufactured separation.
  This is the direction the 2026-08-12/13 sandbox work identifies as the
  bead-collapse signature.
* ratio **< 1** -> the map is less pure than reality: lost structure.

The band
--------
Two-sided, on the **UNFOLDED** ``log`` ratio.  Never fold a two-sided ratio
about 1.0 -- review-0222-01 (§ "(b) The *calibration* is not right") showed the
fold reflects about ``r = 1``, which is not the family centre
(``r_geo = 1.0115`` at n = 29), manufactures ``|z|`` on one side and roughly
triples the defining-cell failure rate.  R0255's own artifact publishes the
purity entries of ``registered_floors`` as ``null`` for exactly this reason:

    "the gated purity criterion is the two-sided band on the UNFOLDED ratio
     below.  A consumer reading a folded one-sided purity floor would fail
     cells this gate passes -- the trap review-0231-01 found in R0231."

    centre     = median( log r_i )                    over the family
    scale      = MAD_n( log r_i ) = 1.4826 * median| log r_i - centre |
    band       = exp( centre +/- k2 * scale )

``k2`` is the R0234 Gaussian-null calibrated TWO-SIDED multiplier for
``median +/- k * MAD_n`` at 95 % content / 95 % confidence, recalibrated at
``n = 29``.

Provenance of every carried number is in ``SEALED`` below.  Sealed artifacts are
read only; this module never writes to them.

CPU only.  Run everything with ``CUDA_VISIBLE_DEVICES=""``.

CLI
---
    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/k256_two_sided.py calibrate
    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/metrics/k256_two_sided.py verdicts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.special import ndtr

# --------------------------------------------------------------------------
# Carried numbers.  Every one of these is read from, or reproduced bitwise
# against, a sealed artifact.  Paths are `gsv:` local paths, read-only.
# --------------------------------------------------------------------------

SEALED = {
    "panel_n29": {
        "path": (
            "/data/latent-basemap/runs/round-0255/queue/artifacts/"
            "minilm-mixed-2m-seed-family-panel-n29-v1/seed-family-panel-n29.json"
        ),
        "sha256": "76590920ff212113d037676ac7dbd2331d6d3a366228d031607d95e58f88bead",
        "supplies": "raw_purity_ratios for the 29 sealed defining cells",
    },
    "floors_n29": {
        "path": (
            "/data/latent-basemap/runs/round-0255/queue/artifacts/"
            "minilm-mixed-2m-calibrated-madn-floors-n29-v1/"
            "minilm-calibrated-madn-floors-n29.json"
        ),
        "supplies": (
            "registered_two_sided_bands, descriptive_folded_purity_floors, "
            "calibration.n29 multipliers, scoring_by_candidate for the 41 cells"
        ),
    },
    "floors_n13": {
        "path": (
            "/data/latent-basemap/runs/round-0234/queue/artifacts/"
            "minilm-mixed-2m-calibrated-robust-floors-n13-v1/"
            "minilm-calibrated-robust-floors-n13.json"
        ),
        "sha256": "999181c5966b1d8987e12d53fd506d53af2914bfaaae39a843434b499cb6129a",
        "supplies": "calibration.n13.candidates.median_minus_k_madn (k1, k2 at n = 13)",
    },
    "one_sided_n29": {
        "path": (
            "/data/latent-basemap/runs/round-0260/queue/artifacts/"
            "minilm-mixed-2m-one-sided-purity-floors-n29-v1/"
            "minilm-one-sided-purity-floors-n29.json"
        ),
        "supplies": "registered_ratio_floors (the criterion currently in force)",
    },
    "readjudication_0257": {
        "path": (
            "/data/latent-basemap/runs/round-0260/queue/artifacts/"
            "minilm-mixed-6250k-rung-readjudication-one-sided-v1/"
            "readjudication-0257-one-sided.json"
        ),
        "supplies": "the three 6.25M rung maps' raw k256 ratios and both verdicts",
    },
    "verdict_0257": {
        "path": (
            "/data/latent-basemap/runs/round-0257/queue-correction-2/artifacts/"
            "minilm-mixed-6250k-rung-gate-verdict-v1/judge_6250k-rung-verdict.json"
        ),
        "supplies": "R0257's published FAILs and the folded k256 values",
    },
}

# R0234, calibration.n13.candidates.median_minus_k_madn -- the validation target.
R0234_N13_K1 = 3.7363661743730416
R0234_N13_K2 = 4.452408030160313  # published in result-0234 as `4.45241`
R0234_CALIBRATION_SEED = 20260809
R0234_FAMILIES = 4_000_000

# R0255, calibration.n29.candidates.median_minus_k_madn -- the sealed n = 29
# multipliers.  R0260's registration node re-derived both bitwise.
R0255_N29_K1 = 2.6934393367626024
R0255_N29_K2 = 3.147763220676551

# R0255, registered_two_sided_bands.purity_fidelity_k256 -- the sealed band.
R0255_K256_BAND = {
    "log_centre": 0.01143437762566317,
    "log_scale": 0.01339863133626119,
    "log_lower": -0.03074134130202412,
    "log_upper": 0.05361009655335046,
    "ratio_geometric_centre": 1.0115,
    "ratio_lower": 0.9697263687993266,
    "ratio_upper": 1.0550731452902518,
}

# R0255, descriptive_folded_purity_floors.purity_fidelity_k256.  This is the
# FOLDED one-sided floor, published descriptive-only.  It is fitted on the raw
# folded scale: median(fold) - k1 * MAD_n(fold).
R0255_K256_FOLDED_FLOOR = 0.962048853686814

# R0260, registered_ratio_floors.purity_fidelity_k256 -- the criterion in force.
R0260_K256_ONE_SIDED_FLOOR = 0.9756474051324426

# panel_v2 rounds each purity ratio to four decimals inside the scorer, so any
# band fitted to these ratios inherits +/- 5e-5 in r.
PANEL_QUANTUM = 5e-5

MADN_CONSTANT = 1.4826  # R0234: "MAD_n = 1.4826 * median|x - median|"

# The 29 sealed defining cells, embedded so the criterion is reproducible
# without /data.  ``load_sealed_family`` verifies these against the artifact
# whenever the artifact is present.
SEALED_FAMILY_SEEDS: List[int] = list(range(42, 71))
SEALED_FAMILY_K256: List[float] = [
    1.0216, 1.0059, 1.0046, 0.9929, 1.0049, 0.9932, 1.0370, 1.0099, 1.0120,
    1.0024, 1.0055, 1.0065, 1.0115, 1.0293, 1.0232, 1.0259, 1.0313, 1.0417,
    1.0171, 1.0187, 0.9988, 0.9885, 1.0152, 1.0208, 1.0096, 1.0180, 0.9979,
    1.0239, 1.0063,
]

# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------


def madn(values: Sequence[float] | np.ndarray, axis: int = -1) -> np.ndarray | float:
    """``MAD_n = 1.4826 * median|x - median(x)|`` (R0234's registered scale)."""
    x = np.asarray(values, dtype=np.float64)
    centre = np.median(x, axis=axis, keepdims=True)
    out = MADN_CONSTANT * np.median(np.abs(x - centre), axis=axis)
    return float(out) if np.ndim(out) == 0 else out


def fold(ratio: float | np.ndarray) -> float | np.ndarray:
    """The folded 'fidelity' view, ``min(r, 1/r)``.

    Present ONLY so the folding regression can be written down.  It must never
    be used to gate a two-sided ratio: review-0222-01.
    """
    r = np.asarray(ratio, dtype=np.float64)
    out = np.minimum(r, 1.0 / r)
    return float(out) if np.ndim(out) == 0 else out


# --------------------------------------------------------------------------
# Gaussian-null calibration (the R0234 method)
# --------------------------------------------------------------------------


def calibrate_multipliers(
    n: int,
    families: int = R0234_FAMILIES,
    seed: int = R0234_CALIBRATION_SEED,
    content: float = 0.95,
    confidence: float = 0.95,
    chunk: int = 250_000,
    bisect_iters: int = 80,
) -> Dict[str, float]:
    """Calibrate ``median +/- k * MAD_n`` on a Gaussian null at size ``n``.

    R0234 §A, verbatim in method:

      * one-sided --  ``P(centre - k*scale <= mu - z_P*sigma) = gamma`` rearranges
        to a quantile, so ``k1`` is the ``gamma``-quantile of ``(centre + z_P)/scale``.
      * two-sided --  the standard content definition ``Phi(U) - Phi(L) >= P``.
        Per family solve for the minimal ``k`` with
        ``Phi(c + k*s) - Phi(c - k*s) >= content``; ``k2`` is the
        ``gamma``-quantile of those minimal ``k``.

    Reads no cell of this program: it is a property of the estimator at ``n``
    under a clean Gaussian null.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if families < 1:
        raise ValueError("families must be >= 1")

    z_content = float(-math.sqrt(2.0) * _erfcinv(2.0 * content))
    rng = np.random.default_rng(seed)

    k1_parts: List[np.ndarray] = []
    k2_parts: List[np.ndarray] = []
    drawn = 0
    while drawn < families:
        m = min(chunk, families - drawn)
        drawn += m
        x = rng.standard_normal((m, n))
        c = np.median(x, axis=1)
        s = madn(x, axis=1)
        if np.any(s <= 0):
            # MAD_n = 0 is possible in principle for a discrete draw; it cannot
            # happen for continuous normals, but refuse rather than divide.
            raise RuntimeError("a simulated family had MAD_n == 0")

        k1_parts.append((c + z_content) / s)

        lo = np.zeros(m)
        hi = (np.abs(c) + 10.0) / s  # content at `hi` is >= Phi(10) - Phi(-10)
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            covered = (ndtr(c + mid * s) - ndtr(c - mid * s)) >= content
            hi = np.where(covered, mid, hi)
            lo = np.where(covered, lo, mid)
        k2_parts.append(0.5 * (lo + hi))

    k1 = np.concatenate(k1_parts)
    k2 = np.concatenate(k2_parts)
    return {
        "n": n,
        "families": families,
        "seed": seed,
        "content": content,
        "confidence": confidence,
        "z_content": z_content,
        "one_sided_multiplier": float(np.quantile(k1, confidence)),
        "two_sided_multiplier": float(np.quantile(k2, confidence)),
    }


def _erfcinv(y: float) -> float:
    from scipy.special import erfcinv as _e

    return float(_e(y))


def calibrate_k2(n: int, **kwargs) -> float:
    """Just the two-sided multiplier."""
    return calibrate_multipliers(n, **kwargs)["two_sided_multiplier"]


# Published multipliers, used as the default so ``fit_band`` is cheap and
# deterministic.  ``fit_band(..., k2=calibrate_k2(29))`` uses a fresh simulation.
PUBLISHED_K2 = {13: R0234_N13_K2, 29: R0255_N29_K2}


def k2_for_n(n: int, families: int = R0234_FAMILIES, seed: int = R0234_CALIBRATION_SEED) -> float:
    """Published calibrated ``k2`` at ``n`` if there is one, else simulate."""
    if n in PUBLISHED_K2:
        return PUBLISHED_K2[n]
    return calibrate_k2(n, families=families, seed=seed)


# --------------------------------------------------------------------------
# The criterion
# --------------------------------------------------------------------------


def fit_band(family_ratios: Iterable[float], k2: Optional[float] = None) -> Dict[str, object]:
    """Fit the two-sided band on the UNFOLDED log ratio.

    Returns ``center`` / ``lower`` / ``upper`` on the RATIO scale (what a
    consumer compares an observation against), plus the log-scale internals.
    """
    r = np.asarray(list(family_ratios), dtype=np.float64)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("family_ratios must be a 1-D family of at least 2 ratios")
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("every purity ratio must be finite and strictly positive")

    n = int(r.size)
    if k2 is None:
        k2 = k2_for_n(n)
    k2 = float(k2)
    if k2 <= 0:
        raise ValueError("k2 must be positive")

    log_r = np.log(r)
    log_centre = float(np.median(log_r))
    log_scale = float(madn(log_r))
    if log_scale <= 0:
        raise ValueError("MAD_n of the log-ratios is zero: the family is degenerate")

    log_lower = log_centre - k2 * log_scale
    log_upper = log_centre + k2 * log_scale
    return {
        "center": float(math.exp(log_centre)),
        "lower": float(math.exp(log_lower)),
        "upper": float(math.exp(log_upper)),
        "k2": k2,
        "n": n,
        "log_center": log_centre,
        "log_scale": log_scale,
        "log_lower": log_lower,
        "log_upper": log_upper,
        "estimator": "median_minus_k_madn",
        "sided": "two_sided",
        "scale": "unfolded log ratio",
        "ideal_ratio": 1.0,
        "quantisation_note": (
            "panel_v2 rounds each purity ratio to four decimals inside the "
            "scorer, so this band inherits +/- 5e-5 in r"
        ),
    }


def judge(ratio: float, band: Dict[str, object]) -> Dict[str, object]:
    """Judge one observed ratio against a fitted two-sided band.

    ``direction`` is the side of the band the observation sits on:
    ``"over"`` (above the upper edge -- the map is purer than the family, the
    manufactured-separation / bead-collapse direction), ``"under"`` (below the
    lower edge -- lost structure) or ``"inside"``.

    ``separation`` is the sealed-artifact convention: where the observation
    sits relative to the perfectly faithful ratio 1.0, independent of pass/fail.
    """
    r = float(ratio)
    if not math.isfinite(r) or r <= 0:
        raise ValueError("the purity ratio must be finite and strictly positive")

    lower = float(band["lower"])
    upper = float(band["upper"])
    log_r = math.log(r)
    log_centre = float(band["log_center"])
    log_scale = float(band["log_scale"])

    if r > upper:
        direction = "over"
    elif r < lower:
        direction = "under"
    else:
        direction = "inside"

    passes = direction == "inside"
    if direction == "over":
        margin = upper - r  # negative
    elif direction == "under":
        margin = r - lower  # negative
    else:
        margin = min(r - lower, upper - r)  # non-negative distance to the nearer edge

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
            "signed distance to the breached edge on the ratio scale; "
            "negative means outside the band"
        ),
        "separation": separation,
        "ratio_lower": lower,
        "ratio_upper": upper,
        "decided_on": "raw unfolded purity ratio, two-sided band",
        "band_direction": {
            "over": "above_band",
            "under": "below_band",
            "inside": "inside_band",
        }[direction],
    }


def judge_one_sided(ratio: float, floor: float = R0260_K256_ONE_SIDED_FLOOR) -> Dict[str, object]:
    """The criterion currently in force (R0260): a lower ratio floor only."""
    r = float(ratio)
    passes = r >= floor
    return {
        "verdict": "PASS" if passes else "FAIL",
        "direction": "under" if not passes else "inside",
        "passes": passes,
        "raw_ratio": r,
        "margin": r - floor,
        "criterion": f">= {floor!r}",
        "kind": "one_sided_lower_ratio_floor",
        "fails_only": "under_separation",
    }


def judge_folded_floor(ratio: float, floor: float = R0255_K256_FOLDED_FLOOR) -> Dict[str, object]:
    """The FOLDED descriptive floor.  Never a gate -- present for the regression."""
    f = float(fold(ratio))
    passes = f >= floor
    return {
        "verdict": "PASS" if passes else "FAIL",
        "raw_ratio": float(ratio),
        "folded_value": f,
        "floor": floor,
        "margin": f - floor,
        "kind": "folded_one_sided_floor_DESCRIPTIVE_ONLY",
    }


# --------------------------------------------------------------------------
# Sealed inputs
# --------------------------------------------------------------------------


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sealed_family() -> Dict[str, object]:
    """The 29 sealed ``k256`` ratios, from the artifact when it is reachable.

    When ``/data`` is present the embedded copy is checked against the artifact
    and the artifact's sha256 against ``SEALED``; a mismatch raises.
    """
    embedded = dict(zip(SEALED_FAMILY_SEEDS, SEALED_FAMILY_K256))
    path = SEALED["panel_n29"]["path"]
    source = "embedded copy (artifact not reachable)"
    if os.path.exists(path):
        digest = _sha256(path)
        if digest != SEALED["panel_n29"]["sha256"]:
            raise RuntimeError(
                f"sealed panel sha256 mismatch: {digest} != {SEALED['panel_n29']['sha256']}"
            )
        with open(path) as fh:
            panel = json.load(fh)
        from_artifact = {
            int(s): float(panel["raw_purity_ratios"][str(s)]["k256"])
            for s in panel["seed_order"]
        }
        if from_artifact != embedded:
            raise RuntimeError("embedded sealed family disagrees with the sealed artifact")
        source = path
    return {"seeds": SEALED_FAMILY_SEEDS, "ratios": SEALED_FAMILY_K256, "source": source}


def load_published_cells() -> List[Dict[str, object]]:
    """Every map with a published ``k256`` ratio, read-only.

    41 cells from R0255's pooled scoring (29 sealed defining + 12 held-out),
    the R0255 replay control, and the three 6.25M rung maps R0257 judged and
    R0260 re-adjudicated.
    """
    cells: List[Dict[str, object]] = []

    floors_path = SEALED["floors_n29"]["path"]
    with open(floors_path) as fh:
        floors = json.load(fh)
    scored = floors["scoring_by_candidate"]["median_minus_k_madn"]["all_cells"]["cells"]
    for cell in scored:
        m = cell["metrics"]["purity_fidelity_k256"]
        cells.append(
            {
                "map": cell["cell_id"],
                "family": cell["family"],
                "N": 2_000_000,
                "defines_the_band": bool(cell["defines_the_floors"]),
                "ratio": float(m["raw_ratio"]),
                "source": floors_path,
                "source_field": (
                    "scoring_by_candidate.median_minus_k_madn.all_cells"
                    ".cells[].metrics.purity_fidelity_k256.raw_ratio"
                ),
            }
        )

    panel_path = SEALED["panel_n29"]["path"]
    with open(panel_path) as fh:
        panel = json.load(fh)
    cells.append(
        {
            "map": "replay-control-seed42",
            "family": "replay-control (not a family cell)",
            "N": 2_000_000,
            "defines_the_band": False,
            "ratio": float(panel["replay_control_cell"]["raw_purity_ratios"]["k256"]),
            "source": panel_path,
            "source_field": "replay_control_cell.raw_purity_ratios.k256",
        }
    )

    readj_path = SEALED["readjudication_0257"]["path"]
    with open(readj_path) as fh:
        readj = json.load(fh)
    for cell_id, entry in readj["re_adjudication"].items():
        m = entry["per_metric"]["purity_fidelity_k256"]
        cells.append(
            {
                "map": cell_id,
                "family": "ladder-6250k-h2048 (judged rung map)",
                "N": 6_250_000,
                "defines_the_band": False,
                "ratio": float(m["observed"]),
                "r0257_verdict": m["r0257_verdict"],
                "r0260_verdict": m["one_sided_verdict"],
                "source": readj_path,
                "source_field": "re_adjudication[].per_metric.purity_fidelity_k256.observed",
            }
        )
    return cells


# --------------------------------------------------------------------------
# Would-be verdicts
# --------------------------------------------------------------------------


def would_be_verdicts(k2: Optional[float] = None) -> Dict[str, object]:
    """The full table: one-sided (in force) vs restored two-sided, per map."""
    family = load_sealed_family()
    band = fit_band(family["ratios"], k2=k2)
    cells = load_published_cells()

    rows: List[Dict[str, object]] = []
    for cell in cells:
        two = judge(cell["ratio"], band)
        one = judge_one_sided(cell["ratio"])
        rows.append(
            {
                "map": cell["map"],
                "family": cell["family"],
                "N": cell["N"],
                "defines_the_band": cell["defines_the_band"],
                "ratio": cell["ratio"],
                "one_sided_verdict": one["verdict"],
                "one_sided_margin": one["margin"],
                "two_sided_verdict": two["verdict"],
                "two_sided_margin": two["margin"],
                "direction": two["direction"],
                "separation": two["separation"],
                "z_madn": two["z_madn"],
                "verdicts_disagree": one["verdict"] != two["verdict"],
                "source": cell["source"],
                "source_field": cell["source_field"],
            }
        )

    flips = [r for r in rows if r["verdicts_disagree"]]
    return {
        "what_this_is": (
            "a PROPOSAL: would-be verdicts under the restored two-sided "
            "purity_fidelity_k256 criterion. It registers nothing, supersedes "
            "nothing, and no round file was touched to produce it."
        ),
        "band": band,
        "band_reproduces_r0255_sealed_band": {
            "sealed": R0255_K256_BAND,
            "delta_ratio_lower": band["lower"] - R0255_K256_BAND["ratio_lower"],
            "delta_ratio_upper": band["upper"] - R0255_K256_BAND["ratio_upper"],
        },
        "one_sided_criterion_in_force": {
            "ratio_floor": R0260_K256_ONE_SIDED_FLOOR,
            "registered_by": "R0260, minilm-mixed-2m-one-sided-purity-floors-n29-v1",
            "path": SEALED["one_sided_n29"]["path"],
        },
        "sealed_family_source": family["source"],
        "rows": rows,
        "flips": flips,
        "flip_count": len(flips),
        "maps_scored": len(rows),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))


def _cmd_calibrate(args: argparse.Namespace) -> int:
    out: Dict[str, object] = {
        "method": "R0234 Gaussian null, median +/- k * MAD_n, 95/95",
        "validation_target_n13_k2": R0234_N13_K2,
        "runs": {},
    }
    for n in args.n:
        res = calibrate_multipliers(n, families=args.families, seed=args.seed)
        res["published_two_sided_multiplier"] = PUBLISHED_K2.get(n)
        if n in PUBLISHED_K2:
            res["relative_error_vs_published"] = (
                res["two_sided_multiplier"] / PUBLISHED_K2[n] - 1.0
            )
        out["runs"][str(n)] = res
        print(
            f"n={n:>3} families={args.families:,} "
            f"k1={res['one_sided_multiplier']:.9f} k2={res['two_sided_multiplier']:.9f}"
            + (
                f"  (published k2={PUBLISHED_K2[n]:.9f}, "
                f"rel.err {res['relative_error_vs_published']:+.4%})"
                if n in PUBLISHED_K2
                else ""
            )
        )
    path = args.out or os.path.join(_HERE, "k256-two-sided-calibration.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"wrote {path}")
    return 0


def _cmd_verdicts(args: argparse.Namespace) -> int:
    k2 = None
    if args.recalibrate:
        k2 = calibrate_k2(29, families=args.families, seed=args.seed)
        print(f"recalibrated k2 at n=29: {k2:.9f}")
    table = would_be_verdicts(k2=k2)
    band = table["band"]
    print(
        f"band  n={band['n']}  k2={band['k2']:.9f}  "
        f"[{band['lower']:.16f}, {band['upper']:.16f}]  centre={band['center']:.6f}"
    )
    header = f"{'map':<28} {'N':>9} {'ratio':>8} {'one-sided':>10} {'two-sided':>10} {'direction':>9}"
    print(header)
    print("-" * len(header))
    for row in table["rows"]:
        mark = "  <-- FLIP" if row["verdicts_disagree"] else ""
        print(
            f"{row['map']:<28} {row['N']:>9,} {row['ratio']:>8.4f} "
            f"{row['one_sided_verdict']:>10} {row['two_sided_verdict']:>10} "
            f"{row['direction']:>9}{mark}"
        )
    print(f"\nflips: {table['flip_count']} of {table['maps_scored']}")
    path = args.out or os.path.join(_HERE, "k256-two-sided-would-be-verdicts.json")
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    print(f"wrote {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", None):
        print(
            "warning: this is a CPU-only criterion; run with CUDA_VISIBLE_DEVICES=''",
            file=sys.stderr,
        )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="recalibrate k1/k2 on a Gaussian null")
    c.add_argument("--n", type=int, nargs="+", default=[13, 29])
    c.add_argument("--families", type=int, default=R0234_FAMILIES)
    c.add_argument("--seed", type=int, default=R0234_CALIBRATION_SEED)
    c.add_argument("--out", default=None)
    c.set_defaults(func=_cmd_calibrate)

    v = sub.add_parser("verdicts", help="would-be verdicts for every published map")
    v.add_argument("--recalibrate", action="store_true", help="simulate k2 instead of using the published value")
    v.add_argument("--families", type=int, default=R0234_FAMILIES)
    v.add_argument("--seed", type=int, default=R0234_CALIBRATION_SEED)
    v.add_argument("--out", default=None)
    v.set_defaults(func=_cmd_verdicts)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
