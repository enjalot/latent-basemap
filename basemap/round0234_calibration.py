"""R0234 — Monte-Carlo calibration of a floor multiplier on the Gaussian null.

Everything in this module is a property of an *estimator* at a sample size. It
reads no cell of this program, so the multipliers, the delivered coverage, the
false-fail rates and the detection power it returns — and therefore the
pre-registered selection among candidates — are fixed before any sealed artifact
is opened.

**One-sided.** A `95/95` lower tolerance bound satisfies
`P(L <= mu - z_P*sigma) = gamma`. For `L = centre - k*scale` that rearranges to
`P(k >= (centre - mu + z_P*sigma)/scale) = gamma`, so with `mu = 0, sigma = 1`
the calibrated multiplier is exactly the `gamma`-quantile of
`(centre + z_P)/scale`. Closed form for `centre = xbar, scale = s`:
`k = t'_{n-1, z_P*sqrt(n)}(gamma)/sqrt(n)`, which this module reproduces as an
external check on the sampler.

**Two-sided.** Content is `Phi(U) - Phi(L) >= P` — the standard definition Howe's
factor approximates, and *not* "the interval brackets `mu +/- z_{(1+P)/2}`",
which is strictly stronger and would inflate the factor. The per-family minimal
`k` is found by bisection and the calibrated factor is its `gamma`-quantile.

**Reported beside every multiplier.** Delivered coverage; the new-cell false-fail
rate `E[P(X < L)]` for `X ~ N(0,1)`, which is what a *commensurate* future cell
pays; and detection power `E[P(X < L)]` for `X ~ N(-delta, 1)`, which is what a
genuinely regressed cell pays. At a fixed 95% confidence, power is the only
honest way to compare two estimators: a floor can only buy fewer false failures
by also catching fewer real ones.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

#: Fixed so an independent reviewer re-derives the same multipliers bit for bit.
CALIBRATION_SEED = 20260809
CALIBRATION_FAMILIES = 4_000_000
CALIBRATION_CHUNK = 250_000
BISECTION_ITERATIONS = 44
BISECTION_UPPER = 80.0

MAD_CONSISTENCY = 1.4826
SN_CONSISTENCY = 1.1926
QN_CONSISTENCY = 2.2219
IQR_CONSISTENCY = 1.3489795003921634


def _sn_correction(n: int) -> float:
    table = {2: 0.743, 3: 1.851, 4: 0.954, 5: 1.351, 6: 0.993, 7: 1.198, 8: 1.005,
             9: 1.131}
    if n in table:
        return table[n]
    return n / (n - 0.9) if n % 2 else 1.0


def _qn_correction(n: int) -> float:
    table = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844, 6: 0.611, 7: 0.857, 8: 0.669,
             9: 0.872}
    if n in table:
        return table[n]
    return n / (n + 1.4) if n % 2 else n / (n + 3.8)


ALL_ESTIMATORS: tuple[str, ...] = (
    "mean_minus_k_sample_sd",
    "median_minus_k_madn",
    "trimmed1_mean_minus_k_trimmed_sd",
    "median_minus_k_sn",
    "median_minus_k_qn",
    "median_minus_k_iqrn",
)


def centre_scale_arrays(
    x: np.ndarray, n: int, names: tuple[str, ...] = ALL_ESTIMATORS
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Vectorised twins of `round0234_calibrated_floors.centre_and_scale`.

    `names` exists so a ladder over large `n` never pays for the `O(n^2)` pairwise
    estimators it is not asking for, and so a degenerate `n` never evaluates an
    estimator that is undefined there.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wanted = set(names)
    median = np.median(x, axis=1)
    if "mean_minus_k_sample_sd" in wanted:
        out["mean_minus_k_sample_sd"] = (x.mean(axis=1), x.std(axis=1, ddof=1))
    if "median_minus_k_madn" in wanted:
        out["median_minus_k_madn"] = (
            median,
            MAD_CONSISTENCY * np.median(np.abs(x - median[:, None]), axis=1),
        )
    if "trimmed1_mean_minus_k_trimmed_sd" in wanted:
        if n < 5:
            raise ValueError("the 1-trimmed family needs at least five cells")
        kept = np.sort(x, axis=1)[:, 1:-1]
        out["trimmed1_mean_minus_k_trimmed_sd"] = (
            kept.mean(axis=1), kept.std(axis=1, ddof=1)
        )
    if wanted & {"median_minus_k_sn", "median_minus_k_qn"}:
        distances = np.abs(x[:, :, None] - x[:, None, :])
        if "median_minus_k_sn" in wanted:
            himed = np.sort(distances, axis=2)[:, :, n // 2]
            out["median_minus_k_sn"] = (
                median,
                _sn_correction(n) * SN_CONSISTENCY
                * np.sort(himed, axis=1)[:, (n + 1) // 2 - 1],
            )
        if "median_minus_k_qn" in wanted:
            rows, cols = np.triu_indices(n, 1)
            pairs = distances[:, rows, cols]
            half = n // 2 + 1
            rank = half * (half - 1) // 2
            out["median_minus_k_qn"] = (
                median,
                _qn_correction(n) * QN_CONSISTENCY
                * np.partition(pairs, rank - 1, axis=1)[:, rank - 1],
            )
    if "median_minus_k_iqrn" in wanted:
        out["median_minus_k_iqrn"] = (
            median,
            (np.quantile(x, 0.75, axis=1) - np.quantile(x, 0.25, axis=1))
            / IQR_CONSISTENCY,
        )
    return out


def _two_sided_minimal_k(
    centre: np.ndarray, scale: np.ndarray, content: float
) -> np.ndarray:
    low = np.zeros_like(centre)
    high = np.full_like(centre, BISECTION_UPPER)
    for _ in range(BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        covered = stats.norm.cdf(centre + mid * scale) - stats.norm.cdf(
            centre - mid * scale
        )
        short = covered < content
        low = np.where(short, mid, low)
        high = np.where(short, high, mid)
    return 0.5 * (low + high)


def draw_centres_and_scales(
    n: int,
    *,
    families: int = CALIBRATION_FAMILIES,
    seed: int = CALIBRATION_SEED,
    chunk: int = CALIBRATION_CHUNK,
    names: tuple[str, ...] = ALL_ESTIMATORS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    parts: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    drawn = 0
    while drawn < families:
        size = min(chunk, families - drawn)
        sample = rng.standard_normal((size, n))
        for name, (centre, scale) in centre_scale_arrays(sample, n, names).items():
            bucket = parts.setdefault(name, ([], []))
            bucket[0].append(centre)
            bucket[1].append(scale)
        drawn += size
    return {
        name: (np.concatenate(centres), np.concatenate(scales))
        for name, (centres, scales) in parts.items()
    }


def summarise(
    centre: np.ndarray,
    scale: np.ndarray,
    multiplier: float,
    *,
    content: float,
    alternatives: tuple[float, ...],
) -> dict[str, Any]:
    z_content = float(stats.norm.ppf(content))
    lower = centre - float(multiplier) * scale
    return {
        "multiplier": float(multiplier),
        "delivered_coverage": float(np.mean(lower <= -z_content)),
        "new_cell_false_fail_rate": float(stats.norm.cdf(lower).mean()),
        "detection_power": {
            f"minus_{delta:g}_sigma": float(stats.norm.cdf(lower + delta).mean())
            for delta in alternatives
        },
    }


def summarise_two_sided(
    centre: np.ndarray,
    scale: np.ndarray,
    multiplier: float,
    *,
    content: float,
) -> dict[str, Any]:
    covered = stats.norm.cdf(centre + float(multiplier) * scale) - stats.norm.cdf(
        centre - float(multiplier) * scale
    )
    return {
        "multiplier": float(multiplier),
        "delivered_confidence_at_content": float(np.mean(covered >= content)),
        "new_cell_false_fail_rate": float((1.0 - covered).mean()),
    }


def calibrate(
    n: int,
    *,
    content: float = 0.95,
    confidence: float = 0.95,
    alternatives: tuple[float, ...] = (1.0, 2.0, 3.0),
    families: int = CALIBRATION_FAMILIES,
    seed: int = CALIBRATION_SEED,
    chunk: int = CALIBRATION_CHUNK,
    two_sided: bool = True,
    names: tuple[str, ...] = ALL_ESTIMATORS,
) -> dict[str, Any]:
    """Calibrated one- and two-sided multipliers for every candidate at `n`."""
    z_content = float(stats.norm.ppf(content))
    drawn = draw_centres_and_scales(
        n, families=families, seed=seed, chunk=chunk, names=names
    )
    results: dict[str, Any] = {}
    for name, (centre, scale) in drawn.items():
        k_one = float(np.quantile((centre + z_content) / scale, confidence))
        entry: dict[str, Any] = {
            "estimator": name,
            "n": int(n),
            "families_simulated": int(families),
            "one_sided": {
                "calibrated_multiplier": k_one,
                **summarise(
                    centre, scale, k_one,
                    content=content, alternatives=alternatives,
                ),
            },
        }
        if two_sided:
            k_two = float(
                np.quantile(_two_sided_minimal_k(centre, scale, content), confidence)
            )
            entry["two_sided"] = {
                "calibrated_multiplier": k_two,
                **summarise_two_sided(centre, scale, k_two, content=content),
            }
        results[name] = entry
    return {
        "n": int(n),
        "content": float(content),
        "confidence": float(confidence),
        "families_simulated": int(families),
        "seed": int(seed),
        "z_content": z_content,
        "candidates": results,
        "_arrays": drawn,
    }


def fixed_multiplier_report(
    drawn: dict[str, tuple[np.ndarray, np.ndarray]],
    estimator: str,
    multiplier: float,
    *,
    content: float = 0.95,
    alternatives: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, Any]:
    centre, scale = drawn[estimator]
    return {
        "estimator": estimator,
        **summarise(
            centre, scale, multiplier, content=content, alternatives=alternatives
        ),
    }


def fixed_two_sided_report(
    drawn: dict[str, tuple[np.ndarray, np.ndarray]],
    estimator: str,
    multiplier: float,
    *,
    content: float = 0.95,
) -> dict[str, Any]:
    centre, scale = drawn[estimator]
    return {
        "estimator": estimator,
        **summarise_two_sided(centre, scale, multiplier, content=content),
    }


def nct_one_sided_factor(n: int, *, content: float = 0.95, confidence: float = 0.95) -> float:
    z_content = float(stats.norm.ppf(content))
    return float(
        stats.nct.ppf(confidence, n - 1, z_content * math.sqrt(n))
    ) / math.sqrt(n)


def howe_two_sided_factor(n: int, *, content: float = 0.95, confidence: float = 0.95) -> float:
    z = float(stats.norm.ppf((1.0 + content) / 2.0))
    chi = float(stats.chi2.ppf(1.0 - confidence, n - 1))
    return (
        z
        * math.sqrt(1.0 + 1.0 / n)
        * math.sqrt((n - 1) / chi)
        * math.sqrt(1.0 + (n - 3.0 - chi) / (2.0 * (n + 1.0) ** 2))
    )


def power_ladder(
    estimator: str,
    *,
    sizes: tuple[int, ...],
    content: float = 0.95,
    confidence: float = 0.95,
    alternative: float = 2.0,
    families: int = 600_000,
    chunk: int = 100_000,
    seed: int = CALIBRATION_SEED,
) -> list[dict[str, Any]]:
    """Calibrated multiplier and detection power for one estimator across `n`.

    This is how "no estimator achieves both at `n = 13`, and `n` must rise to X"
    becomes a number instead of a sentiment.
    """
    z_content = float(stats.norm.ppf(content))
    rows: list[dict[str, Any]] = []
    for size in sizes:
        rng = np.random.default_rng(seed + size)
        centres: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        drawn = 0
        while drawn < families:
            count = min(chunk, families - drawn)
            sample = rng.standard_normal((count, size))
            centre, scale = centre_scale_arrays(sample, size, (estimator,))[estimator]
            centres.append(centre)
            scales.append(scale)
            drawn += count
        centre = np.concatenate(centres)
        scale = np.concatenate(scales)
        k = float(np.quantile((centre + z_content) / scale, confidence))
        lower = centre - k * scale
        rows.append({
            "n": int(size),
            "calibrated_multiplier": k,
            "delivered_coverage": float(np.mean(lower <= -z_content)),
            "new_cell_false_fail_rate": float(stats.norm.cdf(lower).mean()),
            "detection_power_at_minus_2_sigma": float(
                stats.norm.cdf(lower + alternative).mean()
            ),
            "families_simulated": int(families),
        })
    return rows


__all__ = [
    "ALL_ESTIMATORS",
    "BISECTION_ITERATIONS",
    "CALIBRATION_CHUNK",
    "CALIBRATION_FAMILIES",
    "CALIBRATION_SEED",
    "calibrate",
    "centre_scale_arrays",
    "draw_centres_and_scales",
    "fixed_multiplier_report",
    "fixed_two_sided_report",
    "howe_two_sided_factor",
    "nct_one_sided_factor",
    "power_ladder",
    "summarise",
    "summarise_two_sided",
]
