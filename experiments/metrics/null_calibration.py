"""Gaussian-null calibration of the ``median - k * MAD_n`` multiplier.

This re-implements the method R0234 used to calibrate its n = 13 floors and
R0255 re-applied at n = 29, so that a floor proposed for a NEW metric
(collapse, fog) uses the program's own convention rather than a fresh one.

Method (95/95 one-sided tolerance bound, by simulation)
-------------------------------------------------------
Draw ``families`` independent families of ``n`` standard normals. For each
family compute the candidate floor ``L = median - k * MAD_n``. The floor's
CONTENT is the probability a fresh conforming cell clears it,
``P(Z > L) = 1 - Phi(L)``. We want content >= 0.95 with confidence 0.95,
i.e. the smallest k such that

    P_families( 1 - Phi(median - k * MAD_n) >= 0.95 ) = 0.95.

Because ``1 - Phi(L) >= 0.95  <=>  L <= z_0.05 = -1.6448536...``, this
inverts in closed form per family:

    L <= -z  <=>  k >= (median + z) / MAD_n     (MAD_n > 0)

so the calibrated k is simply the 95th percentile of
``T = (median + z_0.95) / MAD_n`` across the simulated families -- no
bisection, no MC search noise beyond the quantile itself.

Two derived numbers are reported alongside, matching R0234's schema:
  * ``new_cell_false_fail_rate`` = E[Phi(L)], the average probability that
    a fresh CONFORMING cell is rejected by the floor.
  * ``detection_power[-d sigma]`` = E[Phi(L + d)], the probability a cell
    whose true mean sits d sigma below the family mean is caught.

Validation gate
---------------
``validate_r0234()`` reproduces R0234's published one-sided n = 13
``median_minus_k_madn`` multiplier and its false-fail / power ladder. The
sealed value is k = 3.7363661743730416, read from
  /data/latent-basemap/runs/round-0234/queue/artifacts/
    minilm-mixed-2m-calibrated-robust-floors-n13-v1/
    minilm-calibrated-robust-floors-n13.json
  -> calibration.n13.candidates.median_minus_k_madn.one_sided.calibrated_multiplier
The n = 29 sealed value k = 2.6934393367626024 is read from
  /data/latent-basemap/runs/round-0255/queue/artifacts/
    minilm-mixed-2m-calibrated-madn-floors-n29-v1/
    minilm-calibrated-madn-floors-n29.json
  -> calibration.n29.candidates.median_minus_k_madn.one_sided.calibrated_multiplier
Both artifacts are SEALED and are only ever read here.

CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

try:  # works both as a package module and as a bare script in this directory
    from .collapse_fog import MADN_CONSISTENCY
except ImportError:  # pragma: no cover
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collapse_fog import MADN_CONSISTENCY

__all__ = [
    "calibrate_one_sided",
    "validate_r0234",
    "R0234_N13_SEALED_PATH",
    "R0255_N29_SEALED_PATH",
]

R0234_N13_SEALED_PATH = Path(
    "/data/latent-basemap/runs/round-0234/queue/artifacts"
    "/minilm-mixed-2m-calibrated-robust-floors-n13-v1"
    "/minilm-calibrated-robust-floors-n13.json")
R0255_N29_SEALED_PATH = Path(
    "/data/latent-basemap/runs/round-0255/queue/artifacts"
    "/minilm-mixed-2m-calibrated-madn-floors-n29-v1"
    "/minilm-calibrated-madn-floors-n29.json")

DEFAULT_FAMILIES = 4_000_000
DEFAULT_SEED = 20260809  # R0234/R0255's own simulation seed


def sealed_multiplier(path: Path, n_key: str) -> float:
    """Read a sealed calibrated one-sided MAD_n multiplier. Read-only."""
    d = json.loads(Path(path).read_text())
    return float(d["calibration"][n_key]["candidates"]["median_minus_k_madn"]
                 ["one_sided"]["calibrated_multiplier"])


def calibrate_one_sided(
    n: int,
    families: int = DEFAULT_FAMILIES,
    seed: int = DEFAULT_SEED,
    content: float = 0.95,
    confidence: float = 0.95,
    chunk: int = 200_000,
    power_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict:
    """Calibrate k for ``median - k * MAD_n`` at the given family size."""
    if families < 1_000_000:
        raise ValueError("the program's convention is >= 1,000,000 null families")
    z_content = float(norm.ppf(content))
    rng = np.random.default_rng(seed)

    t_parts: list[np.ndarray] = []
    done = 0
    while done < families:
        m = int(min(chunk, families - done))
        done += m
        x = rng.standard_normal((m, n))
        med = np.median(x, axis=1)
        scale = MADN_CONSISTENCY * np.median(np.abs(x - med[:, None]), axis=1)
        t_parts.append((med + z_content) / scale)
    t = np.concatenate(t_parts)
    k = float(np.quantile(t, confidence))

    # Second pass with the calibrated k for the false-fail / power ladder.
    rng2 = np.random.default_rng(seed + 1)
    acc = np.zeros(1 + len(power_sigmas), dtype=np.float64)
    done = 0
    while done < families:
        m = int(min(chunk, families - done))
        done += m
        x = rng2.standard_normal((m, n))
        med = np.median(x, axis=1)
        scale = MADN_CONSISTENCY * np.median(np.abs(x - med[:, None]), axis=1)
        floor = med - k * scale
        acc[0] += norm.cdf(floor).sum()
        for j, d in enumerate(power_sigmas):
            acc[j + 1] += norm.cdf(floor + d).sum()
    acc /= families

    return {
        "estimator": "median_minus_k_madn",
        "madn_consistency": MADN_CONSISTENCY,
        "n": int(n),
        "families_simulated": int(families),
        "seed": int(seed),
        "content": content,
        "confidence": confidence,
        "z_content": z_content,
        "calibrated_multiplier": k,
        "delivered_coverage": content,
        "new_cell_false_fail_rate": float(acc[0]),
        "detection_power": {f"minus_{int(d)}_sigma": float(acc[j + 1])
                            for j, d in enumerate(power_sigmas)},
    }


def validate_r0234(families: int = DEFAULT_FAMILIES,
                   seed: int = DEFAULT_SEED,
                   tolerance: float = 0.02) -> dict:
    """Reproduce R0234's sealed n = 13 multiplier before trusting n = 29.

    The program's standing rule: a calibrator that has never reproduced a
    published number is an untested calibrator.
    """
    sealed = sealed_multiplier(R0234_N13_SEALED_PATH, "n13")
    got = calibrate_one_sided(13, families=families, seed=seed)
    rel = abs(got["calibrated_multiplier"] - sealed) / sealed
    return {
        "gate": "reproduce_r0234_n13_one_sided_madn_multiplier",
        "sealed_value": sealed,
        "sealed_source": str(R0234_N13_SEALED_PATH),
        "sealed_key": "calibration.n13.candidates.median_minus_k_madn."
                      "one_sided.calibrated_multiplier",
        "reproduced_value": got["calibrated_multiplier"],
        "relative_error": rel,
        "tolerance": tolerance,
        "passes": bool(rel <= tolerance),
        "reproduced_false_fail_rate": got["new_cell_false_fail_rate"],
        "reproduced_detection_power": got["detection_power"],
    }
