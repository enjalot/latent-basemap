"""Frozen native-prompted density evidence recovery for Round 0157."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from experiments.round0085_nodes import density_v2_calibration


ROUND_ID = "0157"
CAPABILITY = "jina-fineweb-2m-native-prompted-density-v2-v1"
ROWS = 1_993_761
DIMENSION = 768
ANCHORS = 10_000
ANCHOR_SEED = 123
K_DENSITY = 15
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 10_801
NULL_DRAWS = 1_000
NULL_SEED = 10_802
RAW_UNIVERSE_CONTEXT_FLOOR = 0.17589389755990817
SEEDS = (42, 43)


class Round0157Error(RuntimeError):
    """Raised when accepted prompted-map evidence changes."""


def density_v2_from_radii(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Apply the frozen R0085/R0108 density statistic to one prompted map."""
    high = np.asarray(high_radius, dtype=np.float64)
    low = np.asarray(low_radius, dtype=np.float64)
    if (
        high.shape != (ANCHORS,)
        or low.shape != (ANCHORS,)
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
        or np.any(high <= 0)
        or np.any(low <= 0)
    ):
        raise Round0157Error("prompted density radii changed")
    return density_v2_calibration(
        high,
        low,
        bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=BOOTSTRAP_SEED,
        null_draws=NULL_DRAWS,
        null_seed=NULL_SEED,
    )


def transcribe_native_prompted_score(
    score: Mapping[str, Any],
    *,
    seed: int,
    expected_coordinates: Mapping[str, Any],
) -> dict[str, Any]:
    """Transcribe accepted native-panel evidence without reinterpreting it."""
    panel = score.get("panel")
    metrics = score.get("metrics")
    coordinates = score.get("coordinates")
    if (
        seed not in SEEDS
        or score.get("arm") != "document"
        or not isinstance(panel, Mapping)
        or not isinstance(metrics, Mapping)
        or not isinstance(coordinates, Mapping)
        or coordinates.get("training") != dict(expected_coordinates)
        or int(panel.get("n", -1)) != ROWS
        or int(panel.get("n_dims_hi", -1)) != DIMENSION
        or int(panel.get("k_density", -1)) != K_DENSITY
        or panel.get("provenance", {}).get("arm") != "document"
    ):
        raise Round0157Error(f"seed-{seed} native prompted score changed")
    required = (
        "density",
        "ffr",
        "oos_recall_at_10",
        "oos_recall_at_50",
        "recall_at_10",
    )
    if set(required) - set(metrics):
        raise Round0157Error(f"seed-{seed} prompted metrics are incomplete")
    values = {key: float(metrics[key]) for key in required}
    if not np.isfinite(tuple(values.values())).all():
        raise Round0157Error(f"seed-{seed} prompted metrics are nonfinite")
    return {
        "seed": seed,
        "native_embedding_convention": "Document: ",
        "native_graph": True,
        "native_training": True,
        "training_rows": ROWS,
        "accepted_panel_anchor_count": int(panel["n_anchors"]),
        "accepted_panel_metrics": values,
        "accepted_execution_gates": dict(score.get("execution_gates") or {}),
        "accepted_ood_diagnostics": dict(score.get("ood") or {}),
        "accepted_train_receipt": dict(panel["provenance"]["train_receipt"]),
    }

