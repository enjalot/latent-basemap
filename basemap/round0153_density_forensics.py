"""Frozen CPU-side density-v2 forensics for Round 0153."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from basemap.round0108_evaluation import seal
from experiments.round0085_nodes import density_v2_calibration


ROUND_ID = "0153"
CAPABILITY = "jina-2m-track-a-density-forensics-v1"
REGISTERED_FLOOR = 0.17589389755990817
ROWS = 2_000_000
REPRESENTATIVE_ROWS = 1_996_279
ANCHORS = 10_000
K_DENSITY = 15
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 10_801
NULL_DRAWS = 1_000
NULL_SEED = 10_802

HISTORICAL_ROW_CELLS = (
    "r0140_current_graph_current_host",
    "r0140_historical_graph_current_host",
    "r0140_historical_graph_device_reproduction",
)
CURRENT_POPULATION_REFERENCES = (
    "r0115_current_2m_seed42",
    "r0117_current_2m_seed43",
)


class Round0153Error(RuntimeError):
    """Raised when the preregistered R0153 evidence contract changes."""


def exact_low_radius_cpu(
    coordinates: np.ndarray,
    retained_global_rows: np.ndarray,
    anchor_compact_rows: np.ndarray,
    *,
    workers: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return exact mean-k15 fp32 radii on the R0040 representative universe.

    ``cKDTree`` is used only to identify exact Euclidean candidates. Distances
    are recomputed from the selected coordinates in fp32, matching the frozen
    R0108 low-dimensional arithmetic. Two guard candidates are requested so
    the exact anchor row can always be removed without changing k.
    """
    coordinates = np.asarray(coordinates)
    retained_global_rows = np.asarray(retained_global_rows, dtype=np.int64)
    anchor_compact_rows = np.asarray(anchor_compact_rows, dtype=np.int64)
    if (
        coordinates.shape != (ROWS, 2)
        or coordinates.dtype != np.float32
        or retained_global_rows.shape != (REPRESENTATIVE_ROWS,)
        or anchor_compact_rows.shape != (ANCHORS,)
        or not np.array_equal(
            retained_global_rows,
            np.unique(retained_global_rows),
        )
        or int(retained_global_rows[0]) < 0
        or int(retained_global_rows[-1]) >= ROWS
        or int(anchor_compact_rows.min()) < 0
        or int(anchor_compact_rows.max()) >= REPRESENTATIVE_ROWS
        or workers < 1
    ):
        raise Round0153Error("density-v2 row geometry changed")

    representative_coordinates = np.asarray(
        coordinates[retained_global_rows], dtype=np.float32
    )
    if not np.isfinite(representative_coordinates).all():
        raise Round0153Error("density-v2 coordinates are nonfinite")

    tree = cKDTree(
        representative_coordinates,
        compact_nodes=True,
        balanced_tree=True,
    )
    _distances, candidates = tree.query(
        representative_coordinates[anchor_compact_rows],
        k=K_DENSITY + 2,
        eps=0.0,
        workers=workers,
    )
    candidates = np.asarray(candidates, dtype=np.int64)
    neighbors = np.empty((ANCHORS, K_DENSITY), dtype=np.int64)
    for position, (anchor, row) in enumerate(
        zip(anchor_compact_rows, candidates, strict=True)
    ):
        without_self = row[row != anchor]
        if len(without_self) < K_DENSITY:
            raise Round0153Error("exact CPU search did not close self exclusion")
        neighbors[position] = without_self[:K_DENSITY]

    differences = np.asarray(
        representative_coordinates[neighbors]
        - representative_coordinates[anchor_compact_rows, None, :],
        dtype=np.float32,
    )
    squared = np.sum(differences * differences, axis=2, dtype=np.float32)
    distances = np.sqrt(squared, dtype=np.float32)
    low_radius = np.asarray(distances.mean(axis=1), dtype=np.float64)
    if not np.isfinite(low_radius).all() or np.any(low_radius <= 0):
        raise Round0153Error("exact CPU radii are nonpositive or nonfinite")
    return low_radius, {
        "algorithm": "scipy-cKDTree-exact-euclidean-candidates",
        "candidate_eps": 0.0,
        "candidate_count": K_DENSITY + 2,
        "self_exclusion": "exact compact row id",
        "distance_recompute": "fp32 squared-l2 and fp32 sqrt",
        "mean": "fp32 mean promoted to float64",
        "workers": workers,
        "candidate_population_rows": REPRESENTATIVE_ROWS,
        "anchor_rows": ANCHORS,
        "k": K_DENSITY,
    }


def density_v2_from_radii(
    high_radius: np.ndarray,
    low_radius: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    high_radius = np.asarray(high_radius, dtype=np.float64)
    low_radius = np.asarray(low_radius, dtype=np.float64)
    if (
        high_radius.shape != (ANCHORS,)
        or low_radius.shape != (ANCHORS,)
        or not np.isfinite(high_radius).all()
        or not np.isfinite(low_radius).all()
        or np.any(high_radius <= 0)
        or np.any(low_radius <= 0)
    ):
        raise Round0153Error("density-v2 radius contract changed")
    return density_v2_calibration(
        high_radius,
        low_radius,
        bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=BOOTSTRAP_SEED,
        null_draws=NULL_DRAWS,
        null_seed=NULL_SEED,
    )


def diagnostic_values(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Transcribe the complete functional panel without relabelling density."""
    panel = cell.get("panel")
    projection = cell.get("projection")
    if not isinstance(panel, Mapping) or not isinstance(projection, Mapping):
        raise Round0153Error("functional panel cell is malformed")
    purity = panel.get("purity")
    if not isinstance(purity, Mapping):
        raise Round0153Error("functional panel purity is missing")
    values = {
        "ffr": float(panel["ffr"]),
        "recall_at_10_transductive": float(panel["recall@k"]),
        "purity_raw_ratio_k256": float(purity["k256"]),
        "purity_raw_ratio_k1024": float(purity["k1024"]),
        "projection_ffr": float(projection["ffr"]),
        "held_out_recall_at_10": float(projection["recall_at_10"]),
        # This is the original all-row/panel-anchor statistic. It is retained
        # for diagnosis but must never be mistaken for R0108 density-v2.
        "legacy_panel_density_not_density_v2": float(panel["density"]),
    }
    decision = cell.get("decision_metrics")
    if isinstance(decision, Mapping):
        values["registered_decision_metrics"] = {
            str(key): float(value) for key, value in decision.items()
        }
    if not all(
        np.isfinite(value)
        for key, value in values.items()
        if key != "registered_decision_metrics"
    ):
        raise Round0153Error("functional diagnostic contains nonfinite values")
    return values


def classify_density_branch(
    cell_correlations: Mapping[str, float],
    current_reference_correlations: Mapping[str, float],
) -> dict[str, Any]:
    missing_cells = set(HISTORICAL_ROW_CELLS) - set(cell_correlations)
    missing_references = set(CURRENT_POPULATION_REFERENCES) - set(
        current_reference_correlations
    )
    if missing_cells or missing_references:
        raise Round0153Error(
            f"density branch inputs incomplete: cells={sorted(missing_cells)}, "
            f"references={sorted(missing_references)}"
        )
    historical_pass = {
        key: float(cell_correlations[key]) >= REGISTERED_FLOOR
        for key in HISTORICAL_ROW_CELLS
    }
    current_fail = {
        key: float(current_reference_correlations[key]) < REGISTERED_FLOOR
        for key in CURRENT_POPULATION_REFERENCES
    }
    if all(historical_pass.values()) and all(current_fail.values()):
        outcome = "density-restores-with-row-universe"
        track_f_activated = True
    elif not any(historical_pass.values()):
        outcome = "density-does-not-restore"
        track_f_activated = False
    else:
        outcome = "density-mixed-owner-decision-required"
        track_f_activated = False
    return seal({
        "schema": "round0153-density-branch-decision-v1",
        "round_id": ROUND_ID,
        "capability": CAPABILITY,
        "registered_floor": REGISTERED_FLOOR,
        "historical_row_cells_clear_floor": historical_pass,
        "current_population_references_fail_floor": current_fail,
        "outcome": outcome,
        "track_f_activated": track_f_activated,
        "floor_changed": False,
        "training_performed": False,
    })
