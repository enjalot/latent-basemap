"""Rotation/translation/scale-invariant diagnostics for duplicate components."""
from __future__ import annotations

import numpy as np

from .artifact_identity import ordered_array_sha256


def duplicate_component_diagnostics(
    coordinates,
    *,
    excluded_rows: np.ndarray,
    representative_rows: np.ndarray,
    sample_seed: int = 20260719,
    sample_size: int = 200_000,
) -> dict:
    """Measure component offset in the map's own fixed-sample covariance frame."""
    excluded = np.asarray(excluded_rows, dtype=np.int64)
    representatives = np.asarray(representative_rows, dtype=np.int64)
    selected_component = np.sort(np.concatenate([excluded, representatives]))
    rng = np.random.RandomState(sample_seed)
    sample_ids = np.sort(
        rng.choice(len(coordinates), sample_size, replace=False)
    ).astype(np.int64)
    sample_ids = sample_ids[
        ~np.isin(sample_ids, selected_component, assume_unique=True)
    ]
    sample = np.asarray(coordinates[sample_ids], dtype=np.float64)
    component = np.asarray(coordinates[representatives], dtype=np.float64)
    if (
        sample.shape != (len(sample_ids), 2)
        or component.shape != (len(representatives), 2)
        or not np.isfinite(sample).all()
        or not np.isfinite(component).all()
    ):
        raise ValueError("duplicate-component diagnostic received invalid coordinates")
    center = sample.mean(axis=0)
    covariance = np.cov(sample, rowvar=False)
    if covariance.shape != (2, 2) or np.linalg.det(covariance) <= 0:
        raise ValueError("duplicate-component reference covariance is singular")
    inverse = np.linalg.inv(covariance)
    delta = component - center
    distance = np.sqrt(np.einsum("ni,ij,nj->n", delta, inverse, delta))
    sample_delta = sample - center
    sample_distance = np.sqrt(
        np.einsum("ni,ij,nj->n", sample_delta, inverse, sample_delta)
    )
    return {
        "method": "fixed-sample-mahalanobis-excluding-selected-component",
        "sample_seed": sample_seed,
        "sample_size_requested": sample_size,
        "sample_size_effective": len(sample_ids),
        "sample_ids_sha256": ordered_array_sha256(sample_ids),
        "reference_center": center.tolist(),
        "reference_covariance": covariance.tolist(),
        "representative_rows": representatives.tolist(),
        "representative_coordinates": component.tolist(),
        "representative_mahalanobis": distance.tolist(),
        "maximum_representative_mahalanobis": float(distance.max()),
        "sample_median_mahalanobis": float(np.median(sample_distance)),
    }
