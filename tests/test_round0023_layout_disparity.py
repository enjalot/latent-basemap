from __future__ import annotations

import numpy as np

from basemap.round0023_program import train_config_for_seed
from experiments.layout_disparity import (
    knn_indices,
    local_radius,
    optimal_similarity,
    pair_metrics,
    retention_and_jaccard,
)
from experiments import run_round0014_node as node


def test_round0023_seed_config_changes_only_seed_identity():
    cfg43, digest43 = train_config_for_seed(43)
    cfg44, digest44 = train_config_for_seed(44)
    assert cfg43["optimizer"]["seed"] == 43
    assert cfg44["optimizer"]["seed"] == 44
    assert cfg43["model"] == cfg44["model"]
    assert cfg43["execution"]["duplicate_multiplicity"] == cfg44["execution"]["duplicate_multiplicity"]
    assert digest43 != digest44


def test_round0023_configure_uses_job_seed():
    node.configure_round0023(job={"seed": 44})
    assert node.ROUND_ID == "0023"
    assert node.SCHEMA_PREFIX == "round0023-seed44"
    assert node.TRAIN_CONFIG["optimizer"]["seed"] == 44


def test_optimal_similarity_recovers_scaled_rotated_layout():
    rng = np.random.default_rng(23)
    reference = rng.normal(size=(128, 2))
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moving = ((reference - [2.0, -1.0]) @ rotation.T) / 3.5 + [9.0, 4.0]
    aligned, transform = optimal_similarity(reference, moving)
    assert np.max(np.linalg.norm(reference - aligned, axis=1)) < 1e-10
    assert transform["normalized_sum_squared_disparity"] < 1e-20


def test_retention_reports_retention_and_jaccard():
    left = np.array([[1, 2, 3], [3, 4, 5]])
    right = np.array([[2, 3, 4], [6, 7, 8]])
    got = retention_and_jaccard(left, right, k=3)
    assert got["mean_retention"] == (2 / 3 + 0) / 2
    assert got["mean_jaccard"] == (2 / 4 + 0) / 2


def test_pair_metrics_uses_seed42_local_radius_units():
    points = np.stack([np.arange(64), np.zeros(64)], axis=1).astype(float)
    moved = points + np.array([0.0, 0.5])
    radii = local_radius(points, k=15)
    knn_a = knn_indices(points, k=15)
    knn_b = knn_indices(moved, k=15)
    got = pair_metrics(points, moved, radii, knn_a, knn_b, pair=("a", "b"))
    assert got["pair"] == ["a", "b"]
    assert got["median_drift_local_r15"] < 1e-10
    assert got["neighbor_overlap"]["mean_retention"] == 1.0
