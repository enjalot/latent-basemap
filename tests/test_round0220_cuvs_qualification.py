"""Unit tests for R0220's pure measurement code. CPU only, no cuVS, no CUDA."""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0220_cuvs_qualification import (
    BRUTE_FORCE_100M_GPU_HOURS,
    GRAPH_K,
    PROJECTION_ROWS,
    Round0220Error,
    SCALING_SETTING_ID,
    SWEEP,
    graph_validity,
    power_law,
    project_cost,
    setting,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
    validate_truth_probe,
)


def test_sweep_is_a_curve_not_a_point():
    assert len(SWEEP) >= 2
    assert len({item["id"] for item in SWEEP}) == len(SWEEP)
    assert {item["algo"] for item in SWEEP} == {"nn_descent", "cagra"}
    assert setting(SCALING_SETTING_ID)["algo"] == "nn_descent"
    with pytest.raises(Round0220Error):
        setting("no-such-setting")


def test_strict_containment_counts_truth_not_candidates():
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    candidates = np.array([[1, 2, 9], [7, 7, 7]])
    got = strict_containment_rows(candidates, truth, chunk=1)
    assert got.tolist() == [2 / 3, 0.0]


def test_strict_containment_is_immune_to_duplicated_candidates():
    truth = np.array([[1, 2, 3]])
    duplicated = np.array([[1, 1, 1]])
    assert strict_containment_rows(duplicated, truth).tolist() == [1 / 3]


def test_tie_aware_counts_ties_and_dedupes():
    cosines = np.array([[1.0, 1.0, 0.5]])
    ids = np.array([[7, 7, 8]])
    kth = np.array([0.9])
    # 7 counts once even though it appears twice; 8 is below the k-th cosine.
    assert tie_aware_rows(cosines, ids, kth, k=3, tolerance=1e-6).tolist() == [1 / 3]


def test_tie_aware_caps_at_k():
    cosines = np.ones((1, 5))
    ids = np.arange(5)[None, :]
    assert tie_aware_rows(cosines, ids, np.array([1.0]), k=3).tolist() == [1.0]


def test_graph_validity_finds_the_r0215_tripwire():
    ids = np.array([[0, 0, 0], [2, 3, 4], [9, 9, 9]], dtype=np.int64)
    got = graph_validity(ids, rows=5)
    assert got["rows_with_self_loop"] == 1
    assert got["rows_with_out_of_range"] == 1
    assert got["rows_with_duplicates"] == 2
    assert got["zero_degree_rows"] == 2
    assert got["min_usable_degree"] == 0
    assert got["rows_below_k"] == 3


def test_graph_validity_on_a_clean_graph():
    ids = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.uint32)
    got = graph_validity(ids, rows=3)
    assert got["zero_degree_rows"] == 0
    assert got["self_loop_entries"] == 0
    assert got["duplicate_entries"] == 0
    assert got["out_of_range_entries"] == 0


def test_summarize_rejects_non_finite():
    with pytest.raises(Round0220Error):
        summarize(np.array([1.0, np.nan]), label="bad")
    with pytest.raises(Round0220Error):
        summarize(np.array([]), label="empty")
    got = summarize(np.array([0.0, 1.0, 1.0, 1.0]), label="ok")
    assert got["mean"] == 0.75
    assert got["fraction_perfect"] == 0.75
    assert got["min"] == 0.0


def test_power_law_recovers_a_known_exponent():
    sizes = [1_000, 2_000, 4_000, 8_000]
    seconds = [2.0 * n**1.2 for n in sizes]
    fit = power_law(sizes, seconds)
    assert fit["exponent"] == pytest.approx(1.2, abs=1e-6)
    assert fit["coefficient"] == pytest.approx(2.0, rel=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_power_law_needs_positive_matched_points():
    with pytest.raises(Round0220Error):
        power_law([1_000], [1.0])
    with pytest.raises(Round0220Error):
        power_law([1_000, 2_000], [1.0, 0.0])


def test_projection_is_labelled_a_projection():
    fit = power_law([1_000_000, 2_000_000], [10.0, 20.0])
    projection = project_cost(fit, rows=PROJECTION_ROWS)
    assert projection["is_measurement"] is False
    assert projection["kind"] == "projection"
    assert projection["rows"] == PROJECTION_ROWS
    assert "153.6 GB" in projection["caveat"]
    assert projection["projected_seconds"] == pytest.approx(1000.0, rel=1e-9)
    assert projection["brute_force_baseline_gpu_hours"] == BRUTE_FORCE_100M_GPU_HOURS
    assert projection["speedup_vs_brute_force"] > 1.0


def test_truth_probe_floor_is_fail_closed():
    good = {"mean": 0.9999, "p10": 1.0}
    strict = {"mean": 0.9986, "p10": 1.0}
    assert validate_truth_probe(tie_aware=good, strict=strict)["passed"] is True
    with pytest.raises(Round0220Error):
        validate_truth_probe(tie_aware={"mean": 0.5, "p10": 0.0}, strict=strict)
    with pytest.raises(Round0220Error):
        validate_truth_probe(tie_aware={"mean": 0.9999, "p10": 0.5}, strict=strict)


def test_perfect_builder_scores_one_everywhere():
    rng = np.random.default_rng(220)
    truth = rng.choice(1_000, size=(64, GRAPH_K), replace=True)
    truth = np.array([np.unique(row)[:GRAPH_K] for row in truth if np.unique(row).size >= GRAPH_K])
    assert truth.shape[0] > 0
    strict = strict_containment_rows(truth, truth)
    assert np.all(strict == 1.0)
