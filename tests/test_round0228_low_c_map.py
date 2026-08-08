"""Unit tests for R0228's contract, its permutation tests and its geometry.

The statistics get positive controls with answers that can be worked out by
hand, because an exact permutation test that silently computes the wrong null is
exactly the kind of defect a reviewer cannot see from a receipt.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from basemap.round0223_cuvs_graph_map import GRAPH_BEARING_PATHS
from basemap.round0228_geometry import (
    Round0228GeometryError,
    clump_profile,
    density_matched_control,
    displacement_summary,
    map_scale,
    true_neighbour_scatter,
)
from basemap.round0228_low_c_map import (
    CELLS,
    SEED_BEARING_PATHS,
    CLUSTERS_BUILT_HERE,
    CLUSTERS_FROM_R0227,
    CLUSTER_COUNTS,
    C_MIN,
    DESCRIPTIVE_ONLY_METRICS,
    EXACT_FAMILY_SEEDS,
    GATED_METRICS,
    PANEL_METRICS,
    R0217_TREATMENT_INVARIANT_SHA256,
    ROWS,
    Round0228Error,
    SEEDS,
    _label_assignments,
    compare_to_families,
    exact_permutation_trend,
    exact_permutation_two_sample,
    graph_capability,
    map_capability,
    successful_updates_for_edges,
    train_config,
    validate_cluster_spill_graph,
)
from basemap.round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)


PROBE_GRAPH_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0228/x/edges-k15-fuzzy.npz",
    "bytes": 123,
    "sha256": "a" * 64,
}
PROBE_MANIFEST_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0228/x/cluster-spill-graph.json",
    "bytes": 456,
    "sha256": "b" * 64,
}
PROBE_EDGES = 48_100_000


# --------------------------------------------------------------------------- #
# the arms
# --------------------------------------------------------------------------- #
def test_the_registered_arms_are_what_the_round_says() -> None:
    assert CLUSTER_COUNTS == (4, 8, 16)
    assert SEEDS == (42, 43, 44)
    assert len(CELLS) == 9
    assert set(CLUSTERS_BUILT_HERE) | set(CLUSTERS_FROM_R0227) == set(CLUSTER_COUNTS)
    assert not set(CLUSTERS_BUILT_HERE) & set(CLUSTERS_FROM_R0227)
    # c = 4 is the structural floor of the builder, so there is no lower bracket.
    assert min(CLUSTER_COUNTS) == C_MIN
    assert len(set(MAP := [map_capability(c, s) for c, s in CELLS])) == len(MAP)
    assert len({graph_capability(c) for c in CLUSTER_COUNTS}) == len(CLUSTER_COUNTS)


def test_density_v2_is_descriptive_only_and_never_gated() -> None:
    assert "density_v2" in DESCRIPTIVE_ONLY_METRICS
    assert "density_v2" not in GATED_METRICS
    assert set(GATED_METRICS) | set(DESCRIPTIVE_ONLY_METRICS) == set(PANEL_METRICS)


def test_unregistered_cells_are_refused() -> None:
    for clusters, seed in ((5, 42), (4, 45), (32, 42), (8, 99)):
        with pytest.raises(Round0228Error):
            map_capability(clusters, seed)
    with pytest.raises(Round0228Error):
        graph_capability(32)


# --------------------------------------------------------------------------- #
# the treatment identity
# --------------------------------------------------------------------------- #
def _config(clusters: int, seed: int, edges: int = PROBE_EDGES):
    return train_config(
        clusters=clusters,
        seed=seed,
        graph_signature=dict(PROBE_GRAPH_SIGNATURE),
        graph_manifest_signature=dict(PROBE_MANIFEST_SIGNATURE),
        substrate_signature=dict(SEALED_SUBSTRATE_SIGNATURE),
        r0216_graph_signature=dict(SEALED_GRAPH_SIGNATURE),
        r0216_graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        graph_edges=edges,
        rows=ROWS,
    )


def test_every_cell_reproduces_the_cross_round_treatment_digest() -> None:
    digests = set()
    configs = set()
    for clusters, seed in CELLS:
        config, config_sha, invariant = _config(clusters, seed)
        digests.add(invariant)
        configs.add(config_sha)
        assert config["capability"] == map_capability(clusters, seed)
        assert config["graph"]["capability"] == graph_capability(clusters)
        assert config["seed"] == seed
    assert digests == {R0217_TREATMENT_INVARIANT_SHA256}
    assert len(configs) == len(CELLS)


def test_only_registered_paths_move_between_configurations() -> None:
    """Two cells with the same seed and different `c` differ only in graph paths."""
    left, _sha_l, _inv_l = _config(4, 42)
    right, _sha_r, _inv_r = _config(16, 42)
    registered = {".".join(path) for path in GRAPH_BEARING_PATHS}

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        if isinstance(value, dict):
            for key, item in value.items():
                out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        else:
            out[prefix] = value
        return out

    # The capability NAME embeds the cluster count, and every capability-bearing
    # path is one of R0217's registered seed-bearing paths, so both registers are
    # admissible here. Nothing outside the two registers may move.
    registered |= {".".join(path) for path in SEED_BEARING_PATHS}
    flat_left, flat_right = flatten(left), flatten(right)
    assert set(flat_left) == set(flat_right)
    differing = {
        key for key in flat_left if flat_left[key] != flat_right[key]
    }
    assert differing
    for key in differing:
        assert any(
            key == path or key.startswith(f"{path}.") for path in registered
        ), key


def test_the_horizon_is_the_ceil_rule_on_this_graphs_edge_count() -> None:
    for edges in (48_000_000, 48_344_648, 48_360_472, 49_000_000):
        expected = math.ceil(1_000_000 * edges / 603_086_368)
        assert successful_updates_for_edges(edges) == expected
        config, _sha, _inv = _config(4, 42, edges=edges)
        assert config["optimizer"]["successful_positive_lr_updates"] == expected
    # A different edge count legitimately yields a different horizon; that is
    # quantisation of the registered rule, not a deviation (review-0223-01).
    assert successful_updates_for_edges(48_344_648) != successful_updates_for_edges(
        48_360_472
    )


def test_a_graph_below_the_r0171_floors_is_refused() -> None:
    good = dict(
        clusters=8,
        degrees={"zero_degree_rows": 0},
        recall={"mean_recall_at_k": 0.97, "p10_recall_at_k": 0.87},
        edges=48_000_000,
        structural={
            "self_loop_entries": 0,
            "duplicate_entries": 0,
            "out_of_range_entries": 0,
            "rows_below_k": 0,
        },
    )
    assert validate_cluster_spill_graph(**good)["clusters"] == 8
    with pytest.raises(Round0228Error):
        validate_cluster_spill_graph(
            **{**good, "recall": {"mean_recall_at_k": 0.89, "p10_recall_at_k": 0.87}}
        )
    with pytest.raises(Round0228Error):
        validate_cluster_spill_graph(
            **{**good, "recall": {"mean_recall_at_k": 0.97, "p10_recall_at_k": 0.79}}
        )
    with pytest.raises(Round0228Error):
        validate_cluster_spill_graph(
            **{**good, "degrees": {"zero_degree_rows": 1}}
        )
    with pytest.raises(Round0228Error):
        validate_cluster_spill_graph(
            **{
                **good,
                "structural": {**good["structural"], "self_loop_entries": 3},
            }
        )


# --------------------------------------------------------------------------- #
# the statistics, with hand-checkable answers
# --------------------------------------------------------------------------- #
def test_permutation_enumerates_exactly_the_relabellings() -> None:
    small = exact_permutation_two_sample([1.0, 2.0, 3.0], list(range(4, 12)))
    assert small["relabellings"] == math.comb(11, 3)
    pooled = exact_permutation_two_sample(list(range(9)), list(range(20, 28)))
    assert pooled["relabellings"] == math.comb(17, 9)


def test_a_shifted_arm_is_detected_and_an_exchangeable_one_is_not() -> None:
    control = [0.0, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.0]
    far = exact_permutation_two_sample([10.0, 10.1, 9.9], control)
    assert far["p_mean_two_sided"] < 0.01
    assert far["cells_inside_control_observed_range"] == 0
    near = exact_permutation_two_sample([0.01, -0.01, 0.03], control)
    assert near["p_mean_two_sided"] > 0.2
    assert near["cells_inside_control_observed_range"] == 3


def test_a_wider_arm_is_caught_by_the_variance_statistic_not_the_mean() -> None:
    control = [1.0, 1.01, 0.99, 1.02, 0.98, 1.005, 0.995, 1.0]
    wide = exact_permutation_two_sample([0.5, 1.5, 1.0], control)
    # Exactly the nine relabellings that put BOTH extremes in the treatment arm
    # are at least as dispersed; that is the resolution ceiling of C(11,3), not a
    # weak signal, and the test pins it so nobody later reads 0.055 as a miss.
    assert wide["p_variance_ratio_one_sided"] == pytest.approx(9 / 165)
    assert wide["smallest_attainable_p"] == pytest.approx(1 / 165)
    assert wide["p_mean_two_sided"] > 0.5  # the mean is exactly where it was
    assert wide["variance_ratio"] > 100


def test_the_label_space_is_the_multinomial_not_the_permutation_space() -> None:
    assert len(list(_label_assignments([3, 3, 3]))) == 1680
    assert len(list(_label_assignments([2, 2]))) == math.comb(4, 2)


def test_a_perfect_trend_is_significant_and_a_flat_one_is_not() -> None:
    rising = exact_permutation_trend(
        {4: [1.0, 1.1, 1.2], 8: [2.0, 2.1, 2.2], 16: [3.0, 3.1, 3.2]}
    )
    assert rising["distinct_arrangements"] == 1680
    assert rising["pearson_r_vs_log2_c"] > 0.98
    assert rising["p_two_sided"] < 0.01
    assert rising["slope_per_doubling_of_c"] > 0.9
    flat = exact_permutation_trend(
        {4: [1.0, 2.0, 3.0], 8: [1.1, 2.1, 3.1], 16: [0.9, 1.9, 2.9]}
    )
    assert abs(flat["pearson_r_vs_log2_c"]) < 0.2
    assert flat["p_two_sided"] > 0.3


def _synthetic_families(offset: float = 0.0) -> dict[str, Any]:
    rng = np.random.default_rng(228)
    exact_cells = {
        str(seed): {
            metric: float(0.5 + 0.01 * rng.standard_normal())
            for metric in PANEL_METRICS
        }
        for seed in EXACT_FAMILY_SEEDS
    }
    cuvs_cells = {
        str(seed): {
            metric: float(0.5 + 0.01 * rng.standard_normal())
            for metric in PANEL_METRICS
        }
        for seed in (42, 43, 44)
    }
    candidate_cells = {
        str(clusters): {
            str(seed): {
                metric: float(
                    0.5 + offset * math.log2(clusters / 4.0) + 0.01 * rng.standard_normal()
                )
                for metric in PANEL_METRICS
            }
            for seed in SEEDS
        }
        for clusters in CLUSTER_COUNTS
    }
    exact_purity = {
        str(seed): {
            "k256": 1.0 + 0.01 * rng.standard_normal(),
            "k1024": 0.71 + 0.005 * rng.standard_normal(),
        }
        for seed in EXACT_FAMILY_SEEDS
    }
    candidate_purity = {
        str(clusters): {
            str(seed): {
                "k256": 1.0 + 0.01 * rng.standard_normal(),
                "k1024": 0.71 + 0.005 * rng.standard_normal(),
            }
            for seed in SEEDS
        }
        for clusters in CLUSTER_COUNTS
    }
    gates = {
        metric: {
            "mean_minus_2sd": {"floor": 0.4},
            "one_sided_tolerance_95_95": {"floor": 0.35},
            "two_sided_log_ratio_95_95": {
                "k2": 3.7685386134034156,
                "log_ratio_mean": 0.0086,
                "log_ratio_sample_sd_ddof1": 0.0144,
                "log_lower": -0.0457,
                "log_upper": 0.0630,
                "ratio_lower": 0.9553,
                "ratio_upper": 1.0650,
            },
        }
        for metric in PANEL_METRICS
    }
    return {
        "candidate_cells": candidate_cells,
        "exact_cells": exact_cells,
        "cuvs_cells": cuvs_cells,
        "tolerance_gates": gates,
        "candidate_purity_ratios": candidate_purity,
        "exact_purity_ratios": exact_purity,
    }


def test_the_comparison_publishes_every_cell_metric_and_test() -> None:
    result = compare_to_families(**_synthetic_families())
    assert result["adoption_claimed"] is False
    assert result["equivalence_claimed"] is False
    for metric in PANEL_METRICS:
        entry = result["per_metric"][metric]
        assert entry["exact_family"]["n"] == 8
        assert "trend_in_log2_c" in entry
        assert entry["pooled_candidates"]["n"] == 9
        for clusters in CLUSTER_COUNTS:
            cell_block = entry["by_clusters"][str(clusters)]
            assert set(cell_block["cells"]) == {str(seed) for seed in SEEDS}
            assert cell_block["permutation_vs_exact_family"]["relabellings"] == 165
            assert cell_block["permutation_vs_r0223_cuvs"]["relabellings"] == 20
    for metric in ("purity_fidelity_k256", "purity_fidelity_k1024"):
        unfolded = result["per_metric"][metric]["unfolded_two_sided"]
        assert unfolded["scale"].startswith("natural log")
        for clusters in CLUSTER_COUNTS:
            for cell in unfolded["by_clusters"][str(clusters)]["cells"].values():
                assert cell["direction"] in {
                    "over-separates (r > 1)",
                    "under-separates (r < 1)",
                    "matches high-D (r = 1)",
                }
                assert isinstance(cell["inside_band"], bool)


def test_a_planted_trend_is_recovered_by_the_comparison() -> None:
    result = compare_to_families(**_synthetic_families(offset=0.05))
    trend = result["per_metric"]["ffr"]["trend_in_log2_c"]
    assert trend["p_two_sided"] < 0.01
    assert trend["slope_per_doubling_of_c"] > 0.04


def test_the_comparison_refuses_a_wrong_shaped_family() -> None:
    payload = _synthetic_families()
    payload["exact_cells"].pop("49")
    with pytest.raises(Round0228Error):
        compare_to_families(**payload)


# --------------------------------------------------------------------------- #
# the geometry
# --------------------------------------------------------------------------- #
def test_a_planted_clump_is_found_and_a_uniform_map_is_not() -> None:
    rng = np.random.default_rng(7)
    uniform = rng.uniform(-10.0, 10.0, size=(200_000, 2)).astype(np.float32)
    flat = clump_profile(uniform, bins=128)
    clumped = uniform.copy()
    clumped[:20_000] = rng.normal(loc=(5.0, 5.0), scale=0.02, size=(20_000, 2))
    tight = clump_profile(clumped, bins=128)
    assert tight["max_bin_count"] > 20 * flat["max_bin_count"]
    assert tight["largest_component_rows"] > flat["largest_component_rows"]


def test_scatter_is_invariant_to_translation_rotation_and_scale() -> None:
    rng = np.random.default_rng(11)
    coordinates = rng.normal(size=(500, 2)).astype(np.float32)
    truth = np.stack(
        [(np.arange(500) + offset) % 500 for offset in range(1, 6)], axis=1
    ).astype(np.int32)
    rows = np.arange(0, 500, 5)
    base = true_neighbour_scatter(coordinates, truth, rows)
    angle = 0.7
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    moved = (coordinates @ rotation.T) * 3.5 + np.asarray([100.0, -50.0], np.float32)
    assert np.allclose(base, true_neighbour_scatter(moved, truth, rows), rtol=1e-4)
    with pytest.raises(Round0228GeometryError):
        map_scale(np.zeros((10, 2), dtype=np.float32))


def test_the_control_is_matched_on_density_deciles() -> None:
    rng = np.random.default_rng(13)
    cosine = rng.uniform(0.2, 0.9, size=50_000)
    # Loss concentrated in the sparse half, exactly the confound the match kills.
    lost = np.zeros(50_000, dtype=bool)
    sparse = np.argsort(cosine)[:20_000]
    lost[rng.choice(sparse, size=8_000, replace=False)] = True
    selection = density_matched_control(
        lost_mask=lost, kth_cosine=cosine, sample_rows=4_000, deciles=10, seed=1
    )
    assert selection["matched_exactly"] is True
    assert selection["decile_counts_lost"] == selection["decile_counts_control"]
    assert len(selection["lost_sample"]) == 4_000
    assert not set(selection["lost_sample"].tolist()) & set(
        selection["control_sample"].tolist()
    )


def test_the_difference_in_differences_is_zero_when_nothing_moved() -> None:
    lost = {name: [1.0, 1.1, 0.9] for name in ("a", "b", "c", "d", "e")}
    control = {name: [0.8, 0.9, 0.7] for name in ("a", "b", "c", "d", "e")}
    summary = displacement_summary(
        lost_scatter=lost,
        control_scatter=control,
        candidate_maps=["a", "b"],
        exact_maps=["c", "d", "e"],
    )
    assert abs(summary["difference_in_differences"]) < 1e-12
    lost["a"] = [2.0, 2.1, 1.9]
    moved = displacement_summary(
        lost_scatter=lost,
        control_scatter=control,
        candidate_maps=["a", "b"],
        exact_maps=["c", "d", "e"],
    )
    assert moved["difference_in_differences"] > 0.4
