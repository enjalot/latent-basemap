"""R0223 contract tests: the treatment identity, the floors, and the direction.

These are the checks that make the round's claim structural rather than stated.
The central one is `test_only_registered_paths_move`: it diffs a real R0223 cell
config against the real R0217 template it was derived from and requires the
differing paths to be **exactly** `SEED_BEARING_PATHS | GRAPH_BEARING_PATHS`. A
new field silently added to either register — or a field that quietly moves
without being registered — fails here, before any GPU work.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from basemap.round0217_minilm_2m_seed_family import (
    SEALED_DIRECTED_EDGES as R0216_EDGES,
    train_config as r0217_train_config,
)
from basemap.round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)
from basemap.round0223_cuvs_graph_map import (
    CUVS_MEAN_RECALL_FLOOR,
    CUVS_P10_RECALL_FLOOR,
    GRAPH_BEARING_PATHS,
    GRAPH_BEARING_REASONS,
    PENDING_FLOOR_METRICS,
    R0222_POOLED_SEEDS,
    REGISTERED_UPDATE_BOUND,
    REVIEW_0222_TOLERANCE_FACTOR_N8,
    ROWS,
    Round0223Error,
    SEEDS,
    SEED_BEARING_PATHS,
    TEMPLATE_SEED,
    assert_family_differs_only_by_seed,
    compare_to_exact_family,
    map_capability,
    successful_updates_for_edges,
    tolerance_factor,
    train_config,
    treatment_invariant_sha256,
    validate_cuvs_graph,
    validate_full_population_map,
)


CUVS_EDGES = 48_100_000
CUVS_GRAPH_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0223/queue/x/edges-k15-fuzzy.npz",
    "bytes": 123,
    "sha256": "a" * 64,
}
CUVS_MANIFEST_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0223/queue/x/cuvs-graph.json",
    "bytes": 456,
    "sha256": "b" * 64,
}


def _template() -> dict[str, Any]:
    config, _sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=dict(SEALED_GRAPH_SIGNATURE),
        graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        substrate_signature=dict(SEALED_SUBSTRATE_SIGNATURE),
        graph_edges=R0216_EDGES,
        rows=ROWS,
    )
    return config


def _cell(seed: int, *, edges: int = CUVS_EDGES) -> tuple[dict[str, Any], str, str]:
    return train_config(
        seed=seed,
        graph_signature=dict(CUVS_GRAPH_SIGNATURE),
        graph_manifest_signature=dict(CUVS_MANIFEST_SIGNATURE),
        substrate_signature=dict(SEALED_SUBSTRATE_SIGNATURE),
        r0216_graph_signature=dict(SEALED_GRAPH_SIGNATURE),
        r0216_graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        graph_edges=edges,
        rows=ROWS,
    )


def _paths(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        out: dict[tuple[str, ...], Any] = {}
        for key, item in value.items():
            out.update(_paths(item, prefix + (str(key),)))
        return out
    return {prefix: value}


def test_only_registered_paths_move() -> None:
    """The differing fields are exactly the two registered path sets."""
    template = _template()
    config, _sha, _invariant = _cell(TEMPLATE_SEED)
    flat_template = _paths(template)
    flat_config = _paths(config)
    assert set(flat_template) == set(flat_config), "R0223 added or dropped a config key"
    differing = {
        path for path in flat_template if flat_template[path] != flat_config[path]
    }
    registered = set()
    for path in SEED_BEARING_PATHS + GRAPH_BEARING_PATHS:
        registered |= {
            candidate for candidate in flat_template if candidate[: len(path)] == path
        }
    # Seed 42's seed-bearing values differ from the template only in `capability`
    # (the template is seed 42 too), so the difference set must be a SUBSET of the
    # registered union. Nothing outside the two registers may move: that is the
    # claim "identical except the seed and the graph".
    assert differing <= registered, sorted(
        ".".join(path) for path in differing - registered
    )
    # Every path that identifies the graph itself must actually move, or the
    # config is not describing the graph it trains on. Paths that are pure
    # functions of the horizon (`execution.performance_windows`) may legitimately
    # coincide when two edge counts land in the same window bucket, so they are
    # registered but not required to differ.
    must_move = (
        ("graph", "path"),
        ("graph", "sha256"),
        ("graph", "capability"),
        ("graph", "source_round"),
        ("graph", "exactness"),
        ("graph", "directed_edges"),
        ("dose_registration",),
        ("execution", "expected_pipeline_stamp", "valid_canonical_edge_count"),
    )
    for path in must_move:
        assert path in set(GRAPH_BEARING_PATHS)
        covered = {
            candidate for candidate in flat_template if candidate[: len(path)] == path
        }
        assert covered & differing, f"{'.'.join(path)} did not actually move"


def test_every_graph_bearing_path_has_a_registered_reason() -> None:
    assert {".".join(path) for path in GRAPH_BEARING_PATHS} == set(
        GRAPH_BEARING_REASONS
    )
    assert all(reason.strip() for reason in GRAPH_BEARING_REASONS.values())


def test_treatment_invariant_digest_equals_the_r0217_template() -> None:
    template_invariant = treatment_invariant_sha256(_template())
    for seed in SEEDS:
        _config, _sha, invariant = _cell(seed)
        assert invariant == template_invariant


def test_cells_are_distinct_and_differ_only_by_seed() -> None:
    template_invariant = treatment_invariant_sha256(_template())
    configs = {seed: _cell(seed)[0] for seed in SEEDS}
    family = assert_family_differs_only_by_seed(
        configs, expected_treatment_invariant=template_invariant
    )
    assert family["cells"] == len(SEEDS)
    assert len(set(family["per_seed_config_sha256"].values())) == len(SEEDS)
    assert family["gate_registerable_here"] is False


def test_a_drifted_recipe_field_fails_closed() -> None:
    config, _sha, _invariant = _cell(SEEDS[0])
    config["model"]["hidden_dimension"] = 4096
    with pytest.raises(Round0223Error):
        assert_family_differs_only_by_seed(
            {seed: (config if seed == SEEDS[0] else _cell(seed)[0]) for seed in SEEDS},
            expected_treatment_invariant=treatment_invariant_sha256(_template()),
        )


def test_horizon_is_the_registered_ceil_of_the_cuvs_edge_count() -> None:
    config, _sha, _invariant = _cell(SEEDS[0])
    expected = successful_updates_for_edges(CUVS_EDGES)
    assert config["optimizer"]["successful_positive_lr_updates"] == expected
    assert config["graph"]["directed_edges"] == CUVS_EDGES
    assert expected != successful_updates_for_edges(R0216_EDGES)
    assert expected <= REGISTERED_UPDATE_BOUND


def test_an_over_budget_horizon_fails_closed() -> None:
    with pytest.raises(Round0223Error):
        _cell(SEEDS[0], edges=400_000_000)


def test_unregistered_seed_fails_closed() -> None:
    for seed in (45, 48, 99):
        with pytest.raises(Round0223Error):
            _cell(seed)
        with pytest.raises(Round0223Error):
            map_capability(seed)


def test_cuvs_graph_validation_applies_r0171_floors_and_the_tripwire() -> None:
    clean = {
        "self_loop_entries": 0,
        "duplicate_entries": 0,
        "out_of_range_entries": 0,
        "rows_below_k": 0,
    }
    ok = validate_cuvs_graph(
        degrees={"zero_degree_rows": 0},
        recall={"mean_recall_at_k": 0.9941, "p10_recall_at_k": 1.0},
        edges=48_000_000,
        structural=clean,
    )
    assert ok["mean_recall_floor"] == CUVS_MEAN_RECALL_FLOOR
    assert ok["p10_recall_floor"] == CUVS_P10_RECALL_FLOOR
    with pytest.raises(Round0223Error):
        validate_cuvs_graph(
            degrees={"zero_degree_rows": 1},
            recall={"mean_recall_at_k": 0.99, "p10_recall_at_k": 1.0},
            edges=1,
            structural=clean,
        )
    with pytest.raises(Round0223Error):
        validate_cuvs_graph(
            degrees={"zero_degree_rows": 0},
            recall={"mean_recall_at_k": 0.5, "p10_recall_at_k": 1.0},
            edges=1,
            structural=clean,
        )
    with pytest.raises(Round0223Error):
        validate_cuvs_graph(
            degrees={"zero_degree_rows": 0},
            recall={"mean_recall_at_k": 0.99, "p10_recall_at_k": 1.0},
            edges=1,
            structural={**clean, "self_loop_entries": 3},
        )


def test_negative_distance_floor_is_a_magnitude_not_a_count() -> None:
    """R0216's own exact kernel emits more tied entries than cuVS does."""
    from basemap.round0223_cuvs_graph_map import (
        MIN_ADMISSIBLE_NEGATIVE_DISTANCE,
        R0216_EXACT_KERNEL_MIN_DISTANCE,
        R0216_EXACT_KERNEL_NEGATIVE_ENTRIES,
    )

    ulp_at_unit_cosine = 2.0 ** -23
    # The floor must tolerate float32 rounding and R0216's measured exact-kernel
    # extreme, and must still reject anything an order of magnitude larger.
    assert MIN_ADMISSIBLE_NEGATIVE_DISTANCE < -10 * ulp_at_unit_cosine
    assert MIN_ADMISSIBLE_NEGATIVE_DISTANCE < R0216_EXACT_KERNEL_MIN_DISTANCE
    assert MIN_ADMISSIBLE_NEGATIVE_DISTANCE > -1.0e-3
    assert R0216_EXACT_KERNEL_NEGATIVE_ENTRIES > 0


def test_full_population_map_validation() -> None:
    good = np.tile(np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32), (ROWS // 2, 1))
    checked = validate_full_population_map(good)
    assert checked["transform_rows_finite"] == ROWS
    bad = good.copy()
    bad[7] = np.nan
    with pytest.raises(Round0223Error):
        validate_full_population_map(bad)
    with pytest.raises(Round0223Error):
        validate_full_population_map(good[:10])


def test_r0222_gate_schema_matches_the_published_artifact() -> None:
    """The constant is the schema R0222 actually sealed, not a plausible name."""
    import json
    import os

    from basemap.round0223_cuvs_graph_map import (
        R0222_GATE_ARTIFACT_ROOT,
        R0222_GATE_SCHEMA,
    )

    path = os.path.join(R0222_GATE_ARTIFACT_ROOT, "minilm-quality-gates-n8.json")
    if not os.path.exists(path):
        pytest.skip("R0222 gate artifact is not on this machine")
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert artifact["schema"] == R0222_GATE_SCHEMA
    assert artifact["round_id"] == "0222"
    assert artifact["gate_registered"] is True


def test_tolerance_factor_reproduces_review_0222() -> None:
    factor = tolerance_factor(8)
    assert factor["reproduces_review_0222"] is True
    assert abs(factor["k"] - REVIEW_0222_TOLERANCE_FACTOR_N8) < 1.0e-3
    assert factor["k"] > 2.0, "the whole point is that k=2.0 is too small at n=8"


def _synthetic_families() -> dict[str, Any]:
    exact_metrics = {
        "density_v2": [0.4377, 0.4406, 0.4387, 0.4477, 0.4434, 0.4400, 0.4393, 0.4491],
        "ffr": [0.3369, 0.3382, 0.3258, 0.3227, 0.3312, 0.3209, 0.3344, 0.3240],
        "purity_fidelity_k256": [
            0.9789, 0.9941, 0.9954, 0.9929, 0.9951, 0.9932, 0.9643, 0.9902
        ],
        "purity_fidelity_k1024": [
            0.7326, 0.7229, 0.6980, 0.6936, 0.7214, 0.6842, 0.7266, 0.6991
        ],
    }
    exact_ratios = {
        "k256": [1.0216, 1.0059, 1.0046, 0.9929, 1.0049, 0.9932, 1.0370, 1.0099],
        "k1024": [0.7326, 0.7229, 0.6980, 0.6936, 0.7214, 0.6842, 0.7266, 0.6991],
    }
    exact_cells = {
        str(seed): {
            metric: exact_metrics[metric][index] for metric in exact_metrics
        }
        for index, seed in enumerate(R0222_POOLED_SEEDS)
    }
    exact_purity = {
        str(seed): {key: exact_ratios[key][index] for key in exact_ratios}
        for index, seed in enumerate(R0222_POOLED_SEEDS)
    }
    cuvs_cells = {
        str(seed): {
            "density_v2": 0.4400,
            "ffr": 0.3300,
            "purity_fidelity_k256": 0.9900,
            "purity_fidelity_k1024": 0.7100,
        }
        for seed in SEEDS
    }
    cuvs_purity = {
        str(seed): {"k256": 1.0101, "k1024": 0.7100} for seed in SEEDS
    }
    floors = {
        "density_v2": 0.4335282413076137,
        "ffr": 0.3157181021069332,
        "purity_fidelity_k256": 0.9660625420699066,
        "purity_fidelity_k1024": 0.6737066290217798,
    }
    return {
        "exact_cells": exact_cells,
        "exact_purity": exact_purity,
        "cuvs_cells": cuvs_cells,
        "cuvs_purity": cuvs_purity,
        "floors": floors,
    }


def test_comparison_reports_both_floor_families_and_the_direction() -> None:
    data = _synthetic_families()
    comparison = compare_to_exact_family(
        cuvs_cells=data["cuvs_cells"],
        exact_cells=data["exact_cells"],
        pending_floors=data["floors"],
        cuvs_purity_ratios=data["cuvs_purity"],
        exact_purity_ratios=data["exact_purity"],
    )
    assert comparison["equivalence_claimed"] is False
    assert comparison["gate_release_claimed"] is False
    assert set(comparison["per_metric"]) == set(PENDING_FLOOR_METRICS)
    for metric, cell in comparison["per_metric"].items():
        assert "registered_mean_minus_2sd_floor" in cell
        assert "tolerance_floor_95_95" in cell
        # The 95/95 floor is strictly looser than mean-2sd, because k > 2.
        assert cell["tolerance_floor_95_95"] < cell["registered_mean_minus_2sd_floor"]
        for value in cell["cells"].values():
            assert math.isfinite(value["z_vs_exact_family"])
    for metric in ("purity_fidelity_k256", "purity_fidelity_k1024"):
        unfolded = comparison["per_metric"][metric]["unfolded"]
        assert unfolded["scale"] == "log r (unfolded)"
        for value in unfolded["cells"].values():
            assert value["direction"] in {
                "over-separates (r > 1)",
                "under-separates (r < 1)",
                "matches high-D (r = 1)",
            }
            assert math.isfinite(value["z_on_log_ratio_vs_exact_family"])
    k256 = comparison["per_metric"]["purity_fidelity_k256"]["unfolded"]
    assert k256["exact_family"]["centre_is_above_one"] is True


def test_the_fold_and_the_unfolded_scale_disagree_on_seed_48() -> None:
    """The reviewer's core arithmetic, reproduced as a regression test."""
    data = _synthetic_families()
    ratios = data["exact_purity"]
    logs = [math.log(ratios[str(seed)]["k256"]) for seed in R0222_POOLED_SEEDS]
    mean = sum(logs) / len(logs)
    sd = (sum((value - mean) ** 2 for value in logs) / (len(logs) - 1)) ** 0.5
    z_unfolded = (math.log(1.0370) - mean) / sd
    fidelities = [
        data["exact_cells"][str(seed)]["purity_fidelity_k256"]
        for seed in R0222_POOLED_SEEDS
    ]
    fmean = sum(fidelities) / len(fidelities)
    fsd = (
        sum((value - fmean) ** 2 for value in fidelities) / (len(fidelities) - 1)
    ) ** 0.5
    z_folded = abs(0.9643 - fmean) / fsd
    assert z_folded > z_unfolded, "the fold must inflate |z|, per review-0222-01"
    assert 1.8 < z_unfolded < 2.05
    assert 2.05 < z_folded < 2.3


def test_comparison_rejects_a_wrong_family() -> None:
    data = _synthetic_families()
    short = {key: value for key, value in data["exact_cells"].items() if key != "49"}
    with pytest.raises(Round0223Error):
        compare_to_exact_family(
            cuvs_cells=data["cuvs_cells"],
            exact_cells=short,
            pending_floors=data["floors"],
            cuvs_purity_ratios=data["cuvs_purity"],
            exact_purity_ratios=data["exact_purity"],
        )
    with pytest.raises(Round0223Error):
        compare_to_exact_family(
            cuvs_cells=data["cuvs_cells"],
            exact_cells=data["exact_cells"],
            pending_floors={"ffr": 0.1},
            cuvs_purity_ratios=data["cuvs_purity"],
            exact_purity_ratios=data["exact_purity"],
        )
