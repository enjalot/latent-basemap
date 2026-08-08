"""Contract tests for the R0217 MiniLM 2M four-seed family."""
from __future__ import annotations

import copy
import math

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0113_prompt_contrast import (
    BATCH_SIZE,
    NEGATIVE_RNG_SEED_OFFSET,
    POSITIVE_ROWS_PER_UPDATE,
)
from basemap.round0202_h4096_nested_dose_ladder import (
    FULL_GRAPH_EDGES,
    FULL_SUCCESSFUL_UPDATES,
    TARGET_POSITIVE_DRAWS_PER_EDGE,
)
from basemap.round0217_minilm_2m_seed_family import (
    ARCHITECTURE,
    CAPABILITIES,
    DIMENSION,
    GATE_REGISTERABLE_HERE,
    GRAPH_CAPABILITY,
    GRAPH_K,
    HIDDEN_DIMENSION,
    HIDDEN_LAYERS,
    LOW_DIM_KERNEL,
    MIN_PROBE_COORD_STD,
    OUTPUT_DIMENSION,
    ROUND_ID,
    ROWS,
    Round0217Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    USE_AMP,
    achieved_draws_per_edge,
    assert_family_differs_only_by_seed,
    capability_for_seed,
    dose_quantum,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
    validate_dose,
    validate_published_map,
)


def _signature(name: str, digest: str) -> dict[str, object]:
    return {
        "kind": "file",
        "canonical_path": f"/data/latent-basemap/runs/round-0216/{name}",
        "bytes": 1_024,
        "sha256": digest * 64,
    }


GRAPH_SIGNATURE = _signature("edges-k15-fuzzy.npz", "a")
MANIFEST_SIGNATURE = _signature("substrate-graph.json", "b")
SUBSTRATE_SIGNATURE = _signature("substrate.f32.npy", "c")


def _config(seed: int, **overrides: object) -> dict[str, object]:
    kwargs = {
        "seed": seed,
        "graph_signature": GRAPH_SIGNATURE,
        "graph_manifest_signature": MANIFEST_SIGNATURE,
        "substrate_signature": SUBSTRATE_SIGNATURE,
        "graph_edges": SEALED_DIRECTED_EDGES,
        "rows": ROWS,
    }
    kwargs.update(overrides)
    config, _sha = train_config(**kwargs)  # type: ignore[arg-type]
    return config


def test_family_is_exactly_four_seeds() -> None:
    assert SEEDS == (42, 43, 44, 45)
    assert CAPABILITIES == tuple(
        f"minilm-mixed-2m-map-seed{seed}-low-dose-v1" for seed in SEEDS
    )
    assert capability_for_seed(42) == "minilm-mixed-2m-map-seed42-low-dose-v1"
    with pytest.raises(Round0217Error):
        capability_for_seed(46)


def test_dose_is_derived_from_the_sealed_edge_count_not_hardcoded() -> None:
    """The registered R0184/R0202 low-dose rule, applied to R0216's graph."""
    updates = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    assert updates == -(
        -FULL_SUCCESSFUL_UPDATES * SEALED_DIRECTED_EDGES // FULL_GRAPH_EDGES
    )
    assert updates == 80_094
    achieved = achieved_draws_per_edge(
        updates=updates, edge_count=SEALED_DIRECTED_EDGES
    )
    quantum = dose_quantum(SEALED_DIRECTED_EDGES)
    assert quantum == POSITIVE_ROWS_PER_UPDATE / SEALED_DIRECTED_EDGES
    # The rule is a ceil, so the achieved dose lives on a lattice of spacing
    # `quantum`. At this edge count that spacing is 8.47e-06, which is why the
    # registered bound is one lattice step and not an arbitrary 1e-06.
    assert abs(achieved - TARGET_POSITIVE_DRAWS_PER_EDGE) <= quantum
    assert not math.isclose(
        achieved, TARGET_POSITIVE_DRAWS_PER_EDGE, rel_tol=1.0e-6, abs_tol=0.0
    )
    registered = validate_dose(updates=updates, edge_count=SEALED_DIRECTED_EDGES)
    assert registered["successful_updates"] == updates
    assert registered["active_graph_edges"] == SEALED_DIRECTED_EDGES
    assert registered["source_graph_edges"] == FULL_GRAPH_EDGES
    assert registered["source_successful_updates"] == FULL_SUCCESSFUL_UPDATES


def test_dose_check_rejects_an_off_by_one_horizon() -> None:
    exact = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    for delta in (-1, 1):
        with pytest.raises(Round0217Error):
            validate_dose(updates=exact + delta, edge_count=SEALED_DIRECTED_EDGES)


def test_recipe_is_the_minilm_precedent_at_bf16() -> None:
    config = _config(42)
    model = config["model"]
    optimizer = config["optimizer"]
    assert model["architecture"] == ARCHITECTURE == "residual_bottleneck"
    assert model["input_dimension"] == DIMENSION == 384
    assert model["hidden_dimension"] == HIDDEN_DIMENSION == 2048
    assert model["hidden_layers"] == HIDDEN_LAYERS == 3
    assert model["output_dimension"] == OUTPUT_DIMENSION == 2
    assert model["low_dim_kernel"] == LOW_DIM_KERNEL == "legacy_lp"
    assert model["use_batchnorm"] is False
    assert model["use_dropout"] is False
    assert optimizer["positive_target_mode"] == "binary"
    assert optimizer["use_amp"] == USE_AMP == "bf16"
    assert optimizer["batch_size"] == BATCH_SIZE
    assert optimizer["weighted_edge_sampling"] is True
    assert optimizer["successful_positive_lr_updates"] == successful_updates_for_edges(
        SEALED_DIRECTED_EDGES
    )
    assert config["graph"]["k"] == GRAPH_K == 15
    assert config["graph"]["capability"] == GRAPH_CAPABILITY
    assert config["graph"]["directed_edges"] == SEALED_DIRECTED_EDGES
    assert config["round_id"] == ROUND_ID


def test_this_round_registers_no_gate() -> None:
    assert GATE_REGISTERABLE_HERE is False
    for seed in SEEDS:
        assert _config(seed)["seed_family"]["gate_registerable_here"] is False


def test_the_four_cells_are_identical_except_for_the_seed() -> None:
    configs = {seed: _config(seed) for seed in SEEDS}
    family = assert_family_differs_only_by_seed(configs)
    assert family["cells"] == 4
    assert family["seeds"] == list(SEEDS)
    assert family["gate_registerable_here"] is False
    # One digest for all four cells once the seed-bearing fields are masked.
    assert len({seed_invariant_sha256(configs[seed]) for seed in SEEDS}) == 1
    # ...and four distinct configs before masking.
    assert len({
        sha256_bytes(canonical_json(configs[seed])) for seed in SEEDS
    }) == len(SEEDS)
    for seed in SEEDS:
        config = configs[seed]
        assert config["seed"] == seed
        assert config["optimizer"]["seed"] == seed
        assert config["optimizer"]["positive_rng_seed"] == seed
        assert config["optimizer"]["negative_rng_seed"] == (
            seed + NEGATIVE_RNG_SEED_OFFSET
        )
        stamp = config["execution"]["expected_pipeline_stamp"]
        assert stamp["positive_rng_seed"] == seed
        assert stamp["negative_rng_seed"] == seed + NEGATIVE_RNG_SEED_OFFSET
        assert config["capability"] == capability_for_seed(seed)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["model"].__setitem__("hidden_dimension", 4096),
        lambda c: c["optimizer"].__setitem__("use_amp", "fp16"),
        lambda c: c["optimizer"].__setitem__("learning_rate", 0.002),
        lambda c: c["optimizer"].__setitem__("successful_positive_lr_updates", 1),
        lambda c: c["graph"].__setitem__("k", 50),
        lambda c: c["input"].__setitem__("substrate_sha256", "z" * 64),
    ],
)
def test_any_non_seed_difference_fails_closed(mutate) -> None:
    configs = {seed: _config(seed) for seed in SEEDS}
    mutated = copy.deepcopy(configs[45])
    mutate(mutated)
    configs[45] = mutated
    with pytest.raises(Round0217Error):
        assert_family_differs_only_by_seed(configs)


def test_a_swapped_seed_field_fails_closed() -> None:
    configs = {seed: _config(seed) for seed in SEEDS}
    mutated = copy.deepcopy(configs[44])
    mutated["optimizer"]["negative_rng_seed"] = 45 + NEGATIVE_RNG_SEED_OFFSET
    configs[44] = mutated
    with pytest.raises(Round0217Error):
        assert_family_differs_only_by_seed(configs)


def test_family_must_be_all_four_seeds() -> None:
    configs = {seed: _config(seed) for seed in SEEDS[:3]}
    with pytest.raises(Round0217Error):
        assert_family_differs_only_by_seed(configs)


def test_config_refuses_a_graph_that_is_not_the_sealed_one() -> None:
    with pytest.raises(Round0217Error):
        _config(42, graph_edges=SEALED_DIRECTED_EDGES + 1)
    with pytest.raises(Round0217Error):
        _config(42, rows=ROWS + 1)
    with pytest.raises(Round0217Error):
        _config(46)


def test_published_map_check_rejects_collapse_and_nonfinite() -> None:
    import numpy as np

    healthy = np.random.default_rng(0).normal(size=(256, 2)) * 5.0
    report = validate_published_map(healthy)
    assert report["coordinates_finite"] is True
    assert report["collapsed"] is False
    assert min(report["per_axis_std"]) >= MIN_PROBE_COORD_STD

    collapsed = np.zeros((256, 2), dtype=np.float64)
    with pytest.raises(Round0217Error):
        validate_published_map(collapsed)

    one_axis = healthy.copy()
    one_axis[:, 1] = 3.0
    with pytest.raises(Round0217Error):
        validate_published_map(one_axis)

    broken = healthy.copy()
    broken[3, 0] = np.nan
    with pytest.raises(Round0217Error):
        validate_published_map(broken)

    with pytest.raises(Round0217Error):
        validate_published_map(np.zeros((4, 3), dtype=np.float64))
