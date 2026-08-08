"""Registered checks for the R0221 MiniLM 2M seed extension (seeds 46-49).

The load-bearing property is that the four new cells carry R0217's treatment
byte for byte outside the seed, because that is what lets R0222 pool eight cells
into one family. These tests assert it positively and then try to break it from
every direction the contract admits.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0113_prompt_contrast import NEGATIVE_RNG_SEED_OFFSET
from basemap.round0217_minilm_2m_seed_family import (
    SEED_BEARING_PATHS,
    SEEDS as R0217_SEEDS,
    expected_seed_bearing_values,
    seed_invariant_sha256,
    train_config as r0217_train_config,
)
from basemap.round0221_minilm_2m_seed_extension import (
    POOLED_SEEDS,
    R0217_SEED_INVARIANT_SHA256,
    REGISTERED_ACHIEVED_DRAWS_PER_EDGE,
    REGISTERED_SUCCESSFUL_UPDATES,
    ROWS,
    Round0221Error,
    SEALED_DIRECTED_EDGES,
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
    SEEDS,
    TEMPLATE_SEED,
    assert_extension_differs_only_by_seed,
    capability_for_seed,
    seed_bearing_values,
    successful_updates_for_edges,
    train_config,
    validate_full_population_map,
    validate_registered_dose,
)


GRAPH = SEALED_GRAPH_SIGNATURE
MANIFEST = SEALED_GRAPH_MANIFEST_SIGNATURE
SUBSTRATE = SEALED_SUBSTRATE_SIGNATURE


def _config(seed: int) -> dict:
    config, _sha = train_config(
        seed=seed,
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        substrate_signature=SUBSTRATE,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    return config


def test_round0221_seeds_and_pooled_family_are_registered() -> None:
    assert SEEDS == (46, 47, 48, 49)
    assert POOLED_SEEDS == (42, 43, 44, 45, 46, 47, 48, 49)
    assert len(set(POOLED_SEEDS)) == 8
    assert set(R0217_SEEDS).isdisjoint(SEEDS)
    assert TEMPLATE_SEED in R0217_SEEDS


def test_round0221_every_cell_reproduces_r0217_published_seed_invariant() -> None:
    for seed in SEEDS:
        assert seed_invariant_sha256(_config(seed)) == R0217_SEED_INVARIANT_SHA256


def test_round0221_cells_differ_from_r0217_only_in_the_seed_bearing_paths() -> None:
    """The strongest form of the claim: diff against R0217's own config bytes."""
    reference, _sha = r0217_train_config(
        seed=TEMPLATE_SEED,
        graph_signature=GRAPH,
        graph_manifest_signature=MANIFEST,
        substrate_signature=SUBSTRATE,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    for seed in SEEDS:
        mine = _config(seed)
        rebuilt = copy.deepcopy(mine)
        # Put R0217's canonical-cell values back into the nine seed-bearing
        # paths; everything else must then be byte-identical to R0217's config.
        for path, value in expected_seed_bearing_values(TEMPLATE_SEED).items():
            cursor = rebuilt
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
        assert canonical_json(rebuilt) == canonical_json(reference), seed


def test_round0221_seed_bearing_fields_take_the_new_seed() -> None:
    for seed in SEEDS:
        config = _config(seed)
        assert config["seed"] == seed
        assert config["capability"] == capability_for_seed(seed)
        assert config["optimizer"]["seed"] == seed
        assert config["optimizer"]["positive_rng_seed"] == seed
        assert config["optimizer"]["negative_rng_seed"] == (
            seed + NEGATIVE_RNG_SEED_OFFSET
        )
        stamp = config["execution"]["expected_pipeline_stamp"]
        assert stamp["positive_rng_seed"] == seed
        assert stamp["negative_rng_seed"] == seed + NEGATIVE_RNG_SEED_OFFSET
        assert config["seed_family"]["this_seed"] == seed
        assert config["seed_family"]["this_capability"] == capability_for_seed(seed)


def test_round0221_treatment_metadata_is_carried_verbatim_from_r0217() -> None:
    """Deliberate: the treatment bytes stay R0217's, including its round id.

    Changing `round_id` or `seed_family.seeds` here would be truer prose and a
    different treatment. The extended family is recorded in R0221's receipt, not
    in the trained config.
    """
    config = _config(SEEDS[0])
    assert config["round_id"] == "0217"
    assert config["seed_family"]["seeds"] == list(R0217_SEEDS)
    assert config["schema"].startswith("round0217-")


def test_round0221_family_assertion_accepts_the_four_cells() -> None:
    family = assert_extension_differs_only_by_seed(
        {seed: _config(seed) for seed in SEEDS}
    )
    assert family["seed_invariant_sha256"] == R0217_SEED_INVARIANT_SHA256
    assert family["matches_r0217_published_seed_invariant"] is True
    assert family["pooled_seed_family"] == list(POOLED_SEEDS)
    assert len(set(family["per_seed_config_sha256"].values())) == 4
    assert family["gate_registerable_here"] is False


@pytest.mark.parametrize(
    "path,value",
    [
        (("model", "hidden_dimension"), 4096),
        (("optimizer", "use_amp"), "fp16"),
        (("optimizer", "learning_rate"), 0.002),
        (("optimizer", "successful_positive_lr_updates"), 80_000),
        (("graph", "k"), 30),
        (("input", "substrate_sha256"), "d" * 64),
        (("seed_family", "seeds"), [46, 47, 48, 49]),
        (("round_id",), "0221"),
    ],
)
def test_round0221_any_treatment_drift_fails_closed(path, value) -> None:
    configs = {seed: _config(seed) for seed in SEEDS}
    cursor = configs[SEEDS[1]]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(Round0221Error):
        assert_extension_differs_only_by_seed(configs)


def test_round0221_rejects_a_wrong_seed_bearing_value() -> None:
    configs = {seed: _config(seed) for seed in SEEDS}
    configs[SEEDS[0]]["optimizer"]["negative_rng_seed"] = 1
    with pytest.raises(Round0221Error):
        assert_extension_differs_only_by_seed(configs)


def test_round0221_rejects_a_wrong_or_duplicated_cell_set() -> None:
    configs = {seed: _config(seed) for seed in SEEDS}
    with pytest.raises(Round0221Error):
        assert_extension_differs_only_by_seed(
            {seed: configs[seed] for seed in SEEDS[:3]}
        )
    duplicated = {seed: _config(SEEDS[0]) for seed in SEEDS}
    with pytest.raises(Round0221Error):
        assert_extension_differs_only_by_seed(duplicated)


def test_round0221_rejects_an_unregistered_seed() -> None:
    for seed in (42, 45, 50, 0, -1):
        with pytest.raises(Round0221Error):
            capability_for_seed(seed)
        with pytest.raises(Round0221Error):
            train_config(
                seed=seed,
                graph_signature=GRAPH,
                graph_manifest_signature=MANIFEST,
                substrate_signature=SUBSTRATE,
                graph_edges=SEALED_DIRECTED_EDGES,
                rows=ROWS,
            )


def test_round0221_seed_bearing_path_set_matches_r0217() -> None:
    assert set(seed_bearing_values(46)) == set(SEED_BEARING_PATHS)


def test_round0221_dose_must_land_on_the_registered_ceil_value() -> None:
    updates = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    assert updates == REGISTERED_SUCCESSFUL_UPDATES
    dose = validate_registered_dose(
        updates=updates, edge_count=SEALED_DIRECTED_EDGES
    )
    assert dose["successful_updates"] == REGISTERED_SUCCESSFUL_UPDATES
    assert dose["achieved_positive_draws_per_edge"] == (
        REGISTERED_ACHIEVED_DRAWS_PER_EDGE
    )
    assert dose["landed_on_registered_ceil_value"] is True
    # Off by one in either direction, and a different (even self-consistent)
    # graph, both abort.
    with pytest.raises(Exception):
        validate_registered_dose(
            updates=updates + 1, edge_count=SEALED_DIRECTED_EDGES
        )
    other_edges = 48_303_258  # R0216 queue-correction-2, superseded
    with pytest.raises(Exception):
        validate_registered_dose(
            updates=successful_updates_for_edges(other_edges),
            edge_count=other_edges,
        )


def test_round0221_full_population_map_check_requires_every_row() -> None:
    good = np.tile(np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32), (ROWS // 2, 1))
    published = validate_full_population_map(good)
    assert published["transform_rows"] == ROWS
    assert published["transform_rows_finite"] == ROWS
    assert published["full_population_finite"] is True
    assert published["collapsed"] is False

    short = good[: ROWS - 1]
    with pytest.raises(Round0221Error):
        validate_full_population_map(short)

    holed = good.copy()
    holed[ROWS // 3, 1] = np.nan
    with pytest.raises(Round0221Error):
        validate_full_population_map(holed)

    collapsed = np.zeros((ROWS, 2), dtype=np.float32)
    with pytest.raises(Exception):
        validate_full_population_map(collapsed)


def test_round0221_config_digests_are_four_distinct_values() -> None:
    digests = {
        sha256_bytes(canonical_json(_config(seed))) for seed in SEEDS
    }
    assert len(digests) == 4


def test_round0221_refuses_any_substrate_but_the_sealed_one() -> None:
    foreign = {**SEALED_SUBSTRATE_SIGNATURE, "sha256": "d" * 64}
    with pytest.raises(Round0221Error):
        train_config(
            seed=46,
            graph_signature=GRAPH,
            graph_manifest_signature=MANIFEST,
            substrate_signature=foreign,
            graph_edges=SEALED_DIRECTED_EDGES,
            rows=ROWS,
        )
    stale = {
        **SEALED_GRAPH_SIGNATURE,
        "canonical_path": SEALED_GRAPH_SIGNATURE["canonical_path"].replace(
            "queue-correction-3", "queue-correction-2"
        ),
    }
    with pytest.raises(Round0221Error):
        train_config(
            seed=46,
            graph_signature=stale,
            graph_manifest_signature=MANIFEST,
            substrate_signature=SUBSTRATE,
            graph_edges=SEALED_DIRECTED_EDGES,
            rows=ROWS,
        )


def test_round0221_family_matches_r0217_published_digest_on_the_sealed_inputs() -> None:
    family = assert_extension_differs_only_by_seed(
        {seed: _config(seed) for seed in SEEDS},
        expected_seed_invariant=R0217_SEED_INVARIANT_SHA256,
    )
    assert family["seed_invariant_sha256"] == R0217_SEED_INVARIANT_SHA256
    assert family["matches_r0217_published_seed_invariant"] is True
    assert family["checked_against_r0217_sealed_receipts"] is True
    with pytest.raises(Round0221Error):
        assert_extension_differs_only_by_seed(
            {seed: _config(seed) for seed in SEEDS},
            expected_seed_invariant="0" * 64,
        )
