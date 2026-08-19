"""CPU-only contract tests for the matched 50M fneg-off baseline."""
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from basemap import baseline_50m_fneg_off as C
import experiments.prepare_baseline_50m_fneg_off_queue as Q


def _config(seed=C.CANONICAL_SEED):
    return C.control_train_config(
        graph_signature={"canonical_path": "/sealed/graph.npz", "sha256": "b" * 64},
        graph_manifest_signature={
            "canonical_path": "/sealed/graph.json",
            "sha256": "c" * 64,
        },
        substrate_signature={
            "canonical_path": "/sealed/substrate.npy",
            "sha256": "a" * 64,
        },
        graph_edges=C.SEALED_DIRECTED_EDGES,
        rows=C.ROWS,
        seed=seed,
    )[0]


def test_control_is_the_registered_50m_x2_host_int8_recipe_with_fneg_off():
    config = _config()
    recipe = C.assert_registered_control(config)
    assert config["optimizer"]["fneg_weight"] == 0.0
    assert recipe["fneg_active"] is False
    assert recipe["loss_branch"] == "unweighted_binary_cross_entropy"
    assert recipe["dose_multiplier"] == 2
    assert recipe["rows"] == 50_000_000
    assert recipe["x_residency"] == "host_int8"
    assert recipe["only_treatment_delta"] == {
        "path": "optimizer.fneg_weight",
        "parent": 1.0,
        "control": 0.0,
    }


def test_control_predicate_refuses_a_second_training_change():
    config = _config()
    config["optimizer"]["learning_rate"] *= 2
    with pytest.raises(C.Baseline50MRecipeError):
        C.assert_registered_control(config)


def test_registered_seed_family_shares_one_masked_recipe():
    family = C.assert_family_shares_one_recipe({seed: _config(seed) for seed in C.SEEDS})
    assert family["seeds"] == [42, 43, 44]
    assert family["n"] == 3
    assert len(set(family["per_seed"].values())) == 1


def test_recipe_refusal_controls_all_fire():
    controls = C.recipe_refusal_controls()
    assert controls["every_planted_defect_was_refused"] is True
    assert controls["the_honest_control_still_passes"] is True
    assert {entry["control"] for entry in controls["controls"]} == {
        "fneg_left_on",
        "dose_changed",
        "weighted_sampling",
        "fp16_residency",
    }


def test_control_source_is_in_the_training_closure():
    assert "basemap.baseline_50m_fneg_off" in C.TRAIN_CLOSURE_MODULES
    assert len(C.TRAIN_CLOSURE_MODULES) == len(set(C.TRAIN_CLOSURE_MODULES))


def test_queue_budget_matches_documented_fourteen_hours_per_seed():
    assert Q.TRAIN_P90_WALL_S == 14 * 3600
    assert Q.PANEL_P90_WALL_S == 3 * 3600


def test_unregistered_seed_is_refused():
    config = copy.deepcopy(_config())
    config["seed"] = 45
    with pytest.raises(C.Baseline50MRecipeError):
        C.assert_registered_control(config)
