"""CPU-only contract tests for the pinned upstream ParamRepulsor baseline."""
import copy
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from basemap import paramrepulsor_baseline as P
import experiments.prepare_paramrepulsor_2m_queue as Q


def test_recipe_records_upstream_identity_and_defaults():
    recipe = P.recipe(42)
    assert P.assert_registered_recipe(recipe) == recipe
    assert recipe["implementation"] == {
        "kind": "unmodified_upstream_package",
        "repository": "https://github.com/hyhuang00/ParamRepulsor",
        "commit": "be8df72b1ac9041be3aae3d99f16f0d392b492dc",
        "package": "parampacmap",
        "version": "0.1.1rc0",
        "license": "Apache-2.0",
    }
    assert recipe["estimator"] == P.UPSTREAM_DEFAULTS
    assert recipe["rows"] == 2_000_000
    assert recipe["dimension"] == 384


def test_registered_family_shares_one_seed_invariant():
    invariants = {P.seed_invariant_sha256(P.recipe(seed)) for seed in P.SEEDS}
    assert invariants and len(invariants) == 1


def test_recipe_predicate_refuses_a_post_hoc_default_change():
    recipe = copy.deepcopy(P.recipe(42))
    recipe["estimator"]["n_neighbors"] = 15
    with pytest.raises(P.ParamRepulsorBaselineError):
        P.assert_registered_recipe(recipe)


def test_source_closure_covers_device_selection_and_algorithm_modules():
    assert set(P.UPSTREAM_SOURCE_CLOSURE) == {
        "__init__.py",
        "parampacmap.py",
        "training.py",
        "models/__init__.py",
        "models/dataset.py",
        "models/module.py",
        "utils/__init__.py",
        "utils/data.py",
        "utils/utils.py",
    }
    assert all(len(digest) == 64 for digest in P.UPSTREAM_SOURCE_CLOSURE.values())


def test_environment_lock_pins_the_upstream_commit_and_runtime():
    lock = Path(Q.ENV_LOCK).read_text(encoding="utf-8")
    assert f"ParamRepulsor.git@{P.UPSTREAM_COMMIT}" in lock
    assert "torch==2.5.1+cu124" in lock
    assert "numpy==2.0.2" in lock
    assert P.EXPECTED_ENVIRONMENT["tqdm"] == "4.67.1"


def test_default_queue_is_one_measured_cost_pilot():
    assert P.CANONICAL_SEED == 42
    assert P.SEEDS == (42, 43, 44)
    assert Q.TRAIN_P90_WALL_S == 30 * 3600
    assert Q.PANEL_P90_WALL_S == 2 * 3600
