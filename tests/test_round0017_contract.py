"""Fixture-safe CPU checks for the thin Round 0017 metadata adapter."""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

from basemap.round0016_program import (
    NODES as ROUND0016_NODES,
    TRAIN_CONFIG as ROUND0016_TRAIN_CONFIG,
)
from basemap.round0017_program import (
    BASE_COMMIT, BASE_TREE, NODES, ROUND_FILE, ROUND_SHA256,
    SEQUENCED_REVIEW_FILE, SEQUENCED_REVIEW_SHA256, TRAIN_CONFIG,
    derived_node_policy, round0017_release_chain,
)
from basemap.roundwatch_gate import (
    ROUNDWATCH_AUTHORITY_VALUES, _declared_authority,
)
from basemap.source_closure import (
    ROUND0016_RUNTIME_ENTRYPOINTS, ROUND0017_RUNTIME_ENTRYPOINTS,
    runtime_source_closure,
)
from experiments import run_round0014_node


ROOT = Path(__file__).resolve().parents[1]


def test_round0017_binds_exact_contract_review_and_one_commit_release():
    assert hashlib.sha256(Path(ROUND_FILE).read_bytes()).hexdigest() == ROUND_SHA256
    assert hashlib.sha256(Path(SEQUENCED_REVIEW_FILE).read_bytes()).hexdigest() == \
        SEQUENCED_REVIEW_SHA256
    assert BASE_COMMIT == "af7283be749e62e6e18a86f88b11a4d80ea8ebb7"
    assert BASE_TREE == "76e29c3055efdfe2c2c384b4c3d13c612d205278"
    release = "c" * 40
    evidence = round0017_release_chain(release)
    assert evidence["implementation_commits"] == [release]
    assert evidence["ancestry"] == [BASE_COMMIT, release]
    assert evidence["commits_after_base"] == 1


def test_round0017_science_and_training_accounting_path_are_unchanged():
    prior = copy.deepcopy(ROUND0016_TRAIN_CONFIG)
    current = copy.deepcopy(TRAIN_CONFIG)
    assert prior.pop("schema") == "round0016-production-config-v1"
    assert current.pop("schema") == "round0017-production-config-v1"
    assert current == prior
    assert [(vars(node)) for node in NODES] == [vars(node) for node in ROUND0016_NODES]
    assert TRAIN_CONFIG["optimizer"]["seed"] == 42
    assert TRAIN_CONFIG["optimizer"]["successful_positive_lr_updates"] == 500_000
    assert TRAIN_CONFIG["graph"]["sampling"] == "uniform-over-directed-edges"
    assert TRAIN_CONFIG["graph"]["with_replacement"] is True
    assert TRAIN_CONFIG["execution"]["required_pipeline"] == "device_uniform"
    assert all(derived_node_policy(node)["retry_count"] == 0 for node in NODES)
    source = inspect.getsource(run_round0014_node._run_train)
    for receipt_field in (
            '"positive_lr_optimizer_steps": 500_000',
            '"optimizer_steps_succeeded": 500_000',
            '"pipeline_pipeline": "device_uniform"',
            '"pipeline_uniform_with_replacement": True'):
        assert receipt_field in source


def test_round0017_dispatch_and_source_closure_are_target_specific():
    historical = runtime_source_closure(str(ROOT), ROUND0016_RUNTIME_ENTRYPOINTS)
    current = runtime_source_closure(str(ROOT), ROUND0017_RUNTIME_ENTRYPOINTS)
    assert not any("round0017" in path for path in historical)
    assert {
        "basemap/round0017_admission.py", "basemap/round0017_program.py",
        "basemap/round0017_staging.py", "experiments/run_round0017_node.py",
        "basemap/round0016_service.py",
    }.issubset(current)
    run_round0014_node.configure_round0017()
    assert run_round0014_node.ROUND_ID == "0017"
    assert run_round0014_node.RUNTIME_SCRIPT == "experiments/run_round0017_node.py"


def test_round0017_uses_standing_autonomous_admission():
    assert "autonomous-gpu" in ROUNDWATCH_AUTHORITY_VALUES
    assert _declared_authority({"execution_authority": "autonomous-gpu"}) == \
        "autonomous-gpu"
    queue_source = (ROOT / "experiments" / "prepare_round0017_queue.py").read_text()
    admission_source = (ROOT / "basemap" / "round0017_admission.py").read_text()
    assert '"execution_authority": "autonomous-gpu"' in queue_source
    assert 'data["execution_authority"] != "autonomous-gpu"' in admission_source
