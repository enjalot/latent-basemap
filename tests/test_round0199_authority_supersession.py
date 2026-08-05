from __future__ import annotations

import inspect

import pytest

from basemap.round0199_grease_batch_stable import (
    ROUND_ID,
    Round0199Error,
    diagnose_execution,
)
from experiments import prepare_round0199_queue


def _execution() -> dict:
    return {
        "schema": "round0196-grease-batch-stable-cpu-execution-v1",
        "device": "cpu",
        "source_checkpoint_round": "0181",
        "query_rows": 20_000,
        "dimension": 768,
        "probe_rows": 256,
        "reload_tolerance": 1e-4,
        "candidates": {
            "baseline": {
                "grease_batch_max_abs_error": 0.001,
                "numap_batch_max_abs_error": 0.001,
            },
            "fixed_grease": {
                "grease_batch_max_abs_error": 0.0,
                "numap_batch_max_abs_error": 0.0,
            },
            "fixed_grease_and_pumap": {
                "grease_batch_max_abs_error": 0.0,
                "numap_batch_max_abs_error": 0.0,
            },
        },
        "selected_patch": "fixed-256-row-grease-network",
    }


def test_replacement_keeps_frozen_f1_decision_contract() -> None:
    decision = diagnose_execution(_execution())
    assert ROUND_ID == "0199"
    assert decision["passed"] is True
    assert decision["f2_gpu_baseline_activated"] is True
    assert decision["additional_debug_or_f4_authorized"] is False


def test_replacement_translates_contract_failures() -> None:
    value = _execution()
    value["dimension"] = 384
    with pytest.raises(Round0199Error, match="R0199"):
        diagnose_execution(value)


def test_queue_authority_is_the_protocol_value() -> None:
    source = inspect.getsource(prepare_round0199_queue)
    assert 'execution_authority="autonomous-cpu"' in source
    assert "owner-campaign-cpu" not in source
    assert 'frontmatter.get("supersedes") != ["0196"]' in source
    assert '"handler_module": "experiments.round0199_nodes"' in source
