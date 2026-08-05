from __future__ import annotations

import inspect

import pytest

from basemap.round0200_grease_batch_stable import (
    ROUND_ID,
    Round0200Error,
    diagnose_execution,
)
from experiments import prepare_round0200_queue


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
    assert ROUND_ID == "0200"
    assert decision["passed"] is True
    assert decision["f2_gpu_baseline_activated"] is True
    assert decision["additional_debug_or_f4_authorized"] is False


def test_replacement_translates_contract_failures() -> None:
    value = _execution()
    value["dimension"] = 384
    with pytest.raises(Round0200Error, match="R0200"):
        diagnose_execution(value)


def test_queue_authority_is_the_protocol_value() -> None:
    source = inspect.getsource(prepare_round0200_queue)
    assert 'execution_authority="autonomous-cpu"' in source
    assert "owner-campaign-cpu" not in source
    assert '_frontmatter_list(frontmatter, "supersedes") != ["0199"]' in source
    assert '"handler_module": "experiments.round0200_nodes"' in source


def test_issued_round_parses_real_frontmatter_lists(tmp_path, monkeypatch) -> None:
    round_file = tmp_path / "round-0200-2026-08-05.md"
    round_file.write_text(
        """---
round_id: "0200"
status: issued
base_commit: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
execution_authority: autonomous-cpu
required_reviews: ["0181"]
supersedes: ["0199"]
---
# Fixture
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare_round0200_queue, "ROUND_FILE", str(round_file))
    signature = prepare_round0200_queue._issued_round("a" * 40)
    assert signature["canonical_path"] == str(round_file)
