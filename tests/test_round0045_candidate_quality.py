from __future__ import annotations

import json

import pytest

from experiments import round0044_nodes as node


def _body(*, passed: bool) -> dict:
    return {
        "schema": "round0044-candidate-quality-sweep-v1",
        "round_id": "0045",
        "checks": {
            "structural_check": passed,
            "no_training_performed": True,
        },
        "training_performed": False,
    }


def test_r0045_persists_failed_validity_vector_without_raising(tmp_path) -> None:
    result = node._publish_candidate_receipt(
        active={"manifest": {"round_id": "0045"}},
        output=str(tmp_path),
        body=_body(passed=False),
    )
    receipt = tmp_path / "candidate-quality-sweep-v1.json"
    assert receipt.is_file()
    assert json.loads(receipt.read_text())["checks"]["structural_check"] is False
    assert result["receipt"]["canonical_path"] == str(receipt.resolve())


def test_old_r0044_guard_raises_only_after_persisting_receipt(tmp_path) -> None:
    with pytest.raises(
        node.Round0044Error,
        match="diagnostic receipt persisted",
    ):
        node._publish_candidate_receipt(
            active={"manifest": {"round_id": "0044"}},
            output=str(tmp_path),
            body=_body(passed=False),
        )
    receipt = tmp_path / "candidate-quality-sweep-v1.json"
    assert receipt.is_file()
    assert json.loads(receipt.read_text())["checks"]["structural_check"] is False


def test_r0045_handler_is_explicitly_allowlisted() -> None:
    source = node.run_job.__code__
    assert node.FOLLOWUP_ROUND_ID == "0045"
    assert "R0044/R0045 handler received another queue" in source.co_consts
