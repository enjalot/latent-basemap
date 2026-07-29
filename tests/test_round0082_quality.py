from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.round0082_quality import (
    CONFIRMATION_SCHEMA,
    EXPECTED_NPROBE,
    EXPECTED_SHORTLIST_WIDTH,
    MEAN_RECALL_FLOOR,
    SOURCE_QUALIFICATION_IDENTITY,
    Round0082Error,
    load_policy_confirmation,
)
from experiments import prepare_round0082_queue, round0082_nodes


def _confirmation(
    *,
    source: dict[str, object],
    substrate: dict[str, object],
    eligibility: dict[str, object],
    filtered: dict[str, object],
    recall: float = 0.91,
) -> dict[str, object]:
    body = {
        "schema": CONFIRMATION_SCHEMA,
        "round_id": "0082",
        "validity_passed": True,
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "source_qualification": source,
        "source_qualification_identity": SOURCE_QUALIFICATION_IDENTITY,
        "substrate": substrate,
        "eligibility": eligibility,
        "filtered_index": filtered,
        "selected_policy": {
            "nprobe": EXPECTED_NPROBE,
            "shortlist_width": EXPECTED_SHORTLIST_WIDTH,
        },
        "quality": {
            "mean_recall_at_15_unambiguous": recall,
        },
        "checks": {"fresh_mean_recall_at_least_0_90": True},
    }
    return {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }


def test_confirmation_loader_accepts_exact_fresh_policy(
    tmp_path: Path,
) -> None:
    source = {"sha256": "a" * 64}
    substrate = {"sha256": "b" * 64}
    eligibility = {"sha256": "c" * 64}
    filtered = {"sha256": "d" * 64}
    receipt = _confirmation(
        source=source,
        substrate=substrate,
        eligibility=eligibility,
        filtered=filtered,
    )
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = expected_input_signature(str(path))
    loaded = load_policy_confirmation(
        str(path),
        expected_sha256=signature["sha256"],
        source_qualification_signature=source,
        substrate_signature=substrate,
        eligibility_signature=eligibility,
        filtered_index_signature=filtered,
    )
    assert loaded["receipt"]["quality"][
        "mean_recall_at_15_unambiguous"
    ] == 0.91


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nprobe", 192),
        ("shortlist_width", 512),
    ],
)
def test_confirmation_loader_rejects_policy_drift(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    source = {"sha256": "a" * 64}
    substrate = {"sha256": "b" * 64}
    eligibility = {"sha256": "c" * 64}
    filtered = {"sha256": "d" * 64}
    receipt = _confirmation(
        source=source,
        substrate=substrate,
        eligibility=eligibility,
        filtered=filtered,
    )
    receipt["selected_policy"][field] = value
    body = {
        key: item
        for key, item in receipt.items()
        if key != "identity_sha256"
    }
    receipt["identity_sha256"] = sha256_bytes(canonical_json(body))
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = expected_input_signature(str(path))
    with pytest.raises(Round0082Error):
        load_policy_confirmation(
            str(path),
            expected_sha256=signature["sha256"],
            source_qualification_signature=source,
            substrate_signature=substrate,
            eligibility_signature=eligibility,
            filtered_index_signature=filtered,
        )


def test_round0082_is_one_fresh_selected_policy_only_job() -> None:
    source = inspect.getsource(prepare_round0082_queue.prepare_round0082)
    assert "gpu_hours_cap=0.15" in source
    assert source.count(
        '"action": "confirm_balanced_120m_gpu_ivfpq_policy"'
    ) == 1
    assert '"one_selected_policy_only": True' in source
    assert '"no_policy_search": True' in source
    assert '"no_graph": True' in source
    assert '"no_training": True' in source
    assert '"required_reviews"] = ["0065", "0081"]' in source
    assert round0082_nodes.QUALITY_SAMPLE_ROWS == 8_192
    assert round0082_nodes.QUALITY_SEED == 82
    assert round0082_nodes.SOURCE_SAMPLE_SEED == 81
    assert EXPECTED_NPROBE == 128
    assert EXPECTED_SHORTLIST_WIDTH == 256
    assert MEAN_RECALL_FLOOR == 0.90


def test_round0082_accepts_only_an_issued_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_file = tmp_path / "round-0082.md"
    monkeypatch.setattr(
        prepare_round0082_queue,
        "ROUND_FILE",
        str(round_file),
    )
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    prepare_round0082_queue._require_issued_round()
    round_file.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remains draft"):
        prepare_round0082_queue._require_issued_round()


def test_round0082_modules_do_not_mutate_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0082_quality")
    importlib.import_module("experiments.round0082_nodes")
    importlib.import_module("experiments.prepare_round0082_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
