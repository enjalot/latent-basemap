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
from basemap.round0072_quality import (
    MEAN_RECALL_FLOOR,
    NPROBE_GRID,
    QUALIFICATION_SCHEMA,
    Round0072Error,
    load_gpu_qualification,
)
from experiments import prepare_round0072_queue, round0072_nodes


def _qualification(
    *,
    substrate: dict[str, object],
    eligibility: dict[str, object],
    selected_nprobe: int = 40,
) -> dict[str, object]:
    rows = {
        str(value): {
            "nprobe": value,
            "passes_mean_floor": value >= selected_nprobe,
        }
        for value in NPROBE_GRID
    }
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "round_id": "0072",
        "validity_passed": True,
        "training_performed": False,
        "optimizer_updates": 0,
        "tier": "90m",
        "substrate": substrate,
        "eligibility": eligibility,
        "selected_nprobe": selected_nprobe,
        "rows_by_nprobe": rows,
        "quality": {
            "selected": {
                "nprobe": selected_nprobe,
                "passes_mean_floor": True,
                "mean_recall_at_15_unambiguous": 0.91,
            },
        },
        "candidate_universe": {
            "filtered_index": {"sha256": "a" * 64},
        },
        "checks": {"all": True},
    }
    return {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }


def test_qualification_loader_accepts_smallest_passing_nprobe(
    tmp_path: Path,
) -> None:
    substrate = {"sha256": "b" * 64}
    eligibility = {"sha256": "c" * 64}
    receipt = _qualification(
        substrate=substrate,
        eligibility=eligibility,
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = expected_input_signature(str(path))
    loaded = load_gpu_qualification(
        str(path),
        expected_sha256=signature["sha256"],
        substrate_signature=substrate,
        eligibility_signature=eligibility,
    )
    assert loaded["receipt"]["selected_nprobe"] == 40


def test_qualification_loader_rejects_nonminimal_selection(
    tmp_path: Path,
) -> None:
    substrate = {"sha256": "b" * 64}
    eligibility = {"sha256": "c" * 64}
    receipt = _qualification(
        substrate=substrate,
        eligibility=eligibility,
        selected_nprobe=48,
    )
    receipt["rows_by_nprobe"]["40"]["passes_mean_floor"] = True
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    receipt["identity_sha256"] = sha256_bytes(canonical_json(body))
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = expected_input_signature(str(path))
    with pytest.raises(Round0072Error):
        load_gpu_qualification(
            str(path),
            expected_sha256=signature["sha256"],
            substrate_signature=substrate,
            eligibility_signature=eligibility,
        )


def test_round0072_is_one_bounded_no_training_gpu_job() -> None:
    source = inspect.getsource(prepare_round0072_queue.prepare_round0072)
    assert "gpu_hours_cap=0.5" in source
    assert source.count(
        '"action": "qualify_balanced_90m_gpu_ivfpq"'
    ) == 1
    assert '"no_graph": True' in source
    assert '"no_training": True' in source
    assert '"no_scale_decision": True' in source
    assert '"required_reviews"] = ["0059", "0069", "0071"]' in source
    assert "966c7782da5ef9142088eeab114c8d5b7b7086ae981a7c0ded226725095b4476" in source
    assert "minilm-balanced-30m-45m-60m-scale-geometry-v1" in source
    assert round0072_nodes.QUALITY_SAMPLE_ROWS == 4_096
    assert MEAN_RECALL_FLOOR == 0.90


def test_round0072_accepts_only_an_issued_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_file = tmp_path / "round-0072.md"
    monkeypatch.setattr(
        prepare_round0072_queue,
        "ROUND_FILE",
        str(round_file),
    )
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    prepare_round0072_queue._require_issued_round()
    round_file.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remains draft"):
        prepare_round0072_queue._require_issued_round()


def test_round0072_modules_do_not_mutate_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0072_quality")
    importlib.import_module("experiments.round0072_nodes")
    importlib.import_module("experiments.prepare_round0072_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
