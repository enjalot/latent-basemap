from __future__ import annotations

import importlib
import inspect
import json

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0066_quality import (
    NPROBE_GRID,
    Round0066Error,
    load_scale_decision,
)


@pytest.mark.parametrize(
    ("advance", "tier"),
    [(False, "45m"), (True, "120m")],
)
def test_scale_decision_selects_exactly_one_registered_tier(
    tmp_path,
    advance: bool,
    tier: str,
) -> None:
    body = {
        "schema": "round0064-scale-geometry-comparison-v1",
        "round_id": "0064",
        "decision": {
            "advance_to_120m_scale_rung": advance,
            "bisect_at_45m_if_false": not advance,
        },
    }
    receipt = {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }
    path = tmp_path / "scale-comparison.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = __import__(
        "basemap.artifact_identity",
        fromlist=["expected_input_signature"],
    ).expected_input_signature(str(path))
    selected = load_scale_decision(
        str(path),
        expected_sha256=signature["sha256"],
    )
    assert selected["tier"] == tier


def test_ambiguous_scale_decision_is_rejected(tmp_path) -> None:
    body = {
        "schema": "round0064-scale-geometry-comparison-v1",
        "round_id": "0064",
        "decision": {
            "advance_to_120m_scale_rung": True,
            "bisect_at_45m_if_false": True,
        },
    }
    receipt = {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }
    path = tmp_path / "scale-comparison.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = __import__(
        "basemap.artifact_identity",
        fromlist=["expected_input_signature"],
    ).expected_input_signature(str(path))
    with pytest.raises(Round0066Error):
        load_scale_decision(
            str(path),
            expected_sha256=signature["sha256"],
        )


def test_nprobe_grid_is_ordered_and_covers_current_60m_policy() -> None:
    assert NPROBE_GRID == tuple(sorted(set(NPROBE_GRID)))
    assert 40 in NPROBE_GRID
    assert NPROBE_GRID[-1] == 96


def test_round0066_is_one_bounded_no_training_gpu_job() -> None:
    from experiments import prepare_round0066_queue

    source = inspect.getsource(
        prepare_round0066_queue.prepare_round0066
    )
    assert "gpu_hours_cap=1.0" in source
    assert source.count('"action": "qualify_next_rung_gpu_ivfpq"') == 1
    assert '"no_graph": True' in source
    assert '"no_training": True' in source
    assert '"no_scale_decision": True' in source
    assert "tier = decision" in source


def test_round0066_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0066_quality")
    importlib.import_module("experiments.round0066_nodes")
    importlib.import_module("experiments.prepare_round0066_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
