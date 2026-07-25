from __future__ import annotations

import json

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0049_program import Round0049Error
from experiments import prepare_round0050_queue, round0050_nodes
from experiments.round0049_nodes import (
    _seal,
    _validate_shard,
)


def test_round0050_wrapper_binds_round_and_action(monkeypatch):
    observed = {}

    def fake(active, job):
        observed["active"] = active
        observed["job"] = job
        return {"ok": True}

    monkeypatch.setattr(round0050_nodes, "run_build_graph", fake)
    active = {
        "manifest": {"round_id": "0050"},
        "job": {"action": "build_graph"},
    }
    assert round0050_nodes.run_job(active) == {"ok": True}
    assert observed["job"] == active["job"]
    with pytest.raises(Round0049Error):
        round0050_nodes.run_job({
            "manifest": {"round_id": "0049"},
            "job": {"action": "build_graph"},
        })
    with pytest.raises(Round0049Error):
        round0050_nodes.run_job({
            "manifest": {"round_id": "0050"},
            "job": {"action": "train"},
        })


def test_resumed_graph_shard_is_bound_to_successor_round(tmp_path):
    target = tmp_path / "targets.npy"
    value = np.zeros((2, 15), dtype="<i4")
    np.save(target, value, allow_pickle=False)
    target.chmod(0o444)
    body = {
        "schema": "round0049-exact-rerank-graph-shard-v2",
        "round_id": "0050",
        "shard": 0,
        "start": 0,
        "stop": 2,
        "retained_sources": 2,
        "excluded_sources": 0,
        "valid_edges": 30,
        "nprobe": 16,
        "search_width": 128,
        "index_search_width": 129,
        "selected_neighbors": 15,
        "exact_rerank": True,
        "self_returned": 2,
        "search_seconds": 0.1,
        "rerank_seconds": 0.05,
        "wall_seconds": 0.2,
        "targets": expected_input_signature(str(target)),
    }
    receipt = _seal(body)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    assert _validate_shard(
        target_path=str(target),
        receipt_path=str(receipt_path),
        start=0,
        stop=2,
        nprobe=16,
        round_id="0050",
    )["round_id"] == "0050"
    with pytest.raises(Round0049Error):
        _validate_shard(
            target_path=str(target),
            receipt_path=str(receipt_path),
            start=0,
            stop=2,
            nprobe=16,
            round_id="0049",
        )


def test_round0050_binds_post_r0049_protocol_dates():
    assert prepare_round0050_queue.ROUND_FILE.endswith(
        "round-0050-2026-07-26.md"
    )
    source = __import__("inspect").getsource(
        prepare_round0050_queue.prepare_round0050
    )
    assert "review-0049-2026-07-26.md" in source
    assert "review-0049-2026-07-25.md" not in source
