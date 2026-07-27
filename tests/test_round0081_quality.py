from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.round0065_substrates import subset_spec
from basemap.round0081_quality import (
    MEAN_RECALL_FLOOR,
    POLICY_GRID,
    QUALIFICATION_SCHEMA,
    Round0081Error,
    _selected_cell,
    cell_key,
    load_gpu_policy_qualification,
)
from experiments import prepare_round0081_queue, round0081_nodes


def _cell(
    *,
    nprobe: int,
    width: int,
    recall: float,
    wall_per_query: float,
) -> dict[str, object]:
    return {
        "nprobe": nprobe,
        "shortlist_width": width,
        "mean_recall_at_15_unambiguous": recall,
        "passes_mean_floor": recall >= MEAN_RECALL_FLOOR,
        "benchmark": (
            {"median_wall_seconds_per_query": wall_per_query}
            if recall >= MEAN_RECALL_FLOOR
            else None
        ),
    }


def _qualification(
    *,
    substrate: dict[str, object],
    eligibility: dict[str, object],
    filtered_index: dict[str, object],
) -> dict[str, object]:
    cells = {
        cell_key(128, 128): _cell(
            nprobe=128,
            width=128,
            recall=0.872,
            wall_per_query=0.001,
        ),
        cell_key(192, 128): _cell(
            nprobe=192,
            width=128,
            recall=0.905,
            wall_per_query=0.003,
        ),
        cell_key(128, 256): _cell(
            nprobe=128,
            width=256,
            recall=0.915,
            wall_per_query=0.002,
        ),
    }
    selected = _selected_cell({"cells": cells})
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "round_id": "0081",
        "validity_passed": True,
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "substrate": substrate,
        "eligibility": eligibility,
        "filtered_index": filtered_index,
        "selected": selected,
        "cells": cells,
        "checks": {"all": True},
    }
    return {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }


def test_policy_loader_accepts_fastest_passing_cell(
    tmp_path: Path,
) -> None:
    substrate = {"sha256": "a" * 64}
    eligibility = {"sha256": "b" * 64}
    filtered = {"sha256": "c" * 64}
    receipt = _qualification(
        substrate=substrate,
        eligibility=eligibility,
        filtered_index=filtered,
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = expected_input_signature(str(path))
    loaded = load_gpu_policy_qualification(
        str(path),
        expected_sha256=signature["sha256"],
        substrate_signature=substrate,
        eligibility_signature=eligibility,
        filtered_index_signature=filtered,
    )
    assert loaded["receipt"]["selected"]["nprobe"] == 128
    assert loaded["receipt"]["selected"]["shortlist_width"] == 256


def test_policy_loader_rejects_nonfastest_selection(
    tmp_path: Path,
) -> None:
    substrate = {"sha256": "a" * 64}
    eligibility = {"sha256": "b" * 64}
    filtered = {"sha256": "c" * 64}
    receipt = _qualification(
        substrate=substrate,
        eligibility=eligibility,
        filtered_index=filtered,
    )
    receipt["selected"] = receipt["cells"][cell_key(192, 128)]
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    receipt["identity_sha256"] = sha256_bytes(canonical_json(body))
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    signature = expected_input_signature(str(path))
    with pytest.raises(Round0081Error):
        load_gpu_policy_qualification(
            str(path),
            expected_sha256=signature["sha256"],
            substrate_signature=substrate,
            eligibility_signature=eligibility,
            filtered_index_signature=filtered,
        )


def test_round0081_grid_and_universe_are_fixed() -> None:
    spec = subset_spec("120m")
    assert round0081_nodes.ROW_COUNT == 120_000_000
    assert round0081_nodes.INTERVALS == tuple(spec["intervals"])
    assert round0081_nodes.ELIGIBILITY_SUMMARY == (
        spec["eligibility_summary"]
    )
    assert round0081_nodes.QUALITY_SAMPLE_ROWS == 4_096
    assert round0081_nodes.QUALITY_SEED == 81
    assert round0081_nodes.BENCHMARK_ROWS == 10_000
    assert round0081_nodes.BENCHMARK_REPEATS == 3
    assert MEAN_RECALL_FLOOR == 0.90
    assert POLICY_GRID == (
        (128, 128),
        (192, 128),
        (256, 128),
        (384, 128),
        (512, 128),
        (128, 256),
        (192, 256),
        (256, 256),
        (384, 256),
        (512, 256),
        (128, 512),
        (192, 512),
        (256, 512),
        (384, 512),
    )


def test_round0081_is_one_bounded_no_training_gpu_job() -> None:
    source = inspect.getsource(prepare_round0081_queue.prepare_round0081)
    assert "gpu_hours_cap=0.5" in source
    assert source.count(
        '"action": "qualify_balanced_120m_gpu_ivfpq_policy"'
    ) == 1
    assert '"no_graph": True' in source
    assert '"no_training": True' in source
    assert '"no_120m_map_quality_claim": True' in source
    assert '"required_reviews"] = ["0065", "0076", "0077"]' in source
    assert (
        "ac7ef23b2a938f0a3d971cf3c8fb95afc255908e7ddf117d769d2c59baf09f53"
        in source
    )
    assert (
        "3ca5548a552bab6b13a01009eed3d3f9a35151ee42370b98f3a0b849633561a8"
        not in source
    )


def test_round0081_accepts_only_an_issued_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_file = tmp_path / "round-0081.md"
    monkeypatch.setattr(
        prepare_round0081_queue,
        "ROUND_FILE",
        str(round_file),
    )
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    prepare_round0081_queue._require_issued_round()
    round_file.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remains draft"):
        prepare_round0081_queue._require_issued_round()


def test_round0081_queries_are_normalized() -> None:
    encoded = np.asarray([[3, 4], [0, 5]], dtype=np.int8)
    scales = np.asarray([2, 3], dtype=np.float16)
    values = round0081_nodes._queries(
        encoded,
        scales,
        np.asarray([0, 1], dtype=np.int64),
    )
    assert values.dtype == np.float32
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)


def test_round0081_search_requests_self_headroom() -> None:
    source = inspect.getsource(round0081_nodes._search_and_rerank)
    assert "index.search(queries, shortlist_width + 1)" in source
    assert "candidate_count=shortlist_width" in source


def test_round0081_modules_do_not_mutate_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0081_quality")
    importlib.import_module("experiments.round0081_nodes")
    importlib.import_module("experiments.prepare_round0081_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
