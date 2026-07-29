from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from basemap.round0071_substrate import (
    ELIGIBILITY_SUMMARY,
    ROW_COUNT,
)
from experiments import prepare_round0073_queue, round0073_nodes


def test_round0073_graph_is_fixed_balanced_90m() -> None:
    source = inspect.getsource(round0073_nodes.run_build_graph)
    assert "ROW_COUNT" in source
    assert "INTERVALS" in source
    assert "load_gpu_qualification(" in source
    assert "load_scale_decision" not in source
    assert '"scale_decision_made": False' in source
    assert ELIGIBILITY_SUMMARY["retained_row_count"] == 88_945_313
    assert ROW_COUNT == 90_000_000


def test_round0073_queue_is_one_resumable_no_training_job() -> None:
    source = inspect.getsource(prepare_round0073_queue.prepare_round0073)
    assert "gpu_hours_cap=3.0" in source
    assert source.count(
        '"action": "build_balanced_90m_gpu_graph"'
    ) == 1
    assert '"shard_rows": 100_000' in source
    assert '"resumable_shards": True' in source
    assert '"no_training": True' in source
    assert '"no_scale_decision": True' in source


def test_round0073_gpu_clone_preserves_qualified_precision() -> None:
    source = inspect.getsource(round0073_nodes._to_gpu)
    assert "INDICES_64_BIT" in source
    assert "useFloat16 = False" in source
    assert "usePrecomputed = True" in source
    assert "setTempMemory(1 << 30)" in source


def test_round0073_accepts_only_an_issued_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_file = tmp_path / "round-0073.md"
    monkeypatch.setattr(
        prepare_round0073_queue,
        "ROUND_FILE",
        str(round_file),
    )
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    prepare_round0073_queue._require_issued_round()
    round_file.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remains draft"):
        prepare_round0073_queue._require_issued_round()


def test_round0073_modules_do_not_mutate_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0073_graph")
    importlib.import_module("experiments.round0073_nodes")
    importlib.import_module("experiments.prepare_round0073_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
