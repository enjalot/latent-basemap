from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from basemap.round0065_substrates import subset_spec
from experiments import prepare_round0078_queue, round0078_nodes


def test_round0078_graph_is_fixed_balanced_120m() -> None:
    spec = subset_spec("120m")
    source = inspect.getsource(round0078_nodes.run_build_graph)
    assert "ROW_COUNT" in source
    assert "INTERVALS" in source
    assert "load_gpu_qualification(" in source
    assert "load_scale_decision" not in source
    assert '"scale_decision_made": False' in source
    assert round0078_nodes.ROW_COUNT == 120_000_000
    assert round0078_nodes.INTERVALS == tuple(spec["intervals"])
    assert round0078_nodes.ELIGIBILITY_SUMMARY[
        "retained_row_count"
    ] == 118_067_492


def test_round0078_queue_is_one_resumable_no_training_job() -> None:
    source = inspect.getsource(prepare_round0078_queue.prepare_round0078)
    assert "gpu_hours_cap=5.0" in source
    assert source.count(
        '"action": "build_balanced_120m_gpu_graph"'
    ) == 1
    assert '"shard_rows": 100_000' in source
    assert '"resumable_shards": True' in source
    assert '"no_training": True' in source
    assert '"no_scale_decision": True' in source
    assert '"required_reviews"] = ["0065", "0077"]' in source


def test_round0078_gpu_clone_preserves_qualified_precision() -> None:
    source = inspect.getsource(round0078_nodes._to_gpu)
    assert "INDICES_64_BIT" in source
    assert "useFloat16 = False" in source
    assert "usePrecomputed = True" in source
    assert "setTempMemory(1 << 30)" in source


def test_round0078_accepts_only_an_issued_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    round_file = tmp_path / "round-0078.md"
    monkeypatch.setattr(
        prepare_round0078_queue,
        "ROUND_FILE",
        str(round_file),
    )
    round_file.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    prepare_round0078_queue._require_issued_round()
    round_file.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="remains draft"):
        prepare_round0078_queue._require_issued_round()


def test_round0078_modules_do_not_mutate_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0078_graph")
    importlib.import_module("experiments.round0078_nodes")
    importlib.import_module("experiments.prepare_round0078_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
