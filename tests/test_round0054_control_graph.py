from __future__ import annotations

import importlib
import inspect

from experiments import round0049_nodes


def test_generalized_graph_helpers_preserve_r0049_defaults() -> None:
    write = inspect.signature(round0049_nodes._write_shard)
    assemble = inspect.signature(round0049_nodes._assemble_graph)
    assert write.parameters["source_rows"].default == 150_000_000
    assert write.parameters["round_id"].default == "0049"
    assert assemble.parameters["row_count"].default == 60_000_000
    assert assemble.parameters["round_id"].default == "0049"


def test_r0054_is_resumable_cpu_only_graph_work() -> None:
    from experiments import prepare_round0054_queue as queue_prep

    source = inspect.getsource(queue_prep.prepare_round0054)
    assert "gpu_hours_cap=0.0" in source
    assert '"total": 15_000.0' in source
    assert '"action": "build_graph"' in source
    assert '"gpu_required": False' in source
    assert '"resumable_shards": True' in source


def test_round0054_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("experiments.round0054_nodes")
    importlib.import_module("experiments.prepare_round0054_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
