from __future__ import annotations

import importlib
import inspect

from basemap.round0067_graph import GRAPH_RECEIPT_SCHEMA


def test_graph_receipt_has_a_distinct_selected_rung_schema() -> None:
    assert GRAPH_RECEIPT_SCHEMA == (
        "round0067-next-rung-gpu-graph-receipt-v1"
    )


def test_round0067_is_one_dynamic_bounded_graph_job() -> None:
    from experiments import prepare_round0067_queue

    source = inspect.getsource(
        prepare_round0067_queue.prepare_round0067
    )
    assert source.count('"action": "build_selected_gpu_graph"') == 1
    assert 'cap = 2.0 if tier == "45m" else 5.0' in source
    assert 'p90 = 7_200.0 if tier == "45m" else 18_000.0' in source
    assert '"no_training": True' in source
    assert '"no_scale_decision": True' in source
    assert "tier = decision" in source


def test_round0067_uses_the_r0066_selected_nprobe() -> None:
    from experiments import round0067_nodes

    source = inspect.getsource(round0067_nodes.run_build_graph)
    assert 'nprobe = int(receipt["selected_nprobe"])' in source
    assert "nprobe=nprobe" in source
    assert '"exact_rerank": True' in source
    validator = inspect.getsource(
        __import__(
            "basemap.round0067_graph",
            fromlist=["load_gpu_qualification"],
        ).load_gpu_qualification
    )
    assert (
        'receipt.get("scale_decision") != scale_decision_signature'
        in validator
    )


def test_round0067_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0067_graph")
    importlib.import_module("experiments.round0067_nodes")
    importlib.import_module("experiments.prepare_round0067_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
