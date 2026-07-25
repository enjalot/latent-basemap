from __future__ import annotations

import importlib
import inspect

from basemap.round0048_program import train_configs_from_graph


def _manifest() -> dict:
    return {
        "schema": "minilm-canonical-source-major-k15-v1",
        "round_id": "0041",
        "row_count": 30_000_000,
        "input_k": 15,
        "inputs": {
            "eligibility": {
                "sha256": (
                    "834089fcbd9a722cec4f05be6382ed8430d27280e7e23ca085"
                    "5785e3f48ea5e2"
                )
            }
        },
        "summary": {
            "input_edge_count": 450_000_000,
            "eligibility_excluded_source_count": 218_242,
            "eligibility_retained_row_count": 29_781_758,
            "retained_positive_source_count": 29_781_619,
            "zero_degree_retained_source_count": 139,
            "valid_canonical_edge_count": 444_198_115,
            "duplicate_destinations_mapped": 2_524_873,
        },
    }


def test_r0048_pair_changes_only_registered_source_exposure() -> None:
    configs = train_configs_from_graph(
        _manifest(),
        graph_manifest_path="/data/canonical.json",
        graph_manifest_sha256="a" * 64,
    )
    source, source_hash = configs["source_uniform"]
    edge, edge_hash = configs["edge_uniform"]
    assert len(source_hash) == len(edge_hash) == 64
    assert source_hash != edge_hash
    assert source["optimizer"] == edge["optimizer"]
    assert source["optimizer"]["seed"] == 43
    assert source["model"] == edge["model"]
    assert source["row_universe"] == edge["row_universe"]
    assert source["graph"]["path"] == edge["graph"]["path"]
    assert source["graph"]["sha256"] == edge["graph"]["sha256"]
    assert source["execution"]["required_pipeline"] == (
        "device_fp16_canonical"
    )
    assert source["execution"]["expected_pipeline_stamp"]["schema"] == (
        "round0042-device-fp16-canonical-pipeline-v1"
    )
    assert edge["execution"]["required_pipeline"] == (
        "device_fp16_canonical_edge_uniform"
    )
    assert source["execution"]["expected_pipeline_stamp"][
        "negative_sampling"
    ] == edge["execution"]["expected_pipeline_stamp"]["negative_sampling"]
    assert source["execution"]["matched_R0048_pair"][
        "only_intended_change"
    ] == "positive-source exposure law"


def test_round0048_modules_do_not_mutate_cuda_visibility(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0048_program")
    importlib.import_module("experiments.round0048_nodes")
    importlib.import_module("experiments.prepare_round0048_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_r0048_queue_orders_both_trains_before_transforms() -> None:
    from experiments import prepare_round0048_queue as queue_prep

    source = inspect.getsource(queue_prep.prepare_round0048)
    assert "gpu_hours_cap=4.0" in source
    assert '"total": 11_500.0' in source
    assert '"deps": list(train_ids.values())' in source
