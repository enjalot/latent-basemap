from __future__ import annotations

import importlib
import inspect

import numpy as np

from basemap.round0055_program import (
    PIPELINE_SCHEMA,
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments.round0055_nodes import (
    HostInt8Balanced30mCanonicalSampler,
)


def _signature(path: str, digest: str, size: int) -> dict:
    return {
        "canonical_path": path,
        "sha256": digest,
        "bytes": size,
        "kind": "file",
    }


def _capabilities() -> tuple[dict, dict]:
    eligibility = _signature("/data/eligibility.npz", "e" * 64, 123)
    substrate = {
        "schema": "round0053-balanced-30m-int8-substrate-v1",
        "round_id": "0053",
        "row_count": 30_000_000,
        "dimension": 384,
        "global_150m_intervals": [
            [0, 10_000_000],
            [50_000_000, 60_000_000],
            [100_000_000, 110_000_000],
        ],
        "outputs": {
            "int8": _signature(
                "/data/embeddings.i8",
                "a" * 64,
                30_000_000 * 384,
            ),
            "scales": _signature(
                "/data/scales.f16",
                "b" * 64,
                30_000_000 * 2,
            ),
            "eligibility": eligibility,
        },
    }
    graph = {
        "schema": "minilm-canonical-source-major-k15-v1",
        "round_id": "0054",
        "row_count": 30_000_000,
        "input_k": 15,
        "inputs": {"eligibility": eligibility},
        "summary": {
            "eligibility_excluded_source_count": 218_246,
            "eligibility_retained_row_count": 29_781_754,
            "retained_positive_source_count": 29_781_754,
            "zero_degree_retained_source_count": 0,
            "valid_canonical_edge_count": 446_726_310,
            "degree_histogram": {
                "0": 218_246,
                "15": 29_781_754,
            },
        },
    }
    return graph, substrate


def test_matched_control_horizon_and_runtime_stamp() -> None:
    graph, substrate = _capabilities()
    config, digest = train_config_from_capabilities(
        graph,
        graph_manifest_path="/data/graph.json",
        graph_manifest_sha256="c" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path="/data/substrate.json",
        substrate_manifest_sha256="d" * 64,
    )
    assert len(digest) == 64
    assert SUCCESSFUL_UPDATES == 500_003
    assert config["optimizer"]["successful_positive_lr_updates"] == (
        SUCCESSFUL_UPDATES
    )
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["sampler_class"] == (
        "HostInt8Balanced30mCanonicalSampler"
    )
    assert stamp["negative_sampling"] == (
        "uniform-balanced-30m-retained-rows-nonself"
    )
    assert config["execution"]["matched_r0052_scale_control"][
        "evaluation_required_for_scale_law"
    ] is True


def test_30m_sampler_overrides_all_rung_specific_semantics() -> None:
    from basemap.round0034_pipeline import HostInt8MaterializedArray

    encoded = np.ones((20, 384), dtype=np.int8)
    scales = np.ones(20, dtype="<f2")
    dataset = HostInt8MaterializedArray(
        encoded,
        scales,
        device="cpu",
        buffer_rows=8,
    )
    sampler = HostInt8Balanced30mCanonicalSampler(
        dataset,
        targets=np.tile(np.arange(15, dtype="<i4"), (20, 1)),
        degrees=np.full(20, 15, dtype="u1"),
        excluded_rows=np.empty(0, dtype=np.int64),
        positive_source_count=20,
        valid_edge_count=300,
        batch_size=8,
        pos_ratio=0.25,
        random_state=42,
        graph_signature={},
        eligibility_signature={},
    )
    stamp = sampler.execution_stamp()
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["sampler_class"] == (
        "HostInt8Balanced30mCanonicalSampler"
    )
    assert "60m" not in str(stamp).lower()
    assert "R0033" not in str(stamp)


def test_round0055_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0055_program")
    importlib.import_module("experiments.round0055_nodes")
    importlib.import_module("experiments.prepare_round0055_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_r0055_has_no_standalone_canary() -> None:
    from experiments import prepare_round0055_queue as queue_prep

    source = inspect.getsource(queue_prep.prepare_round0055)
    assert "gpu_hours_cap=2.0" in source
    assert '"total": 5_400.0' in source
    assert '"standalone_canary": False' in source
    assert '"action": "canary"' not in source
