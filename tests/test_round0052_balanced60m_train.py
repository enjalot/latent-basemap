from __future__ import annotations

import importlib
import inspect

import numpy as np

from basemap.round0052_program import (
    EXPECTED_RETAINED_ROWS,
    EXPECTED_VALID_EDGES,
    PIPELINE_SCHEMA,
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments.round0052_nodes import (
    HostInt8BalancedCanonicalSampler,
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
        "schema": "round0049-balanced-60m-substrate-v1",
        "round_id": "0049",
        "row_count": 60_000_000,
        "dimension": 384,
        "global_150m_intervals": [
            [0, 20_000_000],
            [50_000_000, 70_000_000],
            [100_000_000, 120_000_000],
        ],
        "outputs": {
            "int8": _signature(
                "/data/embeddings.i8",
                "a" * 64,
                60_000_000 * 384,
            ),
            "scales": _signature(
                "/data/scales.f16",
                "b" * 64,
                60_000_000 * 2,
            ),
            "eligibility": eligibility,
        },
    }
    graph = {
        "schema": "minilm-canonical-source-major-k15-v1",
        "round_id": "0050",
        "row_count": 60_000_000,
        "input_k": 15,
        "inputs": {"eligibility": eligibility},
        "summary": {
            "eligibility_excluded_source_count": 600_712,
            "eligibility_retained_row_count": 59_399_288,
            "retained_positive_source_count": 59_399_288,
            "zero_degree_retained_source_count": 0,
            "valid_canonical_edge_count": 890_989_320,
            "degree_histogram": {
                "0": 600_712,
                "15": 59_399_288,
            },
        },
    }
    return graph, substrate


def test_coverage_horizon_and_constant_degree_equivalence() -> None:
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
    assert SUCCESSFUL_UPDATES == 997_248
    assert config["optimizer"]["successful_positive_lr_updates"] == (
        SUCCESSFUL_UPDATES
    )
    assert config["graph"]["positive_source_rows"] == (
        EXPECTED_RETAINED_ROWS
    )
    assert config["graph"]["valid_canonical_edges"] == (
        EXPECTED_VALID_EDGES
    )
    assert config["graph"]["source_edge_uniform_equivalence"][
        "holds"
    ] is True
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["source_edge_uniform_equivalent"] is True
    assert stamp["negative_sampling"] == (
        "uniform-balanced-60m-retained-rows-nonself"
    )


def test_balanced_sampler_stamp_does_not_claim_r0033_semantics() -> None:
    encoded = np.ones((20, 384), dtype=np.int8)
    scales = np.ones(20, dtype="<f2")
    from basemap.round0034_pipeline import HostInt8MaterializedArray

    dataset = HostInt8MaterializedArray(
        encoded,
        scales,
        device="cpu",
        signatures={
            "int8": _signature("/data/i8", "a" * 64, encoded.nbytes),
            "scales": _signature(
                "/data/f16",
                "b" * 64,
                scales.nbytes,
            ),
        },
        buffer_rows=8,
    )
    targets = np.tile(
        np.arange(15, dtype="<i4"),
        (20, 1),
    )
    sampler = HostInt8BalancedCanonicalSampler(
        dataset,
        targets=targets,
        degrees=np.full(20, 15, dtype="u1"),
        excluded_rows=np.empty(0, dtype=np.int64),
        positive_source_count=20,
        valid_edge_count=300,
        batch_size=8,
        pos_ratio=0.25,
        random_state=42,
        graph_signature=_signature("/data/graph", "c" * 64, 1),
        eligibility_signature=_signature(
            "/data/eligibility",
            "d" * 64,
            1,
        ),
    )
    stamp = sampler.execution_stamp()
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["sampler_class"] == (
        "HostInt8BalancedCanonicalSampler"
    )
    assert "R0033" not in str(stamp)
    assert stamp["graph_degree"].startswith("fixed-15")
    assert stamp["source_edge_uniform_equivalent"] is True


def test_round0052_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0052_program")
    importlib.import_module("experiments.round0052_nodes")
    importlib.import_module("experiments.prepare_round0052_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_queue_uses_live_train_profiler_without_standalone_canary() -> None:
    from experiments import prepare_round0052_queue as queue_prep

    source = inspect.getsource(queue_prep.prepare_round0052)
    assert "review-0049-2026-07-26.md" in source
    assert "review-0049-2026-07-25.md" not in source
    assert "gpu_hours_cap=3.5" in source
    assert '"total": 10_500.0' in source
    assert '"standalone_canary": False' in source
    assert '"id": "train_seed42_balanced_60m"' in source
    assert '"action": "canary"' not in source
