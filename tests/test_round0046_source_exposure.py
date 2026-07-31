from __future__ import annotations

import importlib

import torch

from basemap.round0042_pipeline import edge_ranks_to_source_slots
from basemap.round0042_program import (
    train_config_from_graph as control_config_from_graph,
)
from basemap.round0046_program import train_config_from_graph


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


def test_flat_edge_ranks_map_to_degree_proportional_sources() -> None:
    # Degrees [2, 0, 3, 1] have offsets [0, 2, 2, 5, 6].
    offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int64)
    ranks = torch.arange(6, dtype=torch.int64)
    sources, slots = edge_ranks_to_source_slots(offsets, ranks)
    assert torch.equal(
        sources,
        torch.tensor([0, 0, 2, 2, 2, 3]),
    )
    assert torch.equal(
        slots,
        torch.tensor([0, 1, 0, 1, 2, 0]),
    )


def test_r0046_changes_only_registered_source_exposure_fields() -> None:
    kwargs = {
        "graph_manifest_path": "/data/canonical.json",
        "graph_manifest_sha256": "a" * 64,
    }
    control, _ = control_config_from_graph(_manifest(), **kwargs)
    treatment, digest = train_config_from_graph(_manifest(), **kwargs)
    assert len(digest) == 64
    assert treatment["row_universe"] == control["row_universe"]
    assert treatment["model"] == control["model"]
    assert treatment["optimizer"] == control["optimizer"]
    assert treatment["execution"]["duplicate_control"] == (
        control["execution"]["duplicate_control"]
    )
    stamp = treatment["execution"]["expected_pipeline_stamp"]
    assert treatment["execution"]["required_pipeline"] == (
        "device_fp16_canonical_edge_uniform"
    )
    assert stamp["sampler_class"] == (
        "DeviceEdgeUniformCanonicalSampler"
    )
    assert stamp["positive_sampling"] == (
        "uniform-valid-canonical-edge-with-replacement"
    )
    assert stamp["positive_source_sampling"] == (
        "degree-proportional-over-positive-sources"
    )
    assert stamp["positive_destination_policy"] == (
        control["execution"]["expected_pipeline_stamp"][
            "positive_destination_policy"
        ]
    )
    assert stamp["negative_sampling"] == (
        control["execution"]["expected_pipeline_stamp"][
            "negative_sampling"
        ]
    )


def test_round0046_modules_do_not_mutate_cuda_visibility(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("experiments.round0046_nodes")
    importlib.import_module("experiments.prepare_round0046_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
