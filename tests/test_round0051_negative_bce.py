from __future__ import annotations

import importlib
import inspect

import torch

from basemap.round0051_program import (
    NEGATIVE_MULTIPLIERS,
    train_configs_from_graph,
)
from experiments.round0051_nodes import NormalizedClassWeightedBCELoss


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


def test_multiplier_one_exactly_matches_mean_bce() -> None:
    values = torch.tensor(
        [0.8, 0.3, 0.1, 0.7],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
    loss = NormalizedClassWeightedBCELoss(
        positive_multiplier=1.0,
        negative_multiplier=1.0,
    )(values, targets)
    expected = torch.nn.functional.binary_cross_entropy(values, targets)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)


def test_weighted_loss_uses_declared_normalized_formula() -> None:
    values = torch.tensor(
        [0.8, 0.3, 0.1, 0.7],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
    element = torch.nn.functional.binary_cross_entropy(
        values,
        targets,
        reduction="none",
    )
    weights = torch.tensor([1.0, 0.25, 0.25, 1.0])
    expected = (element * weights).sum() / weights.sum()
    loss_fn = NormalizedClassWeightedBCELoss(
        positive_multiplier=1.0,
        negative_multiplier=0.25,
    )
    actual = loss_fn(values, targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(values.grad).all()
    assert loss_fn.runtime_stamp()["negative_multiplier"] == 0.25


def test_configs_change_only_registered_loss_fields() -> None:
    configs = train_configs_from_graph(
        _manifest(),
        graph_manifest_path="/data/canonical.json",
        graph_manifest_sha256="a" * 64,
    )
    half, half_hash = configs["negative_0p50"]
    quarter, quarter_hash = configs["negative_0p25"]
    assert len(half_hash) == len(quarter_hash) == 64
    assert half_hash != quarter_hash
    assert half["row_universe"] == quarter["row_universe"]
    assert half["model"] == quarter["model"]
    assert half["graph"] == quarter["graph"]
    half_optimizer = dict(half["optimizer"])
    quarter_optimizer = dict(quarter["optimizer"])
    assert half_optimizer.pop("negative_bce_multiplier") == 0.5
    assert quarter_optimizer.pop("negative_bce_multiplier") == 0.25
    assert half_optimizer == quarter_optimizer
    assert NEGATIVE_MULTIPLIERS == {
        "negative_0p50": 0.5,
        "negative_0p25": 0.25,
    }
    assert half["execution"]["expected_pipeline_stamp"] == (
        quarter["execution"]["expected_pipeline_stamp"]
    )
    assert half["execution"]["expected_loss_stamp"][
        "loss_class"
    ] == "NormalizedClassWeightedBCELoss"


def test_round0051_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0051_program")
    importlib.import_module("experiments.round0051_nodes")
    importlib.import_module("experiments.prepare_round0051_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_queue_keeps_two_treatments_inside_autonomous_cap() -> None:
    from experiments import prepare_round0051_queue as queue_prep

    source = inspect.getsource(queue_prep.prepare_round0051)
    assert "gpu_hours_cap=4.0" in source
    assert '"total": 11_500.0' in source
    assert '"deps": list(train_ids.values())' in source
    assert "external_ood_adoption_gate_run" in source
