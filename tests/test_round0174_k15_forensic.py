"""Contract and bounded CPU smoke tests for R0174."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

from basemap.panel_v2 import PanelV2Config, score_panel
from basemap import round0104_training as training
from basemap.round0108_evaluation import validate_seal
from basemap.round0140_subsystem_bisection import RESTORATION_FLOORS
from basemap.round0174_k15_forensic import (
    CELL,
    GRAPH_K,
    ROUND_ID,
    build_decision,
    host_train_config,
)
from experiments import round0140_nodes as base
from experiments import round0174_nodes as nodes


def _cell(values: dict[str, float]) -> dict[str, Any]:
    return {
        "panel": {
            "ffr": values["ffr"],
            "purity": {
                "k256": values["purity_fidelity_k256"],
                "k1024": values["purity_fidelity_k1024"],
            },
        },
        "projection": {
            "ffr": values["projection_ffr"],
            "recall_at_10": values["ood_recall_at_10"],
        },
    }


def test_k15_config_changes_only_registered_graph_stamps(tmp_path) -> None:
    graph = {
        "canonical_path": str(tmp_path / "graph.npz"),
        "kind": "file",
        "bytes": 10,
        "sha256": "a" * 64,
    }
    manifest = {
        "canonical_path": str(tmp_path / "manifest.json"),
        "kind": "file",
        "bytes": 10,
        "sha256": "b" * 64,
    }
    config, config_sha = host_train_config(
        cell=CELL,
        graph_signature=graph,
        graph_manifest_signature=manifest,
        graph_edges=10_000,
    )
    assert len(config_sha) == 64
    assert config["graph"]["k"] == GRAPH_K
    assert config["optimizer"]["successful_positive_lr_updates"] == 500_000
    assert config["optimizer"]["seed"] == 42
    expected = config["execution"]["expected_pipeline_stamp"]
    assert expected["positive_destination_policy"] == "queue-local-fp16-fuzzy-k15"
    assert expected["graph_degree"] == "variable-fuzzy-k15-edge-universe"


def test_sampler_stamp_reports_actual_graph_degree() -> None:
    class Dataset:
        device = "cpu"

        def __len__(self) -> int:
            return 8

        def execution_stamp(self) -> dict[str, Any]:
            return {"source_representation": "fp16-control"}

    sampler = training.PairedHostWeightedJinaSampler(
        Dataset(),
        sources=np.arange(8, dtype=np.int32),
        targets=np.roll(np.arange(8, dtype=np.int32), -1),
        weights=np.ones(8, dtype=np.float32),
        n_nodes=8,
        batch_size=8,
        pos_ratio=0.25,
        random_state=42,
        graph_signature={"sha256": "a" * 64},
        graph_manifest_signature={"sha256": "b" * 64},
        arm="fp16_control",
        graph_k=GRAPH_K,
    )
    stamp = sampler.execution_stamp()
    assert stamp["positive_destination_policy"] == "queue-local-fp16-fuzzy-k15"
    assert stamp["graph_degree"] == "variable-fuzzy-k15-edge-universe"


def test_registered_selector_has_both_preregistered_branches() -> None:
    control = {
        "ffr": 0.5708,
        "purity_fidelity_k256": 0.9082652134423251,
        "purity_fidelity_k1024": 0.9722,
        "projection_ffr": 0.5365,
        "ood_recall_at_10": 0.00985,
    }
    passing = build_decision(treatment=_cell(control), k50_control=_cell(control))
    assert passing["outcome"] == "k15-maintains-restoration-on-historical-rows"
    failing_values = dict(control)
    failing_values["ffr"] = RESTORATION_FLOORS["ffr"] - 0.001
    failing = build_decision(
        treatment=_cell(failing_values), k50_control=_cell(control)
    )
    assert failing["outcome"] == "k15-breaks-restoration-on-historical-rows"
    assert failing["registered_gates"]["ffr"]["passed"] is False


def test_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import torch

    nodes._configure()
    graph_path = tmp_path / "edges-k15-fuzzy.npz"
    graph_path.write_bytes(b"graph")
    manifest_path = tmp_path / "graph-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    graph_signature = {
        "canonical_path": str(graph_path),
        "kind": "file",
        "bytes": 5,
        "sha256": "a" * 64,
    }
    manifest_signature = {
        "canonical_path": str(manifest_path),
        "kind": "file",
        "bytes": 2,
        "sha256": "b" * 64,
    }
    graph = {
        "signature": graph_signature,
        "manifest_signature": manifest_signature,
        "sources": np.arange(16, dtype=np.int32),
        "targets": np.roll(np.arange(16, dtype=np.int32), -1),
        "weights": np.ones(16, dtype=np.float32),
        "n_nodes": 2_000_000,
        "edges": 16,
        "kind": "current-fixed-row",
        "k": GRAPH_K,
    }
    config, _ = host_train_config(
        cell=CELL,
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        graph_edges=16,
    )
    successful = 500_000
    batch = config["optimizer"]["batch_size"]
    expected_rows = successful * batch
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": successful,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": successful,
        "host_prefetch_producer_batches": successful,
        "host_prefetch_consumer_batches": successful,
        "host_prefetch_source_rows_filled": expected_rows,
        "host_prefetch_destination_rows_filled": expected_rows,
    }
    accounting = {
        "lr_horizon": successful,
        "positive_lr_optimizer_steps": successful,
        "scheduler_steps": successful,
        "attempted_batches": successful,
        "finite_loss_batches": successful,
        "optimizer_steps_attempted": successful,
        "optimizer_steps_succeeded": successful,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": 16,
    }
    for key in training._DYNAMIC_PIPELINE_COUNTERS:
        accounting[f"pipeline_{key}"] = runtime[key]

    features = np.random.default_rng(174).normal(size=(240, 8)).astype(np.float32)

    class Source:
        pass

    class Dataset:
        shape = (2_000_000, 768)

        def __init__(self, *args, **kwargs) -> None:
            self.features = features

    class Wrapper:
        def __init__(self, dataset, *args, **kwargs) -> None:
            self.dataset = dataset

        def runtime_stamp(self) -> dict[str, Any]:
            return dict(runtime)

    class Profiler:
        def finalize(self, **kwargs) -> dict[str, Any]:
            return {"aborted": False, "cpu_smoke": True}

    class Model:
        def __init__(self) -> None:
            self.layer = torch.nn.Linear(8, 2)
            self._canary_profiler = Profiler()
            self._bench_seconds = 1.0
            self._setup_seconds = 0.001

        def fit(self, wrapper, **kwargs) -> None:
            x = torch.from_numpy(wrapper.dataset.features)
            target = x[:, :2] - 0.2 * x[:, 2:4]
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.05)
            for _ in range(4):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(self.layer(x), target)
                loss.backward()
                optimizer.step()
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save({
                "weight": self.layer.weight.detach(),
                "bias": self.layer.bias.detach(),
            }, path)

    real_signature = base.expected_input_signature

    def smoke_signature(path: str) -> dict[str, Any]:
        if os.path.realpath(path) == os.path.realpath(base.TRAIN_PATH):
            return {
                "canonical_path": os.path.realpath(path),
                "kind": "file",
                "bytes": base.TRAIN_BYTES,
                "sha256": base.TRAIN_SHA256,
            }
        return real_signature(path)

    monkeypatch.setattr(base, "_graph_bundle", lambda job: graph)
    monkeypatch.setattr(base, "HistoricalFp16Array", Source)
    monkeypatch.setattr(base, "HistoricalHostFp16Array", Dataset)
    monkeypatch.setattr(base.r0104, "Round0104TrainingInput", Wrapper)
    monkeypatch.setattr(base.r0104, "_new_model", lambda config: Model())
    monkeypatch.setattr(base, "expected_input_signature", smoke_signature)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1_000_000, 2_000_000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / "train"
    nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
        {
            "action": "train_host",
            "cell": CELL,
            "outputs": [str(output)],
        },
    )
    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0174 CPU smoke train receipt")
    assert receipt["train_checks"]["exact_update_closure"] is True
    assert receipt["exact_execution_receipt"]["graph_degree"] == (
        "variable-fuzzy-k15-edge-universe"
    )

    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    coordinates = torch.nn.functional.linear(
        torch.from_numpy(features), checkpoint["weight"], checkpoint["bias"]
    ).detach().numpy()
    panel = score_panel(
        features,
        coordinates,
        config=PanelV2Config(
            frac=0.1,
            k_hit=3,
            k_density=3,
            n_anchors=24,
            corpus_chunk=64,
            overselect=4,
            block_elems=100_000,
            rerank_byte_cap=8_000_000,
            peak_byte_cap=16_000_000,
        ),
        provenance={"round": ROUND_ID, "mode": "cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False
