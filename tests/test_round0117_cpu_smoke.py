"""Bounded CPU smoke for the R0117 train -> seal -> reload -> panel path."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0113_prompt_contrast import (
    BATCH_SIZE,
    DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    RETAINED_ROWS,
    SUCCESSFUL_UPDATES,
    _DYNAMIC_PIPELINE_COUNTERS,
    validate_seal,
)
from experiments import round0113_nodes


def test_round0117_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Exercise the expensive post-fit handoff without allocating CUDA."""
    import torch

    producer_batches = SUCCESSFUL_UPDATES + 1
    consumer_batches = SUCCESSFUL_UPDATES
    expected_rows = SUCCESSFUL_UPDATES * BATCH_SIZE
    emitted = producer_batches * POSITIVE_ROWS_PER_UPDATE
    accepted = emitted + 7
    proposals = accepted + 1_000
    expected_stamp = {
        "schema": "round0113-host-weighted-jina-prompt-pipeline-v1",
        "pipeline": "host_weighted_jina_prompt_contrast",
        "sampler_class": "PromptWeightedJinaSampler",
        "arm": "raw",
    }
    runtime = {
        **expected_stamp,
        "endpoint_gather_calls": consumer_batches,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": producer_batches,
        "host_prefetch_producer_batches": producer_batches,
        "host_prefetch_consumer_batches": consumer_batches,
        "host_prefetch_source_rows_filled": producer_batches * BATCH_SIZE,
        "host_prefetch_destination_rows_filled": producer_batches * BATCH_SIZE,
        "weight_proposals": proposals,
        "weight_acceptances": accepted,
        "weight_emitted_draws": emitted,
        "weight_buffered_draws": accepted - emitted,
        "weight_acceptance_rate": accepted / proposals,
        "weight_rejection_iterations": producer_batches,
    }
    accounting: dict[str, Any] = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": 8,
    }
    accounting.update(
        {f"pipeline_{key}": runtime[key] for key in _DYNAMIC_PIPELINE_COUNTERS}
    )
    graph_manifest_path = tmp_path / "accepted-r0115-graph.json"
    graph_manifest_path.write_text("{}", encoding="utf-8")
    graph_signature = {
        "kind": "file",
        "canonical_path": str(tmp_path / "accepted-r0115-graph.npz"),
        "bytes": 1,
        "sha256": "c" * 64,
    }
    manifest_signature = {
        "kind": "file",
        "canonical_path": str(graph_manifest_path),
        "bytes": graph_manifest_path.stat().st_size,
        "sha256": "d" * 64,
    }
    graph = {
        "manifest": {"round_id": "0115"},
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "sources": np.arange(8, dtype=np.int32),
        "targets": np.roll(np.arange(8, dtype=np.int32), -1),
        "weights": np.ones(8, dtype=np.float32),
        "n_nodes": RETAINED_ROWS,
    }
    config = {
        "model": {},
        "optimizer": {
            "seed": 43,
            "positive_rng_seed": 43,
            "negative_rng_seed": 11_300_043,
        },
        "execution": {
            "expected_pipeline_stamp": expected_stamp,
            "minimum_train_upd_s": 70.0,
            "warning_train_upd_s": 80.0,
        },
    }
    assembly_signature = {
        "kind": "file",
        "canonical_path": str(tmp_path / "assembly.json"),
        "bytes": 1,
        "sha256": "e" * 64,
    }
    assembly = {
        "outputs": {"raw": {"sha256": "f" * 64}},
        "mapping": {"sha256": "0" * 64},
    }

    class SmokeDataset:
        def __init__(self, *args, **kwargs) -> None:
            rng = np.random.default_rng(117)
            self.features = rng.normal(size=(240, 8)).astype(np.float32)

    class SmokeWrapper:
        def __init__(self, dataset, *args, **kwargs) -> None:
            self.dataset = dataset

        def runtime_stamp(self) -> dict[str, Any]:
            return dict(runtime)

    class SmokeProfiler:
        def finalize(self, **kwargs) -> dict[str, Any]:
            assert kwargs["bench_seconds"] > 0
            return {"aborted": False, "smoke": True}

    class SmokeModel:
        def __init__(self) -> None:
            self.layer = torch.nn.Linear(8, 2)
            self._canary_profiler = SmokeProfiler()
            self._bench_seconds = 1.0
            self._setup_seconds = 0.001

        def fit(self, wrapper, **kwargs) -> None:
            assert kwargs["random_state"] == 43
            features = torch.from_numpy(wrapper.dataset.features)
            target = features[:, :2] - 0.25 * features[:, 2:4]
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.05)
            for _ in range(4):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(self.layer(features), target)
                loss.backward()
                optimizer.step()
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save(
                {
                    "weight": self.layer.weight.detach(),
                    "bias": self.layer.bias.detach(),
                },
                path,
            )

    monkeypatch.setattr(
        round0113_nodes,
        "_load_assembly",
        lambda job: (assembly, assembly_signature),
    )
    monkeypatch.setattr(round0113_nodes, "load_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(
        round0113_nodes,
        "train_config",
        lambda *args, **kwargs: (
            config,
            "1" * 64,
        )
        if kwargs["seed"] == 43
        else pytest.fail("R0117 smoke did not request seed 43"),
    )
    monkeypatch.setattr(
        round0113_nodes,
        "_open_compact",
        lambda assembly, arm: np.zeros((2, DIMENSION), dtype=np.float16),
    )
    monkeypatch.setattr(
        round0113_nodes, "HostFp16EndpointArray", SmokeDataset
    )
    monkeypatch.setattr(round0113_nodes, "PromptTrainingInput", SmokeWrapper)
    monkeypatch.setattr(
        round0113_nodes, "_new_model", lambda config: SmokeModel()
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (1_000_000_000, 2_000_000_000),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / "train-output"
    result = round0113_nodes.run_train(
        {"manifest": {"round_id": "0117", "release_sha": "a" * 40}},
        {
            "arm": "raw",
            "training_seed": 43,
            "graph_execution_round_id": "0115",
            "graph_manifest": str(graph_manifest_path),
            "outputs": [str(output)],
        },
    )
    assert result["training_seed"] == 43
    assert result["optimizer_updates"] == SUCCESSFUL_UPDATES
    assert result["train_checks"]["weighted_rejection_accounting_closes"]

    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0117 CPU smoke receipt")
    assert receipt["model"]["sha256"] == result["model"]["sha256"]
    with (output / "production-config.json").open(encoding="utf-8") as handle:
        production = json.load(handle)
    assert production["config"]["optimizer"]["seed"] == 43

    checkpoint = torch.load(
        output / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    features = SmokeDataset().features
    coordinates = (
        torch.nn.functional.linear(
            torch.from_numpy(features),
            checkpoint["weight"],
            checkpoint["bias"],
        )
        .detach()
        .numpy()
    )
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
        provenance={"round": "0117", "mode": "cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False
