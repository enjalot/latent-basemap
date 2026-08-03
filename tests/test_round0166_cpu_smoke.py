"""Bounded CPU smoke for the R0166 train -> seal -> reload -> panel path."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from basemap.panel_v2 import PanelV2Config, score_panel
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0166_prompted_8m import (
    DIMENSION,
    MULTIPLICITY_POLICY,
    ROUND_ID,
    SUCCESSFUL_UPDATES,
    scale_train_config,
)
from experiments import round0166_nodes


def test_round0166_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import torch

    rows = 7_952_419
    producer_batches = SUCCESSFUL_UPDATES + 1
    expected_rows = SUCCESSFUL_UPDATES * prompt_contract.BATCH_SIZE
    emitted = producer_batches * prompt_contract.POSITIVE_ROWS_PER_UPDATE
    accepted = emitted + 7
    proposals = accepted + 1_000
    graph_path = tmp_path / "graph.npz"
    graph_path.write_bytes(b"g")
    manifest_path = tmp_path / "graph-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    graph_signature = {
        "kind": "file",
        "canonical_path": str(graph_path),
        "bytes": 1,
        "sha256": "a" * 64,
    }
    manifest_signature = {
        "kind": "file",
        "canonical_path": str(manifest_path),
        "bytes": 2,
        "sha256": "b" * 64,
    }
    config, config_sha = scale_train_config(
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        graph_edges=8,
        retained_rows=rows,
    )
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": SUCCESSFUL_UPDATES,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": producer_batches,
        "host_prefetch_producer_batches": producer_batches,
        "host_prefetch_consumer_batches": SUCCESSFUL_UPDATES,
        "host_prefetch_source_rows_filled": producer_batches
        * prompt_contract.BATCH_SIZE,
        "host_prefetch_destination_rows_filled": producer_batches
        * prompt_contract.BATCH_SIZE,
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
    accounting.update({
        f"pipeline_{key}": runtime[key]
        for key in prompt_contract._DYNAMIC_PIPELINE_COUNTERS
    })
    population_signature = {
        "kind": "file",
        "canonical_path": str(tmp_path / "population.json"),
        "bytes": 1,
        "sha256": "c" * 64,
    }
    source_signature = {
        "kind": "file",
        "canonical_path": str(tmp_path / "source.f16"),
        "bytes": rows * DIMENSION * 2,
        "sha256": "d" * 64,
    }
    mapping_signature = {
        "kind": "file",
        "canonical_path": str(tmp_path / "mapping.npy"),
        "bytes": rows * 8 + 128,
        "sha256": "e" * 64,
    }
    population = {
        "retained_rows": rows,
        "document_compact": source_signature,
        "mapping": mapping_signature,
    }
    graph = {
        "manifest": {
            "population": population_signature,
            "compact_mapping": mapping_signature,
        },
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "sources": np.arange(8, dtype=np.int32),
        "targets": np.roll(np.arange(8, dtype=np.int32), -1),
        "weights": np.ones(8, dtype=np.float32),
        "n_nodes": rows,
    }

    class SmokeDataset:
        def __init__(self, *args, **kwargs) -> None:
            self.shape = (rows, DIMENSION)
            self.features = np.random.default_rng(166).normal(size=(240, 8)).astype(np.float32)

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
            assert kwargs["random_state"] == 42
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
            torch.save({
                "weight": self.layer.weight.detach(),
                "bias": self.layer.bias.detach(),
            }, path)

    monkeypatch.setattr(
        round0166_nodes,
        "_read_population",
        lambda job: (population, population_signature),
    )
    monkeypatch.setattr(round0166_nodes, "_load_graph", lambda path: graph)
    monkeypatch.setattr(
        round0166_nodes,
        "scale_train_config",
        lambda **kwargs: (config, config_sha),
    )
    monkeypatch.setattr(
        round0166_nodes,
        "_open_source",
        lambda population: np.zeros((2, DIMENSION), dtype=np.float16),
    )
    monkeypatch.setattr(prompt_contract, "HostFp16EndpointArray", SmokeDataset)
    monkeypatch.setattr(round0166_nodes, "ScalePromptTrainingInput", SmokeWrapper)
    monkeypatch.setattr(round0166_nodes.prompt_nodes, "_new_model", lambda config: SmokeModel())
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1_000_000_000, 2_000_000_000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / "train-output"
    round0166_nodes.run_train(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
        {"graph_manifest": str(manifest_path), "outputs": [str(output)]},
    )
    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0166 CPU smoke receipt")
    assert receipt["optimizer_updates"] == SUCCESSFUL_UPDATES
    assert receipt["exact_execution_receipt"]["multiplicity_policy"] == MULTIPLICITY_POLICY
    assert receipt["train_checks"]["weighted_rejection_accounting_closes"] is True

    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    features = SmokeDataset().features
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
