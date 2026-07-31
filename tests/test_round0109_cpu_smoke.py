"""Bounded CPU smoke for the R0109 train -> seal -> reload -> panel path."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from basemap.panel_v2 import PanelV2Config, score_panel
from basemap.round0107_training import (
    _DYNAMIC_PIPELINE_COUNTERS,
    validate_seal,
)
from experiments import round0107_nodes


def test_round0109_train_seal_reload_panel_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Exercise every post-fit R0109 statement without allocating CUDA.

    The tiny trainer is deliberately synthetic; the production sampler and
    optimizer have their own focused tests.  This smoke protects the expensive
    boundary that previously failed only after a full train: accounting,
    profiling, model publication, receipt sealing, checkpoint reload, and the
    panel scorer entry point.
    """
    import torch

    updates = 201
    positive_rows = 409
    batch_size = round0107_nodes.BATCH_SIZE
    expected_rows = updates * batch_size
    emitted = updates * positive_rows
    accepted = emitted + 3
    proposals = accepted + 1_000
    expected_stamp = {
        "schema": "round0109-cpu-smoke-pipeline-v1",
        "pipeline": "round0109_cpu_smoke",
        "sampler_class": "Round0109CpuSmokeSampler",
    }
    runtime = {
        **expected_stamp,
        "endpoint_gather_calls": updates,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": updates,
        "host_prefetch_producer_batches": updates,
        "host_prefetch_consumer_batches": updates,
        "host_prefetch_source_rows_filled": expected_rows,
        "host_prefetch_destination_rows_filled": expected_rows,
        "weight_proposals": proposals,
        "weight_acceptances": accepted,
        "weight_emitted_draws": emitted,
        "weight_buffered_draws": accepted - emitted,
        "weight_acceptance_rate": accepted / proposals,
        "weight_rejection_iterations": updates,
    }
    accounting: dict[str, Any] = {
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": emitted - 7,
    }
    accounting.update(
        {f"pipeline_{key}": runtime[key] for key in _DYNAMIC_PIPELINE_COUNTERS}
    )
    graph = {
        "manifest": {
            "release_sha": "b" * 40,
            "directed_edge_count": accounting["n_pos_edges"],
            "outputs": {"sources": {}, "targets": {}, "weights": {}},
            "compact_mapping": {},
        },
        "signature": {
            "kind": "file",
            "canonical_path": "/data/r0109-smoke/graph-manifest.json",
            "bytes": 1,
            "sha256": "c" * 64,
        },
        "successful_updates": updates,
        "arrays": {
            "mapping": np.arange(8, dtype=np.int64),
            "sources": np.arange(8, dtype=np.int32),
            "targets": np.roll(np.arange(8, dtype=np.int32), -1),
            "weights": np.ones(8, dtype=np.float32),
        },
    }
    config = {
        "optimizer": {
            "successful_positive_lr_updates": updates,
            "positive_rows_per_update": positive_rows,
            "update_rule": "cpu-smoke-fixed-successful-updates",
        },
        "execution": {
            "expected_pipeline_stamp": expected_stamp,
            "minimum_train_upd_s": 70.0,
            "warning_train_upd_s": 80.0,
        },
    }

    class SmokeDataset:
        def __init__(self, *args, **kwargs) -> None:
            rng = np.random.default_rng(109)
            self.features = rng.normal(size=(240, 8)).astype(np.float32)
            self.substrate = {
                "signature": {
                    "kind": "file",
                    "canonical_path": "/data/r0109-smoke/substrate.json",
                    "bytes": 1,
                    "sha256": "d" * 64,
                }
            }

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
            self._bench_seconds = 0.01
            self._setup_seconds = 0.001

        def fit(self, wrapper, **kwargs) -> None:
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
        round0107_nodes,
        "_graph",
        lambda active, job, **kwargs: graph,
    )
    monkeypatch.setattr(
        round0107_nodes,
        "train_config",
        lambda **kwargs: (config, "e" * 64),
    )
    monkeypatch.setattr(
        round0107_nodes,
        "CompactHostInt8MaterializedArray",
        SmokeDataset,
    )
    monkeypatch.setattr(
        round0107_nodes,
        "Round0107TrainingInput",
        SmokeWrapper,
    )
    monkeypatch.setattr(
        round0107_nodes,
        "_new_model",
        lambda config, **kwargs: SmokeModel(),
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
    result = round0107_nodes.run_train_contract(
        {"manifest": {"release_sha": "a" * 40}},
        {"outputs": [str(output)]},
        round_id="0109",
        seed=43,
        train_config_schema="round0109-diverse-jina-train-config-v1",
        production_config_schema="round0109-production-config-v1",
        train_receipt_schema="round0109-diverse-jina-train-receipt-v1",
        output_label="R0109 CPU smoke output",
    )
    assert result["optimizer_updates"] == updates
    assert result["train_checks"]["weighted_rejection_accounting_closes"]

    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0109 CPU smoke receipt")
    assert receipt["model"]["sha256"] == result["model"]["sha256"]

    checkpoint = torch.load(
        output / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    features = SmokeDataset().features
    coordinates = torch.nn.functional.linear(
        torch.from_numpy(features),
        checkpoint["weight"],
        checkpoint["bias"],
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
        provenance={"round": "0109", "mode": "cpu-smoke"},
    )
    assert panel["guards"]["coords_finite"] is True
    assert panel["guards"]["coords_collapsed"] is False
