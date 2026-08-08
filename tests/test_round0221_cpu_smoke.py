"""Bounded CPU smoke for the R0221 train -> seal -> publish -> reload path.

Same shape as R0217's smoke, extended to the one thing R0221 adds: the reloaded
checkpoint projects the **full population** and every coordinate is checked. Only
the GPU kernel, the endpoint array and the sampler are stubbed; the dose
assertion, the R0217-template config construction, the seed-invariant digest
equality, the post-fit accounting, the checkpoint publish and the receipt seal
are all the real code.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import basemap.pumap.parametric_umap as pumap
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0221_minilm_2m_seed_extension import (
    BATCH_SIZE,
    DIMENSION,
    POSITIVE_ROWS_PER_UPDATE,
    R0217_SEED_INVARIANT_SHA256,
    REGISTERED_SUCCESSFUL_UPDATES,
    ROUND_ID,
    ROWS,
    Round0221Error,
    SEALED_DIRECTED_EDGES,
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
    SEEDS,
    capability_for_seed,
    seed_invariant_sha256,
    successful_updates_for_edges,
    train_config,
)
from experiments import round0221_nodes


def _signature(path: str, digest: str) -> dict[str, Any]:
    return {"kind": "file", "canonical_path": path, "bytes": 1_024, "sha256": digest * 64}


def _smoke(monkeypatch: pytest.MonkeyPatch, tmp_path, *, seed: int) -> dict[str, Any]:
    import torch

    updates = successful_updates_for_edges(SEALED_DIRECTED_EDGES)
    # The registered sealed R0216 signatures, so the smoke reaches the real
    # digest-equality check against R0217's published value rather than a stub.
    graph_signature = dict(SEALED_GRAPH_SIGNATURE)
    manifest_signature = dict(SEALED_GRAPH_MANIFEST_SIGNATURE)
    substrate_signature = dict(SEALED_SUBSTRATE_SIGNATURE)
    config, config_sha = train_config(
        seed=seed,
        graph_signature=graph_signature,
        graph_manifest_signature=manifest_signature,
        substrate_signature=substrate_signature,
        graph_edges=SEALED_DIRECTED_EDGES,
        rows=ROWS,
    )
    invariant = seed_invariant_sha256(config)
    assert invariant == R0217_SEED_INVARIANT_SHA256

    producer_batches = updates + 1
    expected_rows = updates * BATCH_SIZE
    emitted = producer_batches * POSITIVE_ROWS_PER_UPDATE
    accepted = emitted + 7
    proposals = accepted + 1_000
    runtime = {
        **config["execution"]["expected_pipeline_stamp"],
        "endpoint_gather_calls": updates,
        "source_rows_gathered": expected_rows,
        "destination_rows_gathered": expected_rows,
        "host_prefetch_batches_filled": producer_batches,
        "host_prefetch_producer_batches": producer_batches,
        "host_prefetch_consumer_batches": updates,
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
        "n_pos_edges": SEALED_DIRECTED_EDGES,
    }
    accounting.update({
        f"pipeline_{key}": runtime[key]
        for key in prompt_contract._DYNAMIC_PIPELINE_COUNTERS
    })
    graph = {
        "manifest": {"substrate": substrate_signature},
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "sources": np.arange(8, dtype=np.int32),
        "targets": np.roll(np.arange(8, dtype=np.int32), -1),
        "weights": np.ones(8, dtype=np.float32),
        "n_nodes": ROWS,
        "directed_edges": SEALED_DIRECTED_EDGES,
    }
    #: A small real substrate stands in for the 3.07 GB memmap; the reload stub
    #: below returns a full (ROWS, 2) array regardless, so the registered
    #: full-population finiteness check runs on its real shape.
    source = np.random.default_rng(221).normal(size=(64, DIMENSION)).astype(np.float32)
    full_coordinates = np.tile(
        np.asarray([[2.0, -3.0], [-1.0, 5.0]], dtype=np.float32), (ROWS // 2, 1)
    )

    class SmokeDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.shape = (ROWS, DIMENSION)

    class SmokeWrapper:
        def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
            self.dataset = dataset

        def runtime_stamp(self) -> dict[str, Any]:
            return dict(runtime)

    class SmokeProfiler:
        def finalize(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["bench_seconds"] > 0
            return {"aborted": False, "smoke": True}

    class SmokeModel:
        def __init__(self) -> None:
            self.layer = torch.nn.Linear(DIMENSION, 2)
            self._canary_profiler = SmokeProfiler()
            self._bench_seconds = 1.0
            self._setup_seconds = 0.001

        def fit(self, wrapper: Any, **kwargs: Any) -> None:
            assert kwargs["random_state"] == seed
            assert kwargs["precomputed_edges_path"] == graph_signature["canonical_path"]
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save({"seed": seed}, path)

    class SmokeReloaded:
        def transform(self, X: Any, batch_size: int = 8192) -> np.ndarray:
            assert batch_size == round0221_nodes.FULL_TRANSFORM_BATCH
            return full_coordinates

    class SmokeParametricUMAP:
        @classmethod
        def load(cls, path: str, device: str | None = None) -> SmokeReloaded:
            torch.load(path, map_location="cpu", weights_only=True)
            return SmokeReloaded()

    monkeypatch.setattr(round0221_nodes, "_sealed_graph", lambda job: graph)
    monkeypatch.setattr(
        round0221_nodes, "_open_substrate", lambda graph: (source, substrate_signature)
    )
    monkeypatch.setattr(round0221_nodes, "MiniLMHostFp32EndpointArray", SmokeDataset)
    monkeypatch.setattr(round0221_nodes, "MiniLMMixedTrainingInput", SmokeWrapper)
    monkeypatch.setattr(
        round0221_nodes.prompt_nodes, "_new_model", lambda config: SmokeModel()
    )
    monkeypatch.setattr(pumap, "ParametricUMAP", SmokeParametricUMAP)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda value: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (1_000_000_000, 2_000_000_000)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 2_048)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    output = tmp_path / f"train-output-seed{seed}"
    job = {
        "action": round0221_nodes.ACTION,
        "training_seed": seed,
        "capability": capability_for_seed(seed),
        "graph_manifest_signature": manifest_signature,
        "family_seed_invariant_sha256": invariant,
        "registered_dose_bound": 120_000,
        "outputs": [str(output)],
    }
    round0221_nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}}, job
    )
    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0221 CPU smoke receipt")
    with (output / "production-config.json").open(encoding="utf-8") as handle:
        published_config = json.load(handle)
    assert published_config["config_sha256"] == config_sha
    assert published_config["seed_invariant_sha256"] == invariant
    assert published_config["treatment_config_round_id"] == "0217"
    return receipt


def test_round0221_train_seal_publish_reload_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    receipt = _smoke(monkeypatch, tmp_path, seed=46)
    assert receipt["training_seed"] == 46
    assert receipt["round_id"] == ROUND_ID
    assert receipt["treatment_config_round_id"] == "0217"
    assert receipt["capability"] == capability_for_seed(46)
    assert receipt["optimizer_updates"] == REGISTERED_SUCCESSFUL_UPDATES
    assert receipt["directed_edges"] == SEALED_DIRECTED_EDGES
    assert receipt["seed_invariant_sha256"] == R0217_SEED_INVARIANT_SHA256
    assert receipt["dose_registration"]["landed_on_registered_ceil_value"] is True
    assert receipt["gate_registerable_here"] is False
    assert receipt["train_accounting"]["pipeline_runtime"] == receipt[
        "exact_execution_receipt"
    ]
    assert receipt["train_checks"]["treatment_identical_to_r0217_except_seed"] is True
    assert receipt["train_checks"]["all_2m_coordinates_finite"] is True
    published = receipt["published_map_check"]
    assert published["coordinates_finite"] is True
    assert published["collapsed"] is False
    assert published["transform_rows"] == ROWS
    assert published["transform_rows_finite"] == ROWS


def test_round0221_every_seed_runs_the_same_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    invariants = set()
    for seed in SEEDS:
        receipt = _smoke(monkeypatch, tmp_path, seed=seed)
        assert receipt["training_seed"] == seed
        invariants.add(receipt["seed_invariant_sha256"])
    assert invariants == {R0217_SEED_INVARIANT_SHA256}


def test_round0221_rejects_a_foreign_action_queue_and_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    with pytest.raises(Round0221Error):
        round0221_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "something_else"}
        )
    with pytest.raises(Round0221Error):
        round0221_nodes.run_train(
            {"manifest": {"round_id": "0217"}},
            {"action": round0221_nodes.ACTION, "training_seed": 46},
        )
    for seed in (42, 45, 99):
        with pytest.raises(Round0221Error):
            round0221_nodes.run_train(
                {"manifest": {"round_id": ROUND_ID}},
                {
                    "action": round0221_nodes.ACTION,
                    "training_seed": seed,
                    "capability": f"minilm-mixed-2m-map-seed{seed}-low-dose-v1",
                },
            )


def test_round0221_rejects_a_drifted_family_invariant(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import torch

    graph_signature = _signature(str(tmp_path / "edges.npz"), "a")
    manifest_signature = _signature(str(tmp_path / "manifest.json"), "b")
    substrate_signature = _signature(str(tmp_path / "substrate.npy"), "c")
    graph = {
        "manifest": {"substrate": substrate_signature},
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "n_nodes": ROWS,
        "directed_edges": SEALED_DIRECTED_EDGES,
    }
    monkeypatch.setattr(round0221_nodes, "_sealed_graph", lambda job: graph)
    monkeypatch.setattr(
        round0221_nodes,
        "_open_substrate",
        lambda graph: (np.zeros((4, DIMENSION), dtype=np.float32), substrate_signature),
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda value: None)
    with pytest.raises(Round0221Error):
        round0221_nodes.run_train(
            {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
            {
                "action": round0221_nodes.ACTION,
                "training_seed": 47,
                "capability": capability_for_seed(47),
                "graph_manifest_signature": manifest_signature,
                "family_seed_invariant_sha256": "0" * 64,
                "outputs": [str(tmp_path / "never-created")],
            },
        )
