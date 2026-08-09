"""Bounded CPU smoke for R0229 phase 2's train -> seal -> publish -> reload path.

R0228's smoke shape. Only the GPU kernel, the endpoint array and the sampler are
stubbed; the ceil-derived dose, the R0217-template config construction with the
spill-lifted graph moved in, the cross-round treatment-digest equality, the
post-fit accounting, the checkpoint publish, the full-population reload, the
coordinate publication and the receipt seal are the real code.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import basemap.pumap.parametric_umap as pumap
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0221_minilm_2m_seed_extension import (
    SEALED_GRAPH_MANIFEST_SIGNATURE,
    SEALED_GRAPH_SIGNATURE,
    SEALED_SUBSTRATE_SIGNATURE,
)
from basemap.round0228_low_c_map import (
    BATCH_SIZE,
    POSITIVE_ROWS_PER_UPDATE,
    successful_updates_for_edges,
)
from basemap.round0229_train_config import train_config
from basemap.round0229_phase2_contract import (
    ARM_NAME,
    DIMENSION,
    ROUND_ID,
    ROWS,
    SEEDS,
    TREATMENT_INVARIANT_SHA256,
    map_capability,
)
from basemap.round0229_quality_contract import Round0229Error
from experiments import round0229_phase2_nodes as nodes


SPILL_LIFTED_EDGES = 48_120_000
CLUSTERS = 200
SPILL = 8
NN_DESCENT = {
    "graph_degree": 64, "intermediate_graph_degree": 256, "max_iterations": 40,
}
GRAPH_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0229/x/edges-k15-fuzzy.npz",
    "bytes": 123,
    "sha256": "a" * 64,
}
MANIFEST_SIGNATURE: dict[str, Any] = {
    "kind": "file",
    "canonical_path": "/data/latent-basemap/runs/round-0229/x/spill-lifted-graph.json",
    "bytes": 456,
    "sha256": "b" * 64,
}


def _smoke(monkeypatch: pytest.MonkeyPatch, tmp_path, *, seed: int) -> dict[str, Any]:
    import torch

    edges = SPILL_LIFTED_EDGES
    updates = successful_updates_for_edges(edges)
    substrate_signature = dict(SEALED_SUBSTRATE_SIGNATURE)
    config, config_sha, invariant = train_config(
        clusters=CLUSTERS,
        spill=SPILL,
        nn_descent=NN_DESCENT,
        seed=seed,
        graph_signature=dict(GRAPH_SIGNATURE),
        graph_manifest_signature=dict(MANIFEST_SIGNATURE),
        substrate_signature=substrate_signature,
        r0216_graph_signature=dict(SEALED_GRAPH_SIGNATURE),
        r0216_graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        graph_edges=edges,
        rows=ROWS,
    )

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
        "n_pos_edges": edges,
    }
    accounting.update({
        f"pipeline_{key}": runtime[key]
        for key in prompt_contract._DYNAMIC_PIPELINE_COUNTERS
    })
    graph_manifest = {
        "substrate": substrate_signature,
        "clusters": CLUSTERS,
        "spill": SPILL,
        "graph": dict(GRAPH_SIGNATURE),
        "builder": {"nn_descent": dict(NN_DESCENT)},
        "directed_edge_count": edges,
        "degrees": {"zero_degree_rows": 0},
        "recall_against_r0220_exact_truth": {
            "population": "all-2000000-substrate-rows",
            "tie_aware": {"mean": 0.996},
            "rows_carrying_any_loss_fraction": 0.02,
        },
    }
    # The bundle MiniLMMixedTrainingInput actually needs: it checks n_nodes
    # against len(dataset), so a bundle without the edge arrays fails at the
    # wrapper rather than at the config. R0229's first train attempt did.
    bundle = {
        "manifest": graph_manifest,
        "manifest_signature": dict(MANIFEST_SIGNATURE),
        "signature": dict(GRAPH_SIGNATURE),
        "edges_path": GRAPH_SIGNATURE["canonical_path"],
        "sources": np.arange(8, dtype=np.int32),
        "targets": np.roll(np.arange(8, dtype=np.int32), -1),
        "weights": np.ones(8, dtype=np.float32),
        "n_nodes": ROWS,
        "directed_edges": edges,
    }
    source = np.random.default_rng(229).normal(size=(64, DIMENSION)).astype(np.float32)
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
            assert kwargs["precomputed_edges_path"] == GRAPH_SIGNATURE["canonical_path"]
            self._train_stats = dict(accounting)

        def save(self, path: str) -> None:
            torch.save({"seed": seed, "arm": ARM_NAME}, path)

    class SmokeReloaded:
        def transform(self, X: Any, batch_size: int = 8192) -> np.ndarray:
            return full_coordinates

    class SmokeParametricUMAP:
        @classmethod
        def load(cls, path: str, device: str | None = None) -> SmokeReloaded:
            torch.load(path, map_location="cpu", weights_only=True)
            return SmokeReloaded()

    import basemap.round0217_minilm_2m_pipeline as pipeline_module

    monkeypatch.setattr(nodes, "_train_graph", lambda job: bundle)
    monkeypatch.setattr(
        nodes, "_open_substrate", lambda manifest: (source, substrate_signature)
    )
    monkeypatch.setattr(pipeline_module, "MiniLMHostFp32EndpointArray", SmokeDataset)
    monkeypatch.setattr(pipeline_module, "MiniLMMixedTrainingInput", SmokeWrapper)
    monkeypatch.setattr(
        nodes.prompt_nodes, "_new_model", lambda config: SmokeModel()
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
        "action": nodes.TRAIN_ACTION,
        "training_seed": seed,
        "capability": map_capability(seed),
        "graph_manifest_reference": dict(MANIFEST_SIGNATURE),
        "r0216_graph_signature": dict(SEALED_GRAPH_SIGNATURE),
        "r0216_graph_manifest_signature": dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
        "treatment_invariant_sha256": invariant,
        "outputs": [str(output)],
    }
    nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}}, job
    )
    with (output / "train-receipt.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0229 phase-2 CPU smoke receipt")
    with (output / "production-config.json").open(encoding="utf-8") as handle:
        published_config = json.load(handle)
    assert published_config["config_sha256"] == config_sha
    assert published_config["treatment_invariant_sha256"] == invariant
    assert published_config["treatment_config_round_id"] == "0217"
    assert (output / f"coordinates-seed{seed}.npy").exists()
    return receipt


def test_train_seal_publish_reload_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    receipt = _smoke(monkeypatch, tmp_path, seed=42)
    assert receipt["treatment_invariant_sha256"] == TREATMENT_INVARIANT_SHA256
    assert receipt["capability"] == map_capability(42)
    assert receipt["training_performed"] is True
    assert receipt["adoption_claimed"] is False
    assert receipt["gate_registerable_here"] is False
    assert receipt["directed_edges"] == SPILL_LIFTED_EDGES
    assert receipt["optimizer_updates"] == successful_updates_for_edges(
        SPILL_LIFTED_EDGES
    )
    assert all(receipt["train_checks"].values())


def test_every_seed_runs_the_same_treatment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    digests = set()
    for seed in SEEDS:
        receipt = _smoke(monkeypatch, tmp_path, seed=seed)
        digests.add(receipt["treatment_invariant_sha256"])
        assert receipt["training_seed"] == seed
    assert digests == {TREATMENT_INVARIANT_SHA256}


def test_rejects_a_foreign_queue_and_an_unknown_action() -> None:
    with pytest.raises(Round0229Error):
        nodes.run_job({"manifest": {"round_id": "0228"}}, {"action": nodes.TRAIN_ACTION})
    with pytest.raises(Round0229Error):
        nodes.run_job({"manifest": {"round_id": ROUND_ID}}, {"action": "nope"})


def test_rejects_an_unregistered_seed_and_a_mismatched_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(Round0229Error):
        nodes.run_train(
            {"manifest": {"round_id": ROUND_ID}},
            {"training_seed": 99, "capability": map_capability(99)},
        )
    with pytest.raises(Round0229Error):
        nodes.run_train(
            {"manifest": {"round_id": ROUND_ID}},
            {"training_seed": 42, "capability": "wrong"},
        )


def test_rejects_a_drifted_treatment_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config that does not reproduce the cross-round digest must not train.

    Note the r0216 graph signature is itself graph-bearing and therefore masked
    out of the invariant, so drifting *it* correctly changes nothing. What must
    refuse is a config whose treatment projection differs, and that is what is
    forced here.
    """
    from basemap import round0229_train_config as builder

    monkeypatch.setattr(
        builder, "treatment_invariant_sha256", lambda config: "d" * 64
    )
    with pytest.raises(Round0229Error, match="cross-round treatment digest"):
        builder.train_config(
            clusters=CLUSTERS,
            spill=SPILL,
            nn_descent=dict(NN_DESCENT),
            seed=42,
            graph_signature=dict(GRAPH_SIGNATURE),
            graph_manifest_signature=dict(MANIFEST_SIGNATURE),
            substrate_signature=dict(SEALED_SUBSTRATE_SIGNATURE),
            r0216_graph_signature=dict(SEALED_GRAPH_SIGNATURE),
            r0216_graph_manifest_signature=dict(SEALED_GRAPH_MANIFEST_SIGNATURE),
            graph_edges=SPILL_LIFTED_EDGES,
            rows=ROWS,
        )


def test_the_published_exactness_string_describes_this_arm_not_r0226_constants():
    from basemap.round0229_train_config import graph_exactness

    text = graph_exactness(
        clusters=CLUSTERS, spill=SPILL, nn_descent=dict(NN_DESCENT)
    )
    assert "spill 8" in text
    assert "graph_degree 64" in text
    assert "intermediate 256" in text
    assert "max_iterations 40" in text
    assert "c=200" in text


def test_the_train_bundle_carries_everything_the_pipeline_wrapper_needs():
    """Regression: MiniLMMixedTrainingInput checks n_nodes against len(dataset).

    A bundle without `sources` / `targets` / `weights` / `n_nodes` raised
    "R0217 training input geometry changed" at the wrapper, after the build and
    fuzzy nodes had already spent their GPU time.
    """
    import inspect

    source = inspect.getsource(nodes._train_graph)
    for key in ("sources", "targets", "weights", "n_nodes"):
        assert f'"{key}"' in source, key
    assert "load_edge_arrays" in source
