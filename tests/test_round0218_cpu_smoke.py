"""Bounded CPU smoke for the R0218 authenticate -> score -> seal path.

Reaches the real sealed-receipt binding, the real map authentication against the
substrate signatures, the real per-cell metric and per-corpus FFR views, the real
execution checks, the real family evidence builder and the real receipt seal.
Only the GPU scoring kernels (`score_panel`, `build_hiD_reference`, the centroid
k-means, the checkpoint loader) are stubbed. Its job is to catch late
NameErrors, payload-shape drift and serialization failures in milliseconds
instead of after a GPU node.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

import basemap.panel_v2 as panel_v2
import basemap.pumap.parametric_umap as pumap
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.round0218_minilm_2m_panel import (
    CAPABILITY,
    CORPUS_SLUGS,
    EVALUATION_SCHEMA,
    MAP_TRAIN_SCHEMA,
    ROUND_ID,
    Round0218Error,
    SEALED_DIRECTED_EDGES,
    SEEDS,
    map_capability,
)
from experiments import round0218_nodes
from experiments import score_complete_panel


SMOKE_ROWS = 512
SMOKE_DIM = 8
ANCHORS = 64
INVARIANT = "1" * 64


def _stub_panel(seed: int) -> dict[str, Any]:
    offset = (seed - 42) * 0.001
    return {
        "density": 0.6 + offset,
        "ffr": 0.40 + offset,
        "purity": {"k256": 0.97 + offset, "k1024": 1.02 + offset},
        "ffr_by_group": {
            slug: {"anchors": ANCHORS // len(CORPUS_SLUGS), "ffr": 0.40 + offset}
            for slug in CORPUS_SLUGS
        },
        "guards": {
            "coords_finite": True,
            "coords_collapsed": False,
            "emb_finite": True,
            "emb_zero_rows": 0,
        },
        "provenance": {"hiD_reference_reused": True},
    }


def _write_map(tmp_path, seed: int, sealed: dict[str, Any]) -> dict[str, Any]:
    capability = map_capability(seed)
    root = tmp_path / capability
    root.mkdir()
    model_path = root / "model.pt"
    model_path.write_bytes(b"smoke-checkpoint-" + str(seed).encode("ascii"))
    receipt = prompt_contract.seal({
        "schema": MAP_TRAIN_SCHEMA,
        "round_id": "0217",
        "capability": capability,
        "training_seed": seed,
        "training_performed": True,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "rows": SMOKE_ROWS,
        "dimension": SMOKE_DIM,
        "directed_edges": SEALED_DIRECTED_EDGES,
        "graph_capability": "minilm-mixed-2m-substrate-and-exact-k15-graph-v1",
        "seed_invariant_sha256": INVARIANT,
        "substrate": sealed["substrate_signature"],
        "graph_manifest": sealed["manifest_signature"],
        "model": expected_input_signature(str(model_path)),
        "train_checks": {"exact_update_closure": True},
    })
    receipt_path = root / "train-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "seed": seed,
        "capability": capability,
        "train_receipt": expected_input_signature(str(receipt_path)),
    }


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict[str, Any]:
    import torch

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(round0218_nodes, "ROWS", SMOKE_ROWS)
    monkeypatch.setattr(round0218_nodes, "DIMENSION", SMOKE_DIM)

    substrate_path = tmp_path / "substrate.f32.npy"
    substrate = np.random.default_rng(218).normal(
        size=(SMOKE_ROWS, SMOKE_DIM)
    ).astype(np.float32)
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    np.save(substrate_path, substrate)
    provenance_path = tmp_path / "provenance.npy"
    provenance_path.write_bytes(b"smoke-provenance")
    manifest_path = tmp_path / "substrate-graph.json"
    manifest_path.write_bytes(b"smoke-manifest")
    sealed = {
        "manifest": {},
        "manifest_signature": expected_input_signature(str(manifest_path)),
        "substrate_signature": expected_input_signature(str(substrate_path)),
        "provenance_signature": expected_input_signature(str(provenance_path)),
        "directed_edges": SEALED_DIRECTED_EDGES,
        "ordered_substrate_sha256": "a" * 64,
    }
    corpus_of_row = np.repeat(
        np.arange(len(CORPUS_SLUGS), dtype=np.int64), SMOKE_ROWS // len(CORPUS_SLUGS)
    )
    anchors = np.sort(
        np.random.RandomState(0).choice(SMOKE_ROWS, ANCHORS, replace=False)
    ).astype(np.int64)
    # Every corpus must appear among the anchors; force it deterministically.
    anchors[: len(CORPUS_SLUGS)] = np.array(
        [index * (SMOKE_ROWS // len(CORPUS_SLUGS)) for index in range(len(CORPUS_SLUGS))],
        dtype=np.int64,
    )
    anchors = np.unique(anchors)

    reference = {"kf": 16, "key": "b" * 64, "content_sha256": "c" * 64}

    def _save_reference(ref: Any, path: str) -> str:
        with open(path, "wb") as handle:
            handle.write(b"smoke-reference")
        return path

    class SmokeModel:
        def transform(self, X: Any, batch_size: int = 8192) -> np.ndarray:
            values = np.asarray(X, dtype=np.float32)
            return np.stack([values[:, 0], values[:, 1]], axis=1)

    class SmokeParametricUMAP:
        @classmethod
        def load(cls, path: str, device: str | None = None) -> SmokeModel:
            assert os.path.exists(path)
            return SmokeModel()

    scored: list[int] = []

    def _score_panel(X: Any, Z: Any, **kwargs: Any) -> dict[str, Any]:
        seed = int(kwargs["provenance"]["seed"])
        assert kwargs["scale_admission"] is None
        assert kwargs["hiD_reference"] is reference
        assert len(kwargs["ffr_group_labels"]) == len(anchors)
        assert len(Z) == SMOKE_ROWS
        scored.append(seed)
        return _stub_panel(seed)

    monkeypatch.setattr(round0218_nodes, "_sealed_substrate", lambda job: sealed)
    monkeypatch.setattr(round0218_nodes, "_open_substrate", lambda sealed: substrate)
    monkeypatch.setattr(round0218_nodes, "_corpus_of_row", lambda sealed: corpus_of_row)
    monkeypatch.setattr(
        score_complete_panel,
        "frozen_centroids",
        lambda X, ks, cache_dir, seed=0, iters=25: {
            int(k): _write_centroids(cache_dir, int(k)) for k in ks
        },
    )
    monkeypatch.setattr(panel_v2, "sample_anchors", lambda n, cfg: anchors)
    monkeypatch.setattr(
        panel_v2, "build_hiD_reference", lambda *args, **kwargs: reference
    )
    monkeypatch.setattr(panel_v2, "save_hiD_reference", _save_reference)
    monkeypatch.setattr(panel_v2, "score_panel", _score_panel)
    monkeypatch.setattr(panel_v2, "reset_process_cuda_peak", lambda *a, **k: True)
    monkeypatch.setattr(pumap, "ParametricUMAP", SmokeParametricUMAP)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 2_048)
    return {"sealed": sealed, "scored": scored}


def _write_centroids(cache_dir: str, k: int) -> np.ndarray:
    values = np.random.default_rng(k).normal(size=(k, SMOKE_DIM)).astype(np.float32)
    np.save(os.path.join(cache_dir, f"centroids_k{k}.npy"), values)
    return values


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path, *, cells=None) -> dict[str, Any]:
    fixture = _fixture(monkeypatch, tmp_path)
    sealed = fixture["sealed"]
    job_cells = cells if cells is not None else [
        _write_map(tmp_path, seed, sealed) for seed in SEEDS
    ]
    output = tmp_path / "panel-output"
    round0218_nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
        {
            "action": round0218_nodes.ACTION,
            "graph_manifest_signature": sealed["manifest_signature"],
            "cells": job_cells,
            "outputs": [str(output)],
        },
    )
    with (output / "seed-family-panel.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0218 CPU smoke receipt")
    return receipt


def test_round0218_scores_all_four_cells_and_seals(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    receipt = _run(monkeypatch, tmp_path)
    assert receipt["schema"] == EVALUATION_SCHEMA
    assert receipt["capabilities"] == [CAPABILITY]
    assert receipt["seeds"] == list(SEEDS)
    assert receipt["gate_registered"] is False
    assert receipt["gate_registerable_here"] is False
    assert receipt["map_quality_claim_available"] is False
    assert receipt["evaluation_performed"] is True
    assert receipt["training_performed"] is False
    assert all(receipt["execution_checks"].values())
    assert set(receipt["panel_metric_cells"]) == {str(seed) for seed in SEEDS}
    assert set(receipt["corpus_ffr_cells"]["45"]) == set(CORPUS_SLUGS)
    assert receipt["seed_invariant_sha256"] == INVARIANT
    assert set(receipt["anchor_corpus_counts"]) == set(CORPUS_SLUGS)


def test_round0218_rejects_a_foreign_action_and_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    with pytest.raises(Round0218Error):
        round0218_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "something_else"}
        )
    with pytest.raises(Round0218Error):
        round0218_nodes.run_panel(
            {"manifest": {"round_id": "0217"}}, {"action": round0218_nodes.ACTION}
        )


def test_round0218_refuses_a_map_trained_on_other_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    sealed = fixture["sealed"]
    foreign = dict(sealed)
    foreign["substrate_signature"] = dict(sealed["substrate_signature"])
    foreign["substrate_signature"]["sha256"] = "d" * 64
    cells = [_write_map(tmp_path, SEEDS[0], foreign)]
    cells.extend(_write_map(tmp_path, seed, sealed) for seed in SEEDS[1:])
    with pytest.raises(Round0218Error):
        round0218_nodes.run_panel(
            {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
            {
                "action": round0218_nodes.ACTION,
                "graph_manifest_signature": sealed["manifest_signature"],
                "cells": cells,
                "outputs": [str(tmp_path / "never-created")],
            },
        )


def test_round0218_refuses_an_incomplete_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    sealed = fixture["sealed"]
    cells = [_write_map(tmp_path, seed, sealed) for seed in SEEDS[:3]]
    with pytest.raises(Round0218Error):
        round0218_nodes.run_panel(
            {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
            {
                "action": round0218_nodes.ACTION,
                "graph_manifest_signature": sealed["manifest_signature"],
                "cells": cells,
                "outputs": [str(tmp_path / "never-created")],
            },
        )
