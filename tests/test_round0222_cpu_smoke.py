"""Bounded CPU smoke for the R0222 authenticate -> score -> pool -> gate path.

Reaches the real R0221 map authentication, the real byte-identical-reference
binding, the real per-cell metric and per-corpus views, the real pooling of
R0218's four sealed cells with four freshly scored ones, the real n=8 and n=4
gate arithmetic, the real jackknife, the real precedent-artifact retraction check
and the real receipt seal. Only the GPU kernels (`score_panel`, the reference
loader, the checkpoint loader) are stubbed.
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
    CAPABILITY as PANEL_CAPABILITY,
    CENTROID_KS,
    CORPUS_SLUGS,
    EVALUATION_SCHEMA as PANEL_SCHEMA,
    PANEL_METRICS,
    SEALED_DIRECTED_EDGES,
    SEEDS as R0218_SEEDS,
)
from basemap.round0221_minilm_2m_seed_extension import (
    R0217_SEED_INVARIANT_SHA256,
    SEEDS as R0221_SEEDS,
    TRAIN_SCHEMA as R0221_TRAIN_SCHEMA,
    capability_for_seed as r0221_capability_for_seed,
)
from basemap.round0222_minilm_2m_gate_n8 import (
    ACCEPTED_SIX_METRIC_SET,
    CAPABILITY,
    GATE_METRICS,
    GATE_SCHEMA,
    PANEL_EXTENSION_CAPABILITY,
    POOLED_SEEDS,
    PRECEDENT_CAPABILITIES,
    ROUND_ID,
    Round0222Error,
)
from experiments import round0218_nodes, round0222_nodes


SMOKE_ROWS = 512
SMOKE_DIM = 8
ANCHORS = 64
R0218_STUB_CELLS = {
    "42": {"density_v2": 0.4377, "ffr": 0.3369, "pk256": 0.9789, "pk1024": 0.7326},
    "43": {"density_v2": 0.4406, "ffr": 0.3382, "pk256": 0.9941, "pk1024": 0.7229},
    "44": {"density_v2": 0.4387, "ffr": 0.3258, "pk256": 0.9954, "pk1024": 0.6980},
    "45": {"density_v2": 0.4477, "ffr": 0.3227, "pk256": 0.9929, "pk1024": 0.6936},
}


def _stub_panel(seed: int) -> dict[str, Any]:
    offset = (seed - 46) * 0.001
    return {
        "density": 0.44 + offset,
        "ffr": 0.33 + offset,
        "purity": {"k256": 0.99 + offset, "k1024": 0.71 + offset},
        "purity_numerators": {
            "k256": {"hi_D_agreement": 0.3828, "map_agreement": 0.39},
            "k1024": {"hi_D_agreement": 0.2385, "map_agreement": 0.17},
        },
        "ffr_by_group": {
            slug: {"anchors": ANCHORS // len(CORPUS_SLUGS), "ffr": 0.31 + offset}
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


def _write_json(path, payload: dict[str, Any]) -> dict[str, Any]:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return expected_input_signature(str(path))


def _write_precedents(tmp_path) -> dict[str, Any]:
    signatures: dict[str, Any] = {}
    floors = {"0161": 0.19134355783912885, "0193": 0.18616941334799972}
    for round_id, floor in floors.items():
        payload = prompt_contract.seal({
            "schema": f"round{round_id}-quality-gates-v1",
            "round_id": round_id,
            "capability": PRECEDENT_CAPABILITIES[round_id],
            "formula": "family mean - 2 * sample standard deviation (ddof=1)",
            "n": 4,
            "seed_family": [42, 43, 44, 45],
            "gates": {
                key: {
                    "floor": floor if key == "density_v2" else 0.5,
                    "mean": floor + 0.02,
                    "sample_sd_ddof1": 0.01,
                }
                for key in ACCEPTED_SIX_METRIC_SET
            },
        })
        signatures[round_id] = _write_json(
            tmp_path / f"precedent-{round_id}.json", payload
        )
    return signatures


def _write_r0221_map(tmp_path, seed: int, sealed: dict[str, Any]) -> dict[str, Any]:
    capability = r0221_capability_for_seed(seed)
    root = tmp_path / capability
    root.mkdir()
    model_path = root / "model.pt"
    model_path.write_bytes(b"smoke-checkpoint-" + str(seed).encode("ascii"))
    receipt = prompt_contract.seal({
        "schema": R0221_TRAIN_SCHEMA,
        "round_id": "0221",
        "treatment_config_round_id": "0217",
        "capability": capability,
        "training_seed": seed,
        "training_performed": True,
        "gate_registerable_here": False,
        "map_decision_made": False,
        "rows": SMOKE_ROWS,
        "dimension": SMOKE_DIM,
        "directed_edges": SEALED_DIRECTED_EDGES,
        "graph_capability": "minilm-mixed-2m-substrate-and-exact-k15-graph-v1",
        "seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "substrate": sealed["substrate_signature"],
        "graph_manifest": sealed["manifest_signature"],
        "model": expected_input_signature(str(model_path)),
        "train_checks": {"all_2m_coordinates_finite": True},
    })
    return {
        "seed": seed,
        "capability": capability,
        "train_receipt": _write_json(root / "train-receipt.json", receipt),
    }


def _write_r0218_panel(tmp_path, sealed, reference_path, anchor_counts) -> Any:
    centroids: dict[str, Any] = {}
    centroid_root = tmp_path / "centroids"
    centroid_root.mkdir()
    for k in CENTROID_KS:
        values = np.random.default_rng(k).normal(size=(k, SMOKE_DIM)).astype(np.float32)
        path = centroid_root / f"centroids_k{k}.npy"
        np.save(path, values)
        centroids[str(k)] = expected_input_signature(str(path))
    cells: dict[str, Any] = {}
    metric_cells: dict[str, Any] = {}
    corpus_cells: dict[str, Any] = {}
    for seed in R0218_SEEDS:
        stub = R0218_STUB_CELLS[str(seed)]
        model_path = tmp_path / f"r0217-model-seed{seed}.pt"
        model_path.write_bytes(b"r0217-" + str(seed).encode("ascii"))
        cells[str(seed)] = {
            "seed": seed,
            "model": expected_input_signature(str(model_path)),
            "panel": {
                "purity_numerators": {
                    "k256": {"hi_D_agreement": 0.3828, "map_agreement": 0.39},
                    "k1024": {"hi_D_agreement": 0.2385, "map_agreement": 0.17},
                }
            },
        }
        metric_cells[str(seed)] = {
            "density_v2": stub["density_v2"],
            "ffr": stub["ffr"],
            "purity_fidelity_k256": stub["pk256"],
            "purity_fidelity_k1024": stub["pk1024"],
        }
        corpus_cells[str(seed)] = {
            slug: {"anchors": ANCHORS // len(CORPUS_SLUGS), "ffr": 0.30 + 0.001 * seed}
            for slug in CORPUS_SLUGS
        }
    payload = prompt_contract.seal({
        "schema": PANEL_SCHEMA,
        "round_id": "0218",
        "capability": PANEL_CAPABILITY,
        "capabilities": [PANEL_CAPABILITY],
        "seeds": list(R0218_SEEDS),
        "n": len(R0218_SEEDS),
        "metrics": list(PANEL_METRICS),
        "seed_invariant_sha256": R0217_SEED_INVARIANT_SHA256,
        "evaluation_performed": True,
        "gate_registered": False,
        "execution_checks": {"all_four_cells_scored": True},
        "panel_metric_cells": metric_cells,
        "corpus_ffr_cells": corpus_cells,
        "cells": cells,
        "shared_high_d_reference": expected_input_signature(str(reference_path)),
        "high_d_reference_key": "b" * 64,
        "high_d_reference_content_sha256": "c" * 64,
        "centroids": centroids,
        "anchor_corpus_counts": dict(anchor_counts),
        "release_sha": "e" * 40,
    })
    return _write_json(tmp_path / "seed-family-panel.json", payload)


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict[str, Any]:
    import torch

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(round0222_nodes, "ROWS", SMOKE_ROWS)
    monkeypatch.setattr(round0222_nodes, "DIMENSION", SMOKE_DIM)

    substrate_path = tmp_path / "substrate.f32.npy"
    substrate = np.random.default_rng(222).normal(
        size=(SMOKE_ROWS, SMOKE_DIM)
    ).astype(np.float32)
    substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
    np.save(substrate_path, substrate)
    provenance_path = tmp_path / "provenance.npy"
    provenance_path.write_bytes(b"smoke-provenance")
    manifest_path = tmp_path / "substrate-graph.json"
    manifest_path.write_bytes(b"smoke-manifest")
    reference_path = tmp_path / "minilm-2m-high-d-reference.npz"
    reference_path.write_bytes(b"smoke-reference")
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
        np.unique(
            np.concatenate([
                np.asarray(
                    [
                        index * (SMOKE_ROWS // len(CORPUS_SLUGS))
                        for index in range(len(CORPUS_SLUGS))
                    ],
                    dtype=np.int64,
                ),
                np.random.RandomState(0)
                .choice(SMOKE_ROWS, ANCHORS - len(CORPUS_SLUGS), replace=False)
                .astype(np.int64),
            ])
        )
    )
    reference = {
        "kf": 16,
        "key": "b" * 64,
        "content_sha256": "c" * 64,
        "anchor_ids": anchors,
    }

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
    monkeypatch.setattr(panel_v2, "sample_anchors", lambda n, cfg: anchors)
    monkeypatch.setattr(
        panel_v2, "load_hiD_reference", lambda path, **kwargs: reference
    )
    monkeypatch.setattr(
        panel_v2, "hiD_reference_key", lambda *args, **kwargs: (reference["key"], {})
    )
    monkeypatch.setattr(panel_v2, "score_panel", _score_panel)
    monkeypatch.setattr(panel_v2, "reset_process_cuda_peak", lambda *a, **k: True)
    monkeypatch.setattr(pumap, "ParametricUMAP", SmokeParametricUMAP)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 1_024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 2_048)
    anchor_counts = {
        slug: int((corpus_of_row[anchors] == index).sum())
        for index, slug in enumerate(CORPUS_SLUGS)
    }
    panel_signature = _write_r0218_panel(
        tmp_path, sealed, reference_path, anchor_counts
    )
    return {
        "sealed": sealed,
        "scored": scored,
        "panel_signature": panel_signature,
        "reference": reference,
    }


def _job(tmp_path, fixture: dict[str, Any], *, cells=None, output="gate-output"):
    sealed = fixture["sealed"]
    return {
        "action": round0222_nodes.ACTION,
        "graph_manifest_signature": sealed["manifest_signature"],
        "panel_evidence": fixture["panel_signature"]["canonical_path"],
        "cells": cells
        if cells is not None
        else [_write_r0221_map(tmp_path, seed, sealed) for seed in R0221_SEEDS],
        "precedent_gate_signatures": _write_precedents(tmp_path),
        "upstream_review_state": {"required_reviews": ["0217", "0218", "0221"]},
        "outputs": [str(tmp_path / output)],
    }


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict[str, Any]:
    fixture = _fixture(monkeypatch, tmp_path)
    job = _job(tmp_path, fixture)
    round0222_nodes.run_job(
        {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}}, job
    )
    with (tmp_path / "gate-output" / "minilm-quality-gates-n8.json").open(
        encoding="utf-8"
    ) as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0222 CPU smoke receipt")
    assert sorted(fixture["scored"]) == sorted(R0221_SEEDS)
    return receipt


def test_round0222_scores_four_pools_eight_and_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    receipt = _run(monkeypatch, tmp_path)
    assert receipt["schema"] == GATE_SCHEMA
    assert receipt["capabilities"] == [PANEL_EXTENSION_CAPABILITY, CAPABILITY]
    assert receipt["n"] == 8
    assert receipt["seed_family"] == list(POOLED_SEEDS)
    assert set(receipt["gates"]) == set(GATE_METRICS)
    assert "density_v2" in receipt["gates"]
    assert receipt["gates"]["density_v2"]["n"] == 8
    assert receipt["excluded_by_judgement"] == {}
    assert sorted(receipt["unavailable_metrics"]) == [
        "heldout_recall_at_10",
        "projection_ffr",
    ]
    assert receipt["gate_registered"] is True
    assert receipt["training_performed"] is False
    assert receipt["reference_byte_identical_to_r0218"] is True
    assert all(receipt["execution_checks"].values())
    assert set(receipt["pooled_panel_metric_cells"]) == {
        str(seed) for seed in POOLED_SEEDS
    }
    assert set(receipt["n4_vs_n8"]) == set(GATE_METRICS)
    assert set(receipt["jackknife"]) == {"n4", "n8"}
    retraction = receipt["r0219_retraction"]
    assert retraction["precedents"]["0161"]["density_v2_floor"] == pytest.approx(
        0.19134355783912885
    )
    assert retraction["precedents"]["0193"]["density_v2_floor"] == pytest.approx(
        0.18616941334799972
    )


def test_round0222_rejects_a_foreign_action_and_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    with pytest.raises(Round0222Error):
        round0222_nodes.run_job(
            {"manifest": {"round_id": ROUND_ID}}, {"action": "something_else"}
        )
    with pytest.raises(Round0222Error):
        round0222_nodes.run_registration(
            {"manifest": {"round_id": "0218"}}, {"action": round0222_nodes.ACTION}
        )


def test_round0222_aborts_when_the_reference_is_not_r0218s(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The mandate's stop condition: a different reference means no gate."""
    fixture = _fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        panel_v2,
        "load_hiD_reference",
        lambda path, **kwargs: {
            **fixture["reference"],
            "content_sha256": "9" * 64,
        },
    )
    with pytest.raises(Round0222Error) as excinfo:
        round0222_nodes.run_registration(
            {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
            _job(tmp_path, fixture, output="never-created"),
        )
    assert "NOT comparable" in str(excinfo.value)


def test_round0222_aborts_on_an_incommensurate_or_incomplete_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    sealed = fixture["sealed"]
    cells = [_write_r0221_map(tmp_path, seed, sealed) for seed in R0221_SEEDS[:3]]
    with pytest.raises(Round0222Error):
        round0222_nodes.run_registration(
            {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
            _job(tmp_path, fixture, cells=cells, output="never-created"),
        )


def test_round0222_refuses_a_map_trained_on_other_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fixture = _fixture(monkeypatch, tmp_path)
    sealed = fixture["sealed"]
    foreign = dict(sealed)
    foreign["substrate_signature"] = dict(sealed["substrate_signature"])
    foreign["substrate_signature"]["sha256"] = "d" * 64
    cells = [_write_r0221_map(tmp_path, R0221_SEEDS[0], foreign)]
    cells.extend(_write_r0221_map(tmp_path, seed, sealed) for seed in R0221_SEEDS[1:])
    with pytest.raises(Round0222Error):
        round0222_nodes.run_registration(
            {"manifest": {"round_id": ROUND_ID, "release_sha": "f" * 40}},
            _job(tmp_path, fixture, cells=cells, output="never-created"),
        )
