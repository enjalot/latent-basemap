"""Bounded CPU smoke for R0224's benchmark-substrate assembly.

The assembly is ~30 minutes of node time at full scale, and the guard cannot
catch an arithmetic slip in the replacement loop or an off-by-one in the
permuted write. So it is run here end to end on synthetic shards, at 1/1000 of
the registered size, through the real selection loop, the real permuted write,
the real shard-span assertion, the real prefix-composition check and the real
seal.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0224_cuvs_memory import DIMENSION, Round0224Error
from experiments import round0224_nodes


SMALL_COMPOSITION = (("corpus-a", 800), ("corpus-b", 500), ("corpus-c", 500), ("corpus-d", 200))
SCALE = 8
ROWS = sum(rows for _n, rows in SMALL_COMPOSITION) * SCALE  # 16,000


def _synthetic(monkeypatch: pytest.MonkeyPatch, tmp_path, *, shards_per_corpus: int = 5):
    store: dict[str, list[np.ndarray]] = {}
    rng = np.random.default_rng(2240)
    for name, base in SMALL_COMPOSITION:
        per = int(base) * SCALE * 3 // shards_per_corpus + 10
        store[name] = [
            rng.normal(size=(per, DIMENSION)).astype(np.float32)
            for _ in range(shards_per_corpus)
        ]

    def fake_shards(corpus: str):
        return [(f"{corpus}:{i}", int(a.shape[0]), True) for i, a in enumerate(store[corpus])]

    def fake_open(path: str, rows: int, real_npy: bool):
        corpus, index = path.split(":")
        return store[corpus][int(index)]

    monkeypatch.setattr(round0224_nodes, "COMPOSITION", SMALL_COMPOSITION)
    monkeypatch.setattr(round0224_nodes, "BENCHMARK_ROWS", ROWS)
    monkeypatch.setattr(round0224_nodes, "BENCHMARK_COMPOSITION_SCALE", SCALE)
    monkeypatch.setattr(round0224_nodes, "SWEEP_ROWS", (2_000, 4_000, 8_000, ROWS))
    monkeypatch.setattr(round0224_nodes, "_shards", fake_shards)
    monkeypatch.setattr(round0224_nodes, "_open_shard", fake_open)
    monkeypatch.setattr(
        round0224_nodes, "expected_input_signature",
        lambda path: {"kind": "file", "canonical_path": str(path), "bytes": 1, "sha256": "0" * 64},
    )
    return store


def test_round0224_assembly_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _synthetic(monkeypatch, tmp_path)
    output = tmp_path / "substrate"
    round0224_nodes.run_job(
        {"manifest": {"round_id": "0224", "release_sha": "f" * 40}},
        {"action": round0224_nodes.ASSEMBLE_ACTION, "outputs": [str(output)]},
    )
    with (output / "benchmark-substrate.json").open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    prompt_contract.validate_seal(receipt, label="R0224 assembly smoke")
    assert receipt["rows"] == ROWS
    assert receipt["benchmark_only"] is True
    assert receipt["seals_training_capability"] is False

    substrate = np.load(output / "substrate.f32.npy", mmap_mode="r")
    assert substrate.shape == (ROWS, DIMENSION)
    norms = np.linalg.norm(np.asarray(substrate), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "every row must be L2-normalized"
    assert np.isfinite(np.asarray(substrate)).all()
    # No destination may have been left unwritten: a permutation bug would leave
    # exact-zero rows behind, which the norm check above already forbids.

    provenance = np.load(output / "provenance.npy")
    counts = np.bincount(provenance["corpus"].astype(np.int64), minlength=4)
    for index, (_name, base) in enumerate(SMALL_COMPOSITION):
        assert int(counts[index]) == int(base) * SCALE

    # The row order really is permuted: a block layout would put corpus 0 first.
    leading = provenance["corpus"][:2_000].astype(np.int64)
    assert len(set(leading.tolist())) == 4, "prefix is not composition-representative"

    for rows, cell in receipt["prefix_composition"].items():
        for name, deviation in cell["deviations"].items():
            assert abs(deviation) <= cell["tolerances"][name], (rows, name, cell)

    for corpus, span in receipt["selection"]["shard_span"].items():
        assert span["coverage"] >= 0.999, (corpus, span)

    # Provenance points back at a real source row for a sample of destinations.
    store = None
    assert receipt["row_order"]["seed"] == round0224_nodes.BENCHMARK_SHUFFLE_SEED


def test_round0224_assembly_rejects_a_short_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _synthetic(monkeypatch, tmp_path, shards_per_corpus=1)

    def tiny_shards(corpus: str):
        return [(f"{corpus}:0", 10, True)]

    monkeypatch.setattr(round0224_nodes, "_shards", tiny_shards)
    with pytest.raises(Round0224Error):
        round0224_nodes.run_assemble(
            {"manifest": {"round_id": "0224", "release_sha": "f" * 40}},
            {"outputs": [str(tmp_path / "short")]},
        )
