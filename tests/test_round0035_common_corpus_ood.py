"""Focused CPU-only tests for the R0035 Common Corpus OOD extension."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import common_corpus_ood_round0035 as panel
from experiments import prepare_round0035_queue as queue
from experiments.prepare_round0035_queue import jobs, scientific_contract


def test_deterministic_split_is_disjoint_complete_and_probe_bound() -> None:
    code_corpus, code_query = panel.deterministic_split("code")
    code_corpus_again, code_query_again = panel.deterministic_split("code")
    science_corpus, science_query = panel.deterministic_split("science")
    assert np.array_equal(code_corpus, code_corpus_again)
    assert np.array_equal(code_query, code_query_again)
    assert len(code_corpus) == 49_500
    assert len(code_query) == 500
    assert len(np.intersect1d(code_corpus, code_query)) == 0
    assert np.array_equal(
        np.sort(np.concatenate([code_corpus, code_query])), np.arange(50_000)
    )
    assert not np.array_equal(code_query, science_query)
    assert len(science_corpus) == 49_500


def _fixture_paths(tmp_path: Path, *, mismatch: bool = False) -> panel.ProbePaths:
    vectors = tmp_path / "vectors.npy"
    ids_path = tmp_path / "corpus_ids.json"
    manifest_path = tmp_path / "manifest.json"
    chunks = tmp_path / "chunks.parquet"
    np.save(vectors, np.ones((6, 384), dtype=np.float32))
    rows = [
        {
            "identifier": f"row-{index}",
            "chunk_index": index,
            "collection": "Fixture",
            "open_type": "Open Science",
            "language": "English",
            "date": 2026.0,
            "title": f"Title {index}",
            "chunk_text": f"text {index}",
        }
        for index in range(6)
    ]
    pd.DataFrame(rows).to_parquet(chunks)
    sidecar = [{key: row[key] for key in panel.KEEP_METADATA} for row in rows]
    if mismatch:
        sidecar[3]["identifier"] = "wrong-row"
    ids_path.write_text(json.dumps(sidecar), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "collection": "fixture",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dim": 384,
                "n_chunks": 6,
                "dtype": "float32",
                "npy": str(vectors),
                "source_chunks": str(chunks),
            }
        ),
        encoding="utf-8",
    )
    return panel.ProbePaths(
        name="fixture",
        vectors=str(vectors),
        ids=str(ids_path),
        manifest=str(manifest_path),
        chunks=str(chunks),
        vectors_sha256="unused",
        ids_sha256="unused",
        manifest_sha256="unused",
        chunks_sha256="unused",
    )


def test_source_mapping_proves_full_ordered_relationship(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(panel, "PROBE_ROWS", 6)
    vectors, ids, receipt = panel.verify_source_mapping(_fixture_paths(tmp_path))
    assert vectors.shape == (6, 384)
    assert len(ids) == 6
    assert receipt["proved"] is True
    assert receipt["rows_compared"] == 6


def test_source_mapping_fails_on_one_permuted_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(panel, "PROBE_ROWS", 6)
    with pytest.raises(RuntimeError, match="mapping failed at row 3"):
        panel.verify_source_mapping(_fixture_paths(tmp_path, mismatch=True))


def test_projection_ids_are_unique_at_chunk_granularity() -> None:
    first = panel._projection_row_id(
        {"identifier": "same-document", "chunk_index": 0}, 12
    )
    second = panel._projection_row_id(
        {"identifier": "same-document", "chunk_index": 1}, 13
    )
    assert first != second
    assert json.loads(first) == {
        "identifier": "same-document",
        "chunk_index": 0,
        "vector_row": 12,
    }


def test_scoring_preserves_exact_topk_and_retention_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(panel, "CORPUS_ROWS", 10)
    monkeypatch.setattr(panel, "QUERY_ROWS", 2)
    monkeypatch.setattr(panel, "TRUE_K", 2)
    monkeypatch.setattr(panel, "HIT_FRACTION", 0.5)
    corpus = np.eye(10, dtype=np.float32)
    queries = corpus[[1, 7]]
    score = panel._score_one(
        name="fixture",
        corpus_vectors=corpus,
        query_vectors=queries,
        corpus_coords=corpus[:, :2],
        query_coords=queries[:, :2],
        device="cpu",
    )
    assert score["true_k"] == 2
    assert score["hit_k"] == 5
    assert 0.0 <= score["ffr"] <= 1.0
    assert panel.retention_verdict(0.7) == "pass"
    assert panel.retention_verdict(0.5) == "amber"
    assert panel.retention_verdict(0.4999) == "failure"
    assert panel.retention_verdict(None) == "undefined-control-zero"


def test_queue_uses_standalone_handlers_and_no_training(tmp_path: Path) -> None:
    nodes = jobs(artifacts=str(tmp_path / "artifacts"), inputs=[])
    assert [node["id"] for node in nodes] == [
        "common_corpus_source_model_canary",
        "common_corpus_ood_panel",
    ]
    assert nodes[1]["deps"] == [nodes[0]["id"]]
    assert all(
        node["handler_module"] == "experiments.common_corpus_ood_round0035"
        for node in nodes
    )
    assert [node["handler_callable"] for node in nodes] == [
        "run_canary_job",
        "run_panel_job",
    ]
    assert all(node["node_policy"]["gpu_required"] for node in nodes)
    assert not any(node["node_policy"]["training_performed"] for node in nodes)


def test_contract_names_corrected_fp32_semantics_and_fail_closed_canary() -> None:
    contract = scientific_contract()
    assert contract["accepted_semantics"]["probe_and_query_source_dtype"] == "float32"
    assert contract["accepted_semantics"]["tf32_allowed"] is False
    assert contract["split"] == {
        "seed": 20260735,
        "source_rows": 50_000,
        "corpus_rows": 49_500,
        "query_rows": 500,
        "query_disjoint_from_corpus": True,
        "selection": "collection-and-purpose-bound RandomState without replacement",
    }
    assert contract["text_canary"]["mapping_or_model_mismatch_policy"].endswith(
        "fail panel closed"
    )
    assert contract["scoring"]["map_neighbors"].endswith("(1% of corpus)")


def test_queue_uses_exact_released_reviews_and_capability_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "_exact_files", lambda _round_file: [])
    monkeypatch.setattr(queue, "_materialized_chunk_inputs", lambda: [])
    monkeypatch.setattr(queue, "_hf_snapshot_file_inputs", lambda: [])
    monkeypatch.setattr(queue, "atomic_write_new_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue, "create_fresh_directory", lambda path, **_kwargs: path)
    monkeypatch.setattr(queue, "ensure_data_directory", lambda path: path)
    monkeypatch.setattr(
        queue,
        "_base_manifest",
        lambda **kwargs: {
            "round_id": kwargs["round_id"],
            "release_sha": kwargs["release_sha"],
        },
    )
    captured: dict[str, object] = {}

    def capture(_path: str, value: dict, **_kwargs: object) -> None:
        captured.update(value)

    monkeypatch.setattr(queue, "atomic_write_new_json", capture)
    queue.prepare("a" * 40)
    assert captured["required_reviews"] == ["0013", "0019", "0028"]
    assert captured["capability_dependencies"] == [
        "30m-input-pack-v1",
        "30m-minilm-map-seed42-duplicate-cap",
        "universality-panel-v1",
    ]
