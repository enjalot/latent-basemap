"""Contract tests for the CPU-only R0185 disjoint probe view."""
from __future__ import annotations

import json

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0185_prompted_ood_disjoint_pack import CAPABILITY, PACK_SCHEMA
from experiments import round0185_nodes


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path, monkeypatch, *, add_unregistered_overlap: bool):
    monkeypatch.setattr(round0185_nodes, "LANGUAGES", ("x",))
    monkeypatch.setattr(round0185_nodes, "HELDOUT_CORPUS_ROWS", 3)
    monkeypatch.setattr(round0185_nodes, "HELDOUT_QUERY_ROWS", 1)
    monkeypatch.setattr(round0185_nodes, "DIMENSION", 2)
    monkeypatch.setattr(round0185_nodes, "TRAINING_ROWS", 3)
    monkeypatch.setattr(round0185_nodes, "SOURCE_PROBE_ROWS", 4)
    monkeypatch.setattr(round0185_nodes, "RETAINED_PROBE_ROWS", 3)
    monkeypatch.setattr(
        round0185_nodes,
        "EXPECTED_REMOVALS",
        (("x", "corpus", 11, 1),),
    )

    language = tmp_path / "language"
    language.mkdir()
    corpus = np.asarray([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float16)
    queries = np.asarray([[0.25, 0.75]], dtype=np.float16)
    corpus_rows = np.asarray([10, 11, 12], dtype=np.int64)
    query_rows = np.asarray([20], dtype=np.int64)
    np.save(language / "corpus.f16.npy", corpus, allow_pickle=False)
    np.save(language / "queries.f16.npy", queries, allow_pickle=False)
    np.save(language / "corpus-source-rows.i64.npy", corpus_rows, allow_pickle=False)
    np.save(language / "query-source-rows.i64.npy", query_rows, allow_pickle=False)
    payloads = {
        "corpus_embeddings": expected_input_signature(language / "corpus.f16.npy"),
        "query_embeddings": expected_input_signature(language / "queries.f16.npy"),
        "corpus_source_rows": expected_input_signature(
            language / "corpus-source-rows.i64.npy"
        ),
        "query_source_rows": expected_input_signature(
            language / "query-source-rows.i64.npy"
        ),
    }
    receipt = prompt_contract.seal({
        "schema": "round0173-prompted-language-probe-v1",
        "round_id": "0173",
        "language": "x",
        "prompt_applied": True,
        "prompt_prefix": "Document: ",
        **payloads,
    })
    _write_json(language / "receipt.json", receipt)
    registered = {
        "receipt": expected_input_signature(language / "receipt.json"),
        **payloads,
    }

    source_audit = prompt_contract.seal({
        "schema": "round0173-prompted-ood-training-disjoint-v1",
        "round_id": "0173",
        "passed": False,
        "capabilities": [],
        "probe_rows": 4,
        "training_rows": 3,
        "exact_training_family_overlap_count": 1,
        "exact_training_family_overlaps": [{
            "language": "x",
            "split": "corpus",
            "source_row": 11,
            "training_compact_row": 1,
        }],
        "language_outputs": {"x": registered},
    })
    audit_path = tmp_path / "source-audit.json"
    _write_json(audit_path, source_audit)

    training = np.asarray(
        [
            [0.9, 0.1],
            [0, 1],
            [0.25, 0.75] if add_unregistered_overlap else [0.1, 0.9],
        ],
        dtype=np.float16,
    )
    training_path = tmp_path / "training.f16.npy"
    np.save(training_path, training, allow_pickle=False)
    manifest = prompt_contract.seal({
        "schema": "round0168-prompted-diverse-u12-staging-v1",
        "round_id": "0168",
        "embedding_convention": "Document: ",
        "rows": 3,
        "dimension": 2,
        "dtype": "<f2",
        "host_fp16": expected_input_signature(training_path),
        "training_performed": False,
    })
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "action": "filter_and_audit_prompted_ood_pack",
        "source_audit": expected_input_signature(audit_path),
        "staging_manifest": expected_input_signature(manifest_path),
        "outputs": [str(tmp_path / "output")],
    }


def test_filters_registered_family_and_rescans_all_retained_rows(
    tmp_path, monkeypatch
) -> None:
    job = _fixture(tmp_path, monkeypatch, add_unregistered_overlap=False)
    round0185_nodes.run_job(
        {"manifest": {"round_id": "0185", "release_sha": "a" * 40}}, job
    )
    with open(tmp_path / "output" / "pack.json", encoding="utf-8") as handle:
        pack = json.load(handle)
    assert pack["schema"] == PACK_SCHEMA
    assert pack["passed"] is True
    assert pack["capabilities"] == [CAPABILITY]
    assert pack["source_probe_rows"] == 4
    assert pack["retained_probe_rows"] == 3
    assert pack["removed_probe_rows"] == 1
    assert pack["exact_retained_training_family_overlap_count"] == 0
    assert pack["queries_unchanged"] is True
    positions = np.load(
        tmp_path / "output" / "x-corpus-retained-positions.i64.npy"
    )
    assert positions.tolist() == [0, 2]


def test_unregistered_remaining_overlap_fails_closed(tmp_path, monkeypatch) -> None:
    job = _fixture(tmp_path, monkeypatch, add_unregistered_overlap=True)
    with pytest.raises(RuntimeError, match="still has 1 training overlaps"):
        round0185_nodes.run_job(
            {"manifest": {"round_id": "0185", "release_sha": "b" * 40}}, job
        )
    with open(tmp_path / "output" / "pack.json", encoding="utf-8") as handle:
        pack = json.load(handle)
    assert pack["passed"] is False
    assert pack["capabilities"] == []
    assert pack["exact_retained_training_family_overlap_count"] == 1


def test_unknown_action_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="does not authorize"):
        round0185_nodes.run_job(
            {"manifest": {"round_id": "0185"}}, {"action": "embed"}
        )
