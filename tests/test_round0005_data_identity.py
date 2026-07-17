"""Focused CPU-only contracts for the Round 0005 data-identity surface."""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from basemap.artifact_identity import expected_input_signature

from basemap.query_artifact import (
    build_query_artifact, load_query_artifact, round0005_query_convention,
    validate_convention, validate_round0005_query_convention,
)
from basemap.round0005_staging import (
    ROUND0005_DIMENSIONS, ROUND0005_JINA_MODEL_CLOSURE, ROUND0005_MODEL_ID,
    ROUND0005_MODEL_REVISION, ROUND0005_QUERY_CONVERSION, ROUND0005_QUERY_ROWS,
    ROUND0005_QUERY_SEED, TESTBED_SEAL_REFERENCE_SCHEMA, _checkpoint_contract,
    _resolved_snapshot_files, _validate_exact_jina_model_closure,
    build_round0005_testbed_seal, make_testbed_seal_reference,
    validate_round0005_testbed_seal,
)
from experiments.calibrate_jina_embedding import _token_regimes
from experiments.embed_prompted_200k import (
    embed_outer_chunks, inspect_loaded_jina_model, load_model,
)
from experiments.prepare_jina_calibration import select_calibration_rows


TEST_REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_directory_identity_binds_directories_and_rejects_specials_and_hardlinks(
        tmp_path, monkeypatch):
    root = tmp_path / "identity-tree"
    (root / "empty" / "nested").mkdir(parents=True)
    (root / "regular.txt").write_text("regular\n")
    signature = expected_input_signature(root)
    members = {(value["relative_path"], value["kind"])
               for value in signature["members"]}
    assert ("empty", "directory") in members
    assert ("empty/nested", "directory") in members
    assert ("regular.txt", "file") in members

    hardlink = root / "hardlink.txt"
    os.link(root / "regular.txt", hardlink)
    with pytest.raises(ValueError, match="hard-linked"):
        expected_input_signature(root)
    hardlink.unlink()

    fifo = root / "unsupported.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="unsupported fifo"):
        expected_input_signature(root)
    fifo.unlink()

    socket_path = root / "unsupported.socket"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # AF_UNIX limits the encoded address to roughly 108 bytes; pytest's
        # isolated /data path is deliberately much longer.  A relative bind
        # creates the same filesystem member without weakening the test.
        monkeypatch.chdir(root)
        server.bind(socket_path.name)
        with pytest.raises(ValueError, match="unsupported socket"):
            expected_input_signature(root)
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def _small_testbed(tmp_path: Path):
    testbed = tmp_path / "testbed"
    source = tmp_path / "source"
    texts = tmp_path / "texts"
    (testbed / "train").mkdir(parents=True)
    source.mkdir()
    texts.mkdir()
    source_values = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-0.6, 0.8],
        [0.8, -0.6], [-0.8, -0.6],
    ], dtype=np.float16)
    sample_ids = np.array([0, 2, 4, 5], dtype=np.int64)
    np.save(source / "data-00000-of-00001.npy", source_values)
    np.save(testbed / "train" / "data-00000.npy",
            source_values[sample_ids].astype(np.float32))
    np.save(testbed / "sample_indices.npy", sample_ids)
    np.save(testbed / "centroids_k256.npy", np.ones((2, 2), np.float32))
    np.save(testbed / "centroids_k1024.npy", np.ones((3, 2), np.float32))
    pd.DataFrame({"chunk_text": [f"text {index}" for index in range(6)]}).to_parquet(
        texts / "data-00000-of-00001.parquet", index=False)
    contract = {
        "dimensions": 2, "train_rows": 4, "source_rows": 6,
        "source_dtype": "float16", "train_dtype": "float32",
        "source_shards": ["data-00000-of-00001.npy"],
        "source_shard_rows": [6],
        "text_shards": ["data-00000-of-00001.parquet"],
        "centroids": {"k256": 2, "k1024": 3},
        "sample_mapping_conversion": ROUND0005_QUERY_CONVERSION,
    }
    return testbed, source, texts, source_values, sample_ids, contract


def test_empty_prompt_fp16_source_materializes_explicit_fp32_query_and_shared_seal(
        tmp_path, fresh_data_root):
    testbed, source, texts, source_values, _, contract = _small_testbed(tmp_path)
    seal_path = os.path.join(fresh_data_root, "testbed-seal.json")
    built_seal = build_round0005_testbed_seal(
        testbed=str(testbed), source_embeddings=str(source), source_texts=str(texts),
        seal_path=seal_path, _contract=contract)
    validated = validate_round0005_testbed_seal(seal_path, require_round0005=False)
    assert validated["identity_sha256"] == built_seal["identity_sha256"]
    reference = make_testbed_seal_reference(
        seal_path, require_round0005=False, deep=False)
    assert reference["schema"] == TESTBED_SEAL_REFERENCE_SCHEMA

    convention = validate_convention({
        "model_id": "fixture/jina", "model_revision": TEST_REVISION,
        "prompt_policy": "raw_unprompted", "prompt_bytes_hex": "",
        "pooling": "lasttoken", "source_dtype": "float16",
        "storage_dtype": "float32", "dtype": "float32",
        "dtype_conversion": ROUND0005_QUERY_CONVERSION,
        "normalization": "l2", "dimensions": 2,
    })
    report = build_query_artifact(
        testbed=str(testbed), source=str(source),
        out_dir=os.path.join(fresh_data_root, "query"), dim=2, n_holdout=2,
        seed=7, convention=convention, testbed_seal_path=seal_path)
    loaded = load_query_artifact(
        report["manifest_path"], testbed=str(testbed),
        expected_convention=convention, expected_testbed_seal=seal_path)
    assert loaded["Xq"].dtype == np.dtype("float32")
    assert loaded["manifest"]["convention"]["prompt_bytes_hex"] == ""
    assert loaded["manifest"]["convention"]["prompt_policy"] == "raw_unprompted"
    assert loaded["manifest"]["testbed_seal"] == reference
    assert np.array_equal(
        np.asarray(loaded["Xq"]),
        source_values[loaded["query_ids"]].astype(np.float32))


def test_testbed_seal_rejects_future_sample_mapping_mutation(tmp_path, fresh_data_root):
    testbed, source, texts, _, _, contract = _small_testbed(tmp_path)
    seal_path = os.path.join(fresh_data_root, "mapping-seal.json")
    build_round0005_testbed_seal(
        testbed=str(testbed), source_embeddings=str(source), source_texts=str(texts),
        seal_path=seal_path, _contract=contract)
    sample_path = testbed / "sample_indices.npy"
    np.save(sample_path, np.array([0, 1, 4, 5], dtype=np.int64))
    with pytest.raises(ValueError, match="sample mapping|no longer matches"):
        validate_round0005_testbed_seal(seal_path, require_round0005=False)


def test_round0005_query_constants_are_exact_and_empty_prompt_is_not_placeholder():
    convention = round0005_query_convention()
    assert convention == {
        "model_id": ROUND0005_MODEL_ID,
        "model_revision": ROUND0005_MODEL_REVISION,
        "prompt_policy": "raw_unprompted", "prompt_bytes_hex": "",
        "pooling": "lasttoken", "source_dtype": "float16",
        "storage_dtype": "float32", "dtype": "float32",
        "dtype_conversion": ROUND0005_QUERY_CONVERSION,
        "normalization": "l2", "dimensions": ROUND0005_DIMENSIONS,
    }
    assert ROUND0005_QUERY_ROWS == 20_000 and ROUND0005_QUERY_SEED == 123
    changed = dict(convention, prompt_policy="literal_prefix")
    with pytest.raises(ValueError, match="prompt_policy"):
        validate_round0005_query_convention(changed)


def test_real_jina_snapshot_has_exact_resolved_closure_and_semantics():
    root = ("/data/hf/sentence-transformers/models--jinaai--"
            "jina-embeddings-v5-text-nano-retrieval/snapshots/" +
            ROUND0005_MODEL_REVISION)
    if not os.path.isdir(root):
        pytest.skip("pinned local Jina snapshot is not present")
    members = _resolved_snapshot_files(root)
    proof = _validate_exact_jina_model_closure(root, members)
    assert proof["member_count"] == len(ROUND0005_JINA_MODEL_CLOSURE) == 10
    assert proof["dimensions"] == 768 and proof["pooling"] == "lasttoken"
    assert proof["default_prompt_name"] is None


def test_local_model_load_never_substitutes_caller_revision(monkeypatch, tmp_path):
    captured = {}

    class Config:
        _commit_hash = None

    class Transformer:
        auto_model = SimpleNamespace(config=Config())

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def __getitem__(self, index):
            assert index == 0
            return Transformer()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer))
    _, commit = load_model(
        device="cpu", dtype="float32", model_path=str(tmp_path),
        model_revision=TEST_REVISION)
    assert commit is None
    assert captured["local_files_only"] is True


def test_loaded_model_runtime_semantics_are_checked_without_revision_invention():
    EuroBertModel = type("EuroBertModel", (), {})
    Normalize = type("Normalize", (), {})
    transformer = SimpleNamespace(
        auto_model=EuroBertModel(),
    )
    transformer.auto_model.config = SimpleNamespace(hidden_size=768)
    pooling = SimpleNamespace(
        pooling_mode_cls_token=False, pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False, pooling_mode_mean_sqrt_len_tokens=False,
        pooling_mode_weightedmean_tokens=False, pooling_mode_lasttoken=True,
        include_prompt=True,
    )

    class Model:
        prompts = {"query": "Query: ", "document": "Document: "}
        default_prompt_name = None

        def __getitem__(self, index):
            return [transformer, pooling, Normalize()][index]

        def get_sentence_embedding_dimension(self):
            return 768

    proof = inspect_loaded_jina_model(Model())
    assert proof["dimensions"] == 768 and proof["normalized"] is True


def test_checkpoint_metadata_binds_cpu_768_to_2_seed_kernel_a_b(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "model.pt"
    state = {
        "proj_in.weight": torch.zeros((4, 768)),
        "proj_out.weight": torch.zeros((2, 4)),
    }
    torch.save({
        "model_state_dict": state, "input_dim": 768, "n_components": 2,
        "low_dim_kernel": "legacy_lp", "a": 1.0, "b": 1.0,
    }, checkpoint_path)
    proof = _checkpoint_contract(
        str(checkpoint_path), label="legacy_a1b1_s42",
        expectation={"kernel": "legacy_lp", "a": 1.0, "b": 1.0},
        observed_seed=42)
    assert proof["loaded_on"] == "cpu" and proof["weights_only"] is True
    assert (proof["input_dim"], proof["n_components"], proof["seed"]) == (768, 2, 42)
    assert proof["seed_source"] == "results.json:config.data.random_seed"


def _normalized_embed(_model, texts, batch_size, show_progress, return_telemetry):
    values = np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(texts), 1))
    return values, {"requested_batch_size": batch_size,
                    "final_batch_size": batch_size, "oom_retries": 0}


def test_corrupt_future_chunk_is_rejected_before_missing_earlier_chunk_reembeds(
        fresh_data_root):
    root = os.path.join(fresh_data_root, "future-corruption")
    calls = []

    def embed(*args, **kwargs):
        calls.append(len(args[1]))
        return _normalized_embed(*args, **kwargs)

    kwargs = dict(
        model=object(), sample_indices=np.arange(6, dtype=np.int64),
        out_train=os.path.join(root, "train"), receipt_dir=os.path.join(root, "receipts"),
        text_dir="x", text_shards=[], offsets=np.array([0, 6]),
        model_commit=TEST_REVISION, compute_dtype="float32", batch_size=256,
        chunk_rows=3, fetch_fn=lambda ids, *_: [f"text-{value}" for value in ids],
        embed_fn=embed)
    first = embed_outer_chunks(**kwargs)
    assert first["preflight_complete_before_new_work"] is True
    assert set(first["chunks"][0]["phase_wall_s"]) == {
        "source_fetch", "prompt_and_context_hash",
        "model_encode_including_tokenization", "validate_and_output_publish",
        "receipt_publish",
    }
    os.unlink(os.path.join(root, "train", "data-00000.npy"))
    os.unlink(os.path.join(root, "receipts", "chunk-00000.json"))
    future = Path(root, "receipts", "chunk-00001.json")
    os.chmod(future, 0o644)
    future.write_text("{corrupt", encoding="utf-8")
    calls.clear()
    with pytest.raises(RuntimeError, match="corrupt existing embedding chunk receipt"):
        embed_outer_chunks(**kwargs)
    assert calls == []
    assert not os.path.exists(os.path.join(root, "train", "data-00000.npy"))


def test_calibration_selection_is_two_adjacent_aligned_production_chunks():
    positions, ids, selection = select_calibration_rows(
        sample_ids=np.arange(50_000, dtype=np.int64), source_rows=2_890_362,
        seed=20260716, production_contract=True)
    assert len(ids) == 50_000 and np.array_equal(ids, positions)
    assert int(ids[0]) % 25_000 == 0 and int(ids[25_000]) == int(ids[0]) + 25_000
    assert selection["method"] == "seeded_aligned_adjacent_source_chunks"
    assert selection["length_sorting"] is False


def test_token_regimes_bind_median_p95_p99_and_maximum_rows():
    lengths = np.arange(1, 101, dtype=np.int64)
    ids = np.arange(1000, 1100, dtype=np.int64)
    regimes = _token_regimes(lengths, ids)
    assert set(regimes) == {"median", "p95", "p99", "maximum"}
    assert regimes["maximum"]["target_effective_tokens"] == 100
    assert regimes["maximum"]["representative_global_id"] == 1099
