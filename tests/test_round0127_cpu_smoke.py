"""CUDA-hidden node-receipt -> finalizer -> reload smoke for R0127."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from basemap import round0127_prompted_multilingual as contract
from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from experiments import round0127_nodes as nodes


def _write_unit_rows(path: Path, rows: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, contract.DIMENSION)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    np.save(path, values.astype("<f2"))
    return expected_input_signature(str(path))


def test_post_embedding_receipts_finalize_without_cuda(
    monkeypatch,
    tmp_path,
) -> None:
    release_sha = "a" * 40
    rows_per_language = 2
    chunk_rows = 2
    tranches = (
        {
            "node_id": "embed_deu_Latn",
            "language": "deu_Latn",
            "dataset": "fineweb2-deu_Latn-chunked-500-jina-v5-nano",
            "corpus_global_row_range": [0, 2],
            "r0087_global_row_range": [100, 102],
        },
        {
            "node_id": "embed_ell_Grek",
            "language": "ell_Grek",
            "dataset": "fineweb2-ell_Grek-chunked-500-jina-v5-nano",
            "corpus_global_row_range": [2, 4],
            "r0087_global_row_range": [102, 104],
        },
        {
            "node_id": "embed_fra_Latn",
            "language": "fra_Latn",
            "dataset": "fineweb2-fra_Latn-chunked-500-jina-v5-nano",
            "corpus_global_row_range": [4, 6],
            "r0087_global_row_range": [104, 106],
        },
    )
    for module in (contract, nodes):
        monkeypatch.setattr(module, "ROWS_PER_LANGUAGE", rows_per_language)
        monkeypatch.setattr(module, "CHUNK_ROWS", chunk_rows)
        monkeypatch.setattr(module, "CORPUS_ROWS", 6)
        monkeypatch.setattr(module, "LANGUAGE_TRANCHES", tranches)
        monkeypatch.setattr(module, "R0087_GLOBAL_START", 100)
        monkeypatch.setattr(module, "R0087_GLOBAL_STOP", 106)
    monkeypatch.setattr(
        nodes,
        "_rehash_boundary_inputs",
        lambda bindings, require_all_sources: [dict(item) for item in bindings],
    )
    monkeypatch.setattr(
        nodes,
        "_verify_semantic_closures",
        lambda job: {
            "inventory_manifest": {"sha256": "1" * 64},
            "r0114_model_prompt_manifest": {"sha256": "2" * 64},
        },
    )

    environment = {
        "schema": "round0116-python-environment-freeze-v1",
        "python_executable": "/smoke/python",
        "python_prefix": "/smoke",
        "python_version": "smoke",
        "packages": [],
        "freeze_sha256": "0" * 64,
    }
    source_layout = []
    boundary_inputs = []
    receipt_paths = []
    for index, tranche in enumerate(tranches):
        text = tmp_path / f"{tranche['language']}.parquet"
        raw = tmp_path / f"{tranche['language']}-raw.npy"
        text.write_bytes(b"source")
        raw.write_bytes(b"identity only")
        text_signature = expected_input_signature(str(text))
        raw_signature = expected_input_signature(str(raw))
        source = {
            "node_id": tranche["node_id"],
            "language": tranche["language"],
            "dataset": tranche["dataset"],
            "dataset_row_range": [0, rows_per_language],
            "dataset_row_start": 0,
            "dataset_row_stop": rows_per_language,
            "corpus_global_row_range": tranche["corpus_global_row_range"],
            "r0087_global_row_range": tranche["r0087_global_row_range"],
            "shard_row_range": [0, rows_per_language],
            "shard_row_start": 0,
            "shard_row_stop": rows_per_language,
            "shard_rows": 2_000_000,
            "text_column": "chunk_text",
            "text_column_type": "large_string",
            "text": text_signature,
            "accepted_raw_embedding": {
                **raw_signature,
                "rows": 2_000_000,
                "dimension": 768,
                "dtype": "<f2",
                "selected_row_range": [0, rows_per_language],
            },
        }
        source_layout.append(source)
        boundary_inputs.extend(
            (
                {"role": "source-parquet", "signature": text_signature},
                {"role": "raw-embedding", "signature": raw_signature},
            )
        )

        output = tmp_path / f"{tranche['node_id']}.npy"
        output_signature = _write_unit_rows(output, rows_per_language, 20 + index)
        model = contract.model_contract()
        model_member = {
            **text_signature,
            "model_relative_path": "model.safetensors",
        }
        model["members"] = [model_member]
        model["runtime_semantics"] = {
            "resolved_sentence_transformers_max_seq_length": (
                contract.NATIVE_MAX_SEQ_LENGTH
            )
        }
        inventory_signature = {
            **text_signature,
            "sha256": contract.INVENTORY_MANIFEST_SHA256,
        }
        model_prompt_signature = {
            **text_signature,
            "sha256": contract.R0114_MANIFEST_SHA256,
        }
        node_boundary = [
            {"role": "round", "signature": text_signature},
            {
                "role": "review-0087",
                "signature": {
                    **text_signature,
                    "sha256": contract.R0087_REVIEW_SHA256,
                },
            },
            {
                "role": "review-0114",
                "signature": {
                    **text_signature,
                    "sha256": contract.R0114_REVIEW_SHA256,
                },
            },
            {"role": "inventory", "signature": inventory_signature},
            {
                "role": "model-prompt-manifest",
                "signature": model_prompt_signature,
            },
            {"role": "model-member", "signature": text_signature},
            {"role": "source-parquet", "signature": text_signature},
            {"role": "raw-embedding", "signature": raw_signature},
        ]
        body = {
            "schema": contract.NODE_SCHEMA,
            "round_id": contract.ROUND_ID,
            "release_sha": release_sha,
            "node_id": tranche["node_id"],
            "language": tranche["language"],
            "dataset": tranche["dataset"],
            "dataset_row_range": [0, rows_per_language],
            "corpus_global_row_range": tranche["corpus_global_row_range"],
            "r0087_global_row_range": tranche["r0087_global_row_range"],
            "source_layout": source,
            "job_boundary_rehash": node_boundary,
            "input_closures": {
                "inventory_manifest": inventory_signature,
                "r0114_model_prompt_manifest": model_prompt_signature,
            },
            "environment_freeze": environment,
            "model": model,
            "dimension": 768,
            "compute_dtype": "float32",
            "output_dtype": "<f2",
            "prompt_prefix": "Document: ",
            "prompt_name_equivalence_passed": True,
            "chunks": [
                {
                    "chunk_index": 0,
                    "language": tranche["language"],
                    "dataset": tranche["dataset"],
                    "dataset_row_range": [0, rows_per_language],
                    "corpus_global_row_range": tranche[
                        "corpus_global_row_range"
                    ],
                    "r0087_global_row_range": tranche[
                        "r0087_global_row_range"
                    ],
                    "source_row_count": rows_per_language,
                    "source_ids_ordered_sha256": ordered_array_sha256(
                        np.arange(rows_per_language, dtype=np.int64)
                    ),
                    "source_text_ordered_sha256": "4" * 64,
                    "document_text_ordered_sha256": "5" * 64,
                    "output": output_signature,
                    "output_shape": [rows_per_language, 768],
                    "output_dtype": "<f2",
                    "stored_norm": {"passed": True},
                }
            ],
            "training_performed": False,
            "optimizer_updates": 0,
            "performance": {
                "wall_s": 1.0,
                "document_rows_per_s": 2.0,
                "oom_retries": 0,
                "requested_batch_size": 16,
            },
        }
        receipt = contract.seal(body)
        receipt_path = tmp_path / f"{tranche['node_id']}-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_paths.append(str(receipt_path))

    output = tmp_path / "finalized-multilingual"
    result = nodes.run_finalize(
        {
            "manifest": {
                "round_id": contract.ROUND_ID,
                "release_sha": release_sha,
            }
        },
        {
            "outputs": [str(output)],
            "canonical_source_layout": source_layout,
            "node_receipts": receipt_paths,
            "authenticated_boundary_inputs": boundary_inputs,
            "environment_freeze": environment,
        },
    )
    assert result["row_count"] == 6
    assert result["source_order"] == ["deu_Latn", "ell_Grek", "fra_Latn"]
    assert result["coverage_validation"]["scanned_rows"] == 6
    assert result["coverage_validation"]["gap_free"] is True
    assert result["r0087_selected_global_row_range"] == [100, 106]
    assert result["claims"]["graph_built"] is False
    assert result["claims"]["map_trained"] is False
    assert result["claims"]["complete_sae_training_corpus"] is False
    assert result["training_performed"] is False
    contract.validate_seal(
        {key: value for key, value in result.items() if key != "receipt"},
        label="R0127 CPU smoke final manifest",
    )
    with (output / f"{contract.CORPUS_SCHEMA}.json").open(
        encoding="utf-8"
    ) as handle:
        published = json.load(handle)
    contract.validate_seal(
        published, label="reloaded R0127 CPU smoke final manifest"
    )
    assert published["identity_sha256"] == result["identity_sha256"]

