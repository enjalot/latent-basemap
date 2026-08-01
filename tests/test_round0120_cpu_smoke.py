"""CUDA-hidden post-embedding -> finalizer smoke for R0120."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from basemap import round0120_prompted_pile as contract
from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from experiments import round0120_nodes


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
    rows = 8
    chunk_rows = 2
    offset = 100
    work_ranges = (
        ("embed_pile_00", 0, 2),
        ("embed_pile_01", 2, 4),
        ("embed_pile_02", 4, 6),
        ("embed_pile_03", 6, 8),
    )
    monkeypatch.setattr(contract, "CORPUS_ROWS", rows)
    monkeypatch.setattr(contract, "CHUNK_ROWS", chunk_rows)
    monkeypatch.setattr(contract, "R0087_PILE_GLOBAL_OFFSET", offset)
    monkeypatch.setattr(contract, "R0087_PILE_GLOBAL_STOP", offset + rows)
    monkeypatch.setattr(contract, "WORK_RANGES", work_ranges)
    monkeypatch.setattr(round0120_nodes, "CORPUS_ROWS", rows)
    monkeypatch.setattr(round0120_nodes, "CHUNK_ROWS", chunk_rows)
    monkeypatch.setattr(
        round0120_nodes, "R0087_PILE_GLOBAL_OFFSET", offset
    )
    monkeypatch.setattr(
        round0120_nodes, "R0087_PILE_GLOBAL_STOP", offset + rows
    )
    monkeypatch.setattr(round0120_nodes, "WORK_RANGES", work_ranges)
    monkeypatch.setattr(
        round0120_nodes,
        "_verify_bound_closures",
        lambda job: {
            "inventory_manifest": {"sha256": "1" * 64},
            "r0114_model_prompt_manifest": {"sha256": "2" * 64},
        },
    )

    source = tmp_path / "pile.parquet"
    source.write_bytes(b"authenticated Pile source")
    source_signature = expected_input_signature(str(source))
    source_layout = [
        {
            "dataset": contract.DATASET,
            "dataset_row_start": 0,
            "dataset_row_stop": rows,
            "corpus_global_row_start": 0,
            "corpus_global_row_stop": rows,
            "r0087_global_row_start": offset,
            "r0087_global_row_stop": offset + rows,
            "shard_row_start": 0,
            "shard_row_stop": rows,
            "shard_rows": rows,
            "text_column": "chunk_text",
            "text_column_type": "string",
            "text": source_signature,
            "accepted_raw_embedding": {
                "kind": "file",
                "canonical_path": "/data/pile-raw.npy",
                "bytes": 1,
                "sha256": "3" * 64,
                "rows": rows,
            },
        }
    ]
    environment_freeze = {
        "schema": "round0116-python-environment-freeze-v1",
        "python_executable": "/smoke/python",
        "python_prefix": "/smoke",
        "python_version": "smoke",
        "packages": [],
        "freeze_sha256": "0" * 64,
    }

    receipt_paths = []
    for index, (node_id, start, stop) in enumerate(work_ranges):
        path = tmp_path / f"{node_id}.npy"
        output_signature = _write_unit_rows(path, stop - start, 120 + index)
        model = contract.model_contract()
        model["runtime_semantics"] = {
            "resolved_sentence_transformers_max_seq_length": (
                contract.NATIVE_MAX_SEQ_LENGTH
            )
        }
        node_layout = contract.clip_layout(
            source_layout, start=start, stop=stop
        )
        body = {
            "schema": contract.NODE_SCHEMA,
            "round_id": contract.ROUND_ID,
            "release_sha": release_sha,
            "node_id": node_id,
            "dataset": contract.DATASET,
            "dataset_row_range": [start, stop],
            "corpus_global_row_range": [start, stop],
            "r0087_global_row_range": [offset + start, offset + stop],
            "source_layout": node_layout,
            "source_files_rehashed_at_node_boundary": [source_signature],
            "environment_freeze": environment_freeze,
            "model": model,
            "dimension": contract.DIMENSION,
            "compute_dtype": contract.COMPUTE_DTYPE,
            "output_dtype": "<f2",
            "prompt_prefix": contract.PROMPT_PREFIX,
            "prompt_name_equivalence_passed": True,
            "chunks": [
                {
                    "chunk_index": 0,
                    "dataset": contract.DATASET,
                    "dataset_row_range": [start, stop],
                    "corpus_global_row_range": [start, stop],
                    "r0087_global_row_range": [
                        offset + start,
                        offset + stop,
                    ],
                    "source_row_count": stop - start,
                    "source_ids_ordered_sha256": ordered_array_sha256(
                        np.arange(start, stop, dtype=np.int64)
                    ),
                    "source_text_ordered_sha256": "4" * 64,
                    "document_text_ordered_sha256": "5" * 64,
                    "output": output_signature,
                    "output_shape": [
                        stop - start,
                        contract.DIMENSION,
                    ],
                    "output_dtype": "<f2",
                    "stored_norm": {"passed": True},
                }
            ],
            "training_performed": False,
            "optimizer_updates": 0,
            "performance": {
                "wall_s": 1.0,
                "document_rows_per_s": float(stop - start),
                "oom_retries": 0,
            },
        }
        receipt = contract.seal(body)
        receipt_path = tmp_path / f"{node_id}-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_paths.append(str(receipt_path))

    output = tmp_path / "finalized-pile"
    result = round0120_nodes.run_finalize(
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
            "environment_freeze": environment_freeze,
        },
    )
    assert result["row_count"] == rows
    assert result["coverage_validation"]["scanned_rows"] == rows
    assert result["coverage_validation"]["gap_free"] is True
    assert result["r0087_selected_global_row_range"] == [
        offset,
        offset + rows,
    ]
    assert result["claims"]["graph_built"] is False
    assert result["claims"]["map_trained"] is False
    assert result["claims"]["complete_sae_training_corpus"] is False
    assert result["training_performed"] is False
    contract.validate_seal(
        {
            key: value
            for key, value in result.items()
            if key != "receipt"
        },
        label="R0120 CPU smoke final manifest",
    )

    with (
        output / f"{contract.CORPUS_SCHEMA}.json"
    ).open(encoding="utf-8") as handle:
        published = json.load(handle)
    contract.validate_seal(
        published, label="reloaded R0120 CPU smoke final manifest"
    )
    assert published["identity_sha256"] == result["identity_sha256"]
