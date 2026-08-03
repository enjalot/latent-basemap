"""CUDA-hidden post-embedding -> finalizer smoke for R0116."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basemap import round0116_prompted_corpus as contract
from basemap.artifact_identity import expected_input_signature
from experiments import round0116_nodes


def _write_unit_rows(path: Path, rows: int, dimension: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, dimension)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    np.save(path, values.astype("<f2"))
    return expected_input_signature(str(path))


def test_source_files_are_rehashed_at_each_node_boundary(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"authenticated source bytes")
    signature = expected_input_signature(str(source))
    assert round0116_nodes._verify_source_files(
        [{"text": signature}, {"text": signature}]
    ) == [signature]
    source.write_bytes(b"X" * signature["bytes"])
    with pytest.raises(contract.Round0116Error, match="changed"):
        round0116_nodes._verify_source_files([{"text": signature}])


def test_post_embedding_receipts_finalize_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Exercise real file rehash, fp16 scan, coverage, seal, and reload."""
    fineweb = contract.FINEWEB
    redpajama = contract.REDPAJAMA
    release_sha = "a" * 40
    dimension = 4
    chunk_rows = 2
    work_ranges = (
        ("embed_fineweb_tail", fineweb, 2, 6),
        ("embed_redpajama_00", redpajama, 0, 2),
        ("embed_redpajama_01", redpajama, 2, 4),
        ("embed_redpajama_02", redpajama, 4, 6),
    )
    for module in (contract, round0116_nodes):
        monkeypatch.setattr(module, "DIMENSION", dimension)
        monkeypatch.setattr(module, "CHUNK_ROWS", chunk_rows)
        monkeypatch.setattr(
            module, "DATASET_ROWS", {fineweb: 6, redpajama: 6}
        )
        monkeypatch.setattr(
            module,
            "DATASET_GLOBAL_OFFSETS",
            {fineweb: 0, redpajama: 6},
        )
        monkeypatch.setattr(module, "CORPUS_ROWS", 12)
        monkeypatch.setattr(module, "NEW_ROWS", 10)
        monkeypatch.setattr(module, "REUSED_FINEWEB_ROWS", 2)
    monkeypatch.setattr(contract, "WORK_RANGES", work_ranges)

    source_layout = [
        {
            "dataset": fineweb,
            "dataset_row_start": 0,
            "dataset_row_stop": 6,
            "corpus_global_row_start": 0,
            "corpus_global_row_stop": 6,
            "shard_row_start": 0,
            "shard_row_stop": 6,
            "shard_rows": 6,
            "text_column": "chunk_text",
            "text_column_type": "string",
            "text": {
                "kind": "file",
                "canonical_path": "/data/fine.parquet",
                "bytes": 1,
                "sha256": "1" * 64,
            },
            "accepted_raw_embedding": {
                "kind": "file",
                "canonical_path": "/data/fine.npy",
                "bytes": 1,
                "sha256": "2" * 64,
                "rows": 6,
            },
        },
        {
            "dataset": redpajama,
            "dataset_row_start": 0,
            "dataset_row_stop": 6,
            "corpus_global_row_start": 6,
            "corpus_global_row_stop": 12,
            "shard_row_start": 0,
            "shard_row_stop": 6,
            "shard_rows": 6,
            "text_column": "chunk_text",
            "text_column_type": "string",
            "text": {
                "kind": "file",
                "canonical_path": "/data/red.parquet",
                "bytes": 1,
                "sha256": "3" * 64,
            },
            "accepted_raw_embedding": {
                "kind": "file",
                "canonical_path": "/data/red.npy",
                "bytes": 1,
                "sha256": "4" * 64,
                "rows": 6,
            },
        },
    ]

    reused_path = tmp_path / "reused-prefix.npy"
    reused_signature = _write_unit_rows(
        reused_path, 2, dimension, seed=116
    )
    reused = {
        "source_contract": {
            "r0087_inventory_identity_sha256": (
                contract.INVENTORY_IDENTITY_SHA256
            ),
            "source_dataset": fineweb,
            "source_global_rows": [0, 2],
        },
        "conventions": {
            "document": {
                "chunks": [reused_signature],
            }
        },
        "chunk_text_receipts": [
            {
                "source_row_range": [0, 2],
                "source_text_ordered_sha256": "5" * 64,
                "document_text_ordered_sha256": "6" * 64,
            }
        ],
    }
    reused_mapping = contract.validate_reused_mapping(
        source_layout, reused
    )
    environment_freeze = {
        "schema": "round0116-python-environment-freeze-v1",
        "python_executable": "/smoke/python",
        "python_prefix": "/smoke",
        "python_version": "smoke",
        "packages": [],
        "freeze_sha256": "0" * 64,
    }
    reused_manifest_signature = {
        "kind": "file",
        "canonical_path": str(tmp_path / "accepted-r0114.json"),
        "bytes": 1,
        "sha256": "7" * 64,
    }
    monkeypatch.setattr(
        round0116_nodes,
        "load_reused_manifest",
        lambda path: (reused, reused_manifest_signature),
    )

    receipt_paths = []
    seed = 200
    for node_id, dataset, start, stop in work_ranges:
        chunks = []
        for index, chunk_start in enumerate(
            range(start, stop, chunk_rows)
        ):
            chunk_stop = min(chunk_start + chunk_rows, stop)
            path = tmp_path / (
                f"{node_id}-{chunk_start}-{chunk_stop}.npy"
            )
            signature = _write_unit_rows(
                path, chunk_stop - chunk_start, dimension, seed
            )
            seed += 1
            global_start = (
                contract.DATASET_GLOBAL_OFFSETS[dataset] + chunk_start
            )
            global_stop = (
                contract.DATASET_GLOBAL_OFFSETS[dataset] + chunk_stop
            )
            chunks.append(
                {
                    "chunk_index": index,
                    "dataset": dataset,
                    "dataset_row_range": [chunk_start, chunk_stop],
                    "corpus_global_row_range": [
                        global_start,
                        global_stop,
                    ],
                    "source_text_ordered_sha256": "8" * 64,
                    "document_text_ordered_sha256": "9" * 64,
                    "output": signature,
                    "output_shape": [
                        chunk_stop - chunk_start,
                        dimension,
                    ],
                    "output_dtype": "<f2",
                    "stored_norm": {"passed": True},
                }
            )
        model = contract.model_contract()
        model["runtime_semantics"] = {
            "resolved_sentence_transformers_max_seq_length": (
                contract.NATIVE_MAX_SEQ_LENGTH
            )
        }
        body = {
            "schema": contract.NODE_SCHEMA,
            "round_id": contract.ROUND_ID,
            "release_sha": release_sha,
            "node_id": node_id,
            "dataset": dataset,
            "dataset_row_range": [start, stop],
            "corpus_global_row_range": [
                contract.DATASET_GLOBAL_OFFSETS[dataset] + start,
                contract.DATASET_GLOBAL_OFFSETS[dataset] + stop,
            ],
            "source_layout": contract.clip_layout(
                source_layout,
                dataset=dataset,
                start=start,
                stop=stop,
            ),
            "source_files_rehashed_at_node_boundary": [
                item["text"]
                for item in contract.clip_layout(
                    source_layout,
                    dataset=dataset,
                    start=start,
                    stop=stop,
                )
            ],
            "model": model,
            "dimension": dimension,
            "compute_dtype": contract.COMPUTE_DTYPE,
            "output_dtype": "<f2",
            "prompt_prefix": contract.PROMPT_PREFIX,
            "prompt_name_equivalence_passed": True,
            "environment_freeze": environment_freeze,
            "chunks": chunks,
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
        receipt_path.write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        receipt_paths.append(str(receipt_path))

    output = tmp_path / "finalized-corpus"
    result = round0116_nodes.run_finalize(
        {
            "manifest": {
                "round_id": contract.ROUND_ID,
                "release_sha": release_sha,
            }
        },
        {
            "outputs": [str(output)],
            "reused_manifest": str(tmp_path / "accepted-r0114.json"),
            "canonical_source_layout": source_layout,
            "reused_prefix_mapping": reused_mapping,
            "node_receipts": receipt_paths,
            "environment_freeze": environment_freeze,
        },
    )
    assert result["row_count"] == 12
    assert result["coverage_validation"]["scanned_rows"] == 12
    assert result["coverage_validation"]["gap_free"] is True
    assert result["claims"]["graph_built"] is False
    assert result["claims"]["map_trained"] is False
    assert result["training_performed"] is False
    contract.validate_seal(
        {
            key: value
            for key, value in result.items()
            if key != "receipt"
        },
        label="R0116 CPU smoke final manifest",
    )

    with (
        output / f"{contract.CORPUS_SCHEMA}.json"
    ).open(encoding="utf-8") as handle:
        published = json.load(handle)
    contract.validate_seal(
        published, label="reloaded R0116 CPU smoke final manifest"
    )
    assert published["identity_sha256"] == result["identity_sha256"]
