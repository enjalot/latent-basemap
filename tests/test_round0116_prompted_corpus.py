"""CPU-only contract tests for canonical prompted-Jina production."""
from __future__ import annotations

import copy
import os

import pytest

from basemap import round0116_prompted_corpus as contract


def test_registered_work_ranges_cover_only_the_missing_rows() -> None:
    intervals = {}
    total = 0
    for node_id, dataset, start, stop in contract.WORK_RANGES:
        assert contract.expected_work_range(node_id) == (
            dataset,
            start,
            stop,
        )
        intervals.setdefault(dataset, []).append((start, stop))
        total += stop - start
    assert intervals[contract.FINEWEB] == [
        (contract.REUSED_FINEWEB_ROWS, contract.FINEWEB_ROWS)
    ]
    assert intervals[contract.REDPAJAMA] == [
        (0, 1_000_000),
        (1_000_000, 2_000_000),
        (2_000_000, contract.REDPAJAMA_ROWS),
    ]
    assert total == contract.NEW_ROWS == 3_727_340
    assert contract.production_payload_bytes() == 5_725_194_240
    assert contract.required_free_bytes() > contract.production_payload_bytes()
    worst_passing_gpu_s = (
        contract.NEW_ROWS / contract.EMBED_MINIMUM_ROWS_PER_S
        + 300.0 * len(contract.WORK_RANGES)
    )
    assert worst_passing_gpu_s < 6.5 * 3_600.0


def test_source_layout_maps_exact_shards_and_rejects_a_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fineweb = contract.FINEWEB
    redpajama = contract.REDPAJAMA
    monkeypatch.setattr(
        contract, "DATASET_ROWS", {fineweb: 5, redpajama: 4}
    )
    monkeypatch.setattr(
        contract, "DATASET_GLOBAL_OFFSETS", {fineweb: 0, redpajama: 5}
    )
    monkeypatch.setattr(contract, "CORPUS_ROWS", 9)
    text_roots = {
        fineweb: str(tmp_path / "fineweb"),
        redpajama: str(tmp_path / "redpajama"),
    }
    selection = {
        "ranges": [
            {
                "dataset": fineweb,
                "dataset_row_start": 0,
                "dataset_row_stop": 3,
                "global_row_start": 0,
                "global_row_stop": 3,
                "shard_row_start": 0,
                "shard_row_stop": 3,
                "shard": {
                    "canonical_path": "/data/embed/fine-0.npy",
                    "rows": 3,
                    "bytes": 1_000,
                    "sha256": "1" * 64,
                },
            },
            {
                "dataset": fineweb,
                "dataset_row_start": 3,
                "dataset_row_stop": 5,
                "global_row_start": 3,
                "global_row_stop": 5,
                "shard_row_start": 0,
                "shard_row_stop": 2,
                "shard": {
                    "canonical_path": "/data/embed/fine-1.npy",
                    "rows": 2,
                    "bytes": 900,
                    "sha256": "2" * 64,
                },
            },
            {
                "dataset": redpajama,
                "dataset_row_start": 0,
                "dataset_row_stop": 4,
                "global_row_start": 5,
                "global_row_stop": 9,
                "shard_row_start": 0,
                "shard_row_stop": 4,
                "shard": {
                    "canonical_path": "/data/embed/red-0.npy",
                    "rows": 4,
                    "bytes": 1_100,
                    "sha256": "3" * 64,
                },
            },
        ]
    }

    def signature(path: str):
        return {
            "kind": "file",
            "canonical_path": os.path.realpath(path),
            "bytes": 777,
            "sha256": "a" * 64,
        }

    rows = {
        "fine-0.parquet": 3,
        "fine-1.parquet": 2,
        "red-0.parquet": 4,
    }
    layout = contract.source_layout_from_inventory(
        {"selection": selection},
        text_roots=text_roots,
        signature_fn=signature,
        parquet_inspector=lambda path: (
            rows[os.path.basename(path)],
            "string",
        ),
    )
    assert [item["dataset_row_stop"] for item in layout] == [3, 5, 4]
    clipped = contract.clip_layout(
        layout, dataset=fineweb, start=2, stop=5
    )
    assert [
        (
            item["dataset_row_start"],
            item["dataset_row_stop"],
            item["shard_row_start"],
            item["shard_row_stop"],
        )
        for item in clipped
    ] == [(2, 3, 2, 3), (3, 5, 0, 2)]
    monkeypatch.setattr(contract, "REUSED_FINEWEB_ROWS", 5)
    lineage = [
        {
            "global_row_start": item["dataset_row_start"],
            "global_row_stop": item["dataset_row_stop"],
            "shard_row_start": item["shard_row_start"],
            "shard_row_stop": item["shard_row_stop"],
            "text_path": item["text"]["canonical_path"],
            "embedding": item["accepted_raw_embedding"],
        }
        for item in layout
        if item["dataset"] == fineweb
    ]
    reuse_proof = contract.validate_reused_mapping(
        layout,
        {
            "source_contract": {
                "r0087_inventory_identity_sha256": (
                    contract.INVENTORY_IDENTITY_SHA256
                ),
                "source_dataset": fineweb,
                "source_global_rows": [0, 5],
            }
        },
        r0114_source_lineage=lineage,
    )
    assert reuse_proof[
        "r0114_source_lineage_matches_canonical_prefix"
    ] is True
    bad_lineage = copy.deepcopy(lineage)
    bad_lineage[-1]["embedding"]["sha256"] = "f" * 64
    with pytest.raises(contract.Round0116Error, match="source lineage"):
        contract.validate_reused_mapping(
            layout,
            {
                "source_contract": {
                    "r0087_inventory_identity_sha256": (
                        contract.INVENTORY_IDENTITY_SHA256
                    ),
                    "source_dataset": fineweb,
                    "source_global_rows": [0, 5],
                }
            },
            r0114_source_lineage=bad_lineage,
        )

    broken = copy.deepcopy(selection)
    broken["ranges"][1]["dataset_row_start"] = 4
    with pytest.raises(contract.Round0116Error, match="malformed"):
        contract.source_layout_from_inventory(
            {"selection": broken},
            text_roots=text_roots,
            signature_fn=signature,
            parquet_inspector=lambda path: (
                rows[os.path.basename(path)],
                "string",
            ),
        )


def test_coverage_rejects_repeated_output_and_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fineweb = contract.FINEWEB
    redpajama = contract.REDPAJAMA
    monkeypatch.setattr(
        contract, "DATASET_ROWS", {fineweb: 2, redpajama: 2}
    )
    monkeypatch.setattr(
        contract, "DATASET_GLOBAL_OFFSETS", {fineweb: 0, redpajama: 2}
    )
    monkeypatch.setattr(contract, "CORPUS_ROWS", 4)
    monkeypatch.setattr(contract, "CHUNK_ROWS", 2)
    chunks = [
        {
            "dataset": fineweb,
            "dataset_row_range": [0, 2],
            "corpus_global_row_range": [0, 2],
            "output": {"canonical_path": "/data/fine.npy"},
        },
        {
            "dataset": redpajama,
            "dataset_row_range": [0, 2],
            "corpus_global_row_range": [2, 4],
            "output": {"canonical_path": "/data/red.npy"},
        },
    ]
    contract.validate_coverage(chunks)
    repeated = copy.deepcopy(chunks)
    repeated[1]["output"]["canonical_path"] = "/data/fine.npy"
    with pytest.raises(contract.Round0116Error, match="repeated"):
        contract.validate_coverage(repeated)
    overlapping = copy.deepcopy(chunks)
    overlapping[1]["dataset_row_range"] = [1, 2]
    with pytest.raises(contract.Round0116Error, match="gap, overlap"):
        contract.validate_coverage(overlapping)
