"""CPU-only contract tests for R0120 prompted-Pile production."""
from __future__ import annotations

import copy
import json
import os

import pytest

from basemap import round0120_prompted_pile as contract
from basemap.artifact_identity import expected_input_signature
from experiments import prepare_round0120_queue as prepare


def test_work_ranges_are_balanced_gap_free_and_payload_is_exact() -> None:
    cursor = 0
    for node_id, start, stop in contract.WORK_RANGES:
        assert contract.expected_work_range(node_id) == (start, stop)
        assert start == cursor
        assert 849_000 <= stop - start <= 850_000
        cursor = stop
    assert cursor == contract.CORPUS_ROWS == 3_399_036
    assert contract.CHUNK_ROWS == 25_000
    assert contract.production_payload_bytes() == 5_220_919_296
    assert contract.required_free_bytes() > contract.production_payload_bytes()
    worst_passing_gpu_s = (
        contract.CORPUS_ROWS / contract.EMBED_MINIMUM_ROWS_PER_S
        + 300.0 * len(contract.WORK_RANGES)
    )
    assert worst_passing_gpu_s < 6.5 * 3_600.0


def test_source_layout_preserves_r0087_and_local_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(contract, "CORPUS_ROWS", 5)
    monkeypatch.setattr(contract, "R0087_PILE_GLOBAL_OFFSET", 20)
    monkeypatch.setattr(contract, "R0087_PILE_GLOBAL_STOP", 25)
    selection = {
        "ranges": [
            {
                "dataset": contract.DATASET,
                "dataset_row_start": 0,
                "dataset_row_stop": 3,
                "global_row_start": 20,
                "global_row_stop": 23,
                "shard_row_start": 0,
                "shard_row_stop": 3,
                "shard": {
                    "canonical_path": "/data/embed/pile-0.npy",
                    "rows": 3,
                    "bytes": 1_000,
                    "sha256": "1" * 64,
                },
            },
            {
                "dataset": contract.DATASET,
                "dataset_row_start": 3,
                "dataset_row_stop": 5,
                "global_row_start": 23,
                "global_row_stop": 25,
                "shard_row_start": 0,
                "shard_row_stop": 2,
                "shard": {
                    "canonical_path": "/data/embed/pile-1.npy",
                    "rows": 2,
                    "bytes": 900,
                    "sha256": "2" * 64,
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

    layout = contract.source_layout_from_inventory(
        {"selection": selection},
        text_root=str(tmp_path),
        signature_fn=signature,
        parquet_inspector=lambda path: (
            3 if path.endswith("pile-0.parquet") else 2,
            "string",
        ),
    )
    assert [
        (
            item["dataset_row_start"],
            item["dataset_row_stop"],
            item["corpus_global_row_start"],
            item["r0087_global_row_start"],
        )
        for item in layout
    ] == [(0, 3, 0, 20), (3, 5, 3, 23)]
    clipped = contract.clip_layout(layout, start=2, stop=4)
    assert [
        (
            item["dataset_row_start"],
            item["dataset_row_stop"],
            item["shard_row_start"],
            item["r0087_global_row_start"],
        )
        for item in clipped
    ] == [(2, 3, 2, 22), (3, 4, 0, 23)]

    broken = copy.deepcopy(selection)
    broken["ranges"][1]["global_row_start"] = 24
    with pytest.raises(contract.Round0120Error, match="malformed"):
        contract.source_layout_from_inventory(
            {"selection": broken},
            text_root=str(tmp_path),
            signature_fn=signature,
            parquet_inspector=lambda path: (
                3 if path.endswith("pile-0.parquet") else 2,
                "string",
            ),
        )


def test_coverage_rejects_duplicate_output_and_r0087_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "CORPUS_ROWS", 4)
    monkeypatch.setattr(contract, "CHUNK_ROWS", 2)
    monkeypatch.setattr(contract, "R0087_PILE_GLOBAL_OFFSET", 10)
    chunks = [
        {
            "dataset": contract.DATASET,
            "dataset_row_range": [0, 2],
            "corpus_global_row_range": [0, 2],
            "r0087_global_row_range": [10, 12],
            "output": {"canonical_path": "/data/pile-0.npy"},
        },
        {
            "dataset": contract.DATASET,
            "dataset_row_range": [2, 4],
            "corpus_global_row_range": [2, 4],
            "r0087_global_row_range": [12, 14],
            "output": {"canonical_path": "/data/pile-1.npy"},
        },
    ]
    contract.validate_coverage(chunks)
    repeated = copy.deepcopy(chunks)
    repeated[1]["output"]["canonical_path"] = "/data/pile-0.npy"
    with pytest.raises(contract.Round0120Error, match="repeated"):
        contract.validate_coverage(repeated)
    drifted = copy.deepcopy(chunks)
    drifted[1]["r0087_global_row_range"] = [13, 15]
    with pytest.raises(contract.Round0120Error, match="row-order"):
        contract.validate_coverage(drifted)


def test_r0116_ordering_receipt_requires_clean_success(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "runner-terminal.json"
    required_jobs = list(prepare.R0116_REQUIRED_JOBS)
    terminal = {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0116",
        "verdict": "succeeded",
        "stop_reason": None,
        "completed_jobs": required_jobs,
        "required_jobs": required_jobs,
        "release_checkout": {
            "repo_root": prepare.RELEASE_ROOT,
            "head": prepare.R0116_RELEASE_SHA,
            "detached": True,
            "dirty": False,
        },
        "release_checkout_at_finish": {
            "repo_root": prepare.RELEASE_ROOT,
            "head": prepare.R0116_RELEASE_SHA,
            "detached": True,
            "dirty": False,
        },
        "release_checkout_unchanged": True,
        "queue_manifest_sha256": "a" * 64,
        "queue_manifest_sha256_at_finish": "a" * 64,
        "queue_manifest_unchanged": True,
        "boundary_problems": [],
        "validation_problems": [],
        "nodes": [
            {
                "node": node,
                "returncode": 0,
                "validation_problems": [],
            }
            for node in required_jobs
        ],
    }
    path.write_text(json.dumps(terminal), encoding="utf-8")
    signature = expected_input_signature(str(path))
    with pytest.raises(RuntimeError, match="canonical receipt"):
        prepare._require_successful_r0116_terminal(
            str(path), expected_sha256=signature["sha256"]
        )

    monkeypatch.setattr(
        prepare, "R0116_TERMINAL_PATH", signature["canonical_path"]
    )
    observed, observed_signature = prepare._require_successful_r0116_terminal(
        str(path), expected_sha256=signature["sha256"]
    )
    assert observed == terminal
    assert observed_signature == signature

    terminal["verdict"] = "failed"
    path.write_text(json.dumps(terminal), encoding="utf-8")
    signature = expected_input_signature(str(path))
    with pytest.raises(RuntimeError, match="clean terminal"):
        prepare._require_successful_r0116_terminal(
            str(path), expected_sha256=signature["sha256"]
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("required_jobs", []),
        ("completed_jobs", []),
        ("queue_manifest_sha256_at_finish", "b" * 64),
    ],
)
def test_r0116_ordering_receipt_rejects_degenerate_identity(
    tmp_path, monkeypatch, mutation, value
) -> None:
    path = tmp_path / "runner-terminal.json"
    required_jobs = list(prepare.R0116_REQUIRED_JOBS)
    checkout = {
        "repo_root": prepare.RELEASE_ROOT,
        "head": prepare.R0116_RELEASE_SHA,
        "detached": True,
        "dirty": False,
    }
    terminal = {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0116",
        "verdict": "succeeded",
        "stop_reason": None,
        "completed_jobs": required_jobs,
        "required_jobs": required_jobs,
        "release_checkout": checkout,
        "release_checkout_at_finish": checkout,
        "release_checkout_unchanged": True,
        "queue_manifest_sha256": "a" * 64,
        "queue_manifest_sha256_at_finish": "a" * 64,
        "queue_manifest_unchanged": True,
        "boundary_problems": [],
        "validation_problems": [],
        "nodes": [
            {
                "node": node,
                "returncode": 0,
                "validation_problems": [],
            }
            for node in required_jobs
        ],
    }
    terminal[mutation] = value
    path.write_text(json.dumps(terminal), encoding="utf-8")
    signature = expected_input_signature(str(path))
    monkeypatch.setattr(
        prepare, "R0116_TERMINAL_PATH", signature["canonical_path"]
    )
    with pytest.raises(RuntimeError, match="clean terminal"):
        prepare._require_successful_r0116_terminal(
            str(path), expected_sha256=signature["sha256"]
        )
