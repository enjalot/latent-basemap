from __future__ import annotations

import pytest

from basemap.round0114_prompt_recovery import (
    HISTORICAL_SAMPLE_ROWS,
    Round0114Error,
    source_chunk_path,
    source_sample_positions,
    validate_source_failure,
    validate_source_terminal,
)


def _terminal() -> dict:
    return {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0112",
        "verdict": "failed",
        "stop_reason": "node embed_paired_slice_03 exited 1 after 66.4 min",
        "completed_jobs": [
            "embed_paired_slice_00",
            "embed_paired_slice_01",
            "embed_paired_slice_02",
        ],
        "release_checkout": {
            "head": "b43847d744946934ce5d4c8a9037114ec0b81659"
        },
        "release_checkout_at_finish": {
            "head": "b43847d744946934ce5d4c8a9037114ec0b81659"
        },
        "release_checkout_unchanged": True,
        "queue_manifest_unchanged": True,
    }


def test_source_sample_positions_are_fixed_and_complete() -> None:
    positions = source_sample_positions()
    assert len(positions) == HISTORICAL_SAMPLE_ROWS
    assert len(set(positions)) == HISTORICAL_SAMPLE_ROWS
    assert positions == sorted(positions)
    assert positions[0] >= 0
    assert positions[-1] < 2_000_000


def test_source_chunk_paths_preserve_slice_local_numbering() -> None:
    assert source_chunk_path("raw", 0).endswith(
        "paired-embedding-slice-0000000-0500000/raw/data-00000.npy"
    )
    assert source_chunk_path("document", 79).endswith(
        "paired-embedding-slice-1500000-2000000/document/data-00019.npy"
    )


def test_terminal_validation_preserves_honest_failure() -> None:
    validate_source_terminal(_terminal())
    changed = _terminal()
    changed["verdict"] = "succeeded"
    with pytest.raises(Round0114Error):
        validate_source_terminal(changed)


def test_failed_marker_must_name_original_guard() -> None:
    value = {
        "schema": "slim-runner-failed-v2",
        "node": "embed_paired_slice_03",
        "returncode": 1,
        "release_sha": "b43847d744946934ce5d4c8a9037114ec0b81659",
        "log_tail": (
            "fresh raw local embeddings failed the historical alignment guard"
        ),
    }
    validate_source_failure(value)
    value["log_tail"] = "something else"
    with pytest.raises(Round0114Error):
        validate_source_failure(value)
