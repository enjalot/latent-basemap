"""Contracts for recovering the completed R0112 paired embedding bytes.

R0112 embedded all 2M rows under the model's native 8192-token limit, but its
last slice failed a diagnostic against historical embeddings produced with an
explicit 512-token limit.  This module deliberately does not re-embed or
reinterpret R0112 as successful.  It defines a new, CPU-only evidence product
over the preserved paired bytes and binds their actual native-8192 execution
semantics.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes
from .round0112_prompt_substrate import DIMENSION, ROWS


ROUND_ID = "0114"
SOURCE_ROUND_ID = "0112"
SOURCE_RELEASE_SHA = "b43847d744946934ce5d4c8a9037114ec0b81659"
SOURCE_QUEUE_ROOT = "/data/latent-basemap/runs/round-0112/queue"
SOURCE_ARTIFACT_ROOT = f"{SOURCE_QUEUE_ROOT}/artifacts"
SOURCE_TERMINAL_PATH = f"{SOURCE_QUEUE_ROOT}/runner-terminal.json"
SOURCE_FAILED_PATH = (
    f"{SOURCE_ARTIFACT_ROOT}/embed_paired_slice_03.failed.json"
)

NATIVE_MAX_SEQ_LENGTH = 8192
HISTORICAL_MAX_SEQ_LENGTH = 512
HISTORICAL_SAMPLE_ROWS = 256
HISTORICAL_OVERALL_MEAN_FLOOR = 0.98
COMPARABLE_MEAN_FLOOR = 0.98
COMPARABLE_MINIMUM_FLOOR = 0.95
ROW_IDENTITY_RADIUS = 16

RECOVERY_SCHEMA = "jina-fineweb-2m-dual-prompt-native8192-substrate-v2"
CAPABILITY = RECOVERY_SCHEMA


class Round0114Error(RuntimeError):
    """The R0114 recovery evidence does not satisfy its fixed contract."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_source_terminal(value: Mapping[str, Any]) -> None:
    """Require the exact honest R0112 failure this round is allowed to recover."""
    if (
        value.get("schema") != "slim-runner-terminal-v3"
        or value.get("round_id") != SOURCE_ROUND_ID
        or value.get("verdict") != "failed"
        or (value.get("release_checkout") or {}).get("head")
        != SOURCE_RELEASE_SHA
        or (value.get("release_checkout_at_finish") or {}).get("head")
        != SOURCE_RELEASE_SHA
        or value.get("release_checkout_unchanged") is not True
        or value.get("queue_manifest_unchanged") is not True
        or value.get("completed_jobs")
        != [
            "embed_paired_slice_00",
            "embed_paired_slice_01",
            "embed_paired_slice_02",
        ]
        or "embed_paired_slice_03 exited 1"
        not in str(value.get("stop_reason") or "")
    ):
        raise Round0114Error("R0112 terminal receipt is not the registered failure")


def validate_source_failure(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema") != "slim-runner-failed-v2"
        or value.get("node") != "embed_paired_slice_03"
        or int(value.get("returncode", 0)) != 1
        or value.get("release_sha") != SOURCE_RELEASE_SHA
        or "fresh raw local embeddings failed the historical alignment guard"
        not in str(value.get("log_tail") or "")
    ):
        raise Round0114Error("R0112 failed marker is not the registered guard")


def source_slice_root(index: int) -> str:
    if index not in range(4):
        raise ValueError("slice index must be in [0, 4)")
    start = index * 500_000
    return (
        f"{SOURCE_ARTIFACT_ROOT}/"
        f"paired-embedding-slice-{start:07d}-{start + 500_000:07d}"
    )


def source_chunk_path(arm: str, global_chunk: int) -> str:
    if arm not in {"raw", "document"} or global_chunk not in range(80):
        raise ValueError("invalid R0112 arm/chunk")
    slice_index, local_chunk = divmod(global_chunk, 20)
    return (
        f"{source_slice_root(slice_index)}/{arm}/"
        f"data-{local_chunk:05d}.npy"
    )


def source_sample_positions() -> list[int]:
    """Reproduce R0112's preregistered 64 positions in each 500k slice."""
    import numpy as np

    positions: list[int] = []
    for start in range(0, ROWS, 500_000):
        local = np.sort(
            np.random.default_rng(11_200 + start).choice(
                500_000,
                size=64,
                replace=False,
            ).astype(np.int64)
        )
        positions.extend((start + local).tolist())
    if len(positions) != HISTORICAL_SAMPLE_ROWS:
        raise Round0114Error("R0112 sample-position reproduction did not close")
    return positions


def validate_manifest_shape(value: Mapping[str, Any]) -> None:
    conventions = value.get("conventions") or {}
    duplicate = value.get("duplicate_control") or {}
    diagnostics = value.get("diagnostics") or {}
    identity = diagnostics.get("historical_row_identity") or {}
    if (
        value.get("schema") != RECOVERY_SCHEMA
        or value.get("round_id") != ROUND_ID
        or int(value.get("row_count", -1)) != ROWS
        or int(value.get("dimension", -1)) != DIMENSION
        or set(conventions) != {"raw", "document"}
        or any(
            len((conventions[arm] or {}).get("chunks") or []) != 80
            for arm in ("raw", "document")
        )
        or int(duplicate.get("excluded_exact_copy_rows", -1)) != 5_366
        or int(duplicate.get("retained_representative_rows", -1)) != 1_994_634
        or identity.get("same_row_top1_count") != HISTORICAL_SAMPLE_ROWS
        or identity.get("passed") is not True
    ):
        raise Round0114Error("R0114 recovered substrate contract changed")
