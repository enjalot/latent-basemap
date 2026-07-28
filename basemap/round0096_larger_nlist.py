"""Registered geometry for the balanced-150M larger-nlist qualification."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, sha256_bytes


ROUND_ID = "0096"
ROW_COUNT = 150_000_000
RETAINED_ROWS = 147_221_757
DIMENSION = 384
K = 15
NLIST = 32_768
PQ_M = 48
PQ_BITS = 8
TRAIN_ROWS = 40 * NLIST
TRAIN_SEED = 96
TRAIN_SAMPLE_SHA256 = (
    "388093ae4d985f03f55927c7ee12c879c49b17fd44893e6fc855472489dac3ee"
)
QUALITY_ROWS = 4_096
QUALITY_SEED = 86
QUALITY_SAMPLE_SHA256 = (
    "fab1613919b657a8116931b0fc336678576ea25ac3ce875b00576f860fa413fe"
)
GLOBAL_MEAN_FLOOR = 0.90
PER_CORPUS_MEAN_FLOOR = 0.84
POLICY_GRID = tuple(
    (nprobe, width)
    for width in (512, 1_024, 1_536, 2_047)
    for nprobe in (128, 256, 512, 768)
)
CORPUS_RANGES = {
    "fineweb": (0, 50_000_000, 48_529_276),
    "redpajama": (50_000_000, 100_000_000, 49_567_453),
    "pile": (100_000_000, 150_000_000, 49_125_028),
}
TEMPLATE_SCHEMA = "round0096-balanced-150m-ivf32768-template-v1"
SHARD_SCHEMA = "round0096-balanced-150m-ivf32768-shard-v1"
INDEX_SCHEMA = "round0096-balanced-150m-ivf32768-index-v1"
QUALIFICATION_SCHEMA = (
    "round0096-balanced-150m-ivf32768-qualification-v1"
)
DECISION_SCHEMA = "round0096-balanced-150m-search-policy-decision-v1"


class Round0096Error(RuntimeError):
    """The registered larger-nlist treatment or evidence is invalid."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical content-sealed JSON object."""
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def select_cell(
    cells: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Select the fastest cell passing both global and corpus safeguards."""
    passing = [
        cell
        for cell in cells.values()
        if isinstance(cell, dict)
        and cell.get("passes_global_floor") is True
        and cell.get("passes_every_corpus_floor") is True
        and isinstance(cell.get("benchmark"), dict)
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda cell: (
            float(cell["benchmark"]["median_wall_seconds_per_query"]),
            int(cell["shortlist_width"]),
            int(cell["nprobe"]),
        ),
    )
