"""Registered retained-only search qualification for the diverse Jina atlas."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0087_inventory import (
    FINEWEB,
    PILE,
    REDPAJAMA,
)


ROUND_ID = "0105"
ROW_COUNT = 25_000_000
RETAINED_ROWS = 24_948_663
DIMENSION = 768
K = 15

NLIST = 8_192
PQ_M = 96
PQ_BITS = 8
INDEX_TRAIN_ROWS = 40 * NLIST
INDEX_TRAIN_SEED = 105

QUALITY_ROWS_PER_GROUP = 256
QUALITY_SEED = 105
GROUPS = (
    FINEWEB,
    REDPAJAMA,
    PILE,
    "arb_Arab",
    "ces_Latn",
    "cmn_Hani",
    "deu_Latn",
    "ell_Grek",
    "fra_Latn",
    "hin_Deva",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kor_Hang",
    "nld_Latn",
    "por_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "swe_Latn",
    "tha_Thai",
    "tur_Latn",
    "vie_Latn",
)
QUALITY_ROWS = QUALITY_ROWS_PER_GROUP * len(GROUPS)

# Filled from the deterministic selectors before the round is issued. Keeping
# these values in the release makes a changed NumPy RNG path fail closed.
INDEX_TRAIN_SAMPLE_SHA256 = (
    "5d9dfa79b8ac6c28cce92491aee2b59936c4ffec18387f88dd56008323be0fcc"
)
QUALITY_SAMPLE_SHA256 = (
    "4c868d21b6ddd83d47929e7dffd3d607f5d2f9b59f34ebbac03a80eeed4f5b63"
)
QUALITY_GROUP_IDS_SHA256 = (
    "103f878720e2782e1b2eb5df5a1cb69d8a0d0e6d0348df328ca66f1e66aa1108"
)

GLOBAL_MEAN_FLOOR = 0.90
EVERY_GROUP_MEAN_FLOOR = 0.84
POLICY_GRID = tuple(
    (nprobe, width)
    for nprobe in (64, 128, 192)
    for width in (128, 256, 512)
)
BOUNDARY_TIE_ATOL = 1e-7
BENCHMARK_WARMUP_ROWS = 512
BENCHMARK_REPEATS = 3

SUBSTRATE_MANIFEST_PATH = (
    "/data/latent-basemap/runs/round-0103/queue/artifacts/"
    "jina-diverse-25m-full768-int8-substrate/"
    "jina-diverse-25m-full768-int8-substrate-v1.json"
)
SUBSTRATE_MANIFEST_SHA256 = (
    "b01bc7872cbb22e02b64afed1886bed607b21acd9ac0349caaa2fd88713cc7fa"
)
ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/"
    "jina-diverse-25m-eligibility-v1.npz"
)
ELIGIBILITY_SHA256 = (
    "11a9c197f0e20cb1e5d6968bc9ec3a9e2c89fa66c711d252f864959016eac274"
)

INDEX_SCHEMA = "round0105-jina-diverse-25m-ivf8192-pq96x8-index-v1"
QUALIFICATION_SCHEMA = (
    "round0105-jina-diverse-25m-retained-search-qualification-v1"
)
DECISION_SCHEMA = "round0105-jina-diverse-25m-search-decision-v1"


class Round0105Error(RuntimeError):
    """The registered diverse-Jina search treatment or evidence is invalid."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical content-sealed JSON object."""
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def membership(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return membership in a strictly sorted int64 selector."""
    selector = np.asarray(sorted_values, dtype=np.int64)
    query = np.asarray(values, dtype=np.int64)
    flat = query.reshape(-1)
    positions = np.searchsorted(selector, flat)
    present = positions < len(selector)
    indices = np.flatnonzero(present)
    present[indices] = selector[positions[indices]] == flat[indices]
    return present.reshape(query.shape)


def sample_retained_rows(
    excluded: np.ndarray,
    *,
    count: int,
    seed: int,
    row_count: int = ROW_COUNT,
) -> np.ndarray:
    """Draw an unbiased deterministic retained sample without replacement."""
    selector = np.asarray(excluded, dtype=np.int64)
    if (
        selector.ndim != 1
        or count <= 0
        or count > row_count - len(selector)
        or (len(selector) and (
            selector[0] < 0
            or selector[-1] >= row_count
            or np.any(selector[1:] <= selector[:-1])
        ))
    ):
        raise Round0105Error("retained-row selector is malformed")
    rng = np.random.RandomState(seed)
    rows = np.empty(0, dtype=np.int64)
    while len(rows) < count:
        proposed = rng.randint(
            0,
            row_count,
            size=max(2 * (count - len(rows)), 1_024),
            dtype=np.int64,
        )
        proposed = proposed[~membership(selector, proposed)]
        rows = np.unique(np.concatenate((rows, proposed)))
    if len(rows) > count:
        rows = rows[rng.choice(len(rows), size=count, replace=False)]
    return np.sort(rows).astype(np.int64, copy=False)


def group_ranges(substrate_manifest: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Return the 22 contiguous source/language ranges in registered order."""
    labels = substrate_manifest.get("labels") or {}
    vocabulary = labels.get("vocabulary") or {}
    counts = labels.get("counts") or {}
    datasets = list(vocabulary.get("dataset") or [])
    dataset_counts = counts.get("dataset") or {}
    if len(datasets) != len(GROUPS) or set(dataset_counts) != set(datasets):
        raise Round0105Error("substrate dataset vocabulary changed")
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for index, dataset in enumerate(datasets):
        count = int(dataset_counts.get(dataset, -1))
        if count <= 0:
            raise Round0105Error("substrate dataset count is invalid")
        group = dataset if index < 3 else dataset.removeprefix(
            "fineweb2-"
        ).removesuffix("-chunked-500-jina-v5-nano")
        if group != GROUPS[index]:
            raise Round0105Error("substrate group order changed")
        ranges[group] = (cursor, cursor + count)
        cursor += count
    if cursor != ROW_COUNT:
        raise Round0105Error("substrate group ranges do not close")
    return ranges


def sample_stratified_rows(
    excluded: np.ndarray,
    ranges: Mapping[str, tuple[int, int]],
    *,
    rows_per_group: int = QUALITY_ROWS_PER_GROUP,
    seed: int = QUALITY_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw exactly the registered number of retained rows per group."""
    selector = np.asarray(excluded, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rows: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    for group_id, group in enumerate(GROUPS):
        if group not in ranges:
            raise Round0105Error(f"missing registered group {group}")
        start, stop = ranges[group]
        selected = np.empty(0, dtype=np.int64)
        while len(selected) < rows_per_group:
            proposed = rng.randint(
                start,
                stop,
                size=max(2 * (rows_per_group - len(selected)), 1_024),
                dtype=np.int64,
            )
            proposed = proposed[~membership(selector, proposed)]
            selected = np.unique(np.concatenate((selected, proposed)))
        if len(selected) > rows_per_group:
            selected = selected[
                rng.choice(len(selected), size=rows_per_group, replace=False)
            ]
        selected.sort()
        rows.append(selected)
        group_ids.append(
            np.full(rows_per_group, group_id, dtype=np.uint8)
        )
    sample = np.concatenate(rows).astype(np.int64, copy=False)
    ids = np.concatenate(group_ids)
    if (
        len(sample) != QUALITY_ROWS
        or len(np.unique(sample)) != QUALITY_ROWS
        or np.any(membership(selector, sample))
    ):
        raise Round0105Error("stratified quality sample is malformed")
    return sample, ids


def select_cell(cells: Mapping[str, Any]) -> dict[str, Any] | None:
    """Select the fastest measured cell passing all preregistered safeguards."""
    passing = [
        cell
        for cell in cells.values()
        if isinstance(cell, dict)
        and cell.get("passes_global_floor") is True
        and cell.get("passes_every_group_floor") is True
        and cell.get("all_rows_complete") is True
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
