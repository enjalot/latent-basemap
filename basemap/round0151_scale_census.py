"""Pure contract for a diverse-Jina prefix/drop-only scale census.

The candidate deliberately differs from R0132's post-deduplication SHA-ranked
half: allocate a raw 12.5M prefix across the same 22 registered groups, take
the prefix in each group, and then remove R0087-ineligible rows without
replacement.  This mirrors the row-policy package tested at 2M by R0149/0150
without pretending to isolate a unique causal factor.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .round0105_search import GROUPS


ROUND_ID = "0151"
CAPABILITY = "jina-diverse-12p5m-prefix-drop-only-census-v1"
FULL_RAW_ROWS = 25_000_000
RAW_PREFIX_TARGET = 12_500_000
EXPECTED_RETAINED_ROWS = 12_485_206
EXPECTED_DROPPED_ROWS = 14_794
EXPECTED_U12_OVERLAP = 6_243_347
EXPECTED_MAPPING_ORDERED_SHA256 = (
    "0c27e0d2498e8d179cfa43f5828ad5cbc2c0de0b3eee92173033686b9264f8b8"
)
EXPECTED_GROUP_IDS_ORDERED_SHA256 = (
    "b3b5af91467d7900a7f7f0832ef8dd41f029eaba56864334f223ecd21e4cd831"
)


class Round0151Error(RuntimeError):
    """The preregistered prefix/drop-only census is malformed."""


def largest_remainder_prefix_quotas(
    group_counts: Mapping[str, int], *, target: int = RAW_PREFIX_TARGET
) -> dict[str, int]:
    """Allocate an exact raw-prefix target with registered-order tie breaks."""
    if set(group_counts) != set(GROUPS):
        raise Round0151Error("prefix group-count keys changed")
    counts = {group: int(group_counts[group]) for group in GROUPS}
    total = sum(counts.values())
    if any(value <= 0 for value in counts.values()) or not 0 < target < total:
        raise Round0151Error("prefix group counts or target are invalid")
    quotas = {group: target * counts[group] // total for group in GROUPS}
    remaining = target - sum(quotas.values())
    ranked = sorted(
        GROUPS,
        key=lambda group: (
            -((target * counts[group]) % total),
            GROUPS.index(group),
        ),
    )
    for group in ranked[:remaining]:
        quotas[group] += 1
    if (
        sum(quotas.values()) != target
        or any(not 0 < quotas[group] <= counts[group] for group in GROUPS)
    ):
        raise Round0151Error("raw-prefix quotas did not close")
    return quotas


def build_prefix_drop_mapping(
    ranges: Mapping[str, tuple[int, int]],
    excluded_rows: np.ndarray,
    *,
    target: int = RAW_PREFIX_TARGET,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Take each group's allocated raw prefix and drop exclusions in place."""
    if set(ranges) != set(GROUPS):
        raise Round0151Error("prefix group ranges changed")
    cursor = 0
    counts: dict[str, int] = {}
    for group in GROUPS:
        start, stop = (int(value) for value in ranges[group])
        if start != cursor or stop <= start:
            raise Round0151Error("prefix group ranges are not contiguous")
        counts[group] = stop - start
        cursor = stop
    selector = np.asarray(excluded_rows, dtype=np.int64)
    if (
        selector.ndim != 1
        or (len(selector) > 1 and np.any(selector[1:] <= selector[:-1]))
        or (len(selector) and (int(selector[0]) < 0 or int(selector[-1]) >= cursor))
    ):
        raise Round0151Error("R0087 exclusion selector is malformed")

    quotas = largest_remainder_prefix_quotas(counts, target=target)
    selected_groups: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    receipts: dict[str, dict[str, int]] = {}
    for group_id, group in enumerate(GROUPS):
        start, stop = ranges[group]
        prefix_stop = int(start) + quotas[group]
        left = int(np.searchsorted(selector, start, side="left"))
        right = int(np.searchsorted(selector, prefix_stop, side="left"))
        local_excluded = selector[left:right]
        selected = np.arange(start, prefix_stop, dtype=np.int64)
        if len(local_excluded):
            keep = np.ones(len(selected), dtype=bool)
            keep[local_excluded - int(start)] = False
            selected = selected[keep]
        selected_groups.append(selected)
        group_ids.append(np.full(len(selected), group_id, dtype=np.uint8))
        receipts[group] = {
            "global_start": int(start),
            "global_stop": int(stop),
            "raw_group_rows": counts[group],
            "raw_prefix_rows": quotas[group],
            "raw_prefix_stop": prefix_stop,
            "dropped_rows": len(local_excluded),
            "retained_rows": len(selected),
            "replacement_rows": 0,
        }

    mapping = np.concatenate(selected_groups).astype(np.int64, copy=False)
    ids = np.concatenate(group_ids).astype(np.uint8, copy=False)
    dropped = target - len(mapping)
    if (
        len(mapping) == 0
        or ids.shape != mapping.shape
        or np.any(mapping[1:] <= mapping[:-1])
        or sum(item["raw_prefix_rows"] for item in receipts.values()) != target
        or sum(item["dropped_rows"] for item in receipts.values()) != dropped
        or sum(item["retained_rows"] for item in receipts.values()) != len(mapping)
    ):
        raise Round0151Error("prefix/drop-only mapping did not close")
    return mapping, ids, {
        "full_raw_rows": cursor,
        "raw_prefix_target": target,
        "retained_rows": len(mapping),
        "dropped_rows": dropped,
        "replacement_rows": 0,
        "quotas": quotas,
        "groups": receipts,
    }


def compare_to_u12(mapping: np.ndarray, u12: np.ndarray) -> dict[str, Any]:
    """Prove that the candidate is not a re-materialization of R0132 U12."""
    candidate = np.asarray(mapping, dtype=np.int64)
    control = np.asarray(u12, dtype=np.int64)
    if (
        candidate.ndim != 1
        or control.ndim != 1
        or len(candidate) == 0
        or len(control) == 0
        or np.any(candidate[1:] <= candidate[:-1])
        or np.any(control[1:] <= control[:-1])
    ):
        raise Round0151Error("candidate or U12 mapping is malformed")
    positions = np.searchsorted(control, candidate, side="left")
    bounded = positions < len(control)
    matched = np.zeros(len(candidate), dtype=bool)
    matched[bounded] = control[positions[bounded]] == candidate[bounded]
    overlap = int(np.count_nonzero(matched))
    union = len(candidate) + len(control) - overlap
    identical = len(candidate) == len(control) and overlap == len(candidate)
    return {
        "candidate_rows": len(candidate),
        "u12_rows": len(control),
        "row_count_delta": len(candidate) - len(control),
        "overlap_rows": overlap,
        "candidate_only_rows": len(candidate) - overlap,
        "u12_only_rows": len(control) - overlap,
        "jaccard": overlap / union,
        "byte_or_set_identical": identical,
        "distinct": not identical,
    }
