"""Deterministic population contract for the conditional R0135 treatment."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .round0087_inventory import (
    FINEWEB,
    PILE,
    REDPAJAMA,
    language_code,
)
from .round0105_search import GROUPS
from .round0108_evaluation import IN_MIX_LANGUAGES


ROUND_ID = "0135"
FINAL_ROWS = 24_948_663
STAGED_ROWS = 25_000_000
CANDIDATE_ROWS_PER_LANGUAGE = 1_300_000
PADDING_DUPLICATE_ROWS = STAGED_ROWS - FINAL_ROWS
ENGLISH_DATASETS = (FINEWEB, REDPAJAMA, PILE)
LANGUAGE_GROUPS = tuple(sorted(("eng_Latn", *IN_MIX_LANGUAGES)))


class Round0135Error(RuntimeError):
    """The R0135 balanced-population contract was violated."""


def largest_remainder_equal(
    labels: tuple[str, ...], total: int
) -> dict[str, int]:
    if (
        not labels
        or tuple(sorted(labels)) != labels
        or len(set(labels)) != len(labels)
        or total < len(labels)
    ):
        raise Round0135Error("largest-remainder labels/total are malformed")
    base, remainder = divmod(int(total), len(labels))
    quotas = {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }
    if sum(quotas.values()) != total or max(quotas.values()) - min(quotas.values()) > 1:
        raise Round0135Error("largest-remainder allocation did not close")
    return quotas


def language_quotas() -> dict[str, int]:
    return largest_remainder_equal(LANGUAGE_GROUPS, FINAL_ROWS)


def english_quotas(total: int) -> dict[str, int]:
    # This explicit order is inherited from the accepted R0087/R0105 group
    # order, not Python's case-sensitive ordering of dataset names.
    base, remainder = divmod(int(total), len(ENGLISH_DATASETS))
    quotas = {
        dataset: base + (1 if index < remainder else 0)
        for index, dataset in enumerate(ENGLISH_DATASETS)
    }
    if sum(quotas.values()) != total or min(quotas.values()) <= 0:
        raise Round0135Error("English corpus allocation did not close")
    return quotas


def candidate_budgets() -> dict[str, int]:
    budgets = english_quotas(CANDIDATE_ROWS_PER_LANGUAGE)
    budgets.update({
        language: CANDIDATE_ROWS_PER_LANGUAGE
        for language in IN_MIX_LANGUAGES
    })
    if tuple(budgets) != tuple(GROUPS) or sum(budgets.values()) != (
        len(LANGUAGE_GROUPS) * CANDIDATE_ROWS_PER_LANGUAGE
    ):
        raise Round0135Error("candidate budgets changed group order or total")
    return budgets


def final_group_quotas() -> dict[str, int]:
    languages = language_quotas()
    quotas = english_quotas(languages["eng_Latn"])
    quotas.update({
        language: languages[language]
        for language in IN_MIX_LANGUAGES
    })
    if tuple(quotas) != tuple(GROUPS) or sum(quotas.values()) != FINAL_ROWS:
        raise Round0135Error("final group quotas changed order or total")
    return quotas


def build_candidate_selection(
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    budgets = candidate_budgets()
    ranges: list[dict[str, Any]] = []
    cursor = 0
    gaps: list[dict[str, Any]] = []
    for group, budget in budgets.items():
        dataset = (
            group
            if group in ENGLISH_DATASETS
            else f"fineweb2-{group}-chunked-500-jina-v5-nano"
        )
        record = inventory.get(dataset) or {}
        available = int(record.get("rows", 0))
        remaining = min(available, budget)
        dataset_row = 0
        for shard in record.get("shards") or []:
            if remaining <= 0:
                break
            count = min(remaining, int(shard["rows"]))
            ranges.append({
                "dataset": dataset,
                "group": group,
                "language": (
                    "eng_Latn" if dataset in ENGLISH_DATASETS
                    else str(language_code(dataset))
                ),
                "shard": {
                    key: shard[key]
                    for key in ("canonical_path", "sha256", "bytes", "rows")
                },
                "shard_row_start": 0,
                "shard_row_stop": count,
                "dataset_row_start": dataset_row,
                "dataset_row_stop": dataset_row + count,
                "global_row_start": cursor,
                "global_row_stop": cursor + count,
            })
            cursor += count
            dataset_row += count
            remaining -= count
        if available < budget:
            gaps.append({
                "group": group,
                "dataset": dataset,
                "required_rows": budget,
                "available_rows": available,
                "missing_rows": budget - available,
            })
    selection = {
        "candidate_rows": cursor,
        "candidate_rows_per_language": CANDIDATE_ROWS_PER_LANGUAGE,
        "source_order": list(budgets),
        "budgets": budgets,
        "ranges": ranges,
        "gaps": gaps,
        "complete": not gaps and cursor == (
            len(LANGUAGE_GROUPS) * CANDIDATE_ROWS_PER_LANGUAGE
        ),
        "row_order": (
            "accepted R0105 group order; lexicographic shard path within each "
            "dataset; ascending source row"
        ),
    }
    if not selection["complete"]:
        raise Round0135Error(f"candidate windows are incomplete: {gaps}")
    return selection


def _range_stops(selection: Mapping[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    ranges = selection.get("ranges") or []
    stops = np.asarray([int(item["global_row_stop"]) for item in ranges], dtype=np.int64)
    groups = tuple(str(item["group"]) for item in ranges)
    if len(stops) == 0 or stops[-1] != int(selection["candidate_rows"]):
        raise Round0135Error("candidate ranges do not close")
    return stops, groups


def group_ids_for_rows(
    rows: np.ndarray, selection: Mapping[str, Any]
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64)
    stops, range_groups = _range_stops(selection)
    indices = np.searchsorted(stops, values, side="right")
    if (
        values.ndim != 1
        or np.any(values < 0)
        or np.any(values >= int(selection["candidate_rows"]))
        or np.any(indices >= len(range_groups))
    ):
        raise Round0135Error("candidate row outside registered ranges")
    group_to_id = {group: index for index, group in enumerate(GROUPS)}
    try:
        range_ids = np.asarray(
            [group_to_id[group] for group in range_groups], dtype=np.uint8
        )
    except KeyError as exc:
        raise Round0135Error("candidate range has an unknown group") from exc
    return range_ids[indices]


def _validate_census(
    census: Mapping[str, Any], *, candidate_rows: int
) -> dict[str, np.ndarray]:
    arrays = census.get("arrays") or {}
    required = {
        "zero_rows",
        "nonfinite_rows",
        "excluded_rows",
        "duplicate_excluded_rows",
        "duplicate_representative_rows",
        "representative_rows",
        "family_counts",
        "family_offsets",
        "member_rows",
    }
    if set(arrays) != required:
        raise Round0135Error("candidate duplicate census arrays changed")
    output = {
        key: np.asarray(arrays[key], dtype=np.int64)
        for key in required
    }
    excluded = output["excluded_rows"]
    duplicate_rows = output["duplicate_excluded_rows"]
    duplicate_reps = output["duplicate_representative_rows"]
    family_counts = output["family_counts"]
    family_offsets = output["family_offsets"]
    member_rows = output["member_rows"]
    summary = census.get("summary") or {}
    if (
        int(summary.get("row_count", -1)) != candidate_rows
        or int(summary.get("fingerprint_collision_splits", -1)) != 0
        or len(excluded) != int(summary.get("excluded_row_count", -1))
        or len(duplicate_rows) != len(duplicate_reps)
        or len(duplicate_rows)
        != int(summary.get("duplicate_copy_rows_excluded", -1))
        or len(family_offsets) != len(family_counts) + 1
        or not len(family_offsets)
        or family_offsets[0] != 0
        or family_offsets[-1] != len(member_rows)
        or not np.array_equal(np.diff(family_offsets), family_counts)
        or np.any(family_counts < 2)
        or len(output["representative_rows"]) != len(family_counts)
        or (len(excluded) and (
            excluded[0] < 0
            or excluded[-1] >= candidate_rows
            or np.any(excluded[1:] <= excluded[:-1])
        ))
        or (len(duplicate_rows) and (
            duplicate_rows[0] < 0
            or duplicate_rows[-1] >= candidate_rows
            or np.any(duplicate_rows[1:] <= duplicate_rows[:-1])
            or np.any(duplicate_reps < 0)
            or np.any(duplicate_reps >= candidate_rows)
        ))
    ):
        raise Round0135Error("candidate duplicate census is malformed")
    return output


def _membership(sorted_rows: np.ndarray, values: np.ndarray) -> np.ndarray:
    selector = np.asarray(sorted_rows, dtype=np.int64)
    query = np.asarray(values, dtype=np.int64)
    positions = np.searchsorted(selector, query)
    present = positions < len(selector)
    bounded = np.flatnonzero(present)
    present[bounded] = selector[positions[bounded]] == query[bounded]
    return present


def build_balanced_population(
    selection: Mapping[str, Any], census: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_rows = int(selection["candidate_rows"])
    arrays = _validate_census(census, candidate_rows=candidate_rows)
    excluded = arrays["excluded_rows"]
    budgets = candidate_budgets()
    quotas = final_group_quotas()
    if (
        selection.get("complete") is not True
        or selection.get("budgets") != budgets
        or list(selection.get("source_order") or []) != list(GROUPS)
        or candidate_rows != sum(budgets.values())
    ):
        raise Round0135Error("candidate selection is not the frozen population")
    final_parts: list[np.ndarray] = []
    per_group: dict[str, Any] = {}
    group_start = 0
    for group in GROUPS:
        group_stop = group_start + budgets[group]
        start_stop = np.arange(group_start, group_stop, dtype=np.int64)
        group_start = group_stop
        excluded_positions = np.searchsorted(excluded, start_stop)
        is_excluded = excluded_positions < len(excluded)
        valid = np.flatnonzero(is_excluded)
        is_excluded[valid] = excluded[excluded_positions[valid]] == start_stop[valid]
        eligible = start_stop[~is_excluded]
        quota = quotas[group]
        if len(eligible) < quota:
            raise Round0135Error(
                f"{group} has {len(eligible):,} canonical rows for quota {quota:,}"
            )
        chosen = eligible[:quota]
        final_parts.append(chosen)
        per_group[group] = {
            "candidate_rows": len(start_stop),
            "canonical_rows": len(eligible),
            "quota": quota,
            "unused_canonical_rows": len(eligible) - quota,
        }
    if group_start != candidate_rows:
        raise Round0135Error("candidate group ranges do not cover the population")
    final_candidate_rows = np.concatenate(final_parts).astype(np.int64, copy=False)
    if (
        len(final_candidate_rows) != FINAL_ROWS
        or np.any(final_candidate_rows[1:] <= final_candidate_rows[:-1])
        or np.any(_membership(excluded, final_candidate_rows))
    ):
        raise Round0135Error("balanced final representative mapping did not close")
    final_mask = np.zeros(candidate_rows, dtype=np.bool_)
    final_mask[final_candidate_rows] = True
    complement_candidate_rows = np.flatnonzero(~final_mask).astype(
        np.int64, copy=False
    )
    del final_mask
    if len(complement_candidate_rows) != candidate_rows - FINAL_ROWS:
        raise Round0135Error("balanced population complement did not close")

    duplicate_rows = arrays["duplicate_excluded_rows"]
    duplicate_reps = arrays["duplicate_representative_rows"]
    positions = np.searchsorted(final_candidate_rows, duplicate_reps)
    represented = positions < len(final_candidate_rows)
    represented_indices = np.flatnonzero(represented)
    represented[represented_indices] = (
        final_candidate_rows[positions[represented_indices]]
        == duplicate_reps[represented_indices]
    )
    available_padding_rows = duplicate_rows[represented]
    available_padding_reps = duplicate_reps[represented]
    order = np.argsort(available_padding_rows, kind="stable")
    available_padding_rows = available_padding_rows[order]
    available_padding_reps = available_padding_reps[order]
    if len(available_padding_rows) < PADDING_DUPLICATE_ROWS:
        raise Round0135Error(
            "not enough authentic duplicate copies to preserve the 25M substrate ABI"
        )
    padding_candidate_rows = available_padding_rows[:PADDING_DUPLICATE_ROWS]
    padding_candidate_reps = available_padding_reps[:PADDING_DUPLICATE_ROWS]
    rep_positions = np.searchsorted(final_candidate_rows, padding_candidate_reps)
    if not np.array_equal(final_candidate_rows[rep_positions], padding_candidate_reps):
        raise Round0135Error("padding duplicate representative is absent from final rows")
    staged_candidate_rows = np.concatenate(
        (final_candidate_rows, padding_candidate_rows)
    ).astype(np.int64, copy=False)
    if (
        len(staged_candidate_rows) != STAGED_ROWS
        or len(padding_candidate_rows) != PADDING_DUPLICATE_ROWS
        or np.any(padding_candidate_rows[1:] <= padding_candidate_rows[:-1])
        or np.any(_membership(final_candidate_rows, padding_candidate_rows))
    ):
        raise Round0135Error("one candidate row was staged twice")

    staged_excluded = np.arange(FINAL_ROWS, STAGED_ROWS, dtype=np.int64)
    staged_duplicate_reps = rep_positions.astype(np.int64, copy=False)
    family_order = np.lexsort((staged_excluded, staged_duplicate_reps))
    grouped_reps = staged_duplicate_reps[family_order]
    grouped_members = staged_excluded[family_order]
    representatives, starts, counts_without_rep = np.unique(
        grouped_reps, return_index=True, return_counts=True
    )
    family_counts = counts_without_rep.astype(np.int64) + 1
    family_offsets = np.zeros(len(representatives) + 1, dtype=np.int64)
    family_offsets[1:] = np.cumsum(family_counts, dtype=np.int64)
    member_rows = np.empty(int(family_offsets[-1]), dtype=np.int64)
    for index, (representative, start, copies) in enumerate(
        zip(representatives, starts, counts_without_rep, strict=True)
    ):
        left = family_offsets[index]
        right = family_offsets[index + 1]
        member_rows[left] = representative
        member_rows[left + 1:right] = grouped_members[start:start + copies]
    staged_group_ids = group_ids_for_rows(staged_candidate_rows, selection)
    final_language_counts = {
        language: 0 for language in LANGUAGE_GROUPS
    }
    for group, quota in quotas.items():
        language = "eng_Latn" if group in ENGLISH_DATASETS else group
        final_language_counts[language] += quota
    if final_language_counts != language_quotas():
        raise Round0135Error("final language counts do not match equal quotas")
    eligibility = {
        "zero_rows": np.empty(0, dtype=np.int64),
        "nonfinite_rows": np.empty(0, dtype=np.int64),
        "excluded_rows": staged_excluded,
        "duplicate_excluded_rows": staged_excluded.copy(),
        "duplicate_representative_rows": staged_duplicate_reps,
        "representative_rows": representatives,
        "family_counts": family_counts,
        "family_offsets": family_offsets,
        "member_rows": member_rows,
    }
    return {
        "final_candidate_rows": final_candidate_rows,
        "complement_candidate_rows": complement_candidate_rows,
        "padding_candidate_rows": padding_candidate_rows,
        "padding_candidate_representatives": padding_candidate_reps,
        "staged_candidate_rows": staged_candidate_rows,
        "staged_group_ids": staged_group_ids,
        "eligibility": eligibility,
        "per_group": per_group,
        "final_group_quotas": quotas,
        "final_language_quotas": final_language_counts,
        "candidate_duplicate_summary": dict(census["summary"]),
        "available_authentic_padding_duplicates": len(available_padding_rows),
        "checks": {
            "canonicalization_precedes_quota_fill": True,
            "exact_final_rows": len(final_candidate_rows) == FINAL_ROWS,
            "equal_language_weight_within_one_row": (
                max(final_language_counts.values())
                - min(final_language_counts.values()) <= 1
            ),
            "no_final_exact_duplicates_zero_or_nonfinite": True,
            "padding_uses_distinct_authentic_candidate_copies": True,
            "padding_never_enters_compact_training_universe": True,
        },
    }
