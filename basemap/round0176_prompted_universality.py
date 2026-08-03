"""Frozen contract for the negative-Q2-aware prompted OOD panel."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap import round0167_prompted_universality as base
from basemap.round0087_inventory import _fingerprint_fp16


ROUND_ID = "0176"
CAPABILITY = "jina-prompted-universality-panel-v1"
PROMPTED_MAP_ORDER = (
    "r0115-prompted-2m-seed42",
    "r0117-prompted-2m-seed43",
    "r0171-prompted-8m-seed42",
)


class Round0176Error(base.Round0167Error):
    """The R0176 prompted-universality contract changed."""


def _fingerprints(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(values)
    h0 = np.empty(rows, dtype=np.uint64)
    h1 = np.empty(rows, dtype=np.uint64)
    zero = np.empty(rows, dtype=bool)
    nonfinite = np.empty(rows, dtype=bool)
    bits = np.ascontiguousarray(values).view("<u2")
    _fingerprint_fp16(bits, h0, h1, zero, nonfinite)
    return h0, h1, zero, nonfinite


def exact_training_overlap_report(
    *,
    entries: Sequence[Mapping[str, Any]],
    training_sources: Mapping[str, np.ndarray],
    block_rows: int = 65_536,
) -> dict[str, Any]:
    """Resolve fp16 row-family overlap with two-stage hash + byte equality.

    Query and matched-control overlap is a validity blocker.  Corpus overlap is
    retained as a diagnostic so the frozen R0142 selections are never silently
    changed after their prompted bytes are observed.
    """
    if not entries or not training_sources or block_rows <= 0:
        raise Round0176Error("training-overlap audit inputs are incomplete")
    pair_dtype = np.dtype([("h0", "<u8"), ("h1", "<u8")])
    normalized: list[dict[str, Any]] = []
    total_rows = 0
    dimension: int | None = None
    for raw in entries:
        values = np.asarray(raw.get("values"))
        source_rows = np.asarray(raw.get("source_rows"))
        split = str(raw.get("split") or "")
        label = str(raw.get("label") or "")
        if (
            values.ndim != 2
            or values.dtype != np.float16
            or not label
            or split not in {"corpus", "queries", "control"}
            or source_rows.shape != (len(values),)
            or source_rows.dtype.kind not in "iu"
            or len(values) == 0
        ):
            raise Round0176Error("training-overlap probe entry is malformed")
        if dimension is None:
            dimension = int(values.shape[1])
        elif values.shape[1] != dimension:
            raise Round0176Error("training-overlap dimensions disagree")
        normalized.append({
            "label": label,
            "split": split,
            "values": values,
            "source_rows": source_rows,
            "start": total_rows,
            "stop": total_rows + len(values),
        })
        total_rows += len(values)

    probe_pairs = np.empty(total_rows, dtype=pair_dtype)
    for entry in normalized:
        h0, h1, zero, nonfinite = _fingerprints(entry["values"])
        if np.any(zero) or np.any(nonfinite):
            raise Round0176Error("training-overlap probe rows are invalid")
        start, stop = int(entry["start"]), int(entry["stop"])
        probe_pairs["h0"][start:stop] = h0
        probe_pairs["h1"][start:stop] = h1
    unique_probe_pairs = np.unique(probe_pairs)

    exact_overlaps: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for training_label, raw_source in training_sources.items():
        source = np.asarray(raw_source)
        if (
            source.ndim != 2
            or source.shape[1] != dimension
            or source.dtype != np.float16
            or len(source) == 0
        ):
            raise Round0176Error(
                f"training-overlap source {training_label!r} is malformed"
            )
        candidates: dict[tuple[int, int], list[tuple[int, bytes]]] = {}
        for start in range(0, len(source), block_rows):
            stop = min(start + block_rows, len(source))
            block = np.asarray(source[start:stop])
            h0, h1, zero, nonfinite = _fingerprints(block)
            if np.any(zero) or np.any(nonfinite):
                raise Round0176Error(
                    f"training-overlap source {training_label!r} is invalid"
                )
            pairs = np.empty(len(block), dtype=pair_dtype)
            pairs["h0"] = h0
            pairs["h1"] = h1
            positions = np.searchsorted(unique_probe_pairs, pairs)
            in_range = positions < len(unique_probe_pairs)
            hits = np.zeros(len(block), dtype=bool)
            if np.any(in_range):
                hits[in_range] = (
                    unique_probe_pairs[positions[in_range]] == pairs[in_range]
                )
            for local in np.flatnonzero(hits).tolist():
                key = (int(h0[local]), int(h1[local]))
                candidates.setdefault(key, []).append((
                    start + local,
                    np.asarray(block[local]).tobytes(order="C"),
                ))
            if sum(len(items) for items in candidates.values()) > 100_000:
                raise Round0176Error(
                    "training-overlap candidate count is implausibly large"
                )

        source_exact_start = len(exact_overlaps)
        fingerprint_hits = set(candidates)
        if fingerprint_hits:
            for entry in normalized:
                values = entry["values"]
                source_rows = entry["source_rows"]
                for start in range(0, len(values), block_rows):
                    stop = min(start + block_rows, len(values))
                    block = np.asarray(values[start:stop])
                    h0, h1, _zero, _nonfinite = _fingerprints(block)
                    for local in range(len(block)):
                        key = (int(h0[local]), int(h1[local]))
                        if key not in fingerprint_hits:
                            continue
                        raw = np.asarray(block[local]).tobytes(order="C")
                        for training_row, training_raw in candidates[key]:
                            if raw == training_raw:
                                exact_overlaps.append({
                                    "training_source": str(training_label),
                                    "training_row": int(training_row),
                                    "probe": entry["label"],
                                    "split": entry["split"],
                                    "source_row": int(source_rows[start + local]),
                                })
        source_exact = exact_overlaps[source_exact_start:]
        source_summaries[str(training_label)] = {
            "training_rows": int(len(source)),
            "fingerprint_candidate_training_rows": int(
                sum(len(items) for items in candidates.values())
            ),
            "exact_overlap_count": int(len(source_exact)),
            "blocking_overlap_count": int(sum(
                item["split"] in {"queries", "control"}
                for item in source_exact
            )),
        }

    blocking = [
        item for item in exact_overlaps
        if item["split"] in {"queries", "control"}
    ]
    corpus = [item for item in exact_overlaps if item["split"] == "corpus"]
    return {
        "identity": "complete stored prompted-fp16 row bytes",
        "probe_rows": int(total_rows),
        "unique_probe_fingerprints": int(len(unique_probe_pairs)),
        "duplicate_probe_rows": int(total_rows - len(unique_probe_pairs)),
        "training_sources": source_summaries,
        "exact_training_family_overlaps": exact_overlaps,
        "exact_training_family_overlap_count": int(len(exact_overlaps)),
        "blocking_query_or_control_overlap_count": int(len(blocking)),
        "diagnostic_corpus_overlap_count": int(len(corpus)),
        "all_rows_training_disjoint": not exact_overlaps,
        "passed": not blocking,
        "policy": (
            "query/control exact overlap blocks; corpus overlap is reported "
            "without filtering or changing frozen R0142 selections"
        ),
    }


def _configure_base() -> None:
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    base.Round0167Error = Round0176Error


def twonn_correlations(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _configure_base()
    return base.twonn_correlations(cells)


def retention_verdict(value: float) -> str:
    _configure_base()
    return base.retention_verdict(value)


__all__ = [
    "CAPABILITY",
    "PROMPTED_MAP_ORDER",
    "ROUND_ID",
    "Round0176Error",
    "exact_training_overlap_report",
    "retention_verdict",
    "twonn_correlations",
]
