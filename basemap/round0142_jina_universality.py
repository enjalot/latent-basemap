"""Frozen design helpers for the R0142 Jina universality panel.

The panel measures within-probe neighbourhood retention relative to an equally
shaped, raw-Jina FineWeb control.  It is diagnostic: it does not promote either
map or change an atlas-quality gate.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes


ROUND_ID = "0142"
CAPABILITY = "jina-diverse-universality-panel-v1"
DIMENSION = 768
MAX_PROBE_TOTAL_ROWS = 50_000
QUERY_FRACTION = 0.01
MIN_QUERY_ROWS = 32
MAX_QUERY_ROWS = 500
SPLIT_SEED = 20_260_814
CONTROL_SEED = 20_260_815
PASS_RETENTION = 0.70
FAIL_RETENTION = 0.50
EMBED_BATCH_ROWS = 256
EMBED_CHUNK_ROWS = 5_000
EMBED_MINIMUM_ROWS_PER_S = 120.0
EMBED_WARNING_ROWS_PER_S = 170.0

COMMON_CORPUS_ROWS = {
    "cebuano": 3_537,
    "code": 168_040,
    "culture": 200_000,
    "danish": 23_084,
    "government": 200_000,
    "latin": 200_000,
    "science": 200_000,
    "web": 47_400,
}
PROBE_ORDER = (
    *COMMON_CORPUS_ROWS,
    "scifact",
    "trec-covid",
    "dadabase",
)
MAP_ORDER = ("r0107-25m-seed42", "r0132-12p5m-seed42")


class Round0142Error(RuntimeError):
    """The preregistered R0142 contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0142Error(f"{label} identity seal is invalid")


def stable_seed(name: str, purpose: str) -> int:
    digest = hashlib.sha256(
        f"{SPLIT_SEED}:{name}:{purpose}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _row_keys(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values)
    if array.ndim != 2 or array.shape[1] != DIMENSION:
        raise Round0142Error("probe embedding geometry changed")
    return array.view(np.dtype((np.void, array.dtype.itemsize * DIMENSION))).reshape(-1)


def canonical_representatives(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return first-row representatives for exact stored-vector families."""
    keys = _row_keys(values)
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    rows = np.sort(first.astype(np.int64, copy=False))
    repeated = counts[counts > 1]
    return rows, {
        "identity": "complete stored embedding row bytes",
        "input_rows": int(len(keys)),
        "representative_rows": int(len(rows)),
        "excluded_exact_duplicate_rows": int(len(keys) - len(rows)),
        "nontrivial_family_count": int(len(repeated)),
        "maximum_family_size": int(repeated.max()) if len(repeated) else 1,
        "representative_row_ids_sha256": ordered_array_sha256(rows),
    }


def fixed_single_array_split(
    values: np.ndarray,
    *,
    name: str,
    maximum_total_rows: int = MAX_PROBE_TOTAL_ROWS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Canonicalize, cap, then make a deterministic disjoint query split."""
    representatives, duplicate_control = canonical_representatives(values)
    if len(representatives) > maximum_total_rows:
        rng = np.random.RandomState(stable_seed(name, "representative-cap"))
        representatives = np.sort(
            rng.choice(representatives, maximum_total_rows, replace=False)
        ).astype(np.int64)
    total = int(len(representatives))
    query_count = min(
        MAX_QUERY_ROWS,
        max(MIN_QUERY_ROWS, int(round(QUERY_FRACTION * total))),
    )
    if total <= query_count + 50:
        raise Round0142Error(f"{name} has too few canonical rows")
    rng = np.random.RandomState(stable_seed(name, "query-split"))
    positions = np.sort(rng.choice(total, query_count, replace=False))
    query = representatives[positions]
    keep = np.ones(total, dtype=bool)
    keep[positions] = False
    corpus = representatives[keep]
    return corpus, query, {
        "policy": (
            "first exact-byte representative, deterministic cap, then "
            "deterministic disjoint query selection"
        ),
        "seed": stable_seed(name, "query-split"),
        "corpus_rows": int(len(corpus)),
        "query_rows": int(len(query)),
        "corpus_row_ids_sha256": ordered_array_sha256(corpus),
        "query_row_ids_sha256": ordered_array_sha256(query),
        "exact_family_disjoint_by_construction": True,
        "duplicate_control": duplicate_control,
    }


def fixed_separate_split(
    corpus: np.ndarray,
    queries: np.ndarray,
    *,
    name: str,
    maximum_corpus_rows: int = MAX_PROBE_TOTAL_ROWS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Canonicalize registered corpus/query arrays and remove cross copies."""
    corpus_reps, corpus_duplicates = canonical_representatives(corpus)
    query_reps, query_duplicates = canonical_representatives(queries)
    corpus_keys = _row_keys(np.asarray(corpus)[corpus_reps])
    query_keys = _row_keys(np.asarray(queries)[query_reps])
    overlap = np.isin(corpus_keys, query_keys)
    corpus_reps = corpus_reps[~overlap]
    if len(corpus_reps) > maximum_corpus_rows:
        rng = np.random.RandomState(stable_seed(name, "corpus-cap"))
        corpus_reps = np.sort(
            rng.choice(corpus_reps, maximum_corpus_rows, replace=False)
        ).astype(np.int64)
    if len(corpus_reps) < 50 or len(query_reps) < 10:
        raise Round0142Error(f"{name} has too few canonical corpus/query rows")
    return corpus_reps, query_reps, {
        "policy": (
            "first exact-byte representative independently per side; drop "
            "corpus families present in queries; deterministic corpus cap"
        ),
        "corpus_rows": int(len(corpus_reps)),
        "query_rows": int(len(query_reps)),
        "cross_split_families_removed": int(overlap.sum()),
        "corpus_row_ids_sha256": ordered_array_sha256(corpus_reps),
        "query_row_ids_sha256": ordered_array_sha256(query_reps),
        "exact_family_disjoint_by_construction": True,
        "corpus_duplicate_control": corpus_duplicates,
        "query_duplicate_control": query_duplicates,
    }


def shape_matched_control_split(
    values: np.ndarray,
    *,
    name: str,
    corpus_rows: int,
    query_rows: int,
    representatives: np.ndarray | None = None,
    duplicate_control: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if representatives is None:
        representatives, observed_duplicates = canonical_representatives(values)
        duplicate_control = observed_duplicates
    else:
        representatives = np.asarray(representatives, dtype=np.int64)
        if duplicate_control is None:
            raise Round0142Error(
                "precomputed control representatives require duplicate accounting"
            )
    total = int(corpus_rows + query_rows)
    if total > len(representatives):
        raise Round0142Error(
            f"control cannot match {name}: {total} > {len(representatives)}"
        )
    rng = np.random.RandomState(stable_seed(name, "fineweb-control"))
    chosen = np.sort(rng.choice(representatives, total, replace=False))
    query_positions = np.sort(rng.choice(total, query_rows, replace=False))
    query = chosen[query_positions]
    keep = np.ones(total, dtype=bool)
    keep[query_positions] = False
    corpus = chosen[keep]
    return corpus, query, {
        "policy": "shape-matched canonical FineWeb-heldout control",
        "seed": stable_seed(name, "fineweb-control"),
        "corpus_rows": int(len(corpus)),
        "query_rows": int(len(query)),
        "corpus_row_ids_sha256": ordered_array_sha256(corpus),
        "query_row_ids_sha256": ordered_array_sha256(query),
        "exact_family_disjoint_by_construction": True,
        "duplicate_control": dict(duplicate_control),
    }


def retention_verdict(retention: float) -> str:
    if not np.isfinite(retention) or retention < 0:
        raise Round0142Error("retention must be finite and nonnegative")
    if retention >= PASS_RETENTION:
        return "pass"
    if retention < FAIL_RETENTION:
        return "named-failure"
    return "amber"
