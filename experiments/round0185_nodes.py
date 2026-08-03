"""Build a CPU-only, exact-training-disjoint view over the R0173 probe bytes."""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0108_evaluation import HELDOUT_CORPUS_ROWS, HELDOUT_QUERY_ROWS
from basemap.round0169_prompted_diverse import DIMENSION
from basemap.round0185_prompted_ood_disjoint_pack import (
    CAPABILITY,
    EXPECTED_REMOVALS,
    LANGUAGE_PROBE_SCHEMA,
    PACK_SCHEMA,
    RETAINED_PROBE_ROWS,
    ROUND_ID,
    Round0185Error,
    SOURCE_AUDIT_SCHEMA,
    SOURCE_PROBE_ROWS,
    TRAINING_ROWS,
)
from experiments.round0169_nodes import LANGUAGES, _fingerprints


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    observed = expected_input_signature(str(expected.get("canonical_path") or ""))
    if observed != dict(expected):
        raise Round0185Error(f"{label} bytes changed")
    return observed


def _read_source_audit(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = _signature(job["source_audit"], label="accepted failed R0173 audit")
    audit = prompt_contract.read_sealed(
        signature["canonical_path"], label="accepted failed R0173 audit"
    )
    observed = {
        (
            str(item.get("language") or ""),
            str(item.get("split") or ""),
            int(item.get("source_row", -1)),
            int(item.get("training_compact_row", -1)),
        )
        for item in audit.get("exact_training_family_overlaps") or []
        if isinstance(item, Mapping)
    }
    if (
        audit.get("schema") != SOURCE_AUDIT_SCHEMA
        or audit.get("round_id") != "0173"
        or audit.get("passed") is not False
        or audit.get("capabilities") != []
        or int(audit.get("probe_rows", -1)) != SOURCE_PROBE_ROWS
        or int(audit.get("training_rows", -1)) != TRAINING_ROWS
        or int(audit.get("exact_training_family_overlap_count", -1))
        != len(EXPECTED_REMOVALS)
        or observed != set(EXPECTED_REMOVALS)
        or set(audit.get("language_outputs") or {}) != set(LANGUAGES)
    ):
        raise Round0185Error("accepted R0173 failed audit changed")
    return audit, signature


def _read_training(job: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_signature = _signature(
        job["staging_manifest"], label="accepted R0168 staging manifest"
    )
    manifest = prompt_contract.read_sealed(
        manifest_signature["canonical_path"], label="accepted R0168 staging manifest"
    )
    source_signature = _signature(
        manifest.get("host_fp16") or {}, label="accepted R0168 prompted U12 matrix"
    )
    if (
        manifest.get("round_id") != "0168"
        or manifest.get("schema") != "round0168-prompted-diverse-u12-staging-v1"
        or manifest.get("embedding_convention") != "Document: "
        or int(manifest.get("rows", -1)) != TRAINING_ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("dtype") != "<f2"
        or manifest.get("training_performed") is not False
    ):
        raise Round0185Error("accepted R0168 staging contract changed")
    source = np.load(
        source_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        source.shape != (TRAINING_ROWS, DIMENSION)
        or source.dtype != np.float16
        or not source.flags.c_contiguous
    ):
        raise Round0185Error("accepted R0168 prompted matrix geometry changed")
    return source, manifest_signature


def _read_language(
    audit: Mapping[str, Any], language: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    registered = (audit.get("language_outputs") or {}).get(language)
    if not isinstance(registered, Mapping):
        raise Round0185Error(f"R0173 {language} output binding is missing")
    receipt_signature = _signature(
        registered.get("receipt") or {}, label=f"R0173 {language} receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_signature["canonical_path"], label=f"R0173 {language} receipt"
    )
    signatures = {
        key: _signature(
            registered.get(key) or {}, label=f"R0173 {language} {key}"
        )
        for key in (
            "corpus_embeddings",
            "query_embeddings",
            "corpus_source_rows",
            "query_source_rows",
        )
    }
    if (
        receipt.get("schema") != LANGUAGE_PROBE_SCHEMA
        or receipt.get("round_id") != "0173"
        or receipt.get("language") != language
        or receipt.get("prompt_applied") is not True
        or receipt.get("prompt_prefix") != "Document: "
        or any(receipt.get(key) != value for key, value in signatures.items())
    ):
        raise Round0185Error(f"R0173 {language} receipt changed")
    corpus = np.load(
        signatures["corpus_embeddings"]["canonical_path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    queries = np.load(
        signatures["query_embeddings"]["canonical_path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    corpus_rows = np.load(
        signatures["corpus_source_rows"]["canonical_path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    query_rows = np.load(
        signatures["query_source_rows"]["canonical_path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        corpus.shape != (HELDOUT_CORPUS_ROWS, DIMENSION)
        or queries.shape != (HELDOUT_QUERY_ROWS, DIMENSION)
        or corpus.dtype != np.float16
        or queries.dtype != np.float16
        or corpus_rows.shape != (HELDOUT_CORPUS_ROWS,)
        or query_rows.shape != (HELDOUT_QUERY_ROWS,)
        or corpus_rows.dtype != np.int64
        or query_rows.dtype != np.int64
    ):
        raise Round0185Error(f"R0173 {language} probe geometry changed")
    return corpus, queries, corpus_rows, query_rows, {
        "receipt": receipt_signature,
        **signatures,
    }


def run_filter_and_audit(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Remove registered overlap families, then rescan the complete retained pack."""
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0185Error("R0185 handler received another queue")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0185 disjoint prompted OOD pack"
    )
    started = time.monotonic()
    source_audit, source_audit_signature = _read_source_audit(job)
    training, staging_signature = _read_training(job)
    removals_by_language: dict[str, list[tuple[str, int, int]]] = {
        language: [] for language in LANGUAGES
    }
    for language, split, source_row, training_row in EXPECTED_REMOVALS:
        removals_by_language[language].append((split, source_row, training_row))

    pair_dtype = np.dtype([("h0", "<u8"), ("h1", "<u8")])
    retained_pairs = np.empty(RETAINED_PROBE_ROWS, dtype=pair_dtype)
    entries: list[dict[str, Any]] = []
    languages: dict[str, Any] = {}
    removed: list[dict[str, Any]] = []
    cursor = 0
    zero_rows = 0
    nonfinite_rows = 0
    for language in LANGUAGES:
        corpus, queries, corpus_rows, query_rows, signatures = _read_language(
            source_audit, language
        )
        corpus_keep = np.ones(len(corpus), dtype=bool)
        query_keep = np.ones(len(queries), dtype=bool)
        for split, source_row, training_row in removals_by_language[language]:
            values, rows, keep = (
                (corpus, corpus_rows, corpus_keep)
                if split == "corpus"
                else (queries, query_rows, query_keep)
            )
            positions = np.flatnonzero(np.asarray(rows) == source_row)
            if len(positions) != 1 or not keep[int(positions[0])]:
                raise Round0185Error(
                    f"registered removal {language}/{split}/{source_row} is not unique"
                )
            position = int(positions[0])
            if np.asarray(values[position]).tobytes(order="C") != np.asarray(
                training[training_row]
            ).tobytes(order="C"):
                raise Round0185Error("registered R0173 overlap no longer matches training")
            keep[position] = False
            removed.append({
                "language": language,
                "split": split,
                "source_row": source_row,
                "source_position": position,
                "training_compact_row": training_row,
                "complete_stored_fp16_bytes_equal": True,
            })
        corpus_positions = np.flatnonzero(corpus_keep).astype(np.int64)
        query_positions = np.flatnonzero(query_keep).astype(np.int64)
        corpus_positions_path = os.path.join(
            output, f"{language}-corpus-retained-positions.i64.npy"
        )
        query_positions_path = os.path.join(
            output, f"{language}-query-retained-positions.i64.npy"
        )
        atomic_save_new_npy(corpus_positions_path, corpus_positions, immutable=True)
        atomic_save_new_npy(query_positions_path, query_positions, immutable=True)
        languages[language] = {
            "source": signatures,
            "source_corpus_rows": int(len(corpus)),
            "source_query_rows": int(len(queries)),
            "retained_corpus_rows": int(len(corpus_positions)),
            "retained_query_rows": int(len(query_positions)),
            "corpus_retained_positions": expected_input_signature(
                corpus_positions_path
            ),
            "query_retained_positions": expected_input_signature(query_positions_path),
            "removed_source_rows": [
                item[1] for item in removals_by_language[language]
            ],
        }
        for split, values, source_rows, positions in (
            ("corpus", corpus, corpus_rows, corpus_positions),
            ("queries", queries, query_rows, query_positions),
        ):
            retained = np.asarray(values[positions])
            h0, h1, zero, nonfinite = _fingerprints(retained)
            stop = cursor + len(retained)
            retained_pairs["h0"][cursor:stop] = h0
            retained_pairs["h1"][cursor:stop] = h1
            entries.append({
                "language": language,
                "split": split,
                "values": values,
                "source_rows": source_rows,
                "positions": positions,
            })
            cursor = stop
            zero_rows += int(zero.sum())
            nonfinite_rows += int(nonfinite.sum())
    if (
        cursor != RETAINED_PROBE_ROWS
        or len(removed) != len(EXPECTED_REMOVALS)
        or zero_rows
        or nonfinite_rows
    ):
        raise Round0185Error("R0185 retained probe population is invalid")

    unique_pairs = np.unique(retained_pairs)
    fingerprint_candidates: dict[tuple[int, int], list[tuple[int, bytes]]] = {}
    block_rows = 65_536
    for start in range(0, TRAINING_ROWS, block_rows):
        stop = min(start + block_rows, TRAINING_ROWS)
        block = np.asarray(training[start:stop])
        h0, h1, zero, nonfinite = _fingerprints(block)
        if np.any(zero) or np.any(nonfinite):
            raise Round0185Error("R0185 training source contains invalid rows")
        block_pairs = np.empty(len(block), dtype=pair_dtype)
        block_pairs["h0"] = h0
        block_pairs["h1"] = h1
        positions = np.searchsorted(unique_pairs, block_pairs)
        in_range = positions < len(unique_pairs)
        hits = np.zeros(len(block), dtype=bool)
        if np.any(in_range):
            hits[in_range] = unique_pairs[positions[in_range]] == block_pairs[in_range]
        for local in np.flatnonzero(hits).tolist():
            key = (int(h0[local]), int(h1[local]))
            fingerprint_candidates.setdefault(key, []).append(
                (start + local, np.asarray(block[local]).tobytes(order="C"))
            )
        if sum(len(value) for value in fingerprint_candidates.values()) > 100_000:
            raise Round0185Error("R0185 fingerprint candidate count is implausible")

    exact_overlaps: list[dict[str, Any]] = []
    fingerprint_hits = set(fingerprint_candidates)
    if fingerprint_hits:
        for entry in entries:
            values = entry["values"]
            rows = entry["source_rows"]
            positions = entry["positions"]
            for start in range(0, len(positions), block_rows):
                stop = min(start + block_rows, len(positions))
                retained_positions = positions[start:stop]
                block = np.asarray(values[retained_positions])
                h0, h1, _zero, _nonfinite = _fingerprints(block)
                for local in range(len(block)):
                    key = (int(h0[local]), int(h1[local]))
                    if key not in fingerprint_hits:
                        continue
                    raw = np.asarray(block[local]).tobytes(order="C")
                    for training_row, training_raw in fingerprint_candidates[key]:
                        if raw == training_raw:
                            position = int(retained_positions[local])
                            exact_overlaps.append({
                                "language": entry["language"],
                                "split": entry["split"],
                                "source_row": int(rows[position]),
                                "source_position": position,
                                "training_compact_row": int(training_row),
                            })

    receipt = prompt_contract.seal({
        "schema": PACK_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY] if not exact_overlaps else [],
        "source_failed_audit": source_audit_signature,
        "staging_manifest": staging_signature,
        "identity": (
            "R0173 complete stored prompted-fp16 rows plus ordered retained-position "
            "views; exact full-row confirmation against R0168 prompted U12 training"
        ),
        "source_probe_rows": SOURCE_PROBE_ROWS,
        "retained_probe_rows": RETAINED_PROBE_ROWS,
        "removed_probe_rows": len(removed),
        "removed_exact_training_families": removed,
        "unique_retained_fingerprints": int(len(unique_pairs)),
        "duplicate_retained_probe_rows": int(RETAINED_PROBE_ROWS - len(unique_pairs)),
        "training_rows_scanned": TRAINING_ROWS,
        "fingerprint_candidate_training_rows": int(
            sum(len(value) for value in fingerprint_candidates.values())
        ),
        "fingerprint_collision_candidates": int(
            sum(len(value) for value in fingerprint_candidates.values())
            - len({item["training_compact_row"] for item in exact_overlaps})
        ),
        "exact_retained_training_family_overlaps": exact_overlaps,
        "exact_retained_training_family_overlap_count": len(exact_overlaps),
        "language_outputs": languages,
        "queries_unchanged": all(
            cell["source_query_rows"] == cell["retained_query_rows"]
            for cell in languages.values()
        ),
        "passed": not exact_overlaps,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "pack.json"), receipt, immutable=True)
    if exact_overlaps:
        raise Round0185Error(
            f"R0185 retained pack still has {len(exact_overlaps)} training overlaps"
        )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if job.get("action") != "filter_and_audit_prompted_ood_pack":
        raise Round0185Error(f"R0185 does not authorize action {job.get('action')!r}")
    run_filter_and_audit(active, job)


__all__ = ["run_job"]
