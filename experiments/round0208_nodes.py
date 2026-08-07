"""Execute the R0208 prompted multilingual OOD reserve repair (CPU only)."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0087_inventory import _fingerprint_fp16
from basemap.round0108_evaluation import IN_MIX_LANGUAGES, POLISH
from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0208_prompted_ood_repair import (
    CAPABILITY,
    DIMENSION,
    HELD_OUT_LANGUAGE,
    PACK_SCHEMA,
    RETAINED_CORPUS_ROWS,
    RETAINED_QUERY_ROWS,
    ROUND_ID,
    Round0208Error,
    SOURCE_CORPUS_ROWS,
    SOURCE_PROBE_SCHEMA,
    SOURCE_QUERY_ROWS,
    SOURCE_ROUND_ID,
    STAGING_SCHEMA,
    TRAINING_ROWS,
    repair_plan,
    validate_census,
)


LANGUAGES = (*IN_MIX_LANGUAGES, POLISH)
SPLITS = ("corpus", "queries")
PAIR_DTYPE = np.dtype([("h0", "<u8"), ("h1", "<u8")])
BLOCK_ROWS = 65_536


def _signature(value: Any, *, label: str) -> dict[str, Any]:
    try:
        return dict(
            expected_input_signature(prompt_contract.verify_signature(value, label=label))
        )
    except Round0208Error:
        raise
    except Exception as error:  # noqa: BLE001 - re-raised as the round's error
        raise Round0208Error(f"{label} is unavailable or changed") from error


def _path_signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:  # noqa: BLE001
        raise Round0208Error(f"{label} is unavailable or changed") from error


def _fingerprints(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(values)
    h0 = np.empty(rows, dtype=np.uint64)
    h1 = np.empty(rows, dtype=np.uint64)
    zero = np.empty(rows, dtype=bool)
    nonfinite = np.empty(rows, dtype=bool)
    bits = np.ascontiguousarray(values).view("<u2")
    _fingerprint_fp16(bits, h0, h1, zero, nonfinite)
    return h0, h1, zero, nonfinite


def _load_probe(root: str, language: str) -> dict[str, Any]:
    """Open one immutable R0173 language probe by its sealed receipt."""
    receipt_path = os.path.join(root, "receipt.json")
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0173 {language} prompted probe"
    )
    if (
        receipt.get("schema") != SOURCE_PROBE_SCHEMA
        or receipt.get("round_id") != SOURCE_ROUND_ID
        or receipt.get("language") != language
        or receipt.get("prompt_applied") is not True
        or receipt.get("prompt_prefix") != "Document: "
    ):
        raise Round0208Error(f"R0173 {language} prompted probe changed")
    signatures = {
        key: _signature(receipt[key], label=f"R0173 {language} {key}")
        for key in (
            "corpus_embeddings",
            "query_embeddings",
            "corpus_source_rows",
            "query_source_rows",
        )
    }
    arrays = {
        key: np.load(
            signatures[key]["canonical_path"], mmap_mode="r", allow_pickle=False
        )
        for key in signatures
    }
    if (
        arrays["corpus_embeddings"].shape != (SOURCE_CORPUS_ROWS, DIMENSION)
        or arrays["query_embeddings"].shape != (SOURCE_QUERY_ROWS, DIMENSION)
        or arrays["corpus_embeddings"].dtype != np.float16
        or arrays["query_embeddings"].dtype != np.float16
        or arrays["corpus_source_rows"].shape != (SOURCE_CORPUS_ROWS,)
        or arrays["query_source_rows"].shape != (SOURCE_QUERY_ROWS,)
        or arrays["corpus_source_rows"].dtype != np.int64
        or arrays["query_source_rows"].dtype != np.int64
    ):
        raise Round0208Error(f"R0173 {language} prompted probe geometry changed")
    return {
        "receipt": _path_signature(
            receipt_path, label=f"R0173 {language} probe receipt"
        ),
        "signatures": signatures,
        "corpus": arrays["corpus_embeddings"],
        "queries": arrays["query_embeddings"],
        "corpus_source_rows": np.asarray(arrays["corpus_source_rows"]),
        "query_source_rows": np.asarray(arrays["query_source_rows"]),
    }


def _training_source_rows(job: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Resolve the U12 compact rows to per-language source dataset rows."""
    inventory_signature = _signature(
        job["r0087_inventory"], label="accepted R0087 diverse inventory"
    )
    with open(inventory_signature["canonical_path"], encoding="utf-8") as handle:
        inventory = json.load(handle)
    ranges = ((inventory.get("selection") or {}).get("ranges")) or []
    if not ranges:
        raise Round0208Error("R0087 inventory ranges are missing")
    mapping_signature = _signature(
        job["r0132_mapping"], label="accepted R0132 compact-to-global mapping"
    )
    global_rows = np.load(
        mapping_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if global_rows.shape != (TRAINING_ROWS,) or global_rows.dtype != np.int64:
        raise Round0208Error("R0132 compact-to-global mapping changed")
    global_rows = np.asarray(global_rows)
    resolved: dict[str, np.ndarray] = {}
    for language in LANGUAGES:
        dataset = f"fineweb2-{language}-chunked-500-jina-v5-nano"
        pieces = []
        for span in ranges:
            if span.get("dataset") != dataset:
                continue
            low = int(span["global_row_start"])
            high = int(span["global_row_stop"])
            selected = global_rows[(global_rows >= low) & (global_rows < high)]
            pieces.append(selected - low + int(span["dataset_row_start"]))
        resolved[language] = (
            np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
        )
    if resolved[HELD_OUT_LANGUAGE].size:
        raise Round0208Error(
            "R0208 found held-out Polish rows inside the U12 training population"
        )
    return {
        "per_language": resolved,
        "inventory": inventory_signature,
        "mapping": mapping_signature,
    }


def run_repair(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0208Error("R0208 repair handler received another queue")
    started = time.monotonic()
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0208 prompted OOD probe pack v2"
    )

    staging_signature = _signature(
        job["staging_manifest"], label="accepted R0168 U12 staging manifest"
    )
    staging = prompt_contract.read_sealed(
        staging_signature["canonical_path"], label="accepted R0168 U12 staging manifest"
    )
    if (
        staging.get("schema") != STAGING_SCHEMA
        or staging.get("round_id") != "0168"
        or int(staging.get("rows", -1)) != TRAINING_ROWS
        or (staging.get("population") or {}).get("polish_held_out") is not True
        or staging.get("embedding_convention") != "Document: "
    ):
        raise Round0208Error("R0208 accepted U12 staging contract changed")
    population_signature = _signature(
        staging["host_fp16"], label="accepted R0168 U12 host fp16"
    )
    source = np.load(
        population_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if source.shape != (TRAINING_ROWS, DIMENSION) or source.dtype != np.float16:
        raise Round0208Error("R0208 U12 population geometry changed")

    # ---- load every immutable R0173 language probe -------------------------
    probes = {
        language: _load_probe(
            os.path.join(str(job["source_pack_root"]), f"prompted-{language}"), language
        )
        for language in LANGUAGES
    }

    # ---- identity 1+2: stored fp16 row bytes, in-pack and against training --
    total_probe_rows = len(LANGUAGES) * (SOURCE_CORPUS_ROWS + SOURCE_QUERY_ROWS)
    probe_pairs = np.empty(total_probe_rows, dtype=PAIR_DTYPE)
    families: dict[bytes, list[tuple[str, str, int]]] = defaultdict(list)
    cursor = 0
    split_order: list[tuple[str, str]] = []
    for language in LANGUAGES:
        for split in SPLITS:
            values = np.asarray(probes[language][split])
            h0, h1, zero, nonfinite = _fingerprints(values)
            if bool(zero.any()) or bool(nonfinite.any()):
                raise Round0208Error(
                    f"R0173 {language} {split} contains zero or nonfinite rows"
                )
            stop = cursor + len(values)
            probe_pairs["h0"][cursor:stop] = h0
            probe_pairs["h1"][cursor:stop] = h1
            cursor = stop
            split_order.append((language, split))
            for ordinal in range(len(values)):
                families[values[ordinal].tobytes(order="C")].append(
                    (language, split, ordinal)
                )
    if cursor != total_probe_rows:
        raise Round0208Error("R0208 probe fingerprint population is invalid")
    unique_pairs = np.unique(probe_pairs)

    repeated = {key: members for key, members in families.items() if len(members) > 1}
    within_pack_members = sorted(sorted(members) for members in repeated.values())
    cross_split = sum(
        1 for members in repeated.values() if len({item[1] for item in members}) > 1
    )
    cross_language = sum(
        1 for members in repeated.values() if len({item[0] for item in members}) > 1
    )
    within_pack_maximum = max((len(members) for members in repeated.values()), default=0)

    candidates: dict[tuple[int, int], list[tuple[int, bytes]]] = {}
    for start in range(0, TRAINING_ROWS, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, TRAINING_ROWS)
        block = np.asarray(source[start:stop])
        h0, h1, zero, nonfinite = _fingerprints(block)
        if bool(zero.any()) or bool(nonfinite.any()):
            raise Round0208Error("R0208 U12 training source became invalid")
        block_pairs = np.empty(len(block), dtype=PAIR_DTYPE)
        block_pairs["h0"] = h0
        block_pairs["h1"] = h1
        positions = np.searchsorted(unique_pairs, block_pairs)
        in_range = positions < len(unique_pairs)
        hits = np.zeros(len(block), dtype=bool)
        if bool(np.any(in_range)):
            hits[in_range] = unique_pairs[positions[in_range]] == block_pairs[in_range]
        for local in np.flatnonzero(hits).tolist():
            key = (int(h0[local]), int(h1[local]))
            candidates.setdefault(key, []).append(
                (start + local, block[local].tobytes(order="C"))
            )
        if sum(len(value) for value in candidates.values()) > 100_000:
            raise Round0208Error("R0208 training fingerprint candidate count is implausible")

    hit_keys = set(candidates)
    overlaps: list[dict[str, Any]] = []
    if hit_keys:
        for language, split in split_order:
            values = np.asarray(probes[language][split])
            source_rows = probes[language][f"{'corpus' if split == 'corpus' else 'query'}_source_rows"]
            h0, h1, _zero, _nonfinite = _fingerprints(values)
            for ordinal in range(len(values)):
                key = (int(h0[ordinal]), int(h1[ordinal]))
                if key not in hit_keys:
                    continue
                raw = values[ordinal].tobytes(order="C")
                for training_row, training_raw in candidates[key]:
                    if raw == training_raw:
                        overlaps.append({
                            "language": language,
                            "split": split,
                            "ordinal": int(ordinal),
                            "source_row": int(source_rows[ordinal]),
                            "training_compact_row": int(training_row),
                        })

    # ---- identity 3: exact source-row membership ---------------------------
    training_rows = _training_source_rows(job)
    per_language_training = training_rows["per_language"]
    source_row_overlaps: list[dict[str, Any]] = []
    source_row_diagnostics: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        training = per_language_training[language]
        diagnostic: dict[str, Any] = {
            "training_rows_in_population": int(training.size),
            "training_source_row_maximum": int(training.max()) if training.size else None,
        }
        for split, key in (("corpus", "corpus_source_rows"), ("queries", "query_source_rows")):
            probe_rows = probes[language][key]
            shared = np.intersect1d(probe_rows, training)
            diagnostic[f"{split}_source_row_minimum"] = int(probe_rows.min())
            diagnostic[f"{split}_source_row_training_overlap"] = int(shared.size)
            for value in shared.tolist():
                source_row_overlaps.append({
                    "language": language,
                    "split": split,
                    "source_row": int(value),
                })
        source_row_diagnostics[language] = diagnostic

    census = {
        "training_rows": TRAINING_ROWS,
        "probe_rows": total_probe_rows,
        "unique_probe_fingerprints": int(len(unique_pairs)),
        "duplicate_probe_rows": int(total_probe_rows - len(unique_pairs)),
        "fingerprint_candidate_training_rows": int(
            sum(len(value) for value in candidates.values())
        ),
        "exact_training_family_overlaps": overlaps,
        "exact_training_family_overlap_count": len(overlaps),
        "within_pack_exact_families": len(repeated),
        "within_pack_duplicate_rows": sum(len(members) - 1 for members in repeated.values()),
        "within_pack_maximum_family": within_pack_maximum,
        "within_pack_cross_split_families": cross_split,
        "within_pack_cross_language_families": cross_language,
        "within_pack_family_members": within_pack_members,
        "source_row_identity_overlaps": len(source_row_overlaps),
        "source_row_identity_overlap_rows": source_row_overlaps,
        "source_row_diagnostics": source_row_diagnostics,
    }
    validate_census(census)

    # ---- the removal-only repair ------------------------------------------
    excluded: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item in overlaps:
        excluded[(item["language"], item["split"])].add(int(item["ordinal"]))
    order = {pair: index for index, pair in enumerate(split_order)}
    for members in repeated.values():
        ranked = sorted(members, key=lambda item: (order[(item[0], item[1])], item[2]))
        for language, split, ordinal in ranked[1:]:
            excluded[(language, split)].add(int(ordinal))

    languages_block: dict[str, dict[str, Any]] = {}
    retained_total = 0
    for language in LANGUAGES:
        entry: dict[str, Any] = {
            "receipt": probes[language]["receipt"],
            "source_arrays": probes[language]["signatures"],
            "exclusions": {},
            "retained": {},
        }
        for split, rows_key in (("corpus", "corpus_source_rows"), ("queries", "query_source_rows")):
            removed = sorted(excluded.get((language, split), set()))
            retained = repair_plan(
                language=language,
                split=split,
                excluded_ordinals=removed,
                source_rows=SOURCE_CORPUS_ROWS if split == "corpus" else SOURCE_QUERY_ROWS,
            )
            retained_array = np.asarray(retained, dtype=np.int64)
            source_rows = probes[language][rows_key][retained_array]
            prefix = "corpus" if split == "corpus" else "query"
            ordinal_path = atomic_save_new_npy(
                os.path.join(output, f"{language}-{prefix}-retained-ordinals.i64.npy"),
                retained_array,
                immutable=True,
            )
            rows_path = atomic_save_new_npy(
                os.path.join(output, f"{language}-{prefix}-retained-source-rows.i64.npy"),
                np.ascontiguousarray(source_rows, dtype=np.int64),
                immutable=True,
            )
            entry["exclusions"][split] = {
                "removed_ordinals": removed,
                "removed_rows": len(removed),
                "training_family_removals": sum(
                    1
                    for item in overlaps
                    if item["language"] == language and item["split"] == split
                ),
                "within_pack_repeat_removals": len(removed)
                - sum(
                    1
                    for item in overlaps
                    if item["language"] == language and item["split"] == split
                ),
                "equalization_tail_drops": (
                    (SOURCE_CORPUS_ROWS if split == "corpus" else SOURCE_QUERY_ROWS)
                    - len(removed)
                    - len(retained)
                ),
            }
            entry["retained"][split] = {
                "rows": len(retained),
                "ordinals": _path_signature(
                    ordinal_path, label=f"R0208 {language} {split} retained ordinals"
                ),
                "source_rows": _path_signature(
                    rows_path, label=f"R0208 {language} {split} retained source rows"
                ),
                "ordered_source_rows_sha256": ordered_array_sha256(
                    np.ascontiguousarray(source_rows, dtype=np.int64)
                ),
            }
            retained_total += len(retained)
        languages_block[language] = entry

    expected_total = len(LANGUAGES) * (RETAINED_CORPUS_ROWS + RETAINED_QUERY_ROWS)
    if retained_total != expected_total:
        raise Round0208Error("R0208 retained pack row count is not the registered shape")

    pack = prompt_contract.seal({
        "schema": PACK_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "capabilities": [CAPABILITY],
        "supersedes_capability": "jina-prompted-u12-ood-probe-pack-v1",
        "embedding_performed": False,
        "training_performed": False,
        "repair_policy": {
            "kind": "removal-only explicit exclusion, no reselection and no re-embedding",
            "review_0173_branch": "different explicit exclusion policy",
            "removed": [
                "exact stored prompted-fp16 training-family members",
                "later-ordinal within-pack exact stored-fp16 repeats",
                "tail rows required to equalize the per-language corpus shape",
            ],
            "queries_repaired": False,
            "query_ids_unchanged_from_r0173": True,
            "rows_reselected": 0,
            "rows_embedded": 0,
        },
        "shape": {
            "languages": list(LANGUAGES),
            "corpus_rows_per_language": RETAINED_CORPUS_ROWS,
            "query_rows_per_language": RETAINED_QUERY_ROWS,
            "pack_rows": retained_total,
            "source_corpus_rows_per_language": SOURCE_CORPUS_ROWS,
            "source_query_rows_per_language": SOURCE_QUERY_ROWS,
            "source_pack_rows": total_probe_rows,
            "held_out_language": HELD_OUT_LANGUAGE,
        },
        "audit": {
            **{key: value for key, value in census.items()},
            "identities": [
                "complete stored prompted-fp16 row bytes versus the R0168 U12 matrix",
                "complete stored prompted-fp16 row bytes within the pack",
                "exact per-language source-row membership of the U12 population",
            ],
            "passed_after_repair": True,
        },
        "languages": languages_block,
        "sources": {
            "u12_staging_manifest": staging_signature,
            "u12_host_fp16": population_signature,
            "r0132_compact_to_global": training_rows["mapping"],
            "r0087_inventory": training_rows["inventory"],
            "r0173_audit": _signature(job["r0173_audit"], label="R0173 failed audit"),
            "r0173_prompt_canary": _signature(
                job["r0173_canary"], label="R0173 prompt canary"
            ),
        },
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "probe-pack.json"), pack, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "repair_prompted_ood_pack":
        raise Round0208Error("R0208 authorizes only the prompted OOD pack repair")
    run_repair(active, job)


__all__ = ["run_job"]
