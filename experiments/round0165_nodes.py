"""Derive the prompted 8M frozen-prefix population without another census."""
from __future__ import annotations

import hashlib
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0113_prompt_contrast import seal, verify_signature
from basemap.round0164_prompted_population import SCHEMA as R0164_SCHEMA
from basemap.round0165_frozen_prefix_population import (
    CAPABILITY,
    DIMENSION,
    HOST_CAPABILITY,
    PREFIX_STOP,
    ROUND_ID,
    SCHEMA,
    Round0165Error,
    frozen_prefix_extension,
    population_identity,
)
from experiments.round0163_nodes import (
    _iter_texts,
    _read_sealed,
    _source_layouts,
    _validate_signature,
)


def _write_subset(
    path: str,
    *,
    source_path: str,
    source_rows: int,
    positions: np.ndarray,
) -> None:
    source = np.memmap(
        source_path, mode="r", dtype="<f2", shape=(source_rows, DIMENSION)
    )
    written = 0
    with open(path, "wb") as handle:
        for start in range(0, len(positions), 25_000):
            selected = positions[start:min(start + 25_000, len(positions))]
            block = np.asarray(source[selected], dtype=np.float16, order="C")
            block.tofile(handle)
            written += len(block)
        handle.flush()
        os.fsync(handle.fileno())
    if written != len(positions):
        raise Round0165Error("R0165 compact subset write did not close")


def _remove_hashes(sorted_hashes: np.ndarray, removed: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted_hashes)
    drops = np.asarray(removed, dtype="V32")
    positions = np.searchsorted(values, drops)
    if (
        np.any(positions >= len(values))
        or not np.array_equal(values[positions], drops)
        or len(np.unique(positions)) != len(positions)
    ):
        raise Round0165Error("R0165 dropped prefix hashes are not unique members")
    keep = np.ones(len(values), dtype=bool)
    keep[positions] = False
    return np.asarray(values[keep], dtype="V32")


def run_population(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0165Error("R0165 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0165Error("R0165 is CPU-only")
    started = time.monotonic()
    prior_signature = dict(job["r0164_population"])
    prior = _read_sealed(prior_signature, label="accepted R0164 population")
    if (
        prior.get("schema") != R0164_SCHEMA
        or prior.get("outcome") != "prompted-only-8m-population-not-qualified"
        or prior.get("q2_population_released") is not False
        or prior.get("capabilities") != []
        or int(prior.get("retained_rows", -1)) != 7_952_426
        or int((prior.get("accepted_r0113_prefix") or {}).get("observed_rows", -1))
        != 1_993_768
        or (prior.get("accepted_r0113_prefix") or {}).get("exact") is not False
        or int((prior.get("r0163_delta") or {}).get("added_rows", -1)) != 212
        or prior.get("raw_unprompted_relation_used") is not False
    ):
        raise Round0165Error("accepted R0164 negative premise changed")

    prompted_mapping = np.load(
        verify_signature(prior["mapping"], label="R0164 prompted-only mapping"),
        mmap_mode="r",
        allow_pickle=False,
    )
    accepted_signature = expected_input_signature(job["r0113_mapping"]["canonical_path"])
    if accepted_signature != dict(job["r0113_mapping"]):
        raise Round0165Error("accepted R0113 mapping changed")
    accepted_prefix = np.load(
        accepted_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    r0163_signature = expected_input_signature(job["r0163_mapping"]["canonical_path"])
    if r0163_signature != dict(job["r0163_mapping"]):
        raise Round0165Error("accepted R0163 mapping changed")
    r0163_mapping = np.load(
        r0163_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    mapping, excluded, dropped, added, positions, derivation = frozen_prefix_extension(
        accepted_prefix=accepted_prefix,
        prompted_only_mapping=prompted_mapping,
        prior_three_relation_mapping=r0163_mapping,
    )
    if (
        len(dropped) != 7
        or len(added) != 205
        or np.any(dropped >= PREFIX_STOP)
        or not np.array_equal(mapping[mapping < PREFIX_STOP], accepted_prefix)
    ):
        raise Round0165Error("R0165 registered seven-drop/205-add derivation changed")

    r0116 = _read_sealed(job["r0116_manifest"], label="accepted R0116 manifest")
    r0120 = _read_sealed(job["r0120_manifest"], label="accepted R0120 manifest")
    text_layout, _raw, _inputs = _source_layouts(r0116, r0120)
    touched_text_inputs = [
        dict(item["text"])
        for item in text_layout
        if int(np.searchsorted(dropped, int(item["canonical_stop"]), side="left"))
        > int(np.searchsorted(dropped, int(item["canonical_start"]), side="left"))
    ]
    observed_payloads = {
        (item["canonical_path"], int(item["bytes"]), item["sha256"])
        for item in touched_text_inputs
    }
    expected_payloads = {
        (item["canonical_path"], int(item["bytes"]), item["sha256"])
        for item in job["payload_inputs"]
    }
    if observed_payloads != expected_payloads:
        raise Round0165Error("R0165 dropped-row text payload set changed")
    for index, signature in enumerate(job["payload_inputs"]):
        _validate_signature(signature, label=f"R0165 text payload {index}")
    dropped_hashes = np.empty(len(dropped), dtype="V32")
    for index, (row, text) in enumerate(_iter_texts(text_layout, dropped)):
        if row != int(dropped[index]):
            raise Round0165Error("R0165 dropped prefix text order changed")
        dropped_hashes[index] = hashlib.sha256(text.encode("utf-8")).digest()
    old_hashes = np.load(
        verify_signature(
            prior["source_text_hash_index"], label="R0164 source-text hash index"
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    hashes = _remove_hashes(old_hashes, dropped_hashes)
    if (
        hashes.shape != (len(mapping),)
        or (len(hashes) > 1 and np.any(hashes[1:] == hashes[:-1]))
        or not np.array_equal(hashes, np.sort(hashes, kind="stable"))
    ):
        raise Round0165Error("R0165 retained source-text hash index changed")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0165 output")
    mapping_path = os.path.join(output, "compact-to-canonical.i64.npy")
    excluded_path = os.path.join(output, "excluded-canonical.i64.npy")
    dropped_path = os.path.join(output, "dropped-prompted-only-prefix.i64.npy")
    added_path = os.path.join(output, "added-over-r0163-extension.i64.npy")
    hashes_path = os.path.join(output, "source-text-sha256-sorted.v32.npy")
    atomic_save_new_npy(mapping_path, mapping, immutable=True)
    atomic_save_new_npy(excluded_path, excluded, immutable=True)
    atomic_save_new_npy(dropped_path, dropped, immutable=True)
    atomic_save_new_npy(added_path, added, immutable=True)
    atomic_save_new_npy(hashes_path, hashes, immutable=True)

    compact_source = verify_signature(
        prior["document_compact"], label="R0164 prompted-only compact matrix"
    )
    compact_path = os.path.join(output, "document-compact.f16")
    write_started = time.monotonic()
    atomic_build_new_file(
        compact_path,
        lambda temporary: _write_subset(
            temporary,
            source_path=compact_source,
            source_rows=len(prompted_mapping),
            positions=positions,
        ),
        immutable=True,
    )
    write_seconds = time.monotonic() - write_started
    compact_signature = expected_input_signature(compact_path)
    if compact_signature["bytes"] != len(mapping) * DIMENSION * 2:
        raise Round0165Error("R0165 compact byte count changed")

    receipt = seal({
        "schema": SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "outcome": "prompted-8m-frozen-prefix-population-qualified",
        "capabilities": [CAPABILITY, HOST_CAPABILITY],
        "q2_population_released": True,
        "population_identity": population_identity(mapping=mapping, excluded=excluded),
        "source_rows": 8_000_000,
        "retained_rows": len(mapping),
        "excluded_rows": len(excluded),
        "dimension": DIMENSION,
        "dtype": "<f2",
        "mapping": expected_input_signature(mapping_path),
        "excluded": expected_input_signature(excluded_path),
        "document_compact": compact_signature,
        "source_text_hash_index": expected_input_signature(hashes_path),
        "dropped_prompted_only_prefix": expected_input_signature(dropped_path),
        "added_over_r0163_extension": expected_input_signature(added_path),
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "derivation": derivation,
        "lineage": {
            "r0164_population": prior_signature,
            "r0164_mapping": prior["mapping"],
            "r0163_mapping": job["r0163_mapping"],
            "accepted_r0113_prefix": job["r0113_mapping"],
        },
        "proofs": {
            "prefix_byte_exact": True,
            "extension_equals_r0164_prompted_only_rows_at_or_above_2m": True,
            "mapping_is_r0164_subset": True,
            "mapping_is_strict_r0163_superset": True,
            "document_exact_family_uniqueness_inherited_by_subset": True,
            "source_text_exact_family_uniqueness_inherited_and_hash_checked": True,
            "raw_unprompted_relation_used_for_extension": False,
            "multiplicity_is_metadata": True,
        },
        "training_performed": False,
        "graph_built": False,
        "performance": {
            "compact_subset_write_seconds": write_seconds,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "frozen-prefix-population.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "derive_frozen_prefix_population":
        raise Round0165Error("unknown R0165 action")
    run_population(active, job)
