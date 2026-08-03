"""Execute the CPU-only prompted-population redecision for R0164."""
from __future__ import annotations

import gc
import os
import resource
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0162_prompted_english_staging import DIMENSION, DTYPE, VIEW_ROWS
from basemap.round0163_prompted_english_census import (
    embedding_text_relation,
    validate_seal,
)
from basemap.round0164_prompted_population import (
    CAPABILITY,
    FAMILY_SOURCES,
    HOST_CAPABILITY,
    ROUND_ID,
    SCHEMA,
    Round0164Error,
    population_identity,
    prompted_representatives,
    seal,
)
from experiments.round0163_nodes import (
    ChunkedFp16,
    _document_slices,
    _families_from_fp16,
    _read_sealed,
    _source_layouts,
    _text_families,
    _validate_signature,
)


def _mapping_contains(superset: np.ndarray, subset: np.ndarray) -> bool:
    positions = np.searchsorted(superset, subset)
    valid = positions < len(superset)
    return bool(np.all(valid) and np.array_equal(superset[positions], subset))


def _retained_members(mapping: np.ndarray, family: Sequence[int]) -> int:
    values = np.asarray(family, dtype=np.int64)
    positions = np.searchsorted(mapping, values)
    present = positions < len(mapping)
    if np.any(present):
        present[present] = mapping[positions[present]] == values[present]
    return int(present.sum())


def _write_compact_stream(
    path: str, document: ChunkedFp16, mapping: np.ndarray
) -> None:
    """Write sequentially so output pages do not double the resident source."""
    written = 0
    # atomic_build_new_file has already created this private temporary inode.
    with open(path, "wb") as handle:
        for item in document.slices:
            left = int(np.searchsorted(mapping, item.canonical_start, side="left"))
            right = int(np.searchsorted(mapping, item.canonical_stop, side="left"))
            if right <= left:
                continue
            local = mapping[left:right] - item.canonical_start + item.source_start
            values = document._open(item)
            block = np.asarray(values[local], dtype=np.float16, order="C")
            block.tofile(handle)
            written += len(block)
            del block, values
        handle.flush()
        os.fsync(handle.fileno())
    if written != len(mapping):
        raise Round0164Error("prompted compact stream did not close")


def run_population(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0164Error("R0164 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0164Error("R0164 is CPU-only")
    started = time.monotonic()

    prior = _read_sealed(job["r0163_census"], label="accepted R0163 census")
    if (
        prior.get("schema") != "round0163-prompted-english-8m-representatives-v1"
        or prior.get("outcome") != "prompted-english-8m-population-confound-detected"
        or prior.get("q2_population_released") is not False
        or prior.get("cross_source_text_embedding_collisions")
        != {"raw_fp16": 133, "document_fp16": 0}
        or prior.get("capabilities") != []
    ):
        raise Round0164Error("R0163 negative population premise changed")
    validate_seal(prior, label="accepted R0163 census")

    layout = _read_sealed(job["canonical_layout"], label="accepted R0162 layout")
    view = _read_sealed(job["first8m_view"], label="accepted R0162 first8m view")
    r0116 = _read_sealed(job["r0116_manifest"], label="accepted R0116 manifest")
    r0120 = _read_sealed(job["r0120_manifest"], label="accepted R0120 manifest")
    if (
        view.get("layout_identity") != layout.get("layout_identity")
        or int(view.get("rows", -1)) != VIEW_ROWS
        or view.get("dtype") != DTYPE
    ):
        raise Round0164Error("accepted first8m view changed")

    text_layout, _raw_slices, _source_inputs = _source_layouts(r0116, r0120)
    document_slices = _document_slices(view)
    expected_inputs = [
        *[dict(item["text"]) for item in text_layout],
        *[item.signature for item in document_slices],
    ]
    expected_keys = {
        (item["canonical_path"], item["bytes"], item["sha256"])
        for item in job["payload_inputs"]
    }
    observed_keys = {
        (item["canonical_path"], item["bytes"], item["sha256"])
        for item in expected_inputs
    }
    if expected_keys != observed_keys:
        raise Round0164Error("R0164 payload input set changed")
    for index, signature in enumerate(job["payload_inputs"]):
        _validate_signature(signature, label=f"R0164 payload input {index}")

    document = ChunkedFp16(document_slices, label="Document fp16")
    text_families, text_report, text_hashes = _text_families(text_layout)
    document_families, document_report = _families_from_fp16(document)
    families = {
        "source_text": text_families,
        "document_fp16": document_families,
    }
    mapping, excluded, union_report = prompted_representatives(families)
    document_relation = embedding_text_relation(document_families, text_families)

    old_signature = expected_input_signature(job["r0163_mapping"]["canonical_path"])
    if old_signature != dict(job["r0163_mapping"]):
        raise Round0164Error("accepted R0163 mapping changed")
    old_mapping = np.load(old_signature["canonical_path"], mmap_mode="r", allow_pickle=False)
    if old_mapping.dtype != np.int64 or not _mapping_contains(mapping, old_mapping):
        raise Round0164Error("dropping raw relation did not produce a mapping superset")
    added = np.setdiff1d(mapping, old_mapping, assume_unique=True)
    if len(added) == 0:
        raise Round0164Error("R0163 raw-only collision finding produced no row delta")

    accepted_signature = expected_input_signature(
        job["r0113_compact_mapping"]["canonical_path"]
    )
    if accepted_signature != dict(job["r0113_compact_mapping"]):
        raise Round0164Error("accepted R0113 mapping changed")
    accepted_prefix = np.load(
        accepted_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    prefix = mapping[mapping < 2_000_000]
    prefix_exact = bool(
        accepted_prefix.dtype == np.int64
        and accepted_prefix.shape == (1_993_761,)
        and np.array_equal(prefix, accepted_prefix)
    )

    audit = {
        source: {
            "families_before_selection": len(families[source]),
            "families_with_more_than_one_retained_member": sum(
                _retained_members(mapping, family) > 1
                for family in families[source]
            ),
        }
        for source in FAMILY_SOURCES
    }
    qualified = bool(
        prefix_exact
        and int(document_relation["cross_source_text_families"]) == 0
        and all(
            value["families_with_more_than_one_retained_member"] == 0
            for value in audit.values()
        )
    )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0164 output")
    mapping_path = os.path.join(output, "compact-to-canonical.i64.npy")
    excluded_path = os.path.join(output, "excluded-canonical.i64.npy")
    added_path = os.path.join(output, "added-after-raw-relation-drop.i64.npy")
    text_hash_path = os.path.join(output, "source-text-sha256-sorted.v32.npy")
    atomic_save_new_npy(mapping_path, mapping, immutable=True)
    atomic_save_new_npy(excluded_path, excluded, immutable=True)
    atomic_save_new_npy(added_path, added, immutable=True)
    retained_hashes = np.sort(text_hashes[mapping], kind="stable")
    if len(retained_hashes) > 1 and np.any(retained_hashes[1:] == retained_hashes[:-1]):
        raise Round0164Error("prompted-only selector leaves source-text duplicates")
    atomic_save_new_npy(text_hash_path, retained_hashes, immutable=True)

    # Drop the scan mappings before the streamed output pass. This keeps the
    # measured memory bound near one source representation rather than the
    # R0163 source+source+output triple residency.
    document._arrays.clear()
    gc.collect()
    compact_path = os.path.join(output, "document-compact.f16")
    compact_started = time.monotonic()
    atomic_build_new_file(
        compact_path,
        lambda temporary: _write_compact_stream(temporary, document, mapping),
        immutable=True,
    )
    compact_wall = time.monotonic() - compact_started
    compact_signature = expected_input_signature(compact_path)
    if compact_signature["bytes"] != len(mapping) * DIMENSION * 2:
        raise Round0164Error("prompted compact byte count changed")

    identity = population_identity(
        view_identity=str(view["identity_sha256"]),
        mapping=mapping,
        excluded=excluded,
    )
    receipt = seal({
        "schema": SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY, HOST_CAPABILITY] if qualified else [],
        "outcome": (
            "prompted-only-8m-population-qualified"
            if qualified
            else "prompted-only-8m-population-not-qualified"
        ),
        "q2_population_released": qualified,
        "population_identity": identity,
        "source_rows": VIEW_ROWS,
        "retained_rows": len(mapping),
        "excluded_rows": len(excluded),
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "mapping": expected_input_signature(mapping_path),
        "excluded": expected_input_signature(excluded_path),
        "added_after_raw_relation_drop": expected_input_signature(added_path),
        "document_compact": compact_signature,
        "source_text_hash_index": expected_input_signature(text_hash_path),
        "mapping_ordered_sha256": ordered_array_sha256(mapping),
        "excluded_ordered_sha256": ordered_array_sha256(excluded),
        "family_census": {
            "source_text": text_report,
            "document_fp16": document_report,
            "document_text_relation": document_relation,
            "union": union_report,
        },
        "retained_family_audit": audit,
        "accepted_r0113_prefix": {
            "mapping": job["r0113_compact_mapping"],
            "expected_rows": 1_993_761,
            "observed_rows": len(prefix),
            "exact": prefix_exact,
        },
        "r0163_delta": {
            "prior_mapping": job["r0163_mapping"],
            "prior_rows": len(old_mapping),
            "new_mapping_is_strict_superset": True,
            "added_rows": len(added),
            "added_ordered_sha256": ordered_array_sha256(added),
            "reason": "raw unprompted exact-family relation excluded",
        },
        "raw_unprompted_relation_used": False,
        "multiplicity_preserved_as_metadata": True,
        "training_performed": False,
        "graph_built": False,
        "performance": {
            "compact_write_seconds": compact_wall,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "prompted-population.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "decide_prompted_only_population":
        raise Round0164Error("unknown R0164 action")
    run_population(active, job)
