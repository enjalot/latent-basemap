"""Execute the training-disjoint, source-text-aware R0178 recovery."""
from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap import round0167_prompted_universality as contract_base
from basemap.round0108_evaluation import exact_reference_copy_mask
from basemap.round0116_prompted_corpus import validate_environment_freeze
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0178_prompted_universality import (
    CAPABILITY,
    CONTROL_ROWS,
    EXPECTED_CONTROL_DUPLICATE_TEXT_REJECTS,
    EXPECTED_CONTROL_ROWS_SCANNED,
    EXPECTED_CONTROL_SELECTION_SHA256,
    EXPECTED_CONTROL_TRAINING_TEXT_REJECTS,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    Round0178Error,
    exact_training_overlap_report,
)
from experiments import round0167_nodes as base
from experiments import round0176_nodes as audit_base
from experiments.round0116_nodes import (
    _load_document_model,
    _stored_array_guard,
)


def _configure() -> None:
    contract_bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "PROMPTED_MAP_ORDER": PROMPTED_MAP_ORDER,
        "Round0167Error": Round0178Error,
    }
    for name, value in contract_bindings.items():
        setattr(contract_base, name, value)
    node_bindings = {
        **contract_bindings,
        "CANARY_SCHEMA": "round0178-prompt-model-canary-v1",
        "PROBE_SCHEMA": "round0178-prompted-probe-embeddings-v1",
        "CONTROL_SCHEMA": "round0178-prompted-fineweb-control-v1",
        "MAP_PANEL_SCHEMA": (
            "round0178-prompted-universality-map-panel-v1"
        ),
    }
    for name, value in node_bindings.items():
        setattr(base, name, value)
    base.ALLOW_CROSS_SPLIT_FAMILIES = True
    base.DUPLICATE_SENSITIVITY = True
    audit_base.ROUND_ID = ROUND_ID
    audit_base.CAPABILITY = CAPABILITY
    audit_base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    audit_base.Round0176Error = Round0178Error
    audit_base.exact_training_overlap_report = exact_training_overlap_report


def _load_hash_index(
    signature: Mapping[str, Any], *, label: str
) -> tuple[np.ndarray, dict[str, Any]]:
    bound = base._signature(signature, label=label)
    values = np.load(
        bound["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        values.ndim != 1
        or values.dtype != np.dtype("V32")
        or len(values) == 0
        or not np.array_equal(values, np.sort(values, kind="stable"))
        or np.any(values[1:] == values[:-1])
    ):
        raise Round0178Error(f"{label} is not a sorted unique V32 index")
    return values, bound


def _hash_rows(texts: Sequence[str]) -> np.ndarray:
    values = np.empty(len(texts), dtype="V32")
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise Round0178Error("source text is not a string")
        values[index] = hashlib.sha256(text.encode("utf-8")).digest()
    return values


def _membership(sorted_values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if candidates.ndim != 1 or candidates.dtype != np.dtype("V32"):
        raise Round0178Error("source-text membership candidates changed")
    positions = np.searchsorted(sorted_values, candidates)
    in_range = positions < len(sorted_values)
    found = np.zeros(len(candidates), dtype=bool)
    if np.any(in_range):
        found[in_range] = (
            sorted_values[positions[in_range]] == candidates[in_range]
        )
    return found


def run_select_disjoint_control(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0178 FineWeb control selector"
    )
    started = time.monotonic()
    text_source = base._signature(
        job["text_source"], label="R0178 FineWeb control text"
    )
    training: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        str(label): _load_hash_index(
            signature, label=f"R0178 {label} training text index"
        )
        for label, signature in job["training_text_hashes"].items()
    }
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(text_source["canonical_path"])
    selected_rows: list[int] = []
    selected_hashes: list[bytes] = []
    selected_set: set[bytes] = set()
    rows_scanned = 0
    training_rejects = 0
    duplicate_rejects = 0
    per_training_rejects = {label: 0 for label in training}
    for batch in parquet.iter_batches(
        batch_size=65_536, columns=["chunk_text"]
    ):
        for text in batch.column(0).to_pylist():
            if not isinstance(text, str):
                raise Round0178Error("FineWeb control text is not a string")
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            key = np.void(digest)
            memberships: dict[str, bool] = {}
            for label, (index, _signature) in training.items():
                position = int(np.searchsorted(index, key))
                memberships[label] = bool(
                    position < len(index) and index[position] == key
                )
            row = rows_scanned
            rows_scanned += 1
            if any(memberships.values()):
                training_rejects += 1
                for label, present in memberships.items():
                    per_training_rejects[label] += int(present)
                continue
            if digest in selected_set:
                duplicate_rejects += 1
                continue
            selected_rows.append(row)
            selected_hashes.append(digest)
            selected_set.add(digest)
            if len(selected_rows) == CONTROL_ROWS:
                break
        if len(selected_rows) == CONTROL_ROWS:
            break
    rows = np.asarray(selected_rows, dtype=np.int64)
    hashes = np.asarray(selected_hashes, dtype="V32")
    raw_selection_sha256 = sha256_bytes(
        np.asarray(rows, dtype="<i8").tobytes(order="C")
    )
    if (
        rows.shape != (CONTROL_ROWS,)
        or hashes.shape != (CONTROL_ROWS,)
        or rows_scanned != EXPECTED_CONTROL_ROWS_SCANNED
        or training_rejects != EXPECTED_CONTROL_TRAINING_TEXT_REJECTS
        or duplicate_rejects != EXPECTED_CONTROL_DUPLICATE_TEXT_REJECTS
        or raw_selection_sha256 != EXPECTED_CONTROL_SELECTION_SHA256
        or len(np.unique(hashes)) != CONTROL_ROWS
    ):
        raise Round0178Error("R0178 frozen FineWeb control selector changed")
    for index, _signature in training.values():
        if np.any(_membership(index, hashes)):
            raise Round0178Error("R0178 selected control touches map training")
    rows_path = os.path.join(output, "selected-source-rows.i64.npy")
    hashes_path = os.path.join(output, "selected-text-sha256.v32.npy")
    atomic_save_new_npy(rows_path, rows, immutable=True)
    atomic_save_new_npy(hashes_path, hashes, immutable=True)
    receipt = base.seal({
        "schema": "round0178-fineweb-control-selector-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "policy": (
            "first 60000 source-text-unique rows in source order after "
            "rejecting SHA-256 text members of either map-training family"
        ),
        "identity": "complete source-text UTF-8 bytes via SHA-256 index",
        "text_source": text_source,
        "training_text_hashes": {
            label: signature for label, (_index, signature) in training.items()
        },
        "rows_scanned": rows_scanned,
        "rows_selected": int(len(rows)),
        "last_source_row": int(rows[-1]),
        "training_text_rejects": training_rejects,
        "per_training_text_rejects": per_training_rejects,
        "within_control_duplicate_text_rejects": duplicate_rejects,
        "selected_source_rows": expected_input_signature(rows_path),
        "selected_text_hashes": expected_input_signature(hashes_path),
        "selection_raw_i64_sha256": raw_selection_sha256,
        "training_text_disjoint": True,
        "source_text_unique": True,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "selector.json"), receipt, immutable=True
    )


def run_embed_disjoint_control(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0178 prompted FineWeb control"
    )
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    canary = base._read_sealed(
        os.path.join(str(job["canary_output"]), "canary.json"),
        label="R0178 canary",
    )
    if canary.get("passed") is not True:
        raise Round0178Error("R0178 prompted model canary did not pass")
    selector_path = os.path.join(
        str(job["selector_output"]), "selector.json"
    )
    selector = base._read_sealed(
        selector_path, label="R0178 FineWeb control selector"
    )
    if (
        selector.get("round_id") != ROUND_ID
        or selector.get("training_text_disjoint") is not True
        or selector.get("source_text_unique") is not True
    ):
        raise Round0178Error("R0178 FineWeb selector did not pass")
    rows_signature = base._signature(
        selector["selected_source_rows"], label="R0178 control source rows"
    )
    hashes_signature = base._signature(
        selector["selected_text_hashes"], label="R0178 control text hashes"
    )
    rows = np.load(rows_signature["canonical_path"], allow_pickle=False)
    hashes = np.load(hashes_signature["canonical_path"], allow_pickle=False)
    if rows.shape != (CONTROL_ROWS,) or hashes.shape != (CONTROL_ROWS,):
        raise Round0178Error("R0178 selected control geometry changed")
    text_source = base._signature(
        job["text_source"], label="R0178 FineWeb control text"
    )
    texts = base._parquet_texts(
        text_source,
        column="chunk_text",
        rows=np.asarray(rows, dtype=np.int64),
        label="R0178 FineWeb control",
    )
    if not np.array_equal(_hash_rows(texts), hashes):
        raise Round0178Error("R0178 selected control text bytes changed")
    model, runtime, members = _load_document_model()
    base._verify_model_members(members, job["model_members"])
    embeddings, telemetry, embed_wall = base._encode_prompted(
        model, texts, label="FineWeb disjoint control"
    )
    path = os.path.join(output, "embeddings.f16.npy")
    atomic_save_new_npy(path, embeddings, immutable=True)
    receipt = base.seal({
        "schema": "round0178-prompted-fineweb-control-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "prompt_applied": True,
        "prompt_prefix": base.PROMPT_PREFIX,
        "selection": (
            "R0178 source-text-unique, map-training-disjoint FineWeb pool"
        ),
        "selector": expected_input_signature(selector_path),
        "text_source": text_source,
        "source_rows": rows_signature,
        "source_text_hashes": hashes_signature,
        "text_sha256": base._text_sha256(texts),
        "embeddings": expected_input_signature(path),
        "embedding_guard": _stored_array_guard(
            path, expected_rows=CONTROL_ROWS
        ),
        "model_members": members,
        "runtime": runtime,
        "encode_telemetry": telemetry,
        "rows_per_second": CONTROL_ROWS / max(embed_wall, 1e-9),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "receipt.json"), receipt, immutable=True
    )


def _text_copy_mask(
    corpus: Sequence[str], queries: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    counts = Counter(corpus)
    query_counts = Counter(queries)
    mask = np.asarray([text in counts for text in queries], dtype=bool)
    return mask, {
        "identity": "complete source-text Unicode string after source formatting",
        "corpus_rows": len(corpus),
        "query_rows": len(queries),
        "corpus_unique_families": len(counts),
        "query_unique_families": len(query_counts),
        "corpus_duplicate_rows": len(corpus) - len(counts),
        "query_duplicate_rows": len(queries) - len(query_counts),
        "maximum_corpus_family": max(counts.values()),
        "maximum_query_family": max(query_counts.values()),
        "query_rows_with_corpus_copy": int(mask.sum()),
        "corpus_query_disjoint": bool(not np.any(mask)),
    }


def run_seal_sensitivity_masks(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0178 source-text sensitivity masks"
    )
    started = time.monotonic()
    training = {
        str(label): _load_hash_index(
            signature, label=f"R0178 {label} training text index"
        )
        for label, signature in job["training_text_hashes"].items()
    }
    control_receipt_path = os.path.join(
        str(job["control_output"]), "receipt.json"
    )
    control_receipt = base._read_sealed(
        control_receipt_path, label="R0178 prompted control receipt"
    )
    control_signature = base._signature(
        control_receipt["embeddings"], label="R0178 control embeddings"
    )
    control = np.load(
        control_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    control_rows_signature = base._signature(
        control_receipt["source_rows"], label="R0178 control source rows"
    )
    control_hashes_signature = base._signature(
        control_receipt["source_text_hashes"],
        label="R0178 control source-text hashes",
    )
    control_source_rows = np.load(
        control_rows_signature["canonical_path"], allow_pickle=False
    )
    control_hashes = np.load(
        control_hashes_signature["canonical_path"], allow_pickle=False
    )
    control_text_source = base._signature(
        control_receipt["text_source"], label="R0178 control text source"
    )
    control_texts = base._parquet_texts(
        control_text_source,
        column="chunk_text",
        rows=np.asarray(control_source_rows, dtype=np.int64),
        label="R0178 selected control",
    )
    if (
        control.shape != (CONTROL_ROWS, base.DIMENSION)
        or control_source_rows.shape != (CONTROL_ROWS,)
        or control_hashes.shape != (CONTROL_ROWS,)
        or not np.array_equal(_hash_rows(control_texts), control_hashes)
    ):
        raise Round0178Error("R0178 control sensitivity inputs changed")

    probes: dict[str, Any] = {}
    training_summary = {
        label: {
            "query_text_overlap_count": 0,
            "control_text_overlap_count": 0,
            "diagnostic_corpus_text_overlap_count": 0,
        }
        for label in training
    }
    for name in PROBE_ORDER:
        corpus, queries, corpus_rows, query_rows, _inputs = base._load_probe(
            str(job["probe_outputs"][name]), name
        )
        receipt_path = os.path.join(
            str(job["probe_outputs"][name]), "receipt.json"
        )
        receipt = base._read_sealed(
            receipt_path, label=f"R0178 reused R0177 {name} receipt"
        )
        if receipt.get("round_id") != ROUND_ID:
            raise Round0178Error(f"{name} is not a fresh R0178 embedding")
        source_job = dict(job["probe_sources"][name])
        corpus_texts, query_texts, _sources = base._selected_probe_texts(
            source_job,
            np.asarray(corpus_rows, dtype=np.int64),
            np.asarray(query_rows, dtype=np.int64),
        )
        if (
            base._text_sha256(corpus_texts)
            != receipt.get("corpus_text_sha256")
            or base._text_sha256(query_texts)
            != receipt.get("query_text_sha256")
        ):
            raise Round0178Error(f"{name} reused source-text identity changed")
        probe_text_copies, probe_text_audit = _text_copy_mask(
            corpus_texts, query_texts
        )
        probe_byte_copies, probe_byte_audit = exact_reference_copy_mask(
            np.asarray(corpus), np.asarray(queries)
        )
        control_corpus_rows, control_query_rows, _coordinates = (
            base._coordinate_rows(
                job["control_coordinates"][name],
                label=f"{name} control",
                control=True,
            )
        )
        if (
            len(control_corpus_rows) != len(corpus)
            or len(control_query_rows) != len(queries)
        ):
            raise Round0178Error(f"{name} control shape changed")
        control_corpus_texts = [
            control_texts[int(row)] for row in control_corpus_rows
        ]
        control_query_texts = [
            control_texts[int(row)] for row in control_query_rows
        ]
        control_text_copies, control_text_audit = _text_copy_mask(
            control_corpus_texts, control_query_texts
        )
        control_byte_copies, control_byte_audit = exact_reference_copy_mask(
            np.asarray(control[control_corpus_rows]),
            np.asarray(control[control_query_rows]),
        )
        excluded = np.asarray(
            probe_text_copies | control_text_copies, dtype=bool
        )
        keep = ~excluded
        if not np.any(keep):
            raise Round0178Error(f"{name} source-text mask removed all queries")
        retained_probe_text_copies, _retained_audit = _text_copy_mask(
            corpus_texts,
            [text for index, text in enumerate(query_texts) if keep[index]],
        )
        retained_control_text_copies, _retained_control_audit = (
            _text_copy_mask(
                control_corpus_texts,
                [
                    text
                    for index, text in enumerate(control_query_texts)
                    if keep[index]
                ],
            )
        )
        if np.any(retained_probe_text_copies) or np.any(
            retained_control_text_copies
        ):
            raise Round0178Error(f"{name} retained source-text leakage")

        corpus_hashes = _hash_rows(corpus_texts)
        query_hashes = _hash_rows(query_texts)
        selected_control_hashes = np.asarray(
            control_hashes[np.unique(np.concatenate((
                control_corpus_rows, control_query_rows
            )))],
            dtype="V32",
        )
        per_training: dict[str, Any] = {}
        for label, (index, signature) in training.items():
            corpus_training = _membership(index, corpus_hashes)
            query_training = _membership(index, query_hashes)
            control_training = _membership(index, selected_control_hashes)
            training_summary[label][
                "query_text_overlap_count"
            ] += int(query_training.sum())
            training_summary[label][
                "control_text_overlap_count"
            ] += int(control_training.sum())
            training_summary[label][
                "diagnostic_corpus_text_overlap_count"
            ] += int(corpus_training.sum())
            per_training[label] = {
                "training_text_hash_index": signature,
                "query_text_overlap_positions": np.flatnonzero(
                    query_training
                ).astype(np.int64).tolist(),
                "control_text_overlap_count": int(control_training.sum()),
                "diagnostic_corpus_text_overlap_count": int(
                    corpus_training.sum()
                ),
            }
        excluded_positions = np.flatnonzero(excluded).astype(np.int64)
        probe_text_positions = np.flatnonzero(
            probe_text_copies
        ).astype(np.int64)
        control_text_positions = np.flatnonzero(
            control_text_copies
        ).astype(np.int64)
        probe_byte_positions = np.flatnonzero(
            probe_byte_copies
        ).astype(np.int64)
        control_byte_positions = np.flatnonzero(
            control_byte_copies
        ).astype(np.int64)
        probes[name] = {
            "policy": (
                "exclude the union of paired query positions having an exact "
                "source-text corpus copy; stored-fp16 equality is diagnostic"
            ),
            "identity": "complete source text exactly as passed to Document: embedding",
            "original_query_rows": int(len(query_rows)),
            "control_query_rows": int(len(control_query_rows)),
            "retained_query_rows": int(keep.sum()),
            "probe_text_copy_audit": probe_text_audit,
            "control_text_copy_audit": control_text_audit,
            "probe_stored_fp16_copy_audit": probe_byte_audit,
            "control_stored_fp16_copy_audit": control_byte_audit,
            "probe_text_copy_query_positions": probe_text_positions.tolist(),
            "control_text_copy_query_positions": control_text_positions.tolist(),
            "probe_stored_fp16_copy_query_positions": (
                probe_byte_positions.tolist()
            ),
            "control_stored_fp16_copy_query_positions": (
                control_byte_positions.tolist()
            ),
            "text_but_not_stored_fp16_query_positions": np.setdiff1d(
                probe_text_positions,
                probe_byte_positions,
                assume_unique=True,
            ).astype(np.int64).tolist(),
            "excluded_query_positions": excluded_positions.tolist(),
            "excluded_probe_source_rows": np.asarray(
                query_rows, dtype=np.int64
            )[excluded].tolist(),
            "excluded_control_source_rows": np.asarray(
                control_source_rows, dtype=np.int64
            )[np.asarray(control_query_rows, dtype=np.int64)[excluded]].tolist(),
            "training_text_overlap": per_training,
            "retained_source_text_disjoint": True,
        }

    blocking = sum(
        values["query_text_overlap_count"]
        + values["control_text_overlap_count"]
        for values in training_summary.values()
    )
    report = base.seal({
        "schema": "round0178-source-text-sensitivity-masks-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "map_independent": True,
        "mask_identity": (
            "complete source text exactly as formatted before Document: prefix; "
            "paired union applied to probe and control query ordinals"
        ),
        "probe_order": list(PROBE_ORDER),
        "probes": probes,
        "training_text_overlap_summary": training_summary,
        "blocking_query_or_control_training_text_overlap_count": blocking,
        "control_receipt": expected_input_signature(control_receipt_path),
        "control_source_rows": control_rows_signature,
        "control_source_text_hashes": control_hashes_signature,
        "passed": blocking == 0,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "sensitivity-masks.json")
    atomic_write_new_json(path, report, immutable=True)
    if blocking:
        raise Round0178Error(
            "R0178 query/control source texts overlap map training: "
            f"{blocking} memberships"
        )


def run_job(
    active: Mapping[str, Any], job: Mapping[str, Any] | None = None
) -> None:
    _configure()
    if job is None:
        return base.run_job(dict(active), None)
    action = job.get("action")
    if action == "select_disjoint_control":
        return run_select_disjoint_control(active, job)
    if action == "embed_disjoint_control":
        return run_embed_disjoint_control(active, job)
    if action == "seal_sensitivity_masks":
        return run_seal_sensitivity_masks(active, job)
    if action == "audit_training_disjoint":
        return audit_base.run_training_disjoint_audit(active, job)
    return base.run_job(dict(active), dict(job))
