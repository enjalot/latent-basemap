"""GPU embedding/scoring nodes and CPU synthesis for R0167."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0142_jina_universality import PROBE_ORDER
from basemap.round0146_projection_predictors import geometry_predictors
from basemap.round0116_prompted_corpus import validate_environment_freeze
from basemap.round0108_evaluation import (
    exact_reference_copy_mask,
    exact_split_duplicate_diagnostics,
)
from basemap.round0167_prompted_universality import (
    CAPABILITY,
    CONTROL_QUERY_ID_OFFSET,
    DIMENSION,
    EMBED_CHUNK_ROWS,
    EMBED_MINIMUM_ROWS_PER_S,
    PROMPT_PREFIX,
    PROMPTED_MAP_ORDER,
    QUERY_ID_OFFSET,
    ROUND_ID,
    Round0167Error,
    control_rows_from_coordinate_archive,
    retention_verdict,
    seal,
    source_rows_from_coordinate_archive,
    twonn_correlations,
    validate_seal,
)
from experiments.round0108_nodes import _probe_score
from experiments.round0116_nodes import (
    _encode_document,
    _float32_norm_guard,
    _load_document_model,
    _prompt_equivalence,
    _stored_array_guard,
)


CANARY_SCHEMA = "round0167-prompt-model-canary-v1"
PROBE_SCHEMA = "round0167-prompted-probe-embeddings-v1"
CONTROL_SCHEMA = "round0167-prompted-fineweb-control-v1"
MAP_PANEL_SCHEMA = "round0167-prompted-universality-map-panel-v1"
ALLOW_CROSS_SPLIT_FAMILIES = False
DUPLICATE_SENSITIVITY = False


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    value = expected_input_signature(str(expected.get("canonical_path") or ""))
    if value != dict(expected):
        raise Round0167Error(f"{label} bytes changed")
    return value


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0167Error(f"JSON object required: {path}")
    return value


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    value = _read_json(path)
    validate_seal(value, label=label)
    return value


def _verify_model_members(
    actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> None:
    key = lambda item: str(item["canonical_path"])
    observed = sorted((dict(item) for item in actual), key=key)
    bound = sorted((dict(item) for item in expected), key=key)
    if observed != bound:
        raise Round0167Error("loaded Jina model closure differs from queue binding")


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a, b, dtype=np.float64)


def _coordinate_rows(
    signature: Mapping[str, Any],
    *,
    label: str,
    control: bool = False,
    separate_sources: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    bound = _signature(signature, label=f"{label} R0142 coordinates")
    with np.load(bound["canonical_path"], allow_pickle=False) as archive:
        corpus_ids = np.asarray(archive["probe_corpus_ids"], dtype=np.int64)
        query_ids = np.asarray(archive["probe_query_ids"], dtype=np.int64)
        if (
            archive["probe_corpus_coords"].shape != (len(corpus_ids), 2)
            or archive["probe_query_coords"].shape != (len(query_ids), 2)
        ):
            raise Round0167Error(f"{label} R0142 coordinate geometry changed")
    if control:
        corpus, queries = control_rows_from_coordinate_archive(
            corpus_ids, query_ids, label=label
        )
    else:
        corpus, queries = source_rows_from_coordinate_archive(
            corpus_ids,
            query_ids,
            label=label,
            separate_sources=separate_sources,
        )
    return corpus, queries, bound


def _text_sha256(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _encode_prompted(
    model: Any, texts: Sequence[str], *, label: str
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    output = np.empty((len(texts), DIMENSION), dtype=np.float16)
    telemetry: list[dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, len(texts), EMBED_CHUNK_ROWS):
        stop = min(start + EMBED_CHUNK_ROWS, len(texts))
        raw = list(texts[start:stop])
        if len(raw) != stop - start or not all(isinstance(item, str) for item in raw):
            raise Round0167Error(f"{label} text slice is incomplete")
        values, stamp = _encode_document(
            model, [PROMPT_PREFIX + item for item in raw]
        )
        _float32_norm_guard(values, label=f"R0167 {label} [{start},{stop})")
        output[start:stop] = values.astype(np.float16)
        telemetry.append({"row_range": [start, stop], **stamp})
        rate = stop / max(time.monotonic() - started, 1e-9)
        print(f"R0167 {label}: {stop}/{len(texts)} ({rate:.1f} rows/s)", flush=True)
        if stop >= 20_000 and rate < EMBED_MINIMUM_ROWS_PER_S:
            raise Round0167Error(
                f"{label} throughput {rate:.1f} below {EMBED_MINIMUM_ROWS_PER_S:.1f}"
            )
    return output, telemetry, time.monotonic() - started


def _exact_family_audit(corpus: np.ndarray, queries: np.ndarray) -> dict[str, Any]:
    def keys(values: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(values)
        return array.view(np.dtype((np.void, array.dtype.itemsize * DIMENSION))).reshape(-1)

    corpus_keys = keys(corpus)
    query_keys = keys(queries)
    corpus_unique, corpus_counts = np.unique(corpus_keys, return_counts=True)
    query_unique, query_counts = np.unique(query_keys, return_counts=True)
    overlap = np.intersect1d(corpus_unique, query_unique, assume_unique=True)
    diagnostic = exact_split_duplicate_diagnostics(corpus, queries)
    if overlap.size and not ALLOW_CROSS_SPLIT_FAMILIES:
        raise Round0167Error("prompted corpus/query exact families overlap")
    return {
        "identity": "complete stored prompted-fp16 row bytes",
        "corpus_rows": int(len(corpus)),
        "query_rows": int(len(queries)),
        "corpus_unique_families": int(len(corpus_unique)),
        "query_unique_families": int(len(query_unique)),
        "corpus_duplicate_rows": int(len(corpus) - len(corpus_unique)),
        "query_duplicate_rows": int(len(queries) - len(query_unique)),
        "maximum_corpus_family": int(corpus_counts.max()),
        "maximum_query_family": int(query_counts.max()),
        "cross_split_family_overlap": int(len(overlap)),
        "query_rows_with_exact_corpus_copy": int(
            diagnostic["query_rows_with_exact_corpus_copy"]
        ),
        "corpus_query_exact_family_disjoint": bool(
            diagnostic["corpus_query_exact_family_disjoint"]
        ),
        "policy": (
            "diagnostic-primary-plus-query-leakage-controlled-sensitivity"
            if ALLOW_CROSS_SPLIT_FAMILIES
            else "require-corpus-query-exact-family-disjoint"
        ),
        "passed": (
            None if ALLOW_CROSS_SPLIT_FAMILIES
            else bool(diagnostic["corpus_query_exact_family_disjoint"])
        ),
    }


def _duplicate_sensitivity_mask(
    *,
    probe_corpus: np.ndarray,
    probe_queries: np.ndarray,
    control_corpus: np.ndarray,
    control_queries: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exclude matched query positions leaked in either paired universe."""
    probe_copies, probe_audit = exact_reference_copy_mask(
        probe_corpus, probe_queries
    )
    control_copies, control_audit = exact_reference_copy_mask(
        control_corpus, control_queries
    )
    if probe_copies.shape != control_copies.shape:
        raise Round0167Error("duplicate-sensitivity query shapes disagree")
    excluded = np.asarray(probe_copies | control_copies, dtype=bool)
    keep = ~excluded
    if not np.any(keep):
        raise Round0167Error("duplicate-sensitivity removed every query")
    return keep, {
        "policy": (
            "exclude the union of paired query positions having an exact "
            "stored-fp16 corpus copy; retain both frozen corpora"
        ),
        "probe_copy_audit": probe_audit,
        "control_copy_audit": control_audit,
        "original_query_rows": int(len(keep)),
        "retained_query_rows": int(keep.sum()),
        "probe_copy_query_positions": np.flatnonzero(
            probe_copies
        ).astype(np.int64).tolist(),
        "control_copy_query_positions": np.flatnonzero(
            control_copies
        ).astype(np.int64).tolist(),
        "excluded_query_positions": np.flatnonzero(excluded).astype(
            np.int64
        ).tolist(),
    }


def _parquet_texts(
    signature: Mapping[str, Any], *, column: str, rows: np.ndarray, label: str
) -> list[str]:
    bound = _signature(signature, label=f"{label} parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq

    values = pq.read_table(bound["canonical_path"], columns=[column]).column(column)
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= len(values)):
        raise Round0167Error(f"{label} selected parquet row is out of bounds")
    selected = values.take(pa.array(rows, type=pa.int64())).to_pylist()
    if len(selected) != len(rows) or not all(isinstance(item, str) for item in selected):
        raise Round0167Error(f"{label} selected parquet texts are incomplete")
    return selected


def _ids(signature: Mapping[str, Any], *, label: str) -> list[str]:
    bound = _signature(signature, label=label)
    with open(bound["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise Round0167Error(f"{label} must be a JSON string list")
    return value


def _arrow_texts(
    signature: Mapping[str, Any],
    *,
    wanted_ids: Sequence[str],
    label: str,
    include_title: bool,
) -> list[str]:
    bound = _signature(signature, label=f"{label} Arrow")
    import pyarrow as pa
    import pyarrow.ipc as ipc

    wanted = set(wanted_ids)
    found: dict[str, str] = {}
    with pa.memory_map(bound["canonical_path"], "r") as source:
        try:
            reader = ipc.open_stream(source)
        except pa.ArrowInvalid:
            reader = ipc.open_file(source)
        for batch in reader:
            names = batch.schema.names
            row_ids = batch.column(names.index("_id")).to_pylist()
            bodies = batch.column(names.index("text")).to_pylist()
            titles = (
                batch.column(names.index("title")).to_pylist()
                if include_title and "title" in names
                else [""] * len(row_ids)
            )
            for raw_id, title, body in zip(row_ids, titles, bodies, strict=True):
                key = str(raw_id)
                if key in wanted:
                    found[key] = (str(title) + " " + str(body)).strip()
            if len(found) == len(wanted):
                break
    missing = wanted.difference(found)
    if missing:
        raise Round0167Error(f"{label} Arrow is missing {len(missing)} selected IDs")
    return [found[value] for value in wanted_ids]


def run_prompt_canary(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(str(job["outputs"][0]), label="R0167 prompt canary")
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    text_signature = _signature(job["canary_text"], label="canary text")
    document_signature = _signature(job["canary_document"], label="R0114 document")
    positions = np.asarray(job["canary_positions"], dtype=np.int64)
    if positions.shape != (32,) or np.any(positions < 0):
        raise Round0167Error("prompt canary positions changed")
    texts = _parquet_texts(
        text_signature, column="chunk_text", rows=positions, label="canary"
    )
    historical = np.asarray(
        np.load(document_signature["canonical_path"], mmap_mode="r", allow_pickle=False)[positions],
        dtype=np.float32,
    )
    model, runtime, members = _load_document_model()
    _verify_model_members(members, job["model_members"])
    equivalence = _prompt_equivalence(model, texts)
    fresh, telemetry = _encode_document(model, [PROMPT_PREFIX + text for text in texts])
    _float32_norm_guard(fresh, label="R0167 prompted canary")
    cosine = _cosine_rows(fresh, historical)
    if float(cosine.mean()) < 0.995 or float(cosine.min()) < 0.99:
        raise Round0167Error("prompted Jina execution does not reproduce R0114")
    receipt = seal({
        "schema": CANARY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "prompt_applied": True,
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_equivalence": equivalence,
        "model_members": members,
        "runtime": runtime,
        "text_source": text_signature,
        "historical_document_embeddings": document_signature,
        "positions": positions.tolist(),
        "positions_sha256": ordered_array_sha256(positions),
        "mean_cosine": float(cosine.mean()),
        "minimum_cosine": float(cosine.min()),
        "passed": True,
        "encode_telemetry": telemetry,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "canary.json"), receipt, immutable=True)


def _selected_probe_texts(
    job: Mapping[str, Any], corpus_rows: np.ndarray, query_rows: np.ndarray
) -> tuple[list[str], list[str], dict[str, Any]]:
    kind = str(job.get("source_kind") or "")
    name = str(job["probe"])
    if kind in {"common-parquet", "dadabase-parquet"}:
        column = "chunk_text" if kind == "common-parquet" else "joke"
        corpus = _parquet_texts(
            job["text_source"], column=column, rows=corpus_rows, label=f"{name} corpus"
        )
        queries = _parquet_texts(
            job["text_source"], column=column, rows=query_rows, label=f"{name} queries"
        )
        return corpus, queries, {"text_source": _signature(job["text_source"], label=f"{name} text")}
    if kind == "beir-arrow":
        corpus_ids = _ids(job["corpus_ids"], label=f"{name} corpus IDs")
        query_ids = _ids(job["query_ids"], label=f"{name} query IDs")
        if (
            int(corpus_rows.max()) >= len(corpus_ids)
            or int(query_rows.max()) >= len(query_ids)
        ):
            raise Round0167Error(f"{name} selected BEIR row is out of bounds")
        selected_corpus_ids = [corpus_ids[int(row)] for row in corpus_rows]
        selected_query_ids = [query_ids[int(row)] for row in query_rows]
        corpus = _arrow_texts(
            job["corpus_text_source"],
            wanted_ids=selected_corpus_ids,
            label=f"{name} corpus",
            include_title=True,
        )
        queries = _arrow_texts(
            job["query_text_source"],
            wanted_ids=selected_query_ids,
            label=f"{name} queries",
            include_title=False,
        )
        return corpus, queries, {
            "corpus_text_source": _signature(job["corpus_text_source"], label=f"{name} corpus Arrow"),
            "query_text_source": _signature(job["query_text_source"], label=f"{name} query Arrow"),
            "corpus_ids": _signature(job["corpus_ids"], label=f"{name} corpus IDs"),
            "query_ids": _signature(job["query_ids"], label=f"{name} query IDs"),
            "selected_corpus_ids_sha256": sha256_bytes(canonical_json(selected_corpus_ids)),
            "selected_query_ids_sha256": sha256_bytes(canonical_json(selected_query_ids)),
        }
    raise Round0167Error(f"unknown prompted probe source kind {kind!r}")


def run_embed_probe(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    name = str(job.get("probe") or "")
    if name not in PROBE_ORDER:
        raise Round0167Error(f"unknown R0167 probe {name!r}")
    output = create_fresh_directory(str(job["outputs"][0]), label=f"R0167 {name} embeddings")
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    canary = _read_sealed(
        os.path.join(str(job["canary_output"]), "canary.json"), label="R0167 canary"
    )
    if canary.get("passed") is not True or canary.get("prompt_applied") is not True:
        raise Round0167Error("R0167 prompted model canary did not pass")
    corpus_rows, query_rows, coordinates = _coordinate_rows(
        job["r0142_coordinates"],
        label=name,
        separate_sources=job.get("source_kind") == "beir-arrow",
    )
    corpus_texts, query_texts, sources = _selected_probe_texts(
        job, corpus_rows, query_rows
    )
    model, runtime, members = _load_document_model()
    _verify_model_members(members, job["model_members"])
    corpus, corpus_telemetry, corpus_wall = _encode_prompted(
        model, corpus_texts, label=f"{name} corpus"
    )
    queries, query_telemetry, query_wall = _encode_prompted(
        model, query_texts, label=f"{name} queries"
    )
    duplicate_audit = _exact_family_audit(corpus, queries)
    corpus_path = os.path.join(output, "corpus.f16.npy")
    query_path = os.path.join(output, "queries.f16.npy")
    corpus_rows_path = os.path.join(output, "corpus-source-rows.i64.npy")
    query_rows_path = os.path.join(output, "query-source-rows.i64.npy")
    atomic_save_new_npy(corpus_path, corpus, immutable=True)
    atomic_save_new_npy(query_path, queries, immutable=True)
    atomic_save_new_npy(corpus_rows_path, corpus_rows, immutable=True)
    atomic_save_new_npy(query_rows_path, query_rows, immutable=True)
    receipt = seal({
        "schema": PROBE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "probe": name,
        "prompt_applied": True,
        "prompt_prefix": PROMPT_PREFIX,
        "r0142_coordinates": coordinates,
        "sources": sources,
        "corpus_embeddings": expected_input_signature(corpus_path),
        "query_embeddings": expected_input_signature(query_path),
        "corpus_source_rows": expected_input_signature(corpus_rows_path),
        "query_source_rows": expected_input_signature(query_rows_path),
        "corpus_rows_sha256": ordered_array_sha256(corpus_rows),
        "query_rows_sha256": ordered_array_sha256(query_rows),
        "corpus_text_sha256": _text_sha256(corpus_texts),
        "query_text_sha256": _text_sha256(query_texts),
        "corpus_guard": _stored_array_guard(corpus_path, expected_rows=len(corpus_rows)),
        "query_guard": _stored_array_guard(query_path, expected_rows=len(query_rows)),
        "duplicate_audit": duplicate_audit,
        "model_members": members,
        "runtime": runtime,
        "encode_telemetry": {
            "corpus": corpus_telemetry,
            "queries": query_telemetry,
        },
        "rows_per_second": (len(corpus_rows) + len(query_rows)) / max(corpus_wall + query_wall, 1e-9),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "receipt.json"), receipt, immutable=True)


def run_embed_control(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(str(job["outputs"][0]), label="R0167 prompted control")
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    canary = _read_sealed(
        os.path.join(str(job["canary_output"]), "canary.json"), label="R0167 canary"
    )
    if canary.get("passed") is not True:
        raise Round0167Error("R0167 prompted model canary did not pass")
    rows = np.arange(60_000, dtype=np.int64)
    texts = _parquet_texts(
        job["text_source"], column="chunk_text", rows=rows, label="FineWeb control"
    )
    model, runtime, members = _load_document_model()
    _verify_model_members(members, job["model_members"])
    embeddings, telemetry, embed_wall = _encode_prompted(
        model, texts, label="FineWeb control"
    )
    path = os.path.join(output, "embeddings.f16.npy")
    atomic_save_new_npy(path, embeddings, immutable=True)
    receipt = seal({
        "schema": CONTROL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "prompt_applied": True,
        "prompt_prefix": PROMPT_PREFIX,
        "selection": "first 60000 rows; exact R0142 heldout source order",
        "text_source": _signature(job["text_source"], label="FineWeb control text"),
        "text_sha256": _text_sha256(texts),
        "embeddings": expected_input_signature(path),
        "embedding_guard": _stored_array_guard(path, expected_rows=60_000),
        "model_members": members,
        "runtime": runtime,
        "encode_telemetry": telemetry,
        "rows_per_second": 60_000 / max(embed_wall, 1e-9),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "receipt.json"), receipt, immutable=True)


def _load_probe(output: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    receipt_path = os.path.join(output, "receipt.json")
    receipt = _read_sealed(receipt_path, label=f"R0167 {name} receipt")
    if receipt.get("probe") != name or receipt.get("prompt_applied") is not True:
        raise Round0167Error(f"{name} prompted embedding receipt changed")
    corpus_signature = _signature(receipt["corpus_embeddings"], label=f"{name} corpus embeddings")
    query_signature = _signature(receipt["query_embeddings"], label=f"{name} query embeddings")
    corpus_rows_signature = _signature(receipt["corpus_source_rows"], label=f"{name} corpus rows")
    query_rows_signature = _signature(receipt["query_source_rows"], label=f"{name} query rows")
    corpus = np.load(corpus_signature["canonical_path"], mmap_mode="r", allow_pickle=False)
    queries = np.load(query_signature["canonical_path"], mmap_mode="r", allow_pickle=False)
    corpus_rows = np.load(corpus_rows_signature["canonical_path"], allow_pickle=False)
    query_rows = np.load(query_rows_signature["canonical_path"], allow_pickle=False)
    if corpus.shape != (len(corpus_rows), DIMENSION) or queries.shape != (len(query_rows), DIMENSION):
        raise Round0167Error(f"{name} prompted embedding geometry changed")
    return corpus, queries, corpus_rows, query_rows, {
        "embedding_receipt": expected_input_signature(receipt_path),
        "corpus_embeddings": corpus_signature,
        "query_embeddings": query_signature,
        "corpus_source_rows": corpus_rows_signature,
        "query_source_rows": query_rows_signature,
        "duplicate_audit": receipt.get("duplicate_audit"),
        "prompt_applied": True,
        "prompt_prefix": PROMPT_PREFIX,
    }


def run_score_map(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    map_key = str(job.get("map_key") or "")
    if map_key not in PROMPTED_MAP_ORDER:
        raise Round0167Error(f"unknown R0167 map {map_key!r}")
    output = create_fresh_directory(str(job["outputs"][0]), label=f"R0167 score {map_key}")
    started = time.monotonic()
    external_sensitivity_masks: dict[str, Any] | None = None
    sensitivity_masks_output = job.get("sensitivity_masks_output")
    if DUPLICATE_SENSITIVITY and sensitivity_masks_output:
        masks_path = os.path.join(
            str(sensitivity_masks_output), "sensitivity-masks.json"
        )
        external_sensitivity_masks = _read_sealed(
            masks_path, label=f"R{ROUND_ID} source-text sensitivity masks"
        )
        if (
            external_sensitivity_masks.get("round_id") != ROUND_ID
            or external_sensitivity_masks.get("map_independent") is not True
            or external_sensitivity_masks.get("passed") is not True
            or set(external_sensitivity_masks.get("probes") or {})
            != set(PROBE_ORDER)
        ):
            raise Round0167Error(
                "external duplicate-sensitivity mask receipt changed"
            )
    model_signature = _signature(job["model"], label=f"{map_key} model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device="cuda")
    control_receipt_path = os.path.join(str(job["control_output"]), "receipt.json")
    control_receipt = _read_sealed(control_receipt_path, label="R0167 control receipt")
    if control_receipt.get("prompt_applied") is not True:
        raise Round0167Error("R0167 control is not prompted")
    control_signature = _signature(control_receipt["embeddings"], label="prompted control embeddings")
    control = np.load(control_signature["canonical_path"], mmap_mode="r", allow_pickle=False)
    if control.shape != (60_000, DIMENSION):
        raise Round0167Error("prompted FineWeb control geometry changed")

    reports: dict[str, Any] = {}
    for index, name in enumerate(PROBE_ORDER):
        corpus, queries, corpus_rows, query_rows, inputs = _load_probe(
            str(job["probe_outputs"][name]), name
        )
        control_corpus_rows, control_query_rows, control_coordinates = _coordinate_rows(
            job["control_coordinates"][name], label=f"{name} control", control=True
        )
        if len(control_corpus_rows) != len(corpus) or len(control_query_rows) != len(queries):
            raise Round0167Error(f"{name} prompted control shape mismatch")
        primary_duplicate_policy = (
            "diagnostic-only"
            if DUPLICATE_SENSITIVITY
            else "require-corpus-query-exact-family-disjoint"
        )
        control_corpus = np.asarray(control[control_corpus_rows])
        control_queries = np.asarray(control[control_query_rows])
        probe_report = _probe_score(
            name=name,
            corpus=np.asarray(corpus),
            queries=np.asarray(queries),
            corpus_ids=np.asarray(corpus_rows, dtype=np.int64),
            query_ids=QUERY_ID_OFFSET + np.asarray(query_rows, dtype=np.int64),
            model=model,
            output=output,
            inputs=inputs,
            save_coordinates=True,
            duplicate_policy=primary_duplicate_policy,
        )
        control_report = _probe_score(
            name=f"{name}__fineweb-control",
            corpus=control_corpus,
            queries=control_queries,
            corpus_ids=control_corpus_rows,
            query_ids=CONTROL_QUERY_ID_OFFSET + control_query_rows,
            model=model,
            output=output,
            inputs={
                "embedding_receipt": expected_input_signature(control_receipt_path),
                "embeddings": control_signature,
                "prompt_applied": True,
                "prompt_prefix": PROMPT_PREFIX,
                "training_membership": "dedicated heldout artifact",
                "r0142_coordinates": control_coordinates,
            },
            save_coordinates=True,
            duplicate_policy=primary_duplicate_policy,
        )
        probe_ffr = float(probe_report["probe"]["ffr"])
        control_ffr = float(control_report["probe"]["ffr"])
        if control_ffr <= 0:
            raise Round0167Error(f"{name} control FFR is nonpositive")
        control_recall = float(control_report["probe"]["recall_at_10"])
        retention = probe_ffr / control_ffr
        reports[name] = {
            "probe": probe_report,
            "control": control_report,
            "metrics": {
                "probe_ffr": probe_ffr,
                "control_ffr": control_ffr,
                "ffr_retention": retention,
                "recall10_retention": (
                    float(probe_report["probe"]["recall_at_10"]) / control_recall
                    if control_recall > 0
                    else None
                ),
                "verdict": retention_verdict(retention),
            },
        }
        if DUPLICATE_SENSITIVITY:
            if external_sensitivity_masks is None:
                keep, sensitivity_audit = _duplicate_sensitivity_mask(
                    probe_corpus=np.asarray(corpus),
                    probe_queries=np.asarray(queries),
                    control_corpus=control_corpus,
                    control_queries=control_queries,
                )
            else:
                sensitivity_audit = dict(
                    external_sensitivity_masks["probes"][name]
                )
                if (
                    int(sensitivity_audit.get("original_query_rows", -1))
                    != len(queries)
                    or int(sensitivity_audit.get("control_query_rows", -1))
                    != len(control_queries)
                ):
                    raise Round0167Error(
                        f"{name} external sensitivity mask shape changed"
                    )
                positions = np.asarray(
                    sensitivity_audit.get("excluded_query_positions") or [],
                    dtype=np.int64,
                )
                if (
                    positions.ndim != 1
                    or (len(positions) and (
                        int(positions.min()) < 0
                        or int(positions.max()) >= len(queries)
                    ))
                    or len(np.unique(positions)) != len(positions)
                ):
                    raise Round0167Error(
                        f"{name} external sensitivity positions changed"
                    )
                keep = np.ones(len(queries), dtype=bool)
                keep[positions] = False
                if not np.any(keep):
                    raise Round0167Error(
                        f"{name} external sensitivity removed every query"
                    )
                expected_probe_rows = np.asarray(
                    sensitivity_audit.get("excluded_probe_source_rows") or [],
                    dtype=np.int64,
                )
                if not np.array_equal(
                    expected_probe_rows,
                    np.asarray(query_rows, dtype=np.int64)[~keep],
                ):
                    raise Round0167Error(
                        f"{name} external sensitivity source rows changed"
                    )
            excluded = ~keep
            if external_sensitivity_masks is None:
                sensitivity_audit["excluded_probe_source_rows"] = (
                    np.asarray(query_rows, dtype=np.int64)[excluded].tolist()
                )
                sensitivity_audit["excluded_control_source_rows"] = (
                    np.asarray(
                        control_query_rows, dtype=np.int64
                    )[excluded].tolist()
                )
            if bool(np.all(keep)):
                sensitivity_metrics = dict(reports[name]["metrics"])
                sensitivity = {
                    "audit": sensitivity_audit,
                    "identical_to_primary": True,
                    "probe": None,
                    "control": None,
                    "metrics": sensitivity_metrics,
                }
            else:
                sensitivity_probe = _probe_score(
                    name=f"{name}__duplicate-controlled",
                    corpus=np.asarray(corpus),
                    queries=np.asarray(queries)[keep],
                    corpus_ids=np.asarray(corpus_rows, dtype=np.int64),
                    query_ids=(
                        QUERY_ID_OFFSET
                        + np.asarray(query_rows, dtype=np.int64)[keep]
                    ),
                    model=model,
                    output=output,
                    inputs={
                        **inputs,
                        "duplicate_sensitivity": sensitivity_audit,
                    },
                    save_coordinates=True,
                    duplicate_policy=(
                        "require-corpus-query-exact-family-disjoint"
                    ),
                )
                sensitivity_control = _probe_score(
                    name=f"{name}__fineweb-control__duplicate-controlled",
                    corpus=control_corpus,
                    queries=control_queries[keep],
                    corpus_ids=control_corpus_rows,
                    query_ids=(
                        CONTROL_QUERY_ID_OFFSET + control_query_rows[keep]
                    ),
                    model=model,
                    output=output,
                    inputs={
                        "embedding_receipt": expected_input_signature(
                            control_receipt_path
                        ),
                        "embeddings": control_signature,
                        "prompt_applied": True,
                        "prompt_prefix": PROMPT_PREFIX,
                        "training_membership": "dedicated heldout artifact",
                        "r0142_coordinates": control_coordinates,
                        "duplicate_sensitivity": sensitivity_audit,
                    },
                    save_coordinates=True,
                    duplicate_policy=(
                        "require-corpus-query-exact-family-disjoint"
                    ),
                )
                sensitivity_probe_ffr = float(
                    sensitivity_probe["probe"]["ffr"]
                )
                sensitivity_control_ffr = float(
                    sensitivity_control["probe"]["ffr"]
                )
                sensitivity_control_recall = float(
                    sensitivity_control["probe"]["recall_at_10"]
                )
                if sensitivity_control_ffr <= 0:
                    raise Round0167Error(
                        f"{name} sensitivity control FFR is nonpositive"
                    )
                sensitivity_retention = (
                    sensitivity_probe_ffr / sensitivity_control_ffr
                )
                sensitivity = {
                    "audit": sensitivity_audit,
                    "identical_to_primary": False,
                    "probe": sensitivity_probe,
                    "control": sensitivity_control,
                    "metrics": {
                        "probe_ffr": sensitivity_probe_ffr,
                        "control_ffr": sensitivity_control_ffr,
                        "ffr_retention": sensitivity_retention,
                        "recall10_retention": (
                            float(
                                sensitivity_probe["probe"]["recall_at_10"]
                            )
                            / sensitivity_control_recall
                            if sensitivity_control_recall > 0
                            else None
                        ),
                        "verdict": retention_verdict(
                            sensitivity_retention
                        ),
                    },
                }
            reports[name]["duplicate_controlled_sensitivity"] = sensitivity
        print(
            f"R0167 {map_key} {index + 1}/{len(PROBE_ORDER)} {name}: retention={retention:.4f}",
            flush=True,
        )
        del corpus, queries
        gc.collect()
    panel = seal({
        "schema": MAP_PANEL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "map_key": map_key,
        "model": model_signature,
        "probe_order": list(PROBE_ORDER),
        "probes": reports,
        "thresholds": {"pass_at_least": 0.70, "failure_below": 0.50},
        "duplicate_policy": (
            "full frozen panel primary plus paired query-leakage sensitivity"
            if DUPLICATE_SENSITIVITY
            else "require corpus/query exact-family disjointness"
        ),
        "role": "diagnostic-only; no atlas-quality or production gate",
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "universality-panel.json"), panel, immutable=True)


def _accepted_raw_rho(report: Mapping[str, Any]) -> float:
    matches = [
        item for item in report.get("correlations", [])
        if item.get("outcome") == "ffr_retention"
        and item.get("scope") == "pooled-descriptive"
        and item.get("predictor") == "twonn_intrinsic_dimension"
    ]
    if len(matches) != 1:
        raise Round0167Error("accepted R0146 pooled TwoNN cell is missing")
    return float(matches[0]["spearman_rho"])


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(str(job["outputs"][0]), label="R0167 synthesis")
    started = time.monotonic()
    panels: dict[str, dict[str, Any]] = {}
    for map_key in PROMPTED_MAP_ORDER:
        path = os.path.join(str(job["map_outputs"][map_key]), "universality-panel.json")
        panel = _read_sealed(path, label=f"R0167 {map_key} panel")
        if panel.get("map_key") != map_key or panel.get("probe_order") != list(PROBE_ORDER):
            raise Round0167Error(f"{map_key} prompted panel changed")
        panels[map_key] = panel

    raw_predictor_signature = _signature(job["raw_predictors"], label="R0146 predictors")
    raw_predictors = _read_json(raw_predictor_signature["canonical_path"])
    raw_body = {key: value for key, value in raw_predictors.items() if key != "identity_sha256"}
    if raw_predictors.get("identity_sha256") != sha256_bytes(canonical_json(raw_body)):
        raise Round0167Error("R0146 predictor seal changed")
    raw_rho = _accepted_raw_rho(raw_predictors)
    raw_table_signature = _signature(job["raw_retention_table"], label="R0142 retention table")
    raw_table = _read_json(raw_table_signature["canonical_path"])
    training_audit = None
    if job.get("training_disjoint_audit"):
        audit_path = os.path.join(
            str(job["training_disjoint_audit"]), "audit.json"
        )
        audit = _read_sealed(
            audit_path, label=f"R{ROUND_ID} prompted training-overlap audit"
        )
        if audit.get("round_id") != ROUND_ID or audit.get("passed") is not True:
            raise Round0167Error("prompted training-overlap audit did not pass")
        training_audit = {
            "receipt": expected_input_signature(audit_path),
            "all_rows_training_disjoint": bool(
                audit.get("all_rows_training_disjoint")
            ),
            "blocking_query_or_control_overlap_count": int(
                audit.get("blocking_query_or_control_overlap_count", -1)
            ),
            "diagnostic_corpus_overlap_count": int(
                audit.get("diagnostic_corpus_overlap_count", -1)
            ),
            "policy": audit.get("policy"),
        }

    geometries: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []
    duplicate_sensitivity_cells: list[dict[str, Any]] = []
    for name in PROBE_ORDER:
        corpus, _queries, corpus_rows, _query_rows, inputs = _load_probe(
            str(job["probe_outputs"][name]), name
        )
        geometry = geometry_predictors(
            corpus, source_row_ids=corpus_rows, label=f"prompted:{name}"
        )
        geometries[name] = {"inputs": inputs, "geometry": geometry}
        for map_key in PROMPTED_MAP_ORDER:
            metric = panels[map_key]["probes"][name]["metrics"]
            cells.append({
                "map": map_key,
                "probe": name,
                "ffr_retention": float(metric["ffr_retention"]),
                "recall10_retention": (
                    float(metric["recall10_retention"])
                    if metric.get("recall10_retention") is not None
                    else None
                ),
                "twonn_intrinsic_dimension": float(geometry["twonn"]["intrinsic_dimension"]),
            })
            if DUPLICATE_SENSITIVITY:
                sensitivity = panels[map_key]["probes"][name][
                    "duplicate_controlled_sensitivity"
                ]
                sensitivity_metric = sensitivity["metrics"]
                primary_ffr = float(metric["ffr_retention"])
                sensitivity_ffr = float(
                    sensitivity_metric["ffr_retention"]
                )
                primary_recall = (
                    float(metric["recall10_retention"])
                    if metric.get("recall10_retention") is not None
                    else None
                )
                sensitivity_recall = (
                    float(sensitivity_metric["recall10_retention"])
                    if sensitivity_metric.get("recall10_retention")
                    is not None
                    else None
                )
                duplicate_sensitivity_cells.append({
                    "map": map_key,
                    "probe": name,
                    "ffr_retention": sensitivity_ffr,
                    "recall10_retention": sensitivity_recall,
                    "primary_ffr_retention": primary_ffr,
                    "delta_ffr_retention": (
                        sensitivity_ffr - primary_ffr
                    ),
                    "primary_recall10_retention": primary_recall,
                    "delta_recall10_retention": (
                        sensitivity_recall - primary_recall
                        if sensitivity_recall is not None
                        and primary_recall is not None
                        else None
                    ),
                    "primary_verdict": metric["verdict"],
                    "sensitivity_verdict": (
                        sensitivity_metric["verdict"]
                    ),
                    "twonn_intrinsic_dimension": float(
                        geometry["twonn"]["intrinsic_dimension"]
                    ),
                    "excluded_query_positions": list(
                        sensitivity["audit"][
                            "excluded_query_positions"
                        ]
                    ),
                })
    correlations = twonn_correlations(cells)
    duplicate_sensitivity_correlations = (
        twonn_correlations(duplicate_sensitivity_cells)
        if DUPLICATE_SENSITIVITY
        else None
    )
    duplicate_sensitivity_summary = None
    if DUPLICATE_SENSITIVITY:
        sensitivity_recall_deltas = [
            abs(float(item["delta_recall10_retention"]))
            for item in duplicate_sensitivity_cells
            if item["delta_recall10_retention"] is not None
        ]
        duplicate_sensitivity_summary = {
            "cells": len(duplicate_sensitivity_cells),
            "cells_with_query_exclusions": sum(
                bool(item["excluded_query_positions"])
                for item in duplicate_sensitivity_cells
            ),
            "maximum_absolute_ffr_retention_delta": max(
                abs(float(item["delta_ffr_retention"]))
                for item in duplicate_sensitivity_cells
            ),
            "maximum_absolute_recall10_retention_delta": max(
                sensitivity_recall_deltas
            ) if sensitivity_recall_deltas else None,
            "verdict_changes": sum(
                item["primary_verdict"]
                != item["sensitivity_verdict"]
                for item in duplicate_sensitivity_cells
            ),
        }
    prompted_rho = next(
        float(item["spearman_rho"])
        for item in correlations
        if item["outcome"] == "ffr_retention" and item["scope"] == "pooled-descriptive"
    )
    summaries: dict[str, Any] = {}
    for map_key in PROMPTED_MAP_ORDER:
        values = [panels[map_key]["probes"][name]["metrics"] for name in PROBE_ORDER]
        summaries[map_key] = {
            "ffr_retention_median": float(np.median([float(item["ffr_retention"]) for item in values])),
            "pass": sum(item["verdict"] == "pass" for item in values),
            "amber": sum(item["verdict"] == "amber" for item in values),
            "named_failure": sum(item["verdict"] == "named-failure" for item in values),
        }
        if DUPLICATE_SENSITIVITY:
            sensitivity_values = [
                panels[map_key]["probes"][name][
                    "duplicate_controlled_sensitivity"
                ]["metrics"]
                for name in PROBE_ORDER
            ]
            summaries[map_key]["duplicate_controlled_sensitivity"] = {
                "ffr_retention_median": float(np.median([
                    float(item["ffr_retention"])
                    for item in sensitivity_values
                ])),
                "pass": sum(
                    item["verdict"] == "pass"
                    for item in sensitivity_values
                ),
                "amber": sum(
                    item["verdict"] == "amber"
                    for item in sensitivity_values
                ),
                "named_failure": sum(
                    item["verdict"] == "named-failure"
                    for item in sensitivity_values
                ),
            }
    scale_comparisons: dict[str, Any] = {}
    for name in PROBE_ORDER:
        baseline = np.mean([
            float(panels[map_key]["probes"][name]["metrics"]["ffr_retention"])
            for map_key in PROMPTED_MAP_ORDER[:2]
        ])
        eight = float(panels[PROMPTED_MAP_ORDER[2]]["probes"][name]["metrics"]["ffr_retention"])
        scale_comparisons[name] = {
            "mean_2m_seed42_43": float(baseline),
            "prompted_8m_seed42": eight,
            "delta_8m_minus_mean_2m": eight - float(baseline),
        }
        if DUPLICATE_SENSITIVITY:
            sensitivity_baseline = np.mean([
                float(
                    panels[map_key]["probes"][name][
                        "duplicate_controlled_sensitivity"
                    ]["metrics"]["ffr_retention"]
                )
                for map_key in PROMPTED_MAP_ORDER[:2]
            ])
            sensitivity_eight = float(
                panels[PROMPTED_MAP_ORDER[2]]["probes"][name][
                    "duplicate_controlled_sensitivity"
                ]["metrics"]["ffr_retention"]
            )
            scale_comparisons[name][
                "duplicate_controlled_sensitivity"
            ] = {
                "mean_2m_seed42_43": float(sensitivity_baseline),
                "prompted_8m_seed42": sensitivity_eight,
                "delta_8m_minus_mean_2m": (
                    sensitivity_eight - float(sensitivity_baseline)
                ),
            }
    sensitivity_mask_receipt = None
    if job.get("sensitivity_masks_output"):
        sensitivity_mask_receipt = expected_input_signature(
            os.path.join(
                str(job["sensitivity_masks_output"]),
                "sensitivity-masks.json",
            )
        )
    report = seal({
        "schema": CAPABILITY,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "probe_order": list(PROBE_ORDER),
        "map_order": list(PROMPTED_MAP_ORDER),
        "maps": {
            key: expected_input_signature(
                os.path.join(str(job["map_outputs"][key]), "universality-panel.json")
            )
            for key in PROMPTED_MAP_ORDER
        },
        "cells": cells,
        "duplicate_controlled_sensitivity_cells": (
            duplicate_sensitivity_cells
            if DUPLICATE_SENSITIVITY
            else None
        ),
        "summaries": summaries,
        "scale_comparisons": scale_comparisons,
        "prompted_geometry": geometries,
        "twonn_correlations": correlations,
        "duplicate_controlled_sensitivity_correlations": (
            duplicate_sensitivity_correlations
        ),
        "duplicate_controlled_sensitivity_summary": (
            duplicate_sensitivity_summary
        ),
        "source_text_sensitivity_masks": sensitivity_mask_receipt,
        "raw_comparison": {
            "r0142_retention_table": raw_table_signature,
            "r0142_rows": raw_table.get("rows"),
            "r0146_predictors": raw_predictor_signature,
            "raw_pooled_twonn_ffr_rho": raw_rho,
            "prompted_pooled_twonn_ffr_rho": prompted_rho,
            "delta_prompted_minus_raw": prompted_rho - raw_rho,
            "interpretation": (
                "descriptive only: raw R0142 maps are 12.5M/25M while prompted maps are 2M/8M"
            ),
        },
        "training_overlap_audit": training_audit,
        "interpretation": (
            "within-probe FFR divided by an exactly shape-matched prompted FineWeb control; "
            "same accepted R0142 source-row selections"
        ),
        "duplicate_policy": (
            "full frozen panel is primary; paired sensitivity excludes the "
            "union of query positions with an exact corpus copy in either "
            "probe or matched control"
            if DUPLICATE_SENSITIVITY
            else "require corpus/query exact-family disjointness"
        ),
        "diagnostic_only": True,
        "no_causal_prompt_claim": True,
        "no_universal_map_claim": True,
        "no_quality_gate_change": True,
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "prompted-universality-panel.json"), report, immutable=True)


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0167Error("R0167 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "prompt_canary":
        return run_prompt_canary(active, job)
    if action == "embed_probe":
        return run_embed_probe(active, job)
    if action == "embed_control":
        return run_embed_control(active, job)
    if action == "score_map":
        return run_score_map(active, job)
    if action == "assemble":
        return run_assemble(active, job)
    raise Round0167Error(f"unknown R0167 action: {action!r}")
