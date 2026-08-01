"""GPU embedding/scoring nodes and CPU assembler for R0142."""
from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0112_prompt_substrate import OUTPUT_DTYPE
from basemap.round0142_jina_universality import (
    CAPABILITY,
    COMMON_CORPUS_ROWS,
    DIMENSION,
    EMBED_BATCH_ROWS,
    EMBED_CHUNK_ROWS,
    EMBED_MINIMUM_ROWS_PER_S,
    EMBED_WARNING_ROWS_PER_S,
    MAP_ORDER,
    PROBE_ORDER,
    ROUND_ID,
    Round0142Error,
    canonical_representatives,
    fixed_separate_split,
    fixed_single_array_split,
    retention_verdict,
    seal,
    shape_matched_control_split,
    validate_seal,
)
from experiments.round0108_nodes import _probe_score


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    value = expected_input_signature(str(expected.get("canonical_path") or ""))
    if value != dict(expected):
        raise Round0142Error(f"{label} bytes changed")
    return value


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0142Error(f"JSON object required: {path}")
    return value


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    value = _read_json(path)
    validate_seal(value, label=label)
    return value


def _verified_model() -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    from experiments.round0116_nodes import _load_document_model

    model, runtime, members = _load_document_model()
    return model, runtime, [dict(item) for item in members]


def _verify_model_members(
    actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> None:
    sort_key = lambda item: str(item["canonical_path"])
    observed = sorted((dict(item) for item in actual), key=sort_key)
    bound = sorted((dict(item) for item in expected), key=sort_key)
    if observed != bound:
        raise Round0142Error("loaded Jina model closure differs from queue binding")


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a, b, dtype=np.float64)


def run_raw_model_canary(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    from basemap.round0141_prompted_multilingual import validate_environment_freeze
    from experiments.round0116_nodes import _encode_document, _float32_norm_guard

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0142 raw Jina model canary"
    )
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    text_signature = _signature(job["canary_text"], label="canary text parquet")
    raw_signature = _signature(job["canary_raw"], label="R0114 raw embeddings")
    positions = np.asarray(job["canary_positions"], dtype=np.int64)
    if positions.shape != (32,) or np.any(positions < 0):
        raise Round0142Error("raw canary positions changed")
    import pyarrow.parquet as pq

    column = pq.read_table(
        text_signature["canonical_path"], columns=["chunk_text"]
    ).column("chunk_text")
    texts = [column[int(row)].as_py() for row in positions]
    if not all(isinstance(text, str) for text in texts):
        raise Round0142Error("raw canary text mapping is incomplete")
    historical = np.asarray(
        np.load(raw_signature["canonical_path"], mmap_mode="r", allow_pickle=False)[
            positions
        ],
        dtype=np.float32,
    )
    model, runtime, members = _verified_model()
    _verify_model_members(members, job["model_members"])
    fresh, telemetry = _encode_document(
        model, texts, requested_batch_size=EMBED_BATCH_ROWS
    )
    _float32_norm_guard(fresh, label="R0142 raw canary")
    cosines = _cosine_rows(fresh, historical)
    passed = bool(float(cosines.mean()) >= 0.995 and float(cosines.min()) >= 0.99)
    if not passed:
        raise Round0142Error("raw Jina execution does not reproduce R0114 bytes")
    receipt = seal({
        "schema": "round0142-raw-jina-model-canary-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "prompt_applied": False,
        "prompt_semantics": "raw/unprompted; SentenceTransformer default prompt is null",
        "model_members": members,
        "runtime": runtime,
        "text_source": text_signature,
        "historical_raw_embeddings": raw_signature,
        "positions": positions.tolist(),
        "positions_sha256": ordered_array_sha256(positions),
        "mean_cosine": float(cosines.mean()),
        "minimum_cosine": float(cosines.min()),
        "mean_floor": 0.995,
        "minimum_floor": 0.99,
        "encode_telemetry": telemetry,
        "passed": True,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "canary.json"), receipt, immutable=True)


def run_embed_common_corpus(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    from basemap.round0141_prompted_multilingual import validate_environment_freeze
    from experiments.round0116_nodes import (
        _encode_document,
        _float32_norm_guard,
        _stored_array_guard,
    )

    name = str(job.get("probe") or "")
    if name not in COMMON_CORPUS_ROWS:
        raise Round0142Error(f"unknown Common Corpus probe {name!r}")
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0142 {name} raw Jina embeddings"
    )
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    source = _signature(job["source"], label=f"{name} source parquet")
    canary = _read_sealed(
        os.path.join(str(job["canary_output"]), "canary.json"),
        label="R0142 raw model canary",
    )
    if canary.get("passed") is not True or canary.get("prompt_applied") is not False:
        raise Round0142Error("R0142 raw model canary did not pass")
    rows = min(int(job["source_rows"]), int(job["selected_rows"]))
    if rows != min(COMMON_CORPUS_ROWS[name], 50_000):
        raise Round0142Error(f"{name} row-selection contract changed")
    import pyarrow.parquet as pq

    column = pq.read_table(source["canonical_path"], columns=["chunk_text"]).column(
        "chunk_text"
    )
    if len(column) != COMMON_CORPUS_ROWS[name]:
        raise Round0142Error(f"{name} parquet row count changed")
    model, runtime, members = _verified_model()
    _verify_model_members(members, job["model_members"])
    embeddings = np.empty((rows, DIMENSION), dtype=OUTPUT_DTYPE)
    telemetry: list[dict[str, Any]] = []
    for start in range(0, rows, EMBED_CHUNK_ROWS):
        stop = min(start + EMBED_CHUNK_ROWS, rows)
        texts = column.slice(start, stop - start).to_pylist()
        if len(texts) != stop - start or not all(
            isinstance(text, str) for text in texts
        ):
            raise Round0142Error(f"{name} text slice is incomplete")
        values, stamp = _encode_document(
            model, texts, requested_batch_size=EMBED_BATCH_ROWS
        )
        _float32_norm_guard(values, label=f"R0142 {name} [{start},{stop})")
        embeddings[start:stop] = values.astype(OUTPUT_DTYPE)
        telemetry.append({"row_range": [start, stop], **stamp})
        elapsed = time.monotonic() - started
        rate = stop / max(elapsed, 1e-9)
        print(
            f"R0142 {name}: {stop}/{rows} rows ({rate:.1f} rows/s)", flush=True
        )
        if stop >= 20_000 and rate < EMBED_MINIMUM_ROWS_PER_S:
            raise Round0142Error(
                f"{name} throughput {rate:.1f} below {EMBED_MINIMUM_ROWS_PER_S:.1f}"
            )
    path = os.path.join(output, "embeddings.f16.npy")
    atomic_save_new_npy(path, embeddings, immutable=True)
    guard = _stored_array_guard(path, expected_rows=rows)
    wall = time.monotonic() - started
    rate = rows / max(wall, 1e-9)
    receipt = seal({
        "schema": "round0142-common-corpus-raw-jina-probe-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "probe": name,
        "source": source,
        "source_rows": COMMON_CORPUS_ROWS[name],
        "selection": "first min(source_rows, 50000) rows in exact parquet order",
        "selected_source_rows": [0, rows],
        "embedding": expected_input_signature(path),
        "embedding_guard": guard,
        "row_ids": {
            "minimum": 0,
            "maximum": rows - 1,
            "ordered_sha256": ordered_array_sha256(np.arange(rows, dtype=np.int64)),
        },
        "model_members": members,
        "runtime": runtime,
        "prompt_applied": False,
        "prompt_semantics": "raw/unprompted Jina-v5 nano; no Document prefix",
        "canary": expected_input_signature(
            os.path.join(str(job["canary_output"]), "canary.json")
        ),
        "encode_telemetry": telemetry,
        "rows_per_second": rate,
        "warning_below_rows_per_second": EMBED_WARNING_ROWS_PER_S,
        "warning": rate < EMBED_WARNING_ROWS_PER_S,
        "wall_seconds": wall,
        "training_performed": False,
    })
    atomic_write_new_json(os.path.join(output, "receipt.json"), receipt, immutable=True)


def _common_probe(output: str, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    receipt_path = os.path.join(output, "receipt.json")
    receipt = _read_sealed(receipt_path, label=f"R0142 {name} embedding receipt")
    if receipt.get("probe") != name or receipt.get("prompt_applied") is not False:
        raise Round0142Error(f"{name} embedding receipt changed")
    signature = _signature(receipt["embedding"], label=f"{name} embeddings")
    values = np.load(signature["canonical_path"], mmap_mode="r", allow_pickle=False)
    return values, {
        "embedding_receipt": expected_input_signature(receipt_path),
        "embeddings": signature,
        "prompt_applied": False,
        "prompt_semantics": "raw/unprompted Jina-v5 nano",
    }


def _score_one(
    *,
    name: str,
    corpus: np.ndarray,
    queries: np.ndarray,
    corpus_ids: np.ndarray,
    query_ids: np.ndarray,
    control: np.ndarray,
    control_corpus_rows: np.ndarray,
    control_query_rows: np.ndarray,
    model: Any,
    output: str,
    inputs: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    probe = _probe_score(
        name=name,
        corpus=corpus,
        queries=queries,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        model=model,
        output=output,
        inputs=inputs,
        save_coordinates=True,
        duplicate_policy="require-corpus-query-exact-family-disjoint",
    )
    control_report = _probe_score(
        name=f"{name}__fineweb-control",
        corpus=np.asarray(control[control_corpus_rows]),
        queries=np.asarray(control[control_query_rows]),
        corpus_ids=control_corpus_rows,
        query_ids=1_500_000_000 + control_query_rows,
        model=model,
        output=output,
        inputs={
            "embeddings": inputs["control_embeddings"],
            "prompt_applied": False,
            "prompt_semantics": "reviewed raw/unprompted FineWeb heldout",
            "training_membership": "dedicated heldout artifact",
        },
        save_coordinates=True,
        duplicate_policy="require-corpus-query-exact-family-disjoint",
    )
    probe_ffr = float(probe["probe"]["ffr"])
    control_ffr = float(control_report["probe"]["ffr"])
    if control_ffr <= 0:
        raise Round0142Error(f"{name} control FFR is nonpositive")
    retention = probe_ffr / control_ffr
    recall10_control = float(control_report["probe"]["recall_at_10"])
    return {
        "probe": probe,
        "control": control_report,
        "selection": dict(selection),
        "metrics": {
            "probe_ffr": probe_ffr,
            "control_ffr": control_ffr,
            "ffr_retention": retention,
            "recall10_retention": (
                float(probe["probe"]["recall_at_10"]) / recall10_control
                if recall10_control > 0
                else None
            ),
            "verdict": retention_verdict(retention),
        },
    }


def run_score_map(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    map_key = str(job.get("map_key") or "")
    if map_key not in MAP_ORDER:
        raise Round0142Error(f"unknown R0142 map {map_key!r}")
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0142 universality score {map_key}"
    )
    started = time.monotonic()
    model_signature = _signature(job["model"], label=f"{map_key} model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_signature["canonical_path"], device="cuda")
    control_signature = _signature(
        job["control_embeddings"], label="FineWeb heldout control embeddings"
    )
    control = np.load(
        control_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if control.shape != (60_000, DIMENSION) or not np.isfinite(control).all():
        raise Round0142Error("FineWeb control geometry changed")
    control_representatives, control_duplicate_control = canonical_representatives(
        control
    )
    reports: dict[str, Any] = {}
    for index, name in enumerate(PROBE_ORDER):
        if name in COMMON_CORPUS_ROWS:
            values, inputs = _common_probe(str(job["common_outputs"][name]), name)
            corpus_rows, query_rows, split = fixed_single_array_split(
                values, name=name
            )
            corpus = np.asarray(values[corpus_rows])
            queries = np.asarray(values[query_rows])
            corpus_ids = corpus_rows
            query_ids = 1_000_000_000 + query_rows
        elif name == "dadabase":
            source = _signature(job["dadabase"], label="Dadabase embeddings")
            texts = _signature(job["dadabase_texts"], label="Dadabase texts")
            values = np.load(source["canonical_path"], mmap_mode="r", allow_pickle=False)
            corpus_rows, query_rows, split = fixed_single_array_split(
                values, name=name
            )
            corpus = np.asarray(values[corpus_rows])
            queries = np.asarray(values[query_rows])
            corpus_ids = corpus_rows
            query_ids = 1_000_000_000 + query_rows
            inputs = {
                "embeddings": source,
                "texts": texts,
                "prompt_semantics": (
                    "legacy raw Jina-v5 Dadabase artifact; prompt bytes not "
                    "independently sealed"
                ),
                "production_prompt_compatibility_claimed": False,
            }
        else:
            corpus_signature = _signature(
                job["beir"][name]["corpus"], label=f"{name} corpus"
            )
            query_signature = _signature(
                job["beir"][name]["queries"], label=f"{name} queries"
            )
            corpus_ids_signature = _signature(
                job["beir"][name]["corpus_ids"], label=f"{name} corpus IDs"
            )
            query_ids_signature = _signature(
                job["beir"][name]["query_ids"], label=f"{name} query IDs"
            )
            source_corpus = np.load(
                corpus_signature["canonical_path"], mmap_mode="r", allow_pickle=False
            )
            source_queries = np.load(
                query_signature["canonical_path"], mmap_mode="r", allow_pickle=False
            )
            corpus_rows, query_rows, split = fixed_separate_split(
                source_corpus, source_queries, name=name
            )
            corpus = np.asarray(source_corpus[corpus_rows])
            queries = np.asarray(source_queries[query_rows])
            corpus_ids = corpus_rows
            query_ids = 1_000_000_000 + query_rows
            inputs = {
                "corpus_embeddings": corpus_signature,
                "query_embeddings": query_signature,
                "corpus_ids": corpus_ids_signature,
                "query_ids": query_ids_signature,
                "prompt_semantics": (
                    "legacy pooled Jina-v5 artifact; diagnostic raw-map "
                    "compatibility only"
                ),
                "production_prompt_compatibility_claimed": False,
            }
        control_corpus, control_queries, control_split = shape_matched_control_split(
            control,
            name=name,
            corpus_rows=len(corpus_rows),
            query_rows=len(query_rows),
            representatives=control_representatives,
            duplicate_control=control_duplicate_control,
        )
        inputs = {**inputs, "control_embeddings": control_signature}
        reports[name] = _score_one(
            name=name,
            corpus=corpus,
            queries=queries,
            corpus_ids=corpus_ids,
            query_ids=query_ids,
            control=control,
            control_corpus_rows=control_corpus,
            control_query_rows=control_queries,
            model=model,
            output=output,
            inputs=inputs,
            selection={"probe": split, "control": control_split},
        )
        print(
            f"R0142 {map_key} {index + 1}/{len(PROBE_ORDER)} {name}: "
            f"retention={reports[name]['metrics']['ffr_retention']:.4f}",
            flush=True,
        )
        del corpus, queries
        gc.collect()
    panel = seal({
        "schema": "round0142-jina-universality-map-panel-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "map_key": map_key,
        "model": model_signature,
        "probe_order": list(PROBE_ORDER),
        "probes": reports,
        "thresholds": {"pass_at_least": 0.70, "failure_below": 0.50},
        "role": "diagnostic-only; no atlas-quality or production gate",
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "universality-panel.json"), panel, immutable=True
    )


def run_assemble(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0142 immutable retention table"
    )
    started = time.monotonic()
    panels: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for map_key in MAP_ORDER:
        path = os.path.join(
            str(job["map_outputs"][map_key]), "universality-panel.json"
        )
        panel = _read_sealed(path, label=f"R0142 {map_key} panel")
        if panel.get("map_key") != map_key or panel.get("probe_order") != list(
            PROBE_ORDER
        ):
            raise Round0142Error(f"{map_key} universality panel changed")
        panels[map_key] = panel
        for probe in PROBE_ORDER:
            metric = panel["probes"][probe]["metrics"]
            rows.append({"map": map_key, "probe": probe, **metric})
    comparisons = {
        probe: {
            "ffr_retention_25m": float(
                panels[MAP_ORDER[0]]["probes"][probe]["metrics"]["ffr_retention"]
            ),
            "ffr_retention_12p5m": float(
                panels[MAP_ORDER[1]]["probes"][probe]["metrics"]["ffr_retention"]
            ),
        }
        for probe in PROBE_ORDER
    }
    for value in comparisons.values():
        value["delta_25m_minus_12p5m"] = (
            value["ffr_retention_25m"] - value["ffr_retention_12p5m"]
        )
    table = seal({
        "schema": "jina-diverse-universality-panel-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "maps": {
            key: expected_input_signature(
                os.path.join(str(job["map_outputs"][key]), "universality-panel.json")
            )
            for key in MAP_ORDER
        },
        "probe_order": list(PROBE_ORDER),
        "rows": rows,
        "scale_comparisons": comparisons,
        "interpretation": (
            "within-probe FFR divided by an exactly shape-matched canonical "
            "raw-FineWeb control FFR"
        ),
        "role": (
            "diagnostic evidence for B2 predictor modeling; no universal-map, "
            "atlas-quality, prompt-transfer, production, or publishing claim"
        ),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "retention-table.json"), table, immutable=True)


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0142Error("R0142 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "raw_model_canary":
        return run_raw_model_canary(active, job)
    if action == "embed_common_corpus":
        return run_embed_common_corpus(active, job)
    if action == "score_map":
        return run_score_map(active, job)
    if action == "assemble":
        return run_assemble(active, job)
    raise Round0142Error(f"unknown R0142 action: {action!r}")
