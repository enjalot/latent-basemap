"""Execute the conditional prompted-diverse Q3 rung for Round 0169."""
from __future__ import annotations

import gc
import math
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0108_evaluation import (
    HELDOUT_CORPUS_ROWS,
    HELDOUT_QUERY_ROWS,
    IN_MIX_LANGUAGES,
    POLISH,
)
from basemap.round0104_training import L2NormalizedArray
from basemap.round0105_search import GROUPS
from basemap.round0168_prompted_diverse_staging import (
    CAPABILITY as STAGING_CAPABILITY,
    MANIFEST_SCHEMA as STAGING_SCHEMA,
)
from basemap.round0171_prompted_8m import CAPABILITY as Q2_CAPABILITY
from basemap.round0173_prompted_ood_pack import (
    CAPABILITY as OOD_PACK_CAPABILITY_REGISTERED,
    LANGUAGE_PROBE_SCHEMA as OOD_PACK_PROBE_SCHEMA,
    OOD_AUDIT_SCHEMA as OOD_PACK_AUDIT_SCHEMA,
    ROUND_ID as OOD_PACK_ROUND_ID,
)
from basemap.round0169_prompted_diverse import (
    CAPABILITY,
    DIMENSION,
    DiversePromptTrainingInput,
    GRAPH_K,
    GRAPH_EXECUTION,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE,
    GRAPH_NPROBE_GRID,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_QUALITY_ROWS,
    GRAPH_QUALITY_SEED,
    GRAPH_TRAIN_ROWS,
    GRAPH_TRAIN_SEED,
    HOST_RSS_LIMIT_GIB,
    ROUND_ID,
    ROWS,
    SEED,
    SUCCESSFUL_UPDATES,
    Round0169Error,
    diverse_train_config,
    prompted_diverse_decision,
)
from basemap.round0160_prompted_seed_family import METRICS, metric_view
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0166_nodes as q2
from experiments import round0167_nodes as prompted
from experiments import round0108_nodes as ood_nodes
from experiments.round0116_nodes import (
    _encode_document,
    _float32_norm_guard,
    _load_document_model,
    _prompt_equivalence,
    _stored_array_guard,
)
from basemap.round0116_prompted_corpus import validate_environment_freeze
from basemap.round0087_inventory import _fingerprint_fp16


GRAPH_SCHEMA = "round0169-prompted-diverse-u12-fuzzy-graph-v1"
TRAIN_SCHEMA = "round0169-prompted-diverse-u12-train-receipt-v1"
PRODUCTION_CONFIG_SCHEMA = "round0169-prompted-diverse-u12-production-config-v1"
PROMPT_PREFIX = "Document: "
LANGUAGES = (*IN_MIX_LANGUAGES, POLISH)
GRAPH_SHARD_ROWS = 4_000_000
GRAPH_INDEX_DESCRIPTION = (
    "four row-disjoint GPU IndexIVFFlat/IP shards with fp32 vector storage, "
    "one shared coarse quantizer, and exact global top-k merge"
)
EXPECTED_GRAPH_SHARDS = (
    (0, 4_000_000),
    (4_000_000, 8_000_000),
    (8_000_000, 12_000_000),
    (12_000_000, ROWS),
)
CANARY_SCHEMA = "round0169-prompt-model-canary-v1"
LANGUAGE_PROBE_SCHEMA = OOD_PACK_PROBE_SCHEMA
LANGUAGE_RECEIPT_ROUND_ID = OOD_PACK_ROUND_ID
OOD_AUDIT_SCHEMA = OOD_PACK_AUDIT_SCHEMA
OOD_PACK_CAPABILITY: str | None = OOD_PACK_CAPABILITY_REGISTERED


class PromptedDiverseScaleArray(L2NormalizedArray):
    """Lazy normalized Q3 matrix with an exact, self-contained scale identity."""

    round0169_prompted_diverse_view = True

    def __init__(
        self,
        source: np.ndarray,
        *,
        population: Mapping[str, Any],
        population_signature: Mapping[str, Any],
    ) -> None:
        super().__init__(source)
        if source.shape != (ROWS, DIMENSION) or source.dtype != np.float16:
            raise Round0169Error("R0169 scale view source geometry changed")
        self._scale_identity = _seal({
            "schema": "round0169-prompted-diverse-scale-input-v1",
            "round_id": ROUND_ID,
            "row_count": ROWS,
            "dimensions": DIMENSION,
            "source": dict(population["document_compact"]),
            "mapping": dict(population["mapping"]),
            "staging": dict(population_signature),
            "staging_identity_sha256": str(population["staging_identity_sha256"]),
            "population_law": "exact accepted R0132 U12 compact order",
            "duplicate_policy": (
                "exact duplicate families are diagnostic metadata only; "
                "the accepted U12 population is unchanged"
            ),
        })

    def scale_admission_identity(self) -> dict[str, Any]:
        return dict(self._scale_identity)


def _read_population(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = dict(job["staging_manifest"])
    path = prompt_contract.verify_signature(signature, label="accepted R0168 staging")
    staging = prompt_contract.read_sealed(path, label="accepted R0168 staging")
    population = staging.get("population") or {}
    materialization = staging.get("materialization") or {}
    if (
        staging.get("schema") != STAGING_SCHEMA
        or staging.get("round_id") != "0168"
        or staging.get("capability") != STAGING_CAPABILITY
        or int(staging.get("rows", -1)) != ROWS
        or int(staging.get("dimension", -1)) != DIMENSION
        or staging.get("dtype") != "<f2"
        or staging.get("embedding_convention") != "Document: "
        or population.get("exact_r0132_population_match") is not True
        or population.get("polish_held_out") is not True
        or materialization.get("contiguous") is not True
        or materialization.get("immutable") is not True
        or staging.get("graph_built") is not False
        or staging.get("training_performed") is not False
    ):
        raise Round0169Error("accepted R0168 staging contract changed")
    for key, value in (
        ("host_fp16", staging.get("host_fp16")),
        ("mapping", population.get("mapping")),
    ):
        if not isinstance(value, Mapping):
            raise Round0169Error(f"accepted R0168 {key} binding is missing")
        prompt_contract.verify_signature(value, label=f"accepted R0168 {key}")
    return {
        "retained_rows": ROWS,
        "dimension": DIMENSION,
        "document_compact": dict(staging["host_fp16"]),
        "mapping": dict(population["mapping"]),
        "staging_manifest": signature,
        "staging_identity_sha256": staging["identity_sha256"],
        "duplicate_control": staging["duplicate_control"],
    }, signature


def _open_source(population: Mapping[str, Any]) -> np.ndarray:
    signature = dict(population["document_compact"])
    path = prompt_contract.verify_signature(signature, label="R0168 prompted U12 matrix")
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        source.shape != (ROWS, DIMENSION)
        or source.dtype != np.float16
        or not source.flags.c_contiguous
        or int(source.nbytes) + 128 != int(signature["bytes"])
    ):
        raise Round0169Error("R0168 prompted U12 matrix geometry changed")
    return source


def _data_identity(population: Mapping[str, Any]) -> dict[str, Any]:
    source = population["document_compact"]
    return {
        "kind": "ordered_shards",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("<f2").str,
        "shards": [{
            "position": 0,
            "name": os.path.basename(str(source["canonical_path"])),
            "bytes": int(source["bytes"]),
            "sha256": str(source["sha256"]),
        }],
    }


def _fp32_gpu_options(faiss: Any) -> Any:
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    return options


def _graph_execution_ok(graph: Mapping[str, Any]) -> bool:
    search = graph.get("search_qualification") or {}
    execution = search.get("execution") or {}
    shards = execution.get("shards") or []
    observed = tuple(
        (
            int(item.get("start", -1)),
            int(item.get("stop", -1)),
            int(item.get("rows", -1)),
            int(item.get("ntotal", -1)),
        )
        for item in shards
        if isinstance(item, Mapping)
    )
    expected = tuple(
        (start, stop, stop - start, stop - start)
        for start, stop in EXPECTED_GRAPH_SHARDS
    )
    return bool(
        search.get("index") == GRAPH_INDEX_DESCRIPTION
        and int(execution.get("shard_rows", -1)) == GRAPH_SHARD_ROWS
        and execution.get("coarse_quantizer")
        == "one shared trained IVF8192 template"
        and observed == expected
    )


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return prompt_contract.seal(dict(body))


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    value = expected_input_signature(str(expected.get("canonical_path") or ""))
    if value != dict(expected):
        raise Round0169Error(f"{label} bytes changed")
    return value


def run_prompt_canary(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0169Error("R0169 prompt canary received another queue")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0169 prompt canary")
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    text_signature = _signature(job["canary_text"], label="R0169 canary text")
    document_signature = _signature(
        job["canary_document"], label="accepted R0114 prompted canary"
    )
    positions = np.asarray(job["canary_positions"], dtype=np.int64)
    if positions.shape != (32,) or np.any(positions < 0):
        raise Round0169Error("R0169 canary positions changed")
    texts = prompted._parquet_texts(
        text_signature, column="chunk_text", rows=positions, label="R0169 canary"
    )
    historical = np.asarray(
        np.load(
            document_signature["canonical_path"], mmap_mode="r", allow_pickle=False
        )[positions],
        dtype=np.float32,
    )
    model, runtime, members = _load_document_model()
    prompted._verify_model_members(members, job["model_members"])
    equivalence = _prompt_equivalence(model, texts)
    fresh, telemetry = _encode_document(
        model, [PROMPT_PREFIX + text for text in texts]
    )
    _float32_norm_guard(fresh, label="R0169 prompted canary")
    cosine = prompted._cosine_rows(fresh, historical)
    if float(cosine.mean()) < 0.995 or float(cosine.min()) < 0.99:
        raise Round0169Error("R0169 prompt execution does not reproduce R0114")
    receipt = _seal({
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
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "canary.json"), receipt, immutable=True)


def _read_canary(output: str) -> dict[str, Any]:
    path = os.path.join(output, "canary.json")
    receipt = prompt_contract.read_sealed(path, label="R0169 prompt canary")
    if (
        receipt.get("schema") != CANARY_SCHEMA
        or receipt.get("round_id") != LANGUAGE_RECEIPT_ROUND_ID
        or receipt.get("passed") is not True
        or receipt.get("prompt_applied") is not True
        or receipt.get("prompt_prefix") != PROMPT_PREFIX
    ):
        raise Round0169Error("R0169 prompt canary did not pass")
    return receipt


def run_embed_language(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    language = str(job.get("language") or "")
    if language not in LANGUAGES:
        raise Round0169Error(f"unknown R0169 language {language!r}")
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0169Error("R0169 language embed received another queue")
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0169 {language} prompted OOD"
    )
    started = time.monotonic()
    validate_environment_freeze(dict(job["environment_freeze"]))
    _read_canary(str(job["canary_output"]))
    selection = _signature(job["selection"], label="accepted R0108 selectors")
    with np.load(selection["canonical_path"], allow_pickle=False) as selected:
        corpus_rows = np.asarray(selected[f"{language}__corpus"], dtype=np.int64)
        query_rows = np.asarray(selected[f"{language}__queries"], dtype=np.int64)
        original_queries = np.asarray(
            selected[f"{language}__original_queries"], dtype=np.int64
        )
        replacement_mask = np.asarray(
            selected[f"{language}__query_replacement_mask"], dtype=bool
        )
    if (
        corpus_rows.shape != (HELDOUT_CORPUS_ROWS,)
        or query_rows.shape != (HELDOUT_QUERY_ROWS,)
        or original_queries.shape != query_rows.shape
        or replacement_mask.shape != query_rows.shape
        or not np.array_equal(original_queries != query_rows, replacement_mask)
        or np.intersect1d(corpus_rows, query_rows).size
    ):
        raise Round0169Error(f"{language} accepted R0108 selector changed")
    text_source = _signature(job["text_source"], label=f"{language} text source")
    corpus_texts = prompted._parquet_texts(
        text_source,
        column="chunk_text",
        rows=corpus_rows,
        label=f"R0169 {language} corpus",
    )
    query_texts = prompted._parquet_texts(
        text_source,
        column="chunk_text",
        rows=query_rows,
        label=f"R0169 {language} queries",
    )
    model, runtime, members = _load_document_model()
    prompted._verify_model_members(members, job["model_members"])
    corpus, corpus_telemetry, corpus_wall = prompted._encode_prompted(
        model, corpus_texts, label=f"R0169 {language} corpus"
    )
    queries, query_telemetry, query_wall = prompted._encode_prompted(
        model, query_texts, label=f"R0169 {language} queries"
    )
    split_audit = prompted._exact_family_audit(corpus, queries)
    corpus_path = os.path.join(output, "corpus.f16.npy")
    query_path = os.path.join(output, "queries.f16.npy")
    corpus_rows_path = os.path.join(output, "corpus-source-rows.i64.npy")
    query_rows_path = os.path.join(output, "query-source-rows.i64.npy")
    atomic_save_new_npy(corpus_path, corpus, immutable=True)
    atomic_save_new_npy(query_path, queries, immutable=True)
    atomic_save_new_npy(corpus_rows_path, corpus_rows, immutable=True)
    atomic_save_new_npy(query_rows_path, query_rows, immutable=True)
    embed_wall = corpus_wall + query_wall
    receipt = _seal({
        "schema": LANGUAGE_PROBE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "language": language,
        "prompt_applied": True,
        "prompt_prefix": PROMPT_PREFIX,
        "selection": selection,
        "text_source": text_source,
        "corpus_embeddings": expected_input_signature(corpus_path),
        "query_embeddings": expected_input_signature(query_path),
        "corpus_source_rows": expected_input_signature(corpus_rows_path),
        "query_source_rows": expected_input_signature(query_rows_path),
        "corpus_rows_sha256": ordered_array_sha256(corpus_rows),
        "query_rows_sha256": ordered_array_sha256(query_rows),
        "corpus_text_sha256": prompted._text_sha256(corpus_texts),
        "query_text_sha256": prompted._text_sha256(query_texts),
        "corpus_guard": _stored_array_guard(
            corpus_path, expected_rows=HELDOUT_CORPUS_ROWS
        ),
        "query_guard": _stored_array_guard(
            query_path, expected_rows=HELDOUT_QUERY_ROWS
        ),
        "corpus_query_exact_family_audit": split_audit,
        "accepted_raw_selector_replacements": int(replacement_mask.sum()),
        "model_members": members,
        "runtime": runtime,
        "encode_telemetry": {
            "corpus": corpus_telemetry,
            "queries": query_telemetry,
        },
        "rows_per_second": (
            (HELDOUT_CORPUS_ROWS + HELDOUT_QUERY_ROWS) / max(embed_wall, 1e-9)
        ),
        "training_membership": "pending exact prompted-family audit before graph/train",
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "receipt.json"), receipt, immutable=True)


def _load_language_probe(
    output: str, language: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    receipt_path = os.path.join(output, "receipt.json")
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0169 {language} prompted probe"
    )
    if (
        receipt.get("schema") != LANGUAGE_PROBE_SCHEMA
        or receipt.get("round_id") != LANGUAGE_RECEIPT_ROUND_ID
        or receipt.get("language") != language
        or receipt.get("prompt_applied") is not True
    ):
        raise Round0169Error(f"R0169 {language} prompted probe changed")
    signatures = {
        key: _signature(receipt[key], label=f"R0169 {language} {key}")
        for key in (
            "corpus_embeddings",
            "query_embeddings",
            "corpus_source_rows",
            "query_source_rows",
        )
    }
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
        raise Round0169Error(f"R0169 {language} prompted probe geometry changed")
    return corpus, queries, corpus_rows, query_rows, {
        "receipt": expected_input_signature(receipt_path),
        **signatures,
    }


def _fingerprints(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(values)
    h0 = np.empty(rows, dtype=np.uint64)
    h1 = np.empty(rows, dtype=np.uint64)
    zero = np.empty(rows, dtype=bool)
    nonfinite = np.empty(rows, dtype=bool)
    bits = np.ascontiguousarray(values).view("<u2")
    _fingerprint_fp16(bits, h0, h1, zero, nonfinite)
    return h0, h1, zero, nonfinite


def run_audit_probe_training_disjoint(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> None:
    """Prove every prompted OOD row is outside Q3 training by exact fp16 bytes."""
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0169Error("R0169 OOD audit received another queue")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0169 prompted OOD training audit"
    )
    started = time.monotonic()
    population, population_signature = _read_population(job)
    source = _open_source(population)
    pair_dtype = np.dtype([("h0", "<u8"), ("h1", "<u8")])
    total_probe_rows = len(LANGUAGES) * (HELDOUT_CORPUS_ROWS + HELDOUT_QUERY_ROWS)
    probe_pairs = np.empty(total_probe_rows, dtype=pair_dtype)
    entries: list[dict[str, Any]] = []
    language_artifacts: dict[str, dict[str, Any]] = {}
    cursor = 0
    zero_rows = 0
    nonfinite_rows = 0
    for language in LANGUAGES:
        corpus, queries, corpus_rows, query_rows, signatures = _load_language_probe(
            str(job["language_outputs"][language]), language
        )
        language_artifacts[language] = signatures
        for split, values, source_rows in (
            ("corpus", corpus, corpus_rows),
            ("queries", queries, query_rows),
        ):
            h0, h1, zero, nonfinite = _fingerprints(values)
            stop = cursor + len(values)
            probe_pairs["h0"][cursor:stop] = h0
            probe_pairs["h1"][cursor:stop] = h1
            entries.append({
                "language": language,
                "split": split,
                "start": cursor,
                "stop": stop,
                "values": values,
                "source_rows": source_rows,
                "signatures": signatures,
            })
            cursor = stop
            zero_rows += int(zero.sum())
            nonfinite_rows += int(nonfinite.sum())
    if cursor != total_probe_rows or zero_rows or nonfinite_rows:
        raise Round0169Error("R0169 prompted probe fingerprint population is invalid")
    unique_probe_pairs = np.unique(probe_pairs)
    fingerprint_candidates: dict[tuple[int, int], list[tuple[int, bytes]]] = {}
    block_rows = 65_536
    for start in range(0, ROWS, block_rows):
        stop = min(start + block_rows, ROWS)
        block = np.asarray(source[start:stop])
        h0, h1, zero, nonfinite = _fingerprints(block)
        if np.any(zero) or np.any(nonfinite):
            raise Round0169Error("R0169 prompted training source became invalid")
        block_pairs = np.empty(len(block), dtype=pair_dtype)
        block_pairs["h0"] = h0
        block_pairs["h1"] = h1
        positions = np.searchsorted(unique_probe_pairs, block_pairs)
        in_range = positions < len(unique_probe_pairs)
        hits = np.zeros(len(block), dtype=bool)
        if np.any(in_range):
            hits[in_range] = unique_probe_pairs[positions[in_range]] == block_pairs[in_range]
        for local in np.flatnonzero(hits).tolist():
            key = (int(h0[local]), int(h1[local]))
            fingerprint_candidates.setdefault(key, []).append(
                (start + local, np.asarray(block[local]).tobytes(order="C"))
            )
        if sum(len(value) for value in fingerprint_candidates.values()) > 100_000:
            raise Round0169Error("R0169 OOD/training fingerprint candidate count is implausible")
    exact_overlaps: list[dict[str, Any]] = []
    fingerprint_hits = set(fingerprint_candidates)
    if fingerprint_hits:
        for entry in entries:
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
                    for training_row, training_raw in fingerprint_candidates[key]:
                        if raw == training_raw:
                            exact_overlaps.append({
                                "language": entry["language"],
                                "split": entry["split"],
                                "source_row": int(source_rows[start + local]),
                                "training_compact_row": int(training_row),
                            })
    receipt = _seal({
        "schema": OOD_AUDIT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "population": population_signature,
        "probe_rows": total_probe_rows,
        "unique_probe_fingerprints": int(len(unique_probe_pairs)),
        "duplicate_probe_rows": int(total_probe_rows - len(unique_probe_pairs)),
        "training_rows": ROWS,
        "fingerprint_candidate_training_rows": int(
            sum(len(value) for value in fingerprint_candidates.values())
        ),
        "exact_training_family_overlaps": exact_overlaps,
        "exact_training_family_overlap_count": len(exact_overlaps),
        "fingerprint_collision_candidates": int(
            sum(len(value) for value in fingerprint_candidates.values())
            - len({item["training_compact_row"] for item in exact_overlaps})
        ),
        "identity": "complete stored prompted-fp16 row bytes",
        "passed": not exact_overlaps,
        "capabilities": [OOD_PACK_CAPABILITY]
        if not exact_overlaps and OOD_PACK_CAPABILITY
        else [],
        "language_outputs": language_artifacts,
        "prompt_canary": expected_input_signature(
            os.path.join(str(job["canary_output"]), "canary.json")
        ),
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "audit.json"), receipt, immutable=True)
    if exact_overlaps:
        raise Round0169Error(
            f"R0169 OOD rows contain {len(exact_overlaps)} exact prompted training copies"
        )


def _matched_2m_panel(
    *,
    model: Any,
    family: Mapping[str, Any],
    train_signature: Mapping[str, Any],
    output: str,
) -> dict[str, Any]:
    """Run Q2's byte-identical 2M retention panel for the Q3-trained model."""
    from basemap.panel_v2 import load_hiD_reference, load_query_truth, score_panel

    cfg = prompt_contract.panel_config()
    seed42 = family["cells"]["seed42"]
    baseline_metrics = {
        metric: float(seed42["decision_metrics"][metric]) for metric in METRICS
    }
    accepted_score_signature = dict(seed42["native_score"])
    accepted_score = q2._read_sealed(
        accepted_score_signature, label="accepted R0160 seed-42 native score"
    )
    if (
        accepted_score.get("round_id") != "0115"
        or accepted_score.get("arm") != "document"
        or int(accepted_score.get("training_seed", 42)) != 42
    ):
        raise Round0169Error("accepted prompted seed-42 score changed")
    source_signature = dict(family["lineage"]["document_compact"])
    source_path = prompt_contract.verify_signature(
        source_signature, label="accepted R0113 prompted compact matrix"
    )
    source_raw = np.memmap(
        source_path,
        mode="r",
        dtype="<f2",
        shape=(q2.MATCHED_ROWS, DIMENSION),
    )
    source = L2NormalizedArray(source_raw)
    coordinates = np.asarray(
        model.transform(source_raw, batch_size=8192), dtype=np.float32
    )
    accepted_query = q2._read_sealed(
        accepted_score["query_reserve"], label="accepted R0113 query reserve"
    )
    accepted_selection = q2._read_sealed(
        accepted_score["query_selection"], label="accepted seed-42 query selection"
    )
    positions = np.load(
        prompt_contract.verify_signature(
            accepted_selection["positions"], label="accepted query positions"
        ),
        allow_pickle=False,
    )
    reserve = np.load(
        prompt_contract.verify_signature(
            accepted_query["outputs"]["document"],
            label="accepted prompted query reserve",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        positions.shape != (q2.QUERY_ROWS,)
        or positions.dtype != np.int64
        or np.any(positions[1:] <= positions[:-1])
        or reserve.shape != (q2.QUERY_CANDIDATES, DIMENSION)
        or reserve.dtype != np.float16
    ):
        raise Round0169Error("accepted matched query selection changed")
    query_values = np.asarray(reserve[positions], dtype=np.float16)
    query_coordinates = np.asarray(
        model.transform(query_values, batch_size=8192), dtype=np.float32
    )
    if (
        coordinates.shape != (q2.MATCHED_ROWS, 2)
        or query_coordinates.shape != (q2.QUERY_ROWS, 2)
        or not np.isfinite(coordinates).all()
        or not np.isfinite(query_coordinates).all()
    ):
        raise Round0169Error("R0169 matched-2M transform output is invalid")
    coordinate_path = os.path.join(output, "matched-2m-coordinates.npy")
    query_coordinate_path = os.path.join(output, "matched-2m-query-coordinates.npy")
    atomic_save_new_npy(coordinate_path, coordinates, immutable=True)
    atomic_save_new_npy(query_coordinate_path, query_coordinates, immutable=True)
    centroids = q2._centroids(family["centroids"], label="accepted R0160")
    reference = load_hiD_reference(
        prompt_contract.verify_signature(
            family["shared_prompted_reference"],
            label="accepted R0160 prompted high-D reference",
        )
    )
    assembly = q2._read_sealed(
        family["lineage"]["assembly"], label="accepted R0113 compact assembly"
    )
    reference_identity = {
        "data_identity": q2.prompt_nodes._data_identity(assembly, arm="document"),
        "convention": {
            "row_order": "R0113 shared union-representative compact order",
            "distance": "cosine via fp32-L2-normalized squared L2",
            "self_exclusion": True,
            "anchor_namespace": "R0113 compact IDs",
            "embedding_prompt": "document",
        },
    }
    panel = score_panel(
        source,
        coordinates,
        config=cfg,
        centroids_by_k=centroids,
        hiD_reference=reference,
        reference_identity=reference_identity,
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "universe": "accepted-r0113-matched-2m",
            "source": source_signature,
            "train_receipt": dict(train_signature),
            "coordinates": expected_input_signature(coordinate_path),
        },
    )
    truth_signature = dict(accepted_score["combined_query_truth"])
    truth = load_query_truth(
        prompt_contract.verify_signature(
            truth_signature, label="accepted R0115 combined query truth"
        )
    )
    truth_range = accepted_score["projections"]["matched"]["truth_row_range"]
    if truth_range != [0, q2.QUERY_ROWS] or truth["corpus_cardinality"] != q2.MATCHED_ROWS:
        raise Round0169Error("accepted matched query truth changed")
    projection = q2._projection_metrics(
        high10=np.asarray(truth["neighbors"][: q2.QUERY_ROWS], dtype=np.int64),
        query_coordinates=query_coordinates,
        coordinates=coordinates,
        cfg=cfg,
        truth_signature=truth_signature,
        truth_row_range=truth_range,
    )
    metrics = metric_view(
        panel=panel, native_score={"projections": {"matched": projection}}
    )
    return {
        "source": source_signature,
        "coordinates": expected_input_signature(coordinate_path),
        "query_coordinates": expected_input_signature(query_coordinate_path),
        "accepted_seed42_score": accepted_score_signature,
        "accepted_query_truth": truth_signature,
        "panel": panel,
        "projection": projection,
        "decision_metrics": metrics,
        "baseline_seed42_metrics": baseline_metrics,
    }


def _native_training_alignment(
    *,
    model: Any,
    source: np.ndarray,
    coordinates: np.ndarray,
    language_outputs: Mapping[str, str],
    output: str,
) -> dict[str, Any]:
    """Score fixed balanced in-mix and Polish queries against the native atlas."""
    from basemap.panel_v2 import cross_knn
    from basemap.round0108_evaluation import projection_metrics
    from basemap.round0108_evaluation import exact_cosine_topk

    selected: list[np.ndarray] = []
    selected_languages: list[np.ndarray] = []
    for index, language in enumerate(IN_MIX_LANGUAGES):
        _corpus, queries, _corpus_rows, _query_rows, _signatures = _load_language_probe(
            str(language_outputs[language]), language
        )
        count = (
            HELDOUT_QUERY_ROWS // len(IN_MIX_LANGUAGES)
            + (1 if index < HELDOUT_QUERY_ROWS % len(IN_MIX_LANGUAGES) else 0)
        )
        selected.append(np.asarray(queries[:count], dtype=np.float16))
        selected_languages.append(np.full(count, language, dtype="U16"))
    in_mix = np.concatenate(selected, axis=0)
    in_mix_languages = np.concatenate(selected_languages)
    _corpus, polish, _corpus_rows, polish_rows, _signatures = _load_language_probe(
        str(language_outputs[POLISH]), POLISH
    )
    if in_mix.shape != (HELDOUT_QUERY_ROWS, DIMENSION) or polish.shape != (
        HELDOUT_QUERY_ROWS,
        DIMENSION,
    ):
        raise Round0169Error("R0169 native alignment query stack changed")
    queries = np.concatenate((in_mix, np.asarray(polish)), axis=0)
    exact, exact_guard = exact_cosine_topk(
        queries, L2NormalizedArray(source), k=10, candidate_block_rows=100_000
    )
    query_coordinates = np.asarray(
        model.transform(queries, batch_size=8192), dtype=np.float32
    )
    cfg = prompt_contract.panel_config()
    fraction_k = max(50, int(math.ceil(cfg.frac * len(coordinates))))
    low = cross_knn(
        query_coordinates,
        coordinates,
        fraction_k,
        cfg,
        hi_dim=False,
        exact=True,
    )
    split = HELDOUT_QUERY_ROWS
    in_mix_metrics = projection_metrics(
        exact[:split], low[:split], fraction_k=fraction_k
    )
    polish_metrics = projection_metrics(
        exact[split:], low[split:], fraction_k=fraction_k
    )
    arrays_path = os.path.join(output, "native-training-alignment.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        in_mix_query_coordinates=query_coordinates[:split],
        in_mix_query_languages=in_mix_languages,
        in_mix_exact_training_top10=exact[:split],
        in_mix_low_training_top50=low[:split, :50],
        polish_query_coordinates=query_coordinates[split:],
        polish_query_source_rows=np.asarray(polish_rows, dtype=np.int64),
        polish_exact_training_top10=exact[split:],
        polish_low_training_top50=low[split:, :50],
    )
    return {
        "role": "diagnostic-only native-atlas attachment",
        "selection": "first balanced 500 accepted R0108 in-mix queries plus all 500 Polish queries",
        "in_mix_balanced_500": {
            "ffr": float(in_mix_metrics["ffr_diagnostic"]),
            "recall_at_10": float(in_mix_metrics["recall_at_10"]),
            "recall_at_50_of_high10": float(
                in_mix_metrics["recall_at_50_of_high10"]
            ),
        },
        "polish_500": {
            "ffr": float(polish_metrics["ffr_diagnostic"]),
            "recall_at_10": float(polish_metrics["recall_at_10"]),
            "recall_at_50_of_high10": float(
                polish_metrics["recall_at_50_of_high10"]
            ),
        },
        "exact_high_d_search": exact_guard,
        "fraction_k": fraction_k,
        "arrays": expected_input_signature(arrays_path),
    }


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import load_hiD_reference, score_panel

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0169Error("R0169 evaluation received another queue")
    _configure_q2_kernel()
    population, population_signature = _read_population(job)
    family, gates, floors = q2._read_family_and_gates(job)
    model, train, train_signature, graph = q2._authenticate_model(
        job, population, population_signature
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0169 prompted diverse evaluation"
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats("cuda")
    q2_evaluation_signature = _signature(
        job["q2_evaluation"], label="accepted positive R0171 evaluation"
    )
    q2_evaluation = prompt_contract.read_sealed(
        q2_evaluation_signature["canonical_path"],
        label="accepted positive R0171 evaluation",
    )
    if (
        q2_evaluation.get("round_id") != "0171"
        or (q2_evaluation.get("decision") or {}).get("passed") is not True
        or (q2_evaluation.get("decision") or {}).get("outcome")
        != "prompted-english-8m-scale-rung-qualified"
        or q2_evaluation.get("capabilities") != [Q2_CAPABILITY]
    ):
        raise Round0169Error("R0169 requires a positive accepted Q2 evaluation")
    audit_path = os.path.join(str(job["ood_audit_output"]), "audit.json")
    audit_signature = expected_input_signature(audit_path)
    audit = prompt_contract.read_sealed(audit_path, label="R0169 OOD training audit")
    if (
        audit.get("schema") != OOD_AUDIT_SCHEMA
        or audit.get("round_id") != OOD_PACK_ROUND_ID
        or audit.get("passed") is not True
        or audit.get("capabilities") != [OOD_PACK_CAPABILITY_REGISTERED]
        or audit.get("population") != population_signature
        or int(audit.get("probe_rows", -1)) != len(LANGUAGES) * 50_000
        or int(audit.get("exact_training_family_overlap_count", -1)) != 0
    ):
        raise Round0169Error("R0169 OOD training-disjoint audit changed")
    audit_languages = audit.get("language_outputs") or {}
    if set(audit_languages) != set(LANGUAGES):
        raise Round0169Error("R0169 OOD probe-pack language set changed")
    for language in LANGUAGES:
        receipt = expected_input_signature(
            os.path.join(str(job["language_outputs"][language]), "receipt.json")
        )
        if (audit_languages[language] or {}).get("receipt") != receipt:
            raise Round0169Error(
                f"R0169 {language} probe path differs from reviewed OOD pack"
            )

    if not _graph_execution_ok(graph):
        raise Round0169Error("R0169 graph execution contract changed")

    source_raw = _open_source(population)
    source = PromptedDiverseScaleArray(
        source_raw,
        population=population,
        population_signature=population_signature,
    )
    coordinates = np.asarray(
        model.transform(source_raw, batch_size=8192), dtype=np.float32
    )
    if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
        raise Round0169Error("R0169 native transform output is invalid")
    coordinate_path = os.path.join(output, "native-u12-coordinates.npy")
    atomic_save_new_npy(coordinate_path, coordinates, immutable=True)
    group_signature = _signature(job["group_ids"], label="accepted R0132 group IDs")
    group_ids = np.load(
        group_signature["canonical_path"], mmap_mode="r", allow_pickle=False
    )
    if (
        group_ids.shape != (ROWS,)
        or group_ids.dtype != np.uint8
        or set(np.unique(group_ids).tolist()) != set(range(len(GROUPS)))
    ):
        raise Round0169Error("accepted R0132 group IDs changed")
    native_centroids = q2._centroids(graph["centroids"], label="R0169 native")
    native_reference = load_hiD_reference(
        prompt_contract.verify_signature(
            graph["high_d_reference"], label="R0169 native high-D reference"
        ),
        expected_key=str(graph["high_d_reference_key"]),
    )
    anchor_ids = np.asarray(native_reference["anchor_ids"], dtype=np.int64)
    anchor_groups = np.asarray(
        [GROUPS[int(value)] for value in group_ids[anchor_ids]], dtype="U80"
    )
    native_panel = score_panel(
        source,
        coordinates,
        config=prompt_contract.panel_config(),
        centroids_by_k=native_centroids,
        hiD_reference=native_reference,
        reference_identity=graph["reference_identity"],
        ffr_group_labels=anchor_groups,
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "universe": "exact-r0132-u12-prompted",
            "population": population_signature,
            "group_ids": group_signature,
            "train_receipt": train_signature,
            "coordinates": expected_input_signature(coordinate_path),
        },
    )
    group_cells = native_panel.get("ffr_by_group") or {}
    if set(group_cells) != set(GROUPS) or any(
        int(group_cells[name].get("anchors", 0)) <= 0 for name in GROUPS
    ):
        raise Round0169Error("R0169 native group FFR cells are incomplete")
    group_ffr = {name: float(group_cells[name]["ffr"]) for name in GROUPS}
    native_alignment = _native_training_alignment(
        model=model,
        source=source_raw,
        coordinates=coordinates,
        language_outputs=job["language_outputs"],
        output=output,
    )
    native_projection = native_alignment["in_mix_balanced_500"]
    native_metrics = metric_view(
        panel=native_panel,
        native_score={"projections": {"matched": native_projection}},
    )
    matched = _matched_2m_panel(
        model=model,
        family=family,
        train_signature=train_signature,
        output=output,
    )

    ood_reports: dict[str, Any] = {}
    for language in LANGUAGES:
        corpus, queries, corpus_rows, query_rows, signatures = _load_language_probe(
            str(job["language_outputs"][language]), language
        )
        ood_reports[language] = ood_nodes._probe_score(
            name=f"prompted-{language}",
            corpus=np.asarray(corpus),
            queries=np.asarray(queries),
            corpus_ids=np.asarray(corpus_rows, dtype=np.int64),
            query_ids=1_000_000_000 + np.asarray(query_rows, dtype=np.int64),
            model=model,
            output=output,
            inputs={
                **signatures,
                "prompt_applied": True,
                "prompt_prefix": PROMPT_PREFIX,
                "training_disjoint_audit": audit_signature,
            },
            save_coordinates=True,
            duplicate_policy="require-corpus-query-exact-family-disjoint",
        )
        gc.collect()
    in_mix_recall50 = [
        float(ood_reports[language]["probe"]["recall_at_50_of_high10"])
        for language in IN_MIX_LANGUAGES
    ]
    prompted_ood = {
        "polish_recall_at_50_of_high10": float(
            ood_reports[POLISH]["probe"]["recall_at_50_of_high10"]
        ),
        "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_recall50)),
    }
    raw_signature = _signature(job["raw_r0132_ood"], label="accepted R0132 OOD")
    raw = prompt_contract.read_sealed(
        raw_signature["canonical_path"], label="accepted R0132 OOD"
    )
    if (
        raw.get("schema") != "round0132-matched-ood-scale-panel-v1"
        or raw.get("round_id") != "0132"
        or set(raw.get("control_12p5m") or {})
        != {
            "fineweb_recall_at_50_of_high10",
            "in_mix_median_recall_at_50_of_high10",
            "polish_recall_at_50_of_high10",
        }
    ):
        raise Round0169Error("accepted R0132 OOD control changed")
    raw_control = raw.get("control_12p5m") or {}
    raw_ood = {
        "polish_recall_at_50_of_high10": float(
            raw_control["polish_recall_at_50_of_high10"]
        ),
        "in_mix_median_recall_at_50_of_high10": float(
            raw_control["in_mix_median_recall_at_50_of_high10"]
        ),
    }
    decision = prompted_diverse_decision(
        native=native_metrics,
        matched_2m=matched["decision_metrics"],
        baseline_2m_seed42=matched["baseline_seed42_metrics"],
        prompted_floors=floors,
        group_ffr=group_ffr,
        prompted_ood=prompted_ood,
        raw_r0132_ood=raw_ood,
    )
    execution_gates = {
        "q2_positive_reviewed_evaluation_bound": True,
        "train_receipt_closes": all(
            bool(value) for value in (train.get("train_checks") or {}).values()
        ),
        "graph_fixed_nprobe_qualified": (
            ((graph.get("search_qualification") or {}).get("cells") or {})
            .get(str(GRAPH_NPROBE), {})
            .get("passed")
            is True
        ),
        "graph_uses_registered_sharded_fp32_execution": _graph_execution_ok(graph),
        "ood_population_exactly_training_disjoint": audit.get("passed") is True,
        "native_panel_finite_noncollapsed": q2._panel_execution_ok(native_panel),
        "matched_panel_finite_noncollapsed": q2._panel_execution_ok(matched["panel"]),
        "all_twenty_ood_cells_complete": set(ood_reports) == set(LANGUAGES),
    }
    passed = bool(decision["passed"] and all(execution_gates.values()))
    decision = {
        **decision,
        "metric_gates_passed": bool(decision["passed"]),
        "execution_gates": execution_gates,
        "passed": passed,
        "outcome": (
            "prompted-diverse-u12-rung-qualified"
            if passed
            else "prompted-diverse-u12-rung-not-qualified"
        ),
    }
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0169Error(
            f"R0169 evaluation peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = _seal({
        "schema": "round0169-prompted-diverse-u12-evaluation-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY] if passed else [],
        "decision": decision,
        "q2_evaluation": q2_evaluation_signature,
        "population": population_signature,
        "group_ids": group_signature,
        "ood_training_disjoint_audit": audit_signature,
        "graph_manifest": expected_input_signature(str(job["graph_manifest"])),
        "train_receipt": train_signature,
        "prompted_gate_registration": dict(job["gate_registration"]),
        "prompted_seed_family": dict(job["family_evidence"]),
        "native_u12": {
            "coordinates": expected_input_signature(coordinate_path),
            "panel": native_panel,
            "group_ffr": group_ffr,
            "training_alignment": native_alignment,
            "decision_metrics": native_metrics,
            "projection_metrics_role": "diagnostic-only",
        },
        "matched_2m": matched,
        "prompted_ood": {
            "summary": prompted_ood,
            "language_cells": ood_reports,
            "raw_r0132_control": raw_ood,
            "raw_r0132_evidence": raw_signature,
            "projection_ffr_role": "diagnostic-only",
        },
        "prompted_floors": floors,
        "training_performed_in_round": True,
        "evaluation_node_training_performed": False,
        "graph_built_in_round": True,
        "performance": {
            "evaluation_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "scale-evaluation.json"), receipt, immutable=True
    )
    del model, source_raw, source, coordinates, native_centroids, native_reference
    torch.cuda.empty_cache()
    gc.collect()


def _configure_q2_kernel() -> None:
    """Bind the reviewed Q2 graph/train kernel to Q3's frozen population law."""
    bindings = {
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "DIMENSION": DIMENSION,
        "SEED": SEED,
        "SUCCESSFUL_UPDATES": SUCCESSFUL_UPDATES,
        "GRAPH_K": GRAPH_K,
        "GRAPH_NLIST": GRAPH_NLIST,
        "GRAPH_NPROBE": GRAPH_NPROBE,
        "GRAPH_NPROBE_GRID": GRAPH_NPROBE_GRID,
        "GRAPH_TRAIN_ROWS": GRAPH_TRAIN_ROWS,
        "GRAPH_TRAIN_SEED": GRAPH_TRAIN_SEED,
        "GRAPH_QUALITY_ROWS": GRAPH_QUALITY_ROWS,
        "GRAPH_QUALITY_SEED": GRAPH_QUALITY_SEED,
        "GRAPH_MEAN_RECALL_FLOOR": GRAPH_MEAN_RECALL_FLOOR,
        "GRAPH_P10_RECALL_FLOOR": GRAPH_P10_RECALL_FLOOR,
        "HOST_RSS_LIMIT_GIB": HOST_RSS_LIMIT_GIB,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "TRAIN_SCHEMA": TRAIN_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "GRAPH_INDEX_DESCRIPTION": GRAPH_INDEX_DESCRIPTION,
        "GRAPH_SHARD_ROWS": GRAPH_SHARD_ROWS,
        "GRAPH_REFERENCE_ROW_ORDER": "exact accepted R0132 U12 compact order",
        "GRAPH_REFERENCE_ANCHOR_NAMESPACE": "R0132 U12 compact IDs",
        "Round0166Error": Round0169Error,
        "ScalePromptTrainingInput": DiversePromptTrainingInput,
        "scale_train_config": diverse_train_config,
        "_read_population": _read_population,
        "_open_source": _open_source,
        "_data_identity": _data_identity,
        "_faiss_gpu_options": _fp32_gpu_options,
    }
    for name, value in bindings.items():
        setattr(q2, name, value)


def run_build_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_q2_kernel()
    q2.run_build_graph(active, job)


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    _configure_q2_kernel()
    q2.run_train(active, job)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    action = str(job.get("action") or "")
    if action == "prompt_canary":
        return run_prompt_canary(active, job)
    if action == "embed_language_probe":
        return run_embed_language(active, job)
    if action == "audit_probe_training_disjoint":
        return run_audit_probe_training_disjoint(active, job)
    if action == "build_graph_and_reference":
        return run_build_graph(active, job)
    if action == "train_prompted_diverse_u12":
        return run_train(active, job)
    if action == "evaluate_prompted_diverse_u12":
        return run_evaluate(active, job)
    raise Round0169Error(f"unknown R0169 action {action!r}")
