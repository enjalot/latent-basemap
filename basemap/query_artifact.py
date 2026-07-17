"""Content-bound held-out query artifacts for complete-panel scoring."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import numpy as np

from .artifact_identity import (canonical_json, expected_input_signature,
                                ordered_array_sha256, sha256_bytes, sha256_file)
from .output_safety import (atomic_save_new_npy, atomic_write_new_json,
                            create_fresh_directory, refuse_existing)
from .panel_v2 import load_embeddings
from .round0005_staging import (
    ROUND0005_DIMENSIONS, ROUND0005_MODEL_ID, ROUND0005_MODEL_REVISION,
    ROUND0005_NORMALIZATION, ROUND0005_POOLING, ROUND0005_QUERY_CONVERSION,
    ROUND0005_QUERY_ROWS, ROUND0005_QUERY_SEED, ROUND0005_SOURCE_DTYPE,
    ROUND0005_SOURCE_ROWS, ROUND0005_STORAGE_DTYPE, ROUND0005_TRAIN_ROWS,
    make_testbed_seal_reference, validate_testbed_seal_reference,
)

SCHEMA = "basemap_heldout_query.v3"
CONVENTION_FIELDS = (
    "model_id", "model_revision", "prompt_policy", "prompt_bytes_hex", "pooling",
    "source_dtype", "storage_dtype", "dtype", "dtype_conversion",
    "normalization", "dimensions",
)
LEGACY_CONVENTION_FIELDS = (
    "model_id", "model_revision", "prompt_bytes_hex", "pooling", "dtype",
    "normalization", "dimensions",
)
IDENTITY_FIELDS = (
    "schema", "embeddings", "ids", "shape", "ordered_embeddings_sha256",
    "ordered_ids_sha256", "corpus", "source_embeddings", "convention",
    "query_selection", "normalization_proof", "materialization_proof",
    "testbed_seal", "production_contract",
)
L2_ATOL = 1e-4


def validate_convention(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("query convention must be a JSON object")
    fields = set(value)
    if fields == set(LEGACY_CONVENTION_FIELDS):
        storage_dtype = value["dtype"]
        source_dtype = storage_dtype
        prompt_bytes_hex = value["prompt_bytes_hex"]
        out = {
            **value,
            "prompt_policy": "raw_unprompted" if prompt_bytes_hex == "" else "literal_prefix",
            "source_dtype": source_dtype,
            "storage_dtype": storage_dtype,
            "dtype_conversion": "identity_v1",
        }
    elif fields == set(CONVENTION_FIELDS):
        out = {field: value[field] for field in CONVENTION_FIELDS}
    else:
        missing = [field for field in CONVENTION_FIELDS if field not in value]
        extra = sorted(fields - set(CONVENTION_FIELDS))
        raise ValueError(
            f"query convention fields mismatch: missing={missing} extra={extra}")
    for field in CONVENTION_FIELDS:
        if field != "dimensions" and not isinstance(out[field], str):
            raise ValueError(f"query convention {field} must be a string")
    if not out["model_id"] or out["model_id"].lower() in {"model", "placeholder", "unknown"}:
        raise ValueError("query model_id must identify a real model")
    revision = out["model_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or len(set(revision)) == 1:
        raise ValueError("query model_revision must be an immutable 40-hex revision")
    try:
        prompt = bytes.fromhex(out["prompt_bytes_hex"])
    except ValueError as exc:
        raise ValueError("query prompt_bytes_hex is not valid hexadecimal") from exc
    if prompt.hex() != out["prompt_bytes_hex"]:
        raise ValueError("query prompt_bytes_hex must use canonical lowercase hexadecimal")
    expected_prompt_policy = "raw_unprompted" if not prompt else "literal_prefix"
    if out["prompt_policy"] != expected_prompt_policy:
        raise ValueError(
            "query prompt_policy must explicitly distinguish raw empty bytes from a prefix")
    if out["pooling"] not in {"lasttoken", "mean", "cls"}:
        raise ValueError(f"unsupported query pooling {out['pooling']!r}")
    for field in ("source_dtype", "storage_dtype", "dtype"):
        try:
            dtype = np.dtype(out[field])
        except TypeError as exc:
            raise ValueError(f"unsupported query {field} {out[field]!r}") from exc
        if dtype not in {np.dtype("float16"), np.dtype("float32"), np.dtype("float64")}:
            raise ValueError(f"unsupported query {field} {out[field]!r}")
        out[field] = dtype.name
    if out["dtype"] != out["storage_dtype"]:
        raise ValueError("query dtype compatibility alias must equal storage_dtype")
    if out["source_dtype"] == out["storage_dtype"]:
        if out["dtype_conversion"] != "identity_v1":
            raise ValueError("same-dtype query materialization must use identity_v1")
    elif (out["source_dtype"], out["storage_dtype"], out["dtype_conversion"]) != (
            "float16", "float32", ROUND0005_QUERY_CONVERSION):
        raise ValueError("unsupported query source/storage dtype conversion")
    if out["normalization"] not in {"l2", "none"}:
        raise ValueError(f"unsupported query normalization {out['normalization']!r}")
    if not isinstance(out["dimensions"], int) or isinstance(out["dimensions"], bool) \
            or out["dimensions"] <= 0:
        raise ValueError("query convention dimensions must be a positive integer")
    return out


def round0005_query_convention() -> dict:
    return validate_convention({
        "model_id": ROUND0005_MODEL_ID,
        "model_revision": ROUND0005_MODEL_REVISION,
        "prompt_policy": "raw_unprompted",
        "prompt_bytes_hex": "",
        "pooling": ROUND0005_POOLING,
        "source_dtype": ROUND0005_SOURCE_DTYPE,
        "storage_dtype": ROUND0005_STORAGE_DTYPE,
        "dtype": ROUND0005_STORAGE_DTYPE,
        "dtype_conversion": ROUND0005_QUERY_CONVERSION,
        "normalization": ROUND0005_NORMALIZATION,
        "dimensions": ROUND0005_DIMENSIONS,
    })


def validate_round0005_query_convention(value: dict) -> dict:
    convention = validate_convention(value)
    if convention != round0005_query_convention():
        raise ValueError("query convention is not the exact unprompted Round 0005 contract")
    return convention


def _selection(*, train_ids: np.ndarray, source_rows: int, count: int, seed: int) \
        -> tuple[np.ndarray, dict]:
    if count <= 0:
        raise ValueError("query selection count must be positive")
    if train_ids.size and (int(train_ids.min()) < 0 or int(train_ids.max()) >= source_rows):
        raise ValueError("training IDs are outside the source corpus")
    candidates = np.setdiff1d(
        np.arange(source_rows, dtype=np.int64), np.sort(train_ids), assume_unique=False)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} held-out rows for requested {count}")
    rng = np.random.RandomState(seed)
    query_ids = np.sort(rng.choice(candidates, count, replace=False)).astype(np.int64)
    policy = {
        "method": "seeded_without_replacement_from_exact_source_complement",
        "seed": int(seed),
        "requested_count": int(count),
        "candidate_count": int(len(candidates)),
        "source_row_count": int(source_rows),
        "query_ids_ordered_sha256": ordered_array_sha256(query_ids),
        "training_ids_ordered_sha256": ordered_array_sha256(train_ids),
    }
    return query_ids, policy


def _normalization_proof(values: np.ndarray, normalization: str) -> dict:
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    finite = bool(np.isfinite(values).all() and np.isfinite(norms).all())
    if not finite:
        raise ValueError("held-out query embeddings contain non-finite values")
    max_error = float(np.max(np.abs(norms - 1.0))) if len(norms) else float("inf")
    if normalization == "l2" and max_error > L2_ATOL:
        raise ValueError(
            f"held-out query embeddings claim L2 normalization but max norm error is {max_error}")
    return {
        "finite": finite,
        "claimed": normalization,
        "l2_tolerance": L2_ATOL if normalization == "l2" else None,
        "l2_norm_min": float(norms.min()) if len(norms) else None,
        "l2_norm_max": float(norms.max()) if len(norms) else None,
        "l2_norm_mean": float(norms.mean()) if len(norms) else None,
        "l2_max_abs_error": max_error if len(norms) else None,
    }


def _identity(manifest: dict) -> str:
    missing = [key for key in IDENTITY_FIELDS if key not in manifest]
    if missing:
        raise ValueError(f"query artifact identity fields missing {missing}")
    return sha256_bytes(canonical_json({key: manifest[key] for key in IDENTITY_FIELDS}))


def _materialize_source_rows(source, row_ids: np.ndarray, *, source_dtype: str,
                             storage_dtype: str, conversion: str) -> np.ndarray:
    """Gather exact source bytes and perform only the declared dtype conversion."""
    target_dtype = np.dtype(storage_dtype)
    expected_source_dtype = np.dtype(source_dtype)
    if hasattr(source, "mms") and hasattr(source, "offsets"):
        output = np.empty((len(row_ids), int(source.shape[1])), dtype=target_dtype)
        which = np.searchsorted(source.offsets, row_ids, side="right") - 1
        for shard_index in np.unique(which):
            selected = which == shard_index
            local = row_ids[selected] - int(source.offsets[int(shard_index)])
            raw = np.asarray(source.mms[int(shard_index)][local])
            if raw.dtype != expected_source_dtype:
                raise ValueError("query source shard dtype changed during materialization")
            output[selected] = raw.astype(target_dtype, copy=False)
    else:
        raw = np.asarray(source[row_ids])
        if raw.dtype != expected_source_dtype:
            raise ValueError("query source dtype changed during materialization")
        output = np.asarray(raw, dtype=target_dtype)
    if conversion == "identity_v1" and expected_source_dtype != target_dtype:
        raise ValueError("query identity conversion cannot change dtype")
    if conversion == ROUND0005_QUERY_CONVERSION and (
            expected_source_dtype != np.dtype("float16") or
            target_dtype != np.dtype("float32")):
        raise ValueError("query widening policy does not match source/storage dtype")
    return np.ascontiguousarray(output)


def build_query_artifact(*, testbed: str, source: str, out_dir: str, dim: int,
                         n_holdout: int, seed: int, convention: dict,
                         testbed_seal_path: str | None = None,
                         production_contract: bool = False) -> dict:
    """Materialize one explicit ordered held-out query set without model work."""
    out_dir = os.path.abspath(out_dir)
    if not out_dir.startswith("/data/"):
        raise ValueError("query artifact must live under /data")
    refuse_existing(out_dir, label="query artifact root")
    convention = validate_convention(convention)
    if production_contract and convention != round0005_query_convention():
        raise ValueError("production query artifact requires the exact Round 0005 convention")
    if production_contract and (
            dim != ROUND0005_DIMENSIONS or n_holdout != ROUND0005_QUERY_ROWS or
            seed != ROUND0005_QUERY_SEED):
        raise ValueError("production query artifact count/seed/dim constants changed")
    if production_contract and testbed_seal_path is None:
        raise ValueError("production query artifact requires the shared testbed seal")
    seal_reference = (make_testbed_seal_reference(
        testbed_seal_path, require_round0005=production_contract)
                      if testbed_seal_path is not None else None)
    if dim != convention["dimensions"]:
        raise ValueError("query --dim does not match convention dimensions")
    testbed = os.path.realpath(testbed)
    source = os.path.realpath(source)
    train_dir = os.path.join(testbed, "train")
    train_ids_path = os.path.join(testbed, "sample_indices.npy")
    if not os.path.isfile(train_ids_path):
        raise FileNotFoundError(f"query artifact requires {train_ids_path}")
    raw_train_ids = np.load(train_ids_path, mmap_mode="r")
    if raw_train_ids.dtype != np.dtype("int64") or raw_train_ids.ndim != 1:
        raise ValueError("training sample_indices must be a one-dimensional int64 array")
    train_ids = np.asarray(raw_train_ids)
    if len(np.unique(train_ids)) != len(train_ids):
        raise ValueError("training sample_indices contain duplicates")
    src = load_embeddings(source, dim=dim)
    if np.dtype(src.dtype).name != convention["source_dtype"]:
        raise ValueError(
            f"source embedding dtype {src.dtype} != convention {convention['source_dtype']}")
    if len(src.shape) != 2 or int(src.shape[1]) != convention["dimensions"]:
        raise ValueError("source embedding dimensions do not match convention")
    if production_contract and (
            len(train_ids) != ROUND0005_TRAIN_ROWS or len(src) != ROUND0005_SOURCE_ROWS):
        raise ValueError("production query train/source row counts changed")
    query_ids, selection = _selection(
        train_ids=train_ids, source_rows=len(src), count=n_holdout, seed=seed)
    embeddings = _materialize_source_rows(
        src, query_ids, source_dtype=convention["source_dtype"],
        storage_dtype=convention["storage_dtype"],
        conversion=convention["dtype_conversion"])
    if embeddings.dtype.name != convention["storage_dtype"]:
        raise ValueError("selected query embedding dtype changed during materialization")
    proof = _normalization_proof(embeddings, convention["normalization"])

    create_fresh_directory(out_dir, label="query artifact root")
    emb_path = os.path.join(out_dir, "query_embeddings.npy")
    ids_path = os.path.join(out_dir, "query_ids.npy")
    atomic_save_new_npy(emb_path, embeddings, immutable=True)
    atomic_save_new_npy(ids_path, query_ids, immutable=True)
    corpus = {
        "testbed": testbed,
        "ordered_train_embeddings": expected_input_signature(train_dir),
        "ordered_train_ids": expected_input_signature(train_ids_path),
        "n_train": int(len(train_ids)),
        "train_ids_ordered_sha256": ordered_array_sha256(train_ids),
    }
    if seal_reference is not None:
        sealed = _json_file(seal_reference["seal_path"])
        if testbed != sealed.get("testbed_root"):
            raise ValueError("query testbed path differs from the shared testbed seal")
        if source != sealed.get("source_embedding_root"):
            raise ValueError("query source path differs from the shared testbed seal")
        if corpus["ordered_train_embeddings"] != sealed["train"]["root_signature"]:
            raise ValueError("query train corpus differs from the shared testbed seal")
        if corpus["ordered_train_ids"] != sealed["sample_indices"]["signature"]:
            raise ValueError("query sample mapping differs from the shared testbed seal")
    materialization_proof = {
        "source_dtype": convention["source_dtype"],
        "storage_dtype": convention["storage_dtype"],
        "conversion": convention["dtype_conversion"],
        "exact_selected_source_rows": True,
        "contiguous_c_order": bool(embeddings.flags.c_contiguous),
    }
    payload = {
        "schema": SCHEMA,
        "production_contract": bool(production_contract),
        "testbed_seal": seal_reference,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embeddings": expected_input_signature(emb_path),
        "ids": expected_input_signature(ids_path),
        "shape": [int(value) for value in embeddings.shape],
        "ordered_embeddings_sha256": ordered_array_sha256(embeddings),
        "ordered_ids_sha256": ordered_array_sha256(query_ids),
        "corpus": corpus,
        "source_embeddings": expected_input_signature(source),
        "convention": convention,
        "query_selection": selection,
        "normalization_proof": proof,
        "materialization_proof": materialization_proof,
    }
    payload["identity_sha256"] = _identity(payload)
    manifest_path = os.path.join(out_dir, "manifest.json")
    atomic_write_new_json(manifest_path, payload, immutable=True)
    payload["manifest_path"] = manifest_path
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload


def load_query_artifact(manifest_path: str, *, testbed: str,
                        expected_convention: dict,
                        expected_testbed_seal: str | None = None,
                        require_round0005: bool = False) -> dict:
    """Load and independently verify every artifact, selection, and identity byte."""
    manifest_path = os.path.realpath(manifest_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_manifest_fields = set(IDENTITY_FIELDS) | {"created_utc", "identity_sha256"}
    if set(manifest) != expected_manifest_fields:
        raise ValueError(
            f"query artifact manifest fields must be exactly {sorted(expected_manifest_fields)}")
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"query artifact schema must be {SCHEMA}")
    recomputed_identity = _identity(manifest)
    if manifest.get("identity_sha256") != recomputed_identity:
        raise ValueError("query artifact stored identity_sha256 does not match its payload")
    expected_convention = validate_convention(expected_convention)
    actual_convention = validate_convention(manifest.get("convention"))
    if actual_convention != expected_convention:
        raise ValueError(
            f"query convention mismatch: artifact={actual_convention} expected={expected_convention}")
    production_contract = manifest.get("production_contract") is True
    if require_round0005:
        validate_round0005_query_convention(actual_convention)
        if not production_contract:
            raise ValueError("queue requires a production-contract Round 0005 query")
    if production_contract or expected_testbed_seal is not None:
        validate_testbed_seal_reference(
            manifest.get("testbed_seal"), expected_seal=expected_testbed_seal,
            require_round0005=require_round0005)
    for key in ("embeddings", "ids"):
        expected = manifest.get(key)
        if not isinstance(expected, dict) or "canonical_path" not in expected:
            raise ValueError(f"query artifact missing {key} signature")
        observed = expected_input_signature(expected["canonical_path"])
        if observed != expected:
            raise ValueError(f"query artifact {key} bytes do not match manifest")

    current_testbed = os.path.realpath(testbed)
    corpus = manifest.get("corpus") or {}
    if set(corpus) != {"testbed", "ordered_train_embeddings", "ordered_train_ids",
                       "n_train", "train_ids_ordered_sha256"}:
        raise ValueError("query artifact corpus identity fields are incomplete or ambiguous")
    if corpus.get("testbed") != current_testbed:
        raise ValueError("query artifact corpus testbed path mismatch")
    train_dir = os.path.join(current_testbed, "train")
    train_ids_path = os.path.join(current_testbed, "sample_indices.npy")
    if expected_input_signature(train_dir) != corpus.get("ordered_train_embeddings"):
        raise ValueError("query artifact ordered corpus embedding identity mismatch")
    if expected_input_signature(train_ids_path) != corpus.get("ordered_train_ids"):
        raise ValueError("query artifact ordered corpus ID identity mismatch")
    source_path = (manifest.get("source_embeddings") or {}).get("canonical_path")
    if not source_path or expected_input_signature(source_path) != manifest.get("source_embeddings"):
        raise ValueError("query artifact source corpus identity mismatch")

    Xq = np.load(manifest["embeddings"]["canonical_path"], mmap_mode="r")
    query_ids_raw = np.load(manifest["ids"]["canonical_path"], mmap_mode="r")
    if Xq.ndim != 2 or list(Xq.shape) != manifest.get("shape"):
        raise ValueError("query artifact embedding shape mismatch")
    if Xq.dtype.name != actual_convention["storage_dtype"]:
        raise ValueError("query artifact embedding dtype mismatch")
    if Xq.shape[1] != actual_convention["dimensions"]:
        raise ValueError("query artifact embedding dimensions mismatch")
    if query_ids_raw.ndim != 1 or query_ids_raw.dtype != np.dtype("int64"):
        raise ValueError("query artifact IDs must be a one-dimensional int64 array")
    query_ids = np.asarray(query_ids_raw)
    if len(Xq) != len(query_ids) or len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("query artifact IDs are duplicated or length-mismatched")
    if query_ids.size and (int(query_ids.min()) < 0):
        raise ValueError("query artifact IDs contain a negative index")
    if ordered_array_sha256(Xq) != manifest.get("ordered_embeddings_sha256"):
        raise ValueError("query artifact ordered embedding identity mismatch")
    if ordered_array_sha256(query_ids) != manifest.get("ordered_ids_sha256"):
        raise ValueError("query artifact ordered ID identity mismatch")
    proof = _normalization_proof(Xq, actual_convention["normalization"])
    if proof != manifest.get("normalization_proof"):
        raise ValueError("query artifact normalization proof mismatch")

    train_ids_raw = np.load(train_ids_path, mmap_mode="r")
    if train_ids_raw.dtype != np.dtype("int64") or train_ids_raw.ndim != 1:
        raise ValueError("query artifact corpus IDs no longer have exact int64 dtype")
    train_ids = np.asarray(train_ids_raw)
    if len(train_ids) != corpus.get("n_train") or \
            ordered_array_sha256(train_ids) != corpus.get("train_ids_ordered_sha256"):
        raise ValueError("query artifact corpus ID count/order mismatch")
    selection = manifest.get("query_selection") or {}
    expected_ids, expected_selection = _selection(
        train_ids=train_ids,
        source_rows=int(selection.get("source_row_count", -1)),
        count=int(selection.get("requested_count", -1)),
        seed=int(selection.get("seed", -1)),
    )
    if selection != expected_selection or not np.array_equal(query_ids, expected_ids):
        raise ValueError("query artifact exact selection/order mismatch")
    if np.intersect1d(query_ids, train_ids).size:
        raise ValueError("query artifact IDs overlap training IDs")
    source = load_embeddings(source_path, dim=actual_convention["dimensions"])
    if len(source) != selection["source_row_count"] or \
            source.dtype.name != actual_convention["source_dtype"]:
        raise ValueError("query artifact source cardinality/dtype mismatch")
    if query_ids.size and int(query_ids.max()) >= len(source):
        raise ValueError("query artifact IDs exceed source corpus cardinality")
    expected_materialization = {
        "source_dtype": actual_convention["source_dtype"],
        "storage_dtype": actual_convention["storage_dtype"],
        "conversion": actual_convention["dtype_conversion"],
        "exact_selected_source_rows": True,
        "contiguous_c_order": True,
    }
    if manifest.get("materialization_proof") != expected_materialization:
        raise ValueError("query artifact materialization policy/proof mismatch")
    selected = _materialize_source_rows(
        source, query_ids, source_dtype=actual_convention["source_dtype"],
        storage_dtype=actual_convention["storage_dtype"],
        conversion=actual_convention["dtype_conversion"])
    if not np.array_equal(selected, np.asarray(Xq)):
        raise ValueError("query artifact embeddings are not the exact selected source rows")
    if production_contract:
        sealed = _json_file(manifest["testbed_seal"]["seal_path"])
        if (manifest["corpus"]["ordered_train_embeddings"] !=
                sealed["train"]["root_signature"] or
                manifest["corpus"]["ordered_train_ids"] !=
                sealed["sample_indices"]["signature"] or
                manifest["source_embeddings"] !=
                sealed["source_embeddings"]["root_signature"]):
            raise ValueError("query artifact inputs differ from the shared testbed seal")
    if require_round0005 and (
            len(Xq) != ROUND0005_QUERY_ROWS or
            selection.get("seed") != ROUND0005_QUERY_SEED or
            selection.get("source_row_count") != ROUND0005_SOURCE_ROWS or
            corpus.get("n_train") != ROUND0005_TRAIN_ROWS):
        raise ValueError("query artifact count/seed/source/testbed constants changed")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "Xq": Xq,
        "query_ids": query_ids,
        "identity_sha256": recomputed_identity,
    }


def _json_file(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
