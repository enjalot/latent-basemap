"""CPU-only staging and sealing for Round 0005 runtime inputs."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .artifact_identity import (canonical_json, expected_input_signature,
                                ordered_array_sha256, sha256_bytes)
from .output_safety import (atomic_copy_new, atomic_write_new_json,
                            create_fresh_directory, ensure_data_directory,
                            refuse_existing)

MAP_STAGE_SCHEMA = "round0005_map_stage.v1"
MODEL_STAGE_SCHEMA = "round0005_regular_model_stage.v1"
SEMANTIC_NAMESPACE_SCHEMA = "basemap_semantic_id_namespace.v1"
TESTBED_SEAL_SCHEMA = "round0005_testbed_seal.v1"
TESTBED_SEAL_REFERENCE_SCHEMA = "round0005_testbed_seal_reference.v1"
DATA_IDENTITY_CLOSURE_SCHEMA = "round0005_data_identity_closure.v1"

# The certifying Round 0005 substrate is not caller-configurable.  Generic
# helpers remain available for tiny fixtures, while every production CLI and
# strict validator below binds these exact values.
ROUND0005_MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
ROUND0005_MODEL_REVISION = "ac5d898c8d382b17167c33e5c8af644a3519b47d"
ROUND0005_DIMENSIONS = 768
ROUND0005_TRAIN_ROWS = 2_000_000
ROUND0005_SOURCE_ROWS = 2_890_362
ROUND0005_QUERY_ROWS = 20_000
ROUND0005_QUERY_SEED = 123
ROUND0005_CALIBRATION_SEED = 20260716
ROUND0005_POOLING = "lasttoken"
ROUND0005_NORMALIZATION = "l2"
ROUND0005_SOURCE_DTYPE = "float16"
ROUND0005_STORAGE_DTYPE = "float32"
ROUND0005_QUERY_CONVERSION = "exact_float16_to_float32_widening_v1"
ROUND0005_SOURCE_SHARD_ROWS = (
    265_419, 260_422, 260_748, 264_661, 266_637, 263_875,
    264_737, 262_636, 260_251, 260_043, 260_933,
)
ROUND0005_SOURCE_SHARDS = tuple(
    f"data-{index:05d}-of-00099.npy" for index in range(11))
ROUND0005_TEXT_SHARDS = tuple(
    f"data-{index:05d}-of-00099.parquet" for index in range(11))

# Full resolved-file closure of the immutable HF snapshot named by the pinned
# revision above.  Pinning only the snapshot directory name is insufficient:
# the usual HF layout consists almost entirely of mutable symlinks to blobs.
ROUND0005_JINA_MODEL_CLOSURE = {
    "1_Pooling/config.json": (312, "f54132064e1846eab44448285b505446bf085dc950595638fb94f40a31c7d1fd"),
    "README.md": (9223, "9f2433969e7962fffd5cd2bfef18bb319032dc973090d5449628d9bb4d219381"),
    "config.json": (1361, "367857e3a726df6f1997bcb8443a4351e68b29c65f996e5874a4b3e7c5661a16"),
    "config_sentence_transformers.json": (
        274, "916dede36f621cdfecd30fde3d66923dc45336fe75e051b8839374800148b560"),
    "configuration_eurobert.py": (
        12138, "8737378a5cea9e6c7be0e077138d6c725fb5909ad122c979adcdcc1a005a51b3"),
    "model.safetensors": (
        423543712, "e38cb1ff168a60c24d88f9664997ceaf49afcfd93275af1b7f850b23070bbe7d"),
    "modeling_eurobert.py": (
        48990, "d7d54988acaab772bbc1d4ab227937c1d20341d3f84e2eccfc39e2fddbc2fe07"),
    "modules.json": (349, "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"),
    "tokenizer.json": (
        17210235, "98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069"),
    "tokenizer_config.json": (
        487, "6c4640d432db970b2436a4386d3ee992b99e756b62c37446c3f581c8d09cbb05"),
}
REQUIRED_MAP_FILES = (
    "coords.parquet", "model.pt", "config.yaml", "manifest.json", "results.json",
)
MAP_EXPECTATIONS = {
    "legacy_a1b1_s42": {"seed": 42, "kernel": "legacy_lp", "a": 1.0, "b": 1.0},
    "legacy_a1b1_s43": {"seed": 43, "kernel": "legacy_lp", "a": 1.0, "b": 1.0},
    "legacy_a1b1_s44": {"seed": 44, "kernel": "legacy_lp", "a": 1.0, "b": 1.0},
    "umap_a1b1_s42": {"seed": 42, "kernel": "umap", "a": 1.0, "b": 1.0},
    "umap_a1b1_s43": {"seed": 43, "kernel": "umap", "a": 1.0, "b": 1.0},
    "umap_a1b1_s44": {"seed": 44, "kernel": "umap", "a": 1.0, "b": 1.0},
    "umap_stdcurve_s42": {
        "seed": 42, "kernel": "umap", "a": 1.57694346, "b": 0.895060879,
    },
    "umap_stdcurve_s43": {
        "seed": 43, "kernel": "umap", "a": 1.57694346, "b": 0.895060879,
    },
    "umap_stdcurve_s44": {
        "seed": 44, "kernel": "umap", "a": 1.57694346, "b": 0.895060879,
    },
}


def validate_semantic_namespace(value: dict) -> dict:
    required = {"schema", "name", "kind", "corpus_identity_sha256",
                "universe_sha256", "row_count"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"semantic namespace fields must be exactly {sorted(required)}")
    if value["schema"] != SEMANTIC_NAMESPACE_SCHEMA:
        raise ValueError(f"semantic namespace schema must be {SEMANTIC_NAMESPACE_SCHEMA}")
    if (not isinstance(value["name"], str) or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{2,127}", value["name"])):
        raise ValueError("semantic namespace name is invalid")
    if value["kind"] not in {"coordinate_semantic_id", "row_position"}:
        raise ValueError("semantic namespace kind is invalid")
    for key in ("corpus_identity_sha256", "universe_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError(f"semantic namespace {key} must be full SHA-256")
    if not isinstance(value["row_count"], int) or isinstance(value["row_count"], bool) \
            or value["row_count"] <= 0:
        raise ValueError("semantic namespace row_count must be positive")
    return dict(value)


def _json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _tree_content(signature: dict) -> dict:
    return {
        "kind": signature["kind"],
        "bytes": signature["bytes"],
        "sha256": signature["sha256"],
        "members": signature.get("members", []),
    }


def _round0005_testbed_contract() -> dict:
    return {
        "dimensions": ROUND0005_DIMENSIONS,
        "train_rows": ROUND0005_TRAIN_ROWS,
        "source_rows": ROUND0005_SOURCE_ROWS,
        "source_dtype": ROUND0005_SOURCE_DTYPE,
        "train_dtype": ROUND0005_STORAGE_DTYPE,
        "source_shards": list(ROUND0005_SOURCE_SHARDS),
        "source_shard_rows": list(ROUND0005_SOURCE_SHARD_ROWS),
        "text_shards": list(ROUND0005_TEXT_SHARDS),
        "centroids": {"k256": 256, "k1024": 1024},
        "sample_mapping_conversion": ROUND0005_QUERY_CONVERSION,
    }


def _directory_member_signature(root_signature: dict, relative_path: str) -> dict:
    matches = [member for member in root_signature.get("members", [])
               if member.get("relative_path") == relative_path]
    if len(matches) != 1:
        raise ValueError(
            f"directory signature does not contain exactly one {relative_path!r}")
    member = matches[0]
    return {
        "canonical_path": os.path.join(root_signature["canonical_path"], relative_path),
        "kind": "file",
        "bytes": int(member["bytes"]),
        "sha256": member["sha256"],
    }


def _array_contract(path: str, *, rows: int, dimensions: int | None,
                    dtype: str, label: str) -> tuple[np.ndarray, dict]:
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{label} must be a regular .npy file: {path}")
    value = np.load(path, mmap_mode="r")
    expected_shape = (rows,) if dimensions is None else (rows, dimensions)
    if value.shape != expected_shape or value.dtype != np.dtype(dtype):
        raise ValueError(
            f"{label} must be shape={expected_shape} dtype={dtype}; "
            f"observed shape={value.shape} dtype={value.dtype}")
    if dimensions is not None and not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value, {
        "shape": [int(item) for item in value.shape],
        "dtype": value.dtype.name,
    }


def _inspect_testbed_payload(*, testbed: str, source_embeddings: str,
                             source_texts: str, contract: dict,
                             production_contract: bool,
                             mapping_chunk_rows: int = 65_536) -> dict:
    """Inspect and fully join the testbed to its source corpus without CUDA."""
    testbed = os.path.realpath(testbed)
    source_embeddings = os.path.realpath(source_embeddings)
    source_texts = os.path.realpath(source_texts)
    for root, label in ((testbed, "testbed"),
                        (source_embeddings, "source embedding root"),
                        (source_texts, "source text root")):
        if os.path.islink(root) or not os.path.isdir(root):
            raise ValueError(f"Round 0005 {label} must be a regular directory: {root}")
    if mapping_chunk_rows <= 0:
        raise ValueError("sample-mapping verification chunk size must be positive")

    dimensions = int(contract["dimensions"])
    train_rows = int(contract["train_rows"])
    source_rows = int(contract["source_rows"])
    train_root = os.path.join(testbed, "train")
    train_file = os.path.join(train_root, "data-00000.npy")
    sample_path = os.path.join(testbed, "sample_indices.npy")
    train, train_array = _array_contract(
        train_file, rows=train_rows, dimensions=dimensions,
        dtype=contract["train_dtype"], label="Round 0005 train embeddings")
    sample_ids, sample_array = _array_contract(
        sample_path, rows=train_rows, dimensions=None, dtype="int64",
        label="Round 0005 sample mapping")
    if len(np.unique(sample_ids)) != train_rows:
        raise ValueError("Round 0005 sample mapping contains duplicate global IDs")
    if train_rows and (int(sample_ids[0]) < 0 or int(sample_ids[-1]) >= source_rows):
        raise ValueError("Round 0005 sample mapping is outside the source corpus")
    if train_rows > 1 and not bool(np.all(sample_ids[1:] > sample_ids[:-1])):
        raise ValueError("Round 0005 sample mapping must be strictly increasing")

    source_names = sorted(name for name in os.listdir(source_embeddings)
                          if name.endswith(".npy"))
    if source_names != list(contract["source_shards"]):
        raise ValueError(
            f"Round 0005 source shard names mismatch: observed={source_names}")
    source_root_signature = expected_input_signature(source_embeddings)
    source_inventory = []
    source_total = 0
    for name, expected_rows in zip(contract["source_shards"],
                                   contract["source_shard_rows"]):
        path = os.path.join(source_embeddings, name)
        values, array = _array_contract(
            path, rows=int(expected_rows), dimensions=dimensions,
            dtype=contract["source_dtype"], label=f"source shard {name}")
        source_total += len(values)
        source_inventory.append({
            "name": name,
            "row_range": [source_total - len(values), source_total],
            "array": array,
            "signature": _directory_member_signature(source_root_signature, name),
        })
    if source_total != source_rows:
        raise ValueError(
            f"Round 0005 source shards contain {source_total} rows, expected {source_rows}")

    import pyarrow.parquet as pq

    text_inventory = []
    text_total = 0
    for name, expected_rows in zip(contract["text_shards"],
                                   contract["source_shard_rows"]):
        path = os.path.join(source_texts, name)
        if os.path.islink(path) or not os.path.isfile(path):
            raise ValueError(f"matching source text shard is not a regular file: {path}")
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        if rows != int(expected_rows):
            raise ValueError(
                f"matching text shard {name} has {rows} rows, expected {expected_rows}")
        if "chunk_text" not in parquet.schema.names:
            raise ValueError(f"matching text shard {name} has no chunk_text column")
        text_total += rows
        text_inventory.append({
            "name": name,
            "row_range": [text_total - rows, text_total],
            "rows": rows,
            "signature": expected_input_signature(path),
        })
    if text_total != source_rows:
        raise ValueError("matching source text shards do not cover the exact source corpus")

    # Verify every materialized fp32 train row against the fp16 source row
    # named by sample_indices.  This is the explicit position -> global-ID
    # mapping that map coordinates and held-out queries share.
    from .panel_v2 import load_embeddings

    source = load_embeddings(source_embeddings, dim=dimensions)
    if (source.shape != (source_rows, dimensions) or
            np.dtype(source.dtype).name != contract["source_dtype"]):
        raise ValueError("loaded source corpus disagrees with its exact shard inventory")
    for start in range(0, train_rows, mapping_chunk_rows):
        stop = min(train_rows, start + mapping_chunk_rows)
        selected = np.asarray(source[np.asarray(sample_ids[start:stop])],
                              dtype=np.dtype(contract["train_dtype"]))
        materialized = np.asarray(train[start:stop])
        if not np.array_equal(selected, materialized):
            row_diff = np.flatnonzero(np.any(selected != materialized, axis=1))
            first = start + int(row_diff[0]) if len(row_diff) else start
            raise ValueError(
                f"testbed/source sample mapping differs at testbed row {first}")

    centroids = {}
    for name, count in contract["centroids"].items():
        filename = f"centroids_{name}.npy"
        path = os.path.join(testbed, filename)
        _, array = _array_contract(
            path, rows=int(count), dimensions=dimensions, dtype="float32",
            label=f"Round 0005 {name} centroids")
        centroids[name] = {"array": array, "signature": expected_input_signature(path)}

    train_root_signature = expected_input_signature(train_root)
    sample_signature = expected_input_signature(sample_path)
    position_ids = np.arange(train_rows, dtype=np.int64)
    sample_mapping = {
        "policy": "testbed_row_i_equals_float32_source_row_sample_indices_i",
        "conversion": contract["sample_mapping_conversion"],
        "all_rows_verified": True,
        "row_count": train_rows,
        "testbed_positions_ordered_sha256": ordered_array_sha256(position_ids),
        "source_global_ids_ordered_sha256": ordered_array_sha256(sample_ids),
        "source_global_id_range": [int(sample_ids[0]), int(sample_ids[-1])],
        "sample_indices_signature": sample_signature,
    }
    payload = {
        "schema": TESTBED_SEAL_SCHEMA,
        "production_contract": bool(production_contract),
        "contract": contract,
        "testbed_root": testbed,
        "source_embedding_root": source_embeddings,
        "source_text_root": source_texts,
        "train": {
            "root_signature": train_root_signature,
            "embedding_signature": _directory_member_signature(
                train_root_signature, "data-00000.npy"),
            "array": train_array,
        },
        "sample_indices": {"signature": sample_signature, "array": sample_array},
        "centroids": centroids,
        "source_embeddings": {
            "root_signature": source_root_signature,
            "rows": source_total,
            "shards": source_inventory,
        },
        "matching_source_texts": {"rows": text_total, "shards": text_inventory},
        "sample_mapping": sample_mapping,
    }
    payload["corpus_identity_sha256"] = sha256_bytes(canonical_json({
        key: payload[key] for key in (
            "contract", "train", "sample_indices", "centroids",
            "source_embeddings", "matching_source_texts", "sample_mapping")
    }))
    return payload


def build_round0005_testbed_seal(*, testbed: str, source_embeddings: str,
                                 source_texts: str, seal_path: str,
                                 _contract: dict | None = None) -> dict:
    """Build the one immutable substrate seal referenced by every Round 0005 input.

    ``_contract`` exists only for CPU-sized adversarial fixtures.  Such seals
    are explicitly marked non-production and strict queue validation rejects
    them.
    """
    seal_path = os.path.abspath(seal_path)
    if not seal_path.startswith("/data/"):
        raise ValueError("Round 0005 testbed seal must live under /data")
    refuse_existing(seal_path, label="Round 0005 testbed seal")
    production_contract = _contract is None
    contract = _round0005_testbed_contract() if production_contract else dict(_contract)
    payload = _inspect_testbed_payload(
        testbed=testbed, source_embeddings=source_embeddings, source_texts=source_texts,
        contract=contract, production_contract=production_contract)
    payload["identity_sha256"] = sha256_bytes(canonical_json(payload))
    atomic_write_new_json(seal_path, payload, immutable=True)
    payload["seal_path"] = seal_path
    payload["seal_signature"] = expected_input_signature(seal_path)
    return payload


def validate_round0005_testbed_seal(path: str, *,
                                    require_round0005: bool = True) -> dict:
    """Rehash and revalidate the complete substrate and sample mapping."""
    path = os.path.realpath(path)
    report = _json(path)
    if report.get("schema") != TESTBED_SEAL_SCHEMA:
        raise ValueError("invalid Round 0005 testbed seal schema")
    identity = report.get("identity_sha256")
    payload = {key: value for key, value in report.items() if key != "identity_sha256"}
    if sha256_bytes(canonical_json(payload)) != identity:
        raise ValueError("Round 0005 testbed seal identity mismatch")
    if require_round0005:
        if report.get("production_contract") is not True:
            raise ValueError("queue requires the exact production Round 0005 testbed contract")
        if report.get("contract") != _round0005_testbed_contract():
            raise ValueError("Round 0005 testbed contract constants changed")
    observed = _inspect_testbed_payload(
        testbed=report.get("testbed_root", ""),
        source_embeddings=report.get("source_embedding_root", ""),
        source_texts=report.get("source_text_root", ""),
        contract=report.get("contract") or {},
        production_contract=bool(report.get("production_contract")),
    )
    if observed != payload:
        raise ValueError("Round 0005 testbed seal no longer matches its joined inputs")
    return report


def make_testbed_seal_reference(path: str, *, require_round0005: bool = True,
                                deep: bool = True) -> dict:
    report = (validate_round0005_testbed_seal(path, require_round0005=require_round0005)
              if deep else _json(path))
    if report.get("schema") != TESTBED_SEAL_SCHEMA:
        raise ValueError("invalid testbed seal reference target")
    return {
        "schema": TESTBED_SEAL_REFERENCE_SCHEMA,
        "seal_path": os.path.realpath(path),
        "seal_signature": expected_input_signature(path),
        "identity_sha256": report.get("identity_sha256"),
        "corpus_identity_sha256": report.get("corpus_identity_sha256"),
    }


def validate_testbed_seal_reference(value: dict, *, expected_seal: str | None = None,
                                    require_round0005: bool = True) -> dict:
    required = {"schema", "seal_path", "seal_signature", "identity_sha256",
                "corpus_identity_sha256"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Round 0005 testbed seal reference fields are incomplete")
    if value.get("schema") != TESTBED_SEAL_REFERENCE_SCHEMA:
        raise ValueError("invalid Round 0005 testbed seal reference schema")
    seal_path = os.path.realpath(value["seal_path"])
    if expected_seal is not None and seal_path != os.path.realpath(expected_seal):
        raise ValueError("artifact references a different Round 0005 testbed seal")
    if expected_input_signature(seal_path) != value.get("seal_signature"):
        raise ValueError("referenced Round 0005 testbed seal bytes changed")
    report = _json(seal_path)
    if (report.get("identity_sha256") != value.get("identity_sha256") or
            report.get("corpus_identity_sha256") != value.get("corpus_identity_sha256")):
        raise ValueError("artifact testbed-seal identity disagrees with the seal")
    if require_round0005 and (
            report.get("production_contract") is not True or
            report.get("contract") != _round0005_testbed_contract()):
        raise ValueError("artifact does not reference the exact production testbed seal")
    return dict(value)


def _regular_tree_files(root: str) -> list[tuple[str, str]]:
    if not os.path.isdir(root) or os.path.islink(root):
        raise ValueError(f"staging source must be a regular directory: {root}")
    files = []
    for current, dirs, names in os.walk(root, followlinks=False):
        for name in [*dirs, *names]:
            candidate = os.path.join(current, name)
            if os.path.islink(candidate):
                raise ValueError(f"map staging source contains symlink: {candidate}")
        for name in names:
            source = os.path.join(current, name)
            if not os.path.isfile(source):
                raise ValueError(f"staging source contains unsupported member: {source}")
            files.append((str(Path(source).relative_to(root)), source))
    return sorted(files)


def _coordinate_ids(path: str, *, expected_rows: int) -> np.ndarray:
    import pandas as pd

    frame = pd.read_parquet(path)
    columns = [name for name in ("ls_index", "row_id") if name in frame.columns]
    if not columns:
        raise ValueError(f"Round 0005 staged map has no coordinate semantic IDs: {path}")
    first = frame[columns[0]].to_numpy()
    if not np.issubdtype(first.dtype, np.integer):
        raise ValueError(f"coordinate semantic IDs must have integer dtype: {path}")
    ids = np.asarray(first, dtype=np.int64)
    if len(columns) == 2:
        second = np.asarray(frame[columns[1]].to_numpy(), dtype=np.int64)
        if not np.array_equal(ids, second):
            raise ValueError(f"ls_index and row_id disagree: {path}")
    if len(ids) != expected_rows:
        raise ValueError(
            f"Round 0005 map must contain exactly {expected_rows:,} coordinates: {path}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"coordinate semantic IDs are duplicated: {path}")
    return ids


def _checkpoint_contract(path: str, *, label: str, expectation: dict,
                         observed_seed: int) -> dict:
    """CPU-only proof that the persisted projector matches its map metadata."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"map {label} checkpoint is not a metadata dictionary")
    required = {"model_state_dict", "input_dim", "n_components", "low_dim_kernel",
                "a", "b"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"map {label} checkpoint metadata missing {missing}")
    if checkpoint["input_dim"] != ROUND0005_DIMENSIONS or checkpoint["n_components"] != 2:
        raise ValueError(
            f"map {label} checkpoint must be {ROUND0005_DIMENSIONS}->2, observed "
            f"{checkpoint['input_dim']}->{checkpoint['n_components']}")
    if checkpoint["low_dim_kernel"] != expectation["kernel"]:
        raise ValueError(f"map {label} checkpoint kernel mismatch")
    for field in ("a", "b"):
        if not np.isclose(float(checkpoint[field]), float(expectation[field]),
                          rtol=0.0, atol=1e-9):
            raise ValueError(f"map {label} checkpoint {field} mismatch")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"map {label} checkpoint model_state_dict is invalid")
    proj_in = state.get("proj_in.weight")
    proj_out = state.get("proj_out.weight")
    if (proj_in is None or proj_out is None or
            tuple(proj_in.shape)[1:] != (ROUND0005_DIMENSIONS,) or
            tuple(proj_out.shape)[:1] != (2,)):
        raise ValueError(
            f"map {label} checkpoint tensor shapes do not prove "
            f"{ROUND0005_DIMENSIONS}->2")
    # Historical checkpoints do not contain a random_seed field.  The seed is
    # therefore bound from results.json/config and recorded explicitly rather
    # than falsely attributed to model.pt.
    return {
        "loaded_on": "cpu",
        "weights_only": True,
        "input_dim": int(checkpoint["input_dim"]),
        "n_components": int(checkpoint["n_components"]),
        "low_dim_kernel": checkpoint["low_dim_kernel"],
        "a": float(checkpoint["a"]),
        "b": float(checkpoint["b"]),
        "seed": int(observed_seed),
        "seed_source": "results.json:config.data.random_seed",
        "proj_in_weight_shape": [int(value) for value in proj_in.shape],
        "proj_out_weight_shape": [int(value) for value in proj_out.shape],
        "checkpoint_signature": expected_input_signature(path),
    }


def _validate_map_source(label: str, source: str, *, expected_rows: int,
                         verify_checkpoint_metadata: bool = False) -> dict:
    expectation = MAP_EXPECTATIONS[label]
    source = os.path.abspath(source)
    files = dict(_regular_tree_files(source))
    missing = [name for name in REQUIRED_MAP_FILES if name not in files]
    if missing:
        raise FileNotFoundError(f"map {label} is incomplete; missing {missing}: {source}")
    manifest = _json(files["manifest.json"])
    results = _json(files["results.json"])
    config = results.get("config") or {}
    model = config.get("model") or {}
    data = config.get("data") or {}
    observed = {
        "seed": data.get("random_seed"),
        "kernel": model.get("low_dim_kernel"),
        "a": model.get("a"),
        "b": model.get("b"),
        "n_train": (results.get("data") or {}).get("n_train"),
        "manifest_kernel": manifest.get("low_dim_kernel"),
    }
    if observed["seed"] != expectation["seed"]:
        raise ValueError(f"map {label} seed mismatch: {observed}")
    if observed["kernel"] != expectation["kernel"] or observed["manifest_kernel"] != expectation["kernel"]:
        raise ValueError(f"map {label} kernel mismatch: {observed}")
    if observed["n_train"] != expected_rows:
        raise ValueError(f"map {label} result row count mismatch: {observed}")
    for field in ("a", "b"):
        if observed[field] is None or not np.isclose(
                float(observed[field]), float(expectation[field]), rtol=0.0, atol=1e-9):
            raise ValueError(f"map {label} {field} mismatch: {observed[field]!r}")
    ids = _coordinate_ids(files["coords.parquet"], expected_rows=expected_rows)
    checkpoint_contract = (_checkpoint_contract(
        files["model.pt"], label=label, expectation=expectation,
        observed_seed=observed["seed"])
        if verify_checkpoint_metadata else None)
    required_signatures = {
        name: expected_input_signature(files[name]) for name in REQUIRED_MAP_FILES
    }
    return {
        "label": label,
        "source_dir": source,
        "source_signature": expected_input_signature(source),
        "required_files": required_signatures,
        "observed_contract": observed,
        "coordinate_rows": int(len(ids)),
        "coordinate_ids_ordered_sha256": ordered_array_sha256(ids),
        "coordinate_ids_set_sha256": ordered_array_sha256(np.sort(ids)),
        "coordinate_ids": ids,
        "checkpoint_contract": checkpoint_contract,
        "files": list(files.items()),
    }


def stage_round0005_maps(*, sources: dict[str, str], destination_root: str,
                         seal_path: str, corpus_identity_path: str | None = None,
                         namespace_name: str = "jina-en-2m/coordinate-row-id",
                         expected_rows: int = 2_000_000,
                         testbed_seal_path: str | None = None,
                         production_contract: bool = False) -> dict:
    """Validate and copy all nine maps into one fresh regular-file staging root."""
    if set(sources) != set(MAP_EXPECTATIONS):
        missing = sorted(set(MAP_EXPECTATIONS) - set(sources))
        extra = sorted(set(sources) - set(MAP_EXPECTATIONS))
        raise ValueError(f"map sources must name the exact nine maps: missing={missing} extra={extra}")
    destination_root = os.path.abspath(destination_root)
    seal_path = os.path.abspath(seal_path)
    if not destination_root.startswith("/data/") or not seal_path.startswith("/data/"):
        raise ValueError("Round 0005 staged maps and seal must live under /data")
    refuse_existing(destination_root, label="map staging root")
    refuse_existing(seal_path, label="map staging seal")

    # Validate every source and the common semantic universe before creating the
    # destination.  Ignored integration-checkout paths are never returned as
    # runtime inputs.
    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if production_contract and expected_rows != ROUND0005_TRAIN_ROWS:
        raise ValueError("production Round 0005 maps must contain exactly 2,000,000 rows")
    if production_contract and testbed_seal_path is None:
        raise ValueError("production map staging requires the shared testbed seal")
    seal_reference = (make_testbed_seal_reference(
        testbed_seal_path, require_round0005=production_contract)
                      if testbed_seal_path is not None else None)
    if seal_reference is not None:
        testbed_report = _json(seal_reference["seal_path"])
        sample_signature = testbed_report["sample_indices"]["signature"]
        if corpus_identity_path is None:
            corpus_identity_path = sample_signature["canonical_path"]
        elif os.path.realpath(corpus_identity_path) != sample_signature["canonical_path"]:
            raise ValueError("map corpus identity is not the sealed sample mapping")
    if corpus_identity_path is None:
        raise ValueError("map staging requires an explicit corpus/sample identity")
    validated = [_validate_map_source(
        label, sources[label], expected_rows=expected_rows,
        verify_checkpoint_metadata=production_contract)
                 for label in sorted(MAP_EXPECTATIONS)]
    canonical_ids = validated[0]["coordinate_ids"]
    canonical_set_sha = validated[0]["coordinate_ids_set_sha256"]
    for entry in validated[1:]:
        if entry["coordinate_ids_set_sha256"] != canonical_set_sha or not np.array_equal(
                np.sort(entry["coordinate_ids"]), np.sort(canonical_ids)):
            raise ValueError(f"map {entry['label']} coordinate semantic-ID universe differs")
    if production_contract and not np.array_equal(
            np.sort(canonical_ids), np.arange(ROUND0005_TRAIN_ROWS, dtype=np.int64)):
        raise ValueError("production map coordinate IDs must be exact testbed row positions")
    corpus_signature = expected_input_signature(corpus_identity_path)
    namespace = {
        "schema": SEMANTIC_NAMESPACE_SCHEMA,
        "name": namespace_name,
        "kind": "coordinate_semantic_id",
        "corpus_identity_sha256": corpus_signature["sha256"],
        "universe_sha256": canonical_set_sha,
        "row_count": int(len(canonical_ids)),
    }
    validate_semantic_namespace(namespace)

    create_fresh_directory(destination_root, label="map staging root")
    staged_entries = []
    for entry in validated:
        target_dir = os.path.join(destination_root, entry["label"])
        create_fresh_directory(target_dir, label=f"staged map {entry['label']}")
        for relative, source_file in entry.pop("files"):
            target_file = os.path.join(target_dir, relative)
            ensure_data_directory(
                os.path.dirname(target_file), label="staged map member parent")
            atomic_copy_new(source_file, target_file, immutable=True)
        staged_signature = expected_input_signature(target_dir)
        if _tree_content(staged_signature) != _tree_content(entry["source_signature"]):
            raise RuntimeError(f"staged map bytes differ from source: {entry['label']}")
        staged_required = {
            name: expected_input_signature(os.path.join(target_dir, name))
            for name in REQUIRED_MAP_FILES
        }
        staged_entries.append({
            "label": entry["label"],
            "staged_dir": target_dir,
            "staged_signature": staged_signature,
            "source_dir": entry["source_dir"],
            "source_signature": entry["source_signature"],
            "required_files": staged_required,
            "source_required_files": entry["required_files"],
            "observed_contract": entry["observed_contract"],
            "coordinate_rows": entry["coordinate_rows"],
            "coordinate_ids_ordered_sha256": entry["coordinate_ids_ordered_sha256"],
            "coordinate_ids_set_sha256": entry["coordinate_ids_set_sha256"],
            "checkpoint_contract": entry["checkpoint_contract"],
        })
    namespace_path = os.path.join(destination_root, "semantic-id-namespace.json")
    atomic_write_new_json(namespace_path, namespace, immutable=True)
    root_signature = expected_input_signature(destination_root)
    report = {
        "schema": MAP_STAGE_SCHEMA,
        "production_contract": bool(production_contract),
        "testbed_seal": seal_reference,
        "destination_root": destination_root,
        "maps_root_signature": root_signature,
        "corpus_identity": corpus_signature,
        "semantic_id_namespace": namespace,
        "semantic_id_namespace_signature": expected_input_signature(namespace_path),
        "expected_rows": int(expected_rows),
        "maps": staged_entries,
    }
    report["identity_sha256"] = sha256_bytes(canonical_json(report))
    atomic_write_new_json(seal_path, report, immutable=True)
    report["seal_path"] = seal_path
    report["seal_signature"] = expected_input_signature(seal_path)
    return report


def _resolved_snapshot_files(source_root: str) -> list[dict]:
    if not os.path.isdir(source_root):
        raise FileNotFoundError(source_root)
    entries = []
    for member in sorted(Path(source_root).rglob("*")):
        if member.is_symlink() and member.is_dir():
            raise ValueError(f"model snapshot contains a directory symlink: {member}")
        if member.is_dir():
            continue
        if not member.is_file() and not member.is_symlink():
            raise ValueError(f"model snapshot contains unsupported member: {member}")
        resolved = os.path.realpath(member)
        if not os.path.isfile(resolved):
            raise ValueError(f"model snapshot member does not resolve to a file: {member}")
        signature = expected_input_signature(resolved)
        entries.append({
            "relative_path": str(member.relative_to(source_root)),
            "source_path": os.path.abspath(member),
            "resolved_path": resolved,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
        })
    if not entries:
        raise ValueError("model snapshot is empty")
    return entries


def _validate_exact_jina_model_closure(root: str, members: list[dict]) -> dict:
    observed = {
        member["relative_path"]: (int(member["bytes"]), member["sha256"])
        for member in members
    }
    if observed != ROUND0005_JINA_MODEL_CLOSURE:
        missing = sorted(set(ROUND0005_JINA_MODEL_CLOSURE) - set(observed))
        extra = sorted(set(observed) - set(ROUND0005_JINA_MODEL_CLOSURE))
        changed = sorted(name for name in set(observed) & set(ROUND0005_JINA_MODEL_CLOSURE)
                         if observed[name] != ROUND0005_JINA_MODEL_CLOSURE[name])
        raise ValueError(
            f"Jina model closure mismatch: missing={missing} extra={extra} changed={changed}")
    config = _json(os.path.join(root, "config.json"))
    sentence_config = _json(os.path.join(root, "config_sentence_transformers.json"))
    pooling = _json(os.path.join(root, "1_Pooling", "config.json"))
    with open(os.path.join(root, "modules.json"), encoding="utf-8") as handle:
        modules = json.load(handle)
    expected_modules = [
        (0, "sentence_transformers.models.Transformer", ""),
        (1, "sentence_transformers.models.Pooling", "1_Pooling"),
        (2, "sentence_transformers.models.Normalize", "2_Normalize"),
    ]
    observed_modules = [(item.get("idx"), item.get("type"), item.get("path"))
                        for item in modules] if isinstance(modules, list) else None
    if observed_modules != expected_modules:
        raise ValueError("Jina model module closure is not Transformer/lasttoken/Normalize")
    if (config.get("hidden_size") != ROUND0005_DIMENSIONS or
            config.get("architectures") != ["EuroBertModel"] or
            config.get("model_type") != "eurobert"):
        raise ValueError("Jina transformer config does not prove the exact 768d EuroBERT model")
    if sentence_config.get("prompts") != {
            "query": "Query: ", "document": "Document: "}:
        raise ValueError("Jina sentence-transformer prompt bytes changed")
    if sentence_config.get("default_prompt_name") is not None:
        raise ValueError("Jina model unexpectedly applies a default prompt")
    pooling_modes = {
        key: pooling.get(key) for key in (
            "pooling_mode_cls_token", "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken")
    }
    if (pooling.get("word_embedding_dimension") != ROUND0005_DIMENSIONS or
            pooling_modes != {
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": False,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
                "pooling_mode_weightedmean_tokens": False,
                "pooling_mode_lasttoken": True,
            }):
        raise ValueError("Jina pooling config is not exact 768d last-token pooling")
    return {
        "model_id": ROUND0005_MODEL_ID,
        "model_revision": ROUND0005_MODEL_REVISION,
        "member_count": len(observed),
        "members": {name: {"bytes": value[0], "sha256": value[1]}
                    for name, value in sorted(observed.items())},
        "transformer": "EuroBertModel",
        "dimensions": ROUND0005_DIMENSIONS,
        "pooling": ROUND0005_POOLING,
        "normalization_module": "sentence_transformers.models.Normalize",
        "native_prompts": {"query": "Query: ", "document": "Document: "},
        "default_prompt_name": None,
    }


def _cpu_load_staged_jina_model(staged_root: str) -> dict:
    """Load the copied regular tree on CPU and report observed runtime semantics."""
    from experiments.embed_prompted_200k import (inspect_loaded_jina_model,
                                                  load_model)

    model, runtime_commit = load_model(
        device="cpu", dtype="float32", model_path=staged_root)
    proof = inspect_loaded_jina_model(model)
    proof.update({
        "loaded_on": "cpu",
        "runtime_commit_hash": runtime_commit,
        "revision_proof": "exact_resolved_file_closure",
        "revision_substitution_used": False,
    })
    del model
    return proof


def stage_regular_model_snapshot(*, source_snapshot: str, destination_root: str,
                                 seal_path: str, model_id: str,
                                 expected_revision: str,
                                 testbed_seal_path: str | None = None,
                                 production_contract: bool = False,
                                 _cpu_verify_fn=None) -> dict:
    """Resolve a symlinked HF snapshot into a fresh regular-file tree and seal it."""
    if (not re.fullmatch(r"[0-9a-f]{40}", expected_revision) or
            len(set(expected_revision)) == 1):
        raise ValueError("model revision must be a full immutable 40-hex commit")
    if (not isinstance(model_id, str) or not model_id.strip() or
            model_id.lower() in {"model", "placeholder", "unknown"}):
        raise ValueError("staged model_id must identify the real model")
    if production_contract and (
            model_id != ROUND0005_MODEL_ID or expected_revision != ROUND0005_MODEL_REVISION):
        raise ValueError("production staging requires the exact Round 0005 Jina model/revision")
    if production_contract and testbed_seal_path is None:
        raise ValueError("production model staging requires the shared testbed seal")
    seal_reference = (make_testbed_seal_reference(
        testbed_seal_path, require_round0005=production_contract)
                      if testbed_seal_path is not None else None)
    source_snapshot = os.path.abspath(source_snapshot)
    if os.path.basename(os.path.normpath(source_snapshot)) != expected_revision:
        raise ValueError("model snapshot path does not name the expected revision")
    destination_root = os.path.abspath(destination_root)
    seal_path = os.path.abspath(seal_path)
    if not destination_root.startswith("/data/") or not seal_path.startswith("/data/"):
        raise ValueError("staged model and seal must live under /data")
    refuse_existing(destination_root, label="model staging root")
    refuse_existing(seal_path, label="model staging seal")
    source_members = _resolved_snapshot_files(source_snapshot)
    exact_closure = (_validate_exact_jina_model_closure(source_snapshot, source_members)
                     if production_contract else None)
    create_fresh_directory(destination_root, label="model staging root")
    for member in source_members:
        destination = os.path.join(destination_root, member["relative_path"])
        ensure_data_directory(
            os.path.dirname(destination), label="staged model member parent")
        atomic_copy_new(member["resolved_path"], destination, immutable=True)
    staged_signature = expected_input_signature(destination_root)
    staged_members = {member["relative_path"]: member
                      for member in staged_signature["members"]
                      if member["kind"] == "file"}
    for source in source_members:
        staged = staged_members.get(source["relative_path"])
        if staged is None or staged["bytes"] != source["bytes"] or staged["sha256"] != source["sha256"]:
            raise RuntimeError(f"staged model member differs: {source['relative_path']}")
    if any(os.path.islink(path) for path in Path(destination_root).rglob("*")):
        raise RuntimeError("staged model unexpectedly contains a symlink")
    cpu_load_verification = None
    if production_contract:
        verify_fn = _cpu_verify_fn or _cpu_load_staged_jina_model
        cpu_load_verification = verify_fn(destination_root)
        expected_runtime = {
            "dimensions": ROUND0005_DIMENSIONS,
            "pooling": ROUND0005_POOLING,
            "normalized": True,
            "default_prompt_name": None,
        }
        for key, expected in expected_runtime.items():
            if cpu_load_verification.get(key) != expected:
                raise ValueError(
                    f"CPU-loaded staged Jina model {key} mismatch: "
                    f"{cpu_load_verification.get(key)!r} != {expected!r}")
        if cpu_load_verification.get("revision_substitution_used") is not False:
            raise ValueError("staged-model CPU load used revision substitution")
        after_cpu_load = expected_input_signature(destination_root)
        if after_cpu_load != staged_signature:
            raise RuntimeError("CPU model load mutated the staged regular-file closure")
    report = {
        "schema": MODEL_STAGE_SCHEMA,
        "production_contract": bool(production_contract),
        "testbed_seal": seal_reference,
        "model_id": model_id,
        "model_revision": expected_revision,
        "source_snapshot": source_snapshot,
        "source_resolved_members": source_members,
        "staged_root": destination_root,
        "staged_signature": staged_signature,
        "regular_files_only": True,
        "exact_model_closure": exact_closure,
        "cpu_load_verification": cpu_load_verification,
    }
    report["identity_sha256"] = sha256_bytes(canonical_json(report))
    atomic_write_new_json(seal_path, report, immutable=True)
    report["seal_path"] = seal_path
    report["seal_signature"] = expected_input_signature(seal_path)
    return report


def validate_staged_map_seal(path: str, *, expected_root: str | None = None,
                             expected_testbed_seal: str | None = None,
                             require_round0005: bool = False) -> dict:
    report = _json(path)
    if report.get("schema") != MAP_STAGE_SCHEMA or len(report.get("maps") or []) != 9:
        raise ValueError("invalid Round 0005 staged-map seal")
    payload = {key: value for key, value in report.items() if key != "identity_sha256"}
    if sha256_bytes(canonical_json(payload)) != report.get("identity_sha256"):
        raise ValueError("staged-map seal identity mismatch")
    production_contract = report.get("production_contract") is True
    if require_round0005 and not production_contract:
        raise ValueError("queue requires production-contract Round 0005 map staging")
    seal_reference = report.get("testbed_seal")
    if production_contract or expected_testbed_seal is not None:
        validate_testbed_seal_reference(
            seal_reference, expected_seal=expected_testbed_seal,
            require_round0005=require_round0005)
    root = os.path.abspath(expected_root or report["destination_root"])
    if root != report.get("destination_root"):
        raise ValueError("staged-map root mismatch")
    if os.path.islink(root) or not os.path.isdir(root):
        raise ValueError("staged-map root must be a regular directory")
    if expected_input_signature(root) != report.get("maps_root_signature"):
        raise ValueError("staged-map root bytes changed")
    namespace = validate_semantic_namespace(report.get("semantic_id_namespace"))
    expected_rows = report.get("expected_rows")
    if (not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or
            expected_rows <= 0 or namespace["row_count"] != expected_rows):
        raise ValueError("staged-map row count and semantic namespace disagree")
    if require_round0005 and expected_rows != ROUND0005_TRAIN_ROWS:
        raise ValueError("queue requires exact 2,000,000-row Round 0005 maps")
    if production_contract:
        sample_signature = _json(seal_reference["seal_path"])["sample_indices"]["signature"]
        if report.get("corpus_identity") != sample_signature:
            raise ValueError("staged maps are not bound to the sealed sample mapping")
    namespace_path = os.path.join(root, "semantic-id-namespace.json")
    if expected_input_signature(namespace_path) != report.get(
            "semantic_id_namespace_signature"):
        raise ValueError("staged-map semantic namespace proof bytes changed")
    if _json(namespace_path) != namespace:
        raise ValueError("staged-map semantic namespace proof disagrees with seal")
    entries = report.get("maps") or []
    labels = [entry.get("label") for entry in entries if isinstance(entry, dict)]
    if len(entries) != 9 or set(labels) != set(MAP_EXPECTATIONS) or len(set(labels)) != 9:
        raise ValueError("staged-map seal does not name the exact nine maps")
    for entry in entries:
        label = entry["label"]
        staged_dir = os.path.join(root, label)
        if entry.get("staged_dir") != staged_dir:
            raise ValueError(f"staged-map {label} path escapes/collides with its label")
        verified = _validate_map_source(
            label, staged_dir, expected_rows=expected_rows,
            verify_checkpoint_metadata=production_contract)
        if entry.get("staged_signature") != verified["source_signature"]:
            raise ValueError(f"staged-map {label} directory signature mismatch")
        required = entry.get("required_files")
        if not isinstance(required, dict) or set(required) != set(REQUIRED_MAP_FILES):
            raise ValueError(f"staged-map {label} required-file set mismatch")
        if required != verified["required_files"]:
            raise ValueError(f"staged-map {label} required-file bytes changed")
        for key in ("coordinate_rows", "coordinate_ids_ordered_sha256",
                    "coordinate_ids_set_sha256", "observed_contract",
                    "checkpoint_contract"):
            if entry.get(key) != verified[key]:
                raise ValueError(f"staged-map {label} recorded {key} mismatch")
        if entry["coordinate_ids_set_sha256"] != namespace["universe_sha256"]:
            raise ValueError(f"staged-map {label} semantic universe differs from namespace")
    if production_contract and namespace["universe_sha256"] != ordered_array_sha256(
            np.arange(ROUND0005_TRAIN_ROWS, dtype=np.int64)):
        raise ValueError("staged production maps do not use exact testbed row-position IDs")
    return report


def validate_staged_model_seal(path: str, *, expected_root: str | None = None,
                               expected_revision: str | None = None,
                               expected_model_id: str | None = None,
                               expected_testbed_seal: str | None = None,
                               require_round0005: bool = False) -> dict:
    report = _json(path)
    if report.get("schema") != MODEL_STAGE_SCHEMA or report.get("regular_files_only") is not True:
        raise ValueError("invalid regular-file model seal")
    revision = report.get("model_revision")
    if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or len(set(revision)) == 1):
        raise ValueError("staged-model seal has a placeholder/mutable revision")
    model_id = report.get("model_id")
    if (not isinstance(model_id, str) or not model_id.strip() or
            model_id.lower() in {"model", "placeholder", "unknown"}):
        raise ValueError("staged-model seal has an invalid model_id")
    payload = {key: value for key, value in report.items() if key != "identity_sha256"}
    if sha256_bytes(canonical_json(payload)) != report.get("identity_sha256"):
        raise ValueError("staged-model seal identity mismatch")
    production_contract = report.get("production_contract") is True
    if require_round0005 and not production_contract:
        raise ValueError("queue requires production-contract Jina model staging")
    if production_contract or expected_testbed_seal is not None:
        validate_testbed_seal_reference(
            report.get("testbed_seal"), expected_seal=expected_testbed_seal,
            require_round0005=require_round0005)
    root = os.path.abspath(expected_root or report["staged_root"])
    if root != report.get("staged_root"):
        raise ValueError("staged-model root mismatch")
    if expected_revision is not None and report.get("model_revision") != expected_revision:
        raise ValueError("staged-model revision mismatch")
    if expected_model_id is not None and report.get("model_id") != expected_model_id:
        raise ValueError("staged-model model_id mismatch")
    if require_round0005 and (
            report.get("model_id") != ROUND0005_MODEL_ID or
            report.get("model_revision") != ROUND0005_MODEL_REVISION):
        raise ValueError("queue requires the exact Round 0005 Jina model/revision")
    if expected_input_signature(root) != report.get("staged_signature"):
        raise ValueError("staged-model bytes changed")
    if any(os.path.islink(path) for path in Path(root).rglob("*")):
        raise ValueError("staged-model tree contains a symlink")
    staged_members = {member["relative_path"]: member
                      for member in report["staged_signature"].get("members", [])
                      if member.get("kind") == "file"}
    resolved_members = report.get("source_resolved_members")
    if not isinstance(resolved_members, list) or not resolved_members:
        raise ValueError("staged-model seal has no resolved source members")
    for source in resolved_members:
        relative = source.get("relative_path") if isinstance(source, dict) else None
        staged = staged_members.get(relative)
        if (staged is None or staged.get("bytes") != source.get("bytes") or
                staged.get("sha256") != source.get("sha256")):
            raise ValueError(f"staged-model resolved member mismatch: {relative!r}")
    if len(staged_members) != len(resolved_members):
        raise ValueError("staged-model member cardinality mismatch")
    if production_contract:
        exact_closure = _validate_exact_jina_model_closure(root, [
            {
                "relative_path": member["relative_path"],
                "bytes": member["bytes"],
                "sha256": member["sha256"],
            }
            for member in report["staged_signature"].get("members", [])
            if member.get("kind") == "file"
        ])
        if exact_closure != report.get("exact_model_closure"):
            raise ValueError("staged-model exact closure proof mismatch")
        cpu_proof = report.get("cpu_load_verification")
        if (not isinstance(cpu_proof, dict) or cpu_proof.get("loaded_on") != "cpu" or
                cpu_proof.get("dimensions") != ROUND0005_DIMENSIONS or
                cpu_proof.get("pooling") != ROUND0005_POOLING or
                cpu_proof.get("normalized") is not True or
                cpu_proof.get("revision_substitution_used") is not False):
            raise ValueError("staged-model CPU-load proof is absent or invalid")
    return report


def cross_check_round0005_data_identity(*, testbed_seal_path: str,
                                        maps_seal_path: str,
                                        model_seal_path: str,
                                        query_manifest_path: str,
                                        calibration_manifest_path: str,
                                        maps_root: str | None = None,
                                        model_root: str | None = None) -> dict:
    """Revalidate one shared seal reference and return a queue-bindable closure.

    The foundational testbed seal cannot contain signatures of artifacts that
    themselves reference that seal without creating a hash cycle.  This
    function closes the cycle-free graph at queue preparation: it deeply
    verifies the substrate once, verifies every downstream artifact against
    the identical signed reference, and hashes their full signatures and
    immutable identities into one deterministic closure.
    """
    from basemap.query_artifact import (load_query_artifact,
                                        round0005_query_convention)
    from experiments.calibrate_jina_embedding import validate_inventory

    testbed = validate_round0005_testbed_seal(
        testbed_seal_path, require_round0005=True)
    reference = make_testbed_seal_reference(
        testbed_seal_path, require_round0005=True, deep=False)
    maps = validate_staged_map_seal(
        maps_seal_path, expected_root=maps_root,
        expected_testbed_seal=testbed_seal_path, require_round0005=True)
    model = validate_staged_model_seal(
        model_seal_path, expected_root=model_root,
        expected_revision=ROUND0005_MODEL_REVISION,
        expected_model_id=ROUND0005_MODEL_ID,
        expected_testbed_seal=testbed_seal_path, require_round0005=True)
    query = load_query_artifact(
        query_manifest_path, testbed=testbed["testbed_root"],
        expected_convention=round0005_query_convention(),
        expected_testbed_seal=testbed_seal_path, require_round0005=True)
    calibration, _ = validate_inventory(
        calibration_manifest_path, expected_testbed_seal=testbed_seal_path,
        require_round0005=True)
    for label, artifact in (("maps", maps), ("model", model),
                            ("query", query["manifest"]),
                            ("calibration", calibration)):
        if artifact.get("testbed_seal") != reference:
            raise ValueError(f"{label} artifact does not carry the identical testbed seal")
    closure = {
        "schema": DATA_IDENTITY_CLOSURE_SCHEMA,
        "testbed_seal": reference,
        "corpus_identity_sha256": testbed["corpus_identity_sha256"],
        "artifacts": {
            "maps": {
                "seal_signature": expected_input_signature(maps_seal_path),
                "identity_sha256": maps["identity_sha256"],
                "root_signature": maps["maps_root_signature"],
            },
            "model": {
                "seal_signature": expected_input_signature(model_seal_path),
                "identity_sha256": model["identity_sha256"],
                "root_signature": model["staged_signature"],
                "model_id": model["model_id"],
                "model_revision": model["model_revision"],
            },
            "query": {
                "manifest_signature": expected_input_signature(query_manifest_path),
                "identity_sha256": query["identity_sha256"],
                "embeddings": query["manifest"]["embeddings"],
                "ids": query["manifest"]["ids"],
            },
            "calibration": {
                "manifest_signature": expected_input_signature(calibration_manifest_path),
                "identity_sha256": calibration["identity_sha256"],
                "inventory": calibration["inventory"],
            },
        },
    }
    closure["identity_sha256"] = sha256_bytes(canonical_json(closure))
    return closure
