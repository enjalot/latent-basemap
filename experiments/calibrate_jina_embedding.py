"""No-training Jina calibration over two fresh production-shaped chunks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       ordered_array_sha256, sha256_bytes)
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0005_staging import (
    ROUND0005_CALIBRATION_SEED, ROUND0005_DIMENSIONS, ROUND0005_MODEL_ID,
    ROUND0005_MODEL_REVISION, ROUND0005_NORMALIZATION, ROUND0005_POOLING,
    ROUND0005_SOURCE_DTYPE, ROUND0005_SOURCE_ROWS, ROUND0005_SOURCE_SHARD_ROWS,
    ROUND0005_STORAGE_DTYPE,
    validate_staged_model_seal, validate_testbed_seal_reference,
)
from basemap.run_controller import require_active_lease
from experiments.embed_prompted_200k import (
    MODEL_ID, OUTER_CHUNK_ROWS, PROMPT_PREFIX, embed_outer_chunks, embed_texts,
    inspect_loaded_jina_model, load_model, ordered_text_sha256,
)

INVENTORY_SCHEMA = "jina_calibration_inventory.v2"
CALIBRATION_SCHEMA = "round0005_jina_embedding_calibration.v3"


def _quantiles(values) -> dict:
    values = np.asarray(values)
    return {name: int(np.percentile(values, percentile)) for name, percentile in
            (("median", 50), ("p95", 95), ("p99", 99), ("maximum", 100))}


def effective_token_lengths(tokenizer, prompted_texts, *, max_length: int,
                            batch_rows: int = 2048) -> np.ndarray:
    """Measure the actual prompted and effectively truncated token sequences."""
    if max_length <= 0:
        raise ValueError("effective tokenizer max_length must be positive")
    lengths = []
    for start in range(0, len(prompted_texts), batch_rows):
        batch = tokenizer(
            prompted_texts[start:start + batch_rows], padding=False, truncation=True,
            max_length=max_length, add_special_tokens=True)["input_ids"]
        if len(batch) != len(prompted_texts[start:start + batch_rows]):
            raise ValueError("tokenizer changed the calibration row count")
        lengths.extend(len(row) for row in batch)
    values = np.asarray(lengths, dtype=np.int64)
    if len(values) != len(prompted_texts) or np.any(values <= 0) or np.any(values > max_length):
        raise ValueError("tokenizer returned invalid effectively truncated lengths")
    return values


def _inventory_identity(manifest: dict) -> str:
    fields = (
        "production_contract", "testbed_seal", "seed", "selection",
        "source_position_range", "outer_chunk_rows", "inventory",
        "source_sample_indices", "source_text_shards", "source_embedding_shards",
        "source_embedding_rows", "source_embedding_shard_rows", "convention", "phases",
    )
    missing = [field for field in fields if field not in manifest]
    if missing:
        raise ValueError(f"calibration inventory identity missing {missing}")
    return sha256_bytes(canonical_json({field: manifest[field] for field in fields}))


def validate_inventory(manifest_path: str, *, expected_testbed_seal: str | None = None,
                       require_round0005: bool = False) -> tuple[dict, pd.DataFrame]:
    with open(manifest_path, encoding="utf-8") as handle:
        inventory = json.load(handle)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("invalid Jina calibration inventory schema")
    if _inventory_identity(inventory) != inventory.get("identity_sha256"):
        raise ValueError("Jina calibration inventory identity mismatch")
    production_contract = inventory.get("production_contract") is True
    if require_round0005 and not production_contract:
        raise ValueError("queue requires a production-contract calibration inventory")
    if production_contract or expected_testbed_seal is not None:
        validate_testbed_seal_reference(
            inventory.get("testbed_seal"), expected_seal=expected_testbed_seal,
            require_round0005=require_round0005)
    expected_inventory = inventory.get("inventory") or {}
    if expected_input_signature(expected_inventory.get("canonical_path", "")) != expected_inventory:
        raise ValueError("Jina calibration inventory bytes changed")
    if inventory.get("outer_chunk_rows") != OUTER_CHUNK_ROWS:
        raise ValueError("Jina calibration inventory chunk size is not 25,000")
    frame = pd.read_parquet(expected_inventory["canonical_path"])
    required_columns = ["phase", "source_position", "global_id", "text", "character_length"]
    if list(frame.columns) != required_columns:
        raise ValueError(f"calibration inventory columns must be {required_columns}")
    if len(frame) != 2 * OUTER_CHUNK_ROWS:
        raise ValueError("calibration inventory must contain exactly 50,000 rows")
    for column in ("source_position", "global_id", "character_length"):
        if frame[column].dtype != np.dtype("int64"):
            raise ValueError(f"calibration inventory {column} must have exact int64 dtype")
    if not all(isinstance(value, str) for value in frame.text.tolist()):
        raise ValueError("calibration inventory text values must all be strings")
    measured_character_lengths = np.asarray(
        [len(value) for value in frame.text.tolist()], dtype=np.int64)
    if not np.array_equal(frame.character_length.to_numpy(), measured_character_lengths):
        raise ValueError("calibration inventory character lengths do not match text bytes")
    expected_phases = ["calibration"] * OUTER_CHUNK_ROWS + ["heldout"] * OUTER_CHUNK_ROWS
    if frame.phase.tolist() != expected_phases:
        raise ValueError("calibration inventory must contain ordered calibration then heldout rows")
    positions = frame.source_position.to_numpy()
    if not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions))):
        raise ValueError("calibration inventory does not preserve contiguous production order")
    if inventory.get("source_position_range") != [int(positions[0]), int(positions[-1]) + 1]:
        raise ValueError("calibration inventory source-position range mismatch")
    ids = frame.global_id.to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("calibration inventory contains duplicate global IDs")
    expected_phase_reports = {}
    for phase in ("calibration", "heldout"):
        subset = frame[frame.phase == phase]
        expected_phase_reports[phase] = {
            "rows": int(len(subset)),
            "global_ids_ordered_sha256": ordered_array_sha256(
                subset.global_id.to_numpy()),
            "texts_ordered_sha256": ordered_text_sha256(subset.text.tolist()),
            "character_length_quantiles": {
                name: float(np.percentile(subset.character_length, percentile))
                for name, percentile in (
                    ("median", 50), ("p95", 95), ("p99", 99), ("maximum", 100))},
        }
    if inventory.get("phases") != expected_phase_reports:
        raise ValueError("calibration inventory phase identity does not match signed rows")
    for signature in [*inventory.get("source_text_shards", []),
                      *inventory.get("source_embedding_shards", [])]:
        if (not isinstance(signature, dict) or
                expected_input_signature(signature.get("canonical_path", "")) != signature):
            raise ValueError("calibration source shard bytes changed")
    if (sum(inventory.get("source_embedding_shard_rows") or []) !=
            inventory.get("source_embedding_rows")):
        raise ValueError("calibration source embedding row inventory is inconsistent")
    selection = inventory.get("selection") or {}
    if selection.get("source_position_range") != inventory.get("source_position_range"):
        raise ValueError("calibration selection and source-position range disagree")
    if selection.get("chunk_rows") != OUTER_CHUNK_ROWS or \
            selection.get("length_sorting") is not False:
        raise ValueError("calibration selection is not two unsorted production-shaped chunks")
    if production_contract:
        expected_convention = {
            "model_id": ROUND0005_MODEL_ID,
            "model_revision": ROUND0005_MODEL_REVISION,
            "prompt_policy": "literal_prefix",
            "prompt_bytes_hex": PROMPT_PREFIX.encode("utf-8").hex(),
            "pooling": ROUND0005_POOLING,
            "source_dtype": ROUND0005_SOURCE_DTYPE,
            "compute_dtype": ROUND0005_STORAGE_DTYPE,
            "output_dtype": ROUND0005_STORAGE_DTYPE,
            "normalization": ROUND0005_NORMALIZATION,
            "dimensions": ROUND0005_DIMENSIONS,
        }
        if (inventory.get("seed") != ROUND0005_CALIBRATION_SEED or
                inventory.get("source_embedding_rows") != ROUND0005_SOURCE_ROWS or
                inventory.get("source_embedding_shard_rows") !=
                list(ROUND0005_SOURCE_SHARD_ROWS) or
                inventory.get("convention") != expected_convention or
                selection.get("method") != "seeded_aligned_adjacent_source_chunks" or
                int(positions[0]) % OUTER_CHUNK_ROWS != 0 or
                not np.array_equal(ids, positions)):
            raise ValueError("calibration inventory violates exact Round 0005 constants/order")
        with open(inventory["testbed_seal"]["seal_path"], encoding="utf-8") as handle:
            sealed = json.load(handle)
        if (inventory.get("source_sample_indices") !=
                sealed["sample_indices"]["signature"] or
                inventory.get("source_embedding_shards") != [
                    item["signature"] for item in sealed["source_embeddings"]["shards"]] or
                inventory.get("source_text_shards") != [
                    item["signature"] for item in sealed["matching_source_texts"]["shards"]]):
            raise ValueError("calibration inventory sources differ from the shared testbed seal")
    return inventory, frame


def _token_regimes(lengths: np.ndarray, ids: np.ndarray) -> dict:
    """Bind concrete rows for median, p95, p99, and maximum regimes."""
    regimes = {}
    for name, percentile in (("median", 50), ("p95", 95), ("p99", 99),
                             ("maximum", 100)):
        target = int(np.percentile(lengths, percentile))
        distance = np.abs(lengths - target)
        position = int(np.flatnonzero(distance == distance.min())[0])
        regimes[name] = {
            "target_effective_tokens": target,
            "representative_chunk_position": position,
            "representative_global_id": int(ids[position]),
            "representative_effective_tokens": int(lengths[position]),
            "rows_at_or_above_target": int(np.sum(lengths >= target)),
        }
    return regimes


def certify_with_model(*, frame: pd.DataFrame, model, model_revision: str,
                       convention: dict, out_dir: str, batch_size: int = 256,
                       max_prediction_error: float = 0.15, embed_fn=embed_texts,
                       inventory_load_wall_s: float = 0.0,
                       model_load_wall_s: float = 0.0,
                       production_contract: bool = False,
                       testbed_seal: dict | None = None,
                       inventory_identity_sha256: str | None = None) -> dict:
    """Run the full contract; fake models can exercise it with CUDA hidden."""
    if batch_size != 256:
        raise ValueError("Round 0005 representative calibration requires batch size 256")
    if (not isinstance(model_revision, str) or len(model_revision) != 40 or
            any(char not in "0123456789abcdef" for char in model_revision) or
            len(set(model_revision)) == 1):
        raise ValueError("calibration model revision must be immutable, full, and non-placeholder")
    if not 0.0 <= float(max_prediction_error) <= 0.15:
        raise ValueError("calibration heldout prediction-error limit may not exceed 15 percent")
    if len(frame) != 2 * OUTER_CHUNK_ROWS:
        raise ValueError("representative calibration requires exactly 50,000 rows")
    expected_phases = ["calibration"] * OUTER_CHUNK_ROWS + ["heldout"] * OUTER_CHUNK_ROWS
    if frame.phase.tolist() != expected_phases:
        raise ValueError("calibration phases/order are not production-shaped")
    source_positions = frame.source_position.to_numpy(np.int64)
    if not np.array_equal(
            source_positions,
            np.arange(source_positions[0], source_positions[0] + len(source_positions))):
        raise ValueError("calibration rows are length-sorted/permuted rather than production order")
    if (frame.source_position.dtype != np.dtype("int64") or
            frame.global_id.dtype != np.dtype("int64") or
            frame.character_length.dtype != np.dtype("int64")):
        raise ValueError("calibration identity/count columns must have exact int64 dtype")
    if len(np.unique(frame.global_id.to_numpy())) != len(frame):
        raise ValueError("calibration rows contain duplicate global IDs")
    if not all(isinstance(value, str) for value in frame.text.tolist()):
        raise ValueError("calibration texts must all be strings")
    legacy_convention = {
        "model_id": MODEL_ID, "model_revision": model_revision,
        "prompt_bytes_hex": PROMPT_PREFIX.encode("utf-8").hex(),
        "pooling": "lasttoken", "dtype": "float32", "normalization": "l2",
    }
    production_convention = {
        "model_id": ROUND0005_MODEL_ID,
        "model_revision": ROUND0005_MODEL_REVISION,
        "prompt_policy": "literal_prefix",
        "prompt_bytes_hex": PROMPT_PREFIX.encode("utf-8").hex(),
        "pooling": ROUND0005_POOLING,
        "source_dtype": ROUND0005_SOURCE_DTYPE,
        "compute_dtype": ROUND0005_STORAGE_DTYPE,
        "output_dtype": ROUND0005_STORAGE_DTYPE,
        "normalization": ROUND0005_NORMALIZATION,
        "dimensions": ROUND0005_DIMENSIONS,
    }
    expected_convention = production_convention if production_contract else legacy_convention
    if convention != expected_convention:
        raise ValueError("Jina calibration convention mismatch")
    if production_contract:
        validate_testbed_seal_reference(testbed_seal, require_round0005=True)
        if model_revision != ROUND0005_MODEL_REVISION or not inventory_identity_sha256:
            raise ValueError("production calibration lacks exact revision/inventory identity")
    out_dir = os.path.abspath(out_dir)
    if not out_dir.startswith("/data/"):
        raise ValueError("calibration output must live under /data")
    create_fresh_directory(out_dir, label="certifying calibration output root")
    all_started = time.monotonic()
    ids = frame.global_id.to_numpy(np.int64)
    texts = frame.text.tolist()
    text_by_id = {int(row_id): text for row_id, text in zip(ids, texts)}

    token_started = time.monotonic()
    prompted = [PROMPT_PREFIX + text for text in texts]
    max_length = int(getattr(model, "max_seq_length", 0) or
                     getattr(model.tokenizer, "model_max_length", 0))
    token_lengths = effective_token_lengths(
        model.tokenizer, prompted, max_length=max_length)
    token_wall = time.monotonic() - token_started
    token_report = {}
    for phase, start in (("calibration", 0), ("heldout", OUTER_CHUNK_ROWS)):
        stop = start + OUTER_CHUNK_ROWS
        raw_prompted_lengths = np.asarray(
            [len(value) for value in prompted[start:stop]], dtype=np.int64)
        token_report[phase] = {
            "rows": OUTER_CHUNK_ROWS,
            "prompt_applied_before_tokenization": True,
            "effective_truncation": True,
            "max_sequence_length": max_length,
            "token_length_quantiles": _quantiles(token_lengths[start:stop]),
            "rows_at_max_length": int(np.sum(token_lengths[start:stop] == max_length)),
            "global_ids_ordered_sha256": ordered_array_sha256(ids[start:stop]),
            "prompted_texts_ordered_sha256": ordered_text_sha256(prompted[start:stop]),
            "prompted_character_length_quantiles": _quantiles(raw_prompted_lengths),
            "regimes": _token_regimes(token_lengths[start:stop], ids[start:stop]),
        }

    warm_started = time.monotonic()
    _, warm_telemetry = embed_fn(
        model, prompted[:256], batch_size=256, show_progress=False,
        return_telemetry=True)
    warmup_wall = time.monotonic() - warm_started

    def fetch(chunk_ids, _text_dir, _text_shards, _offsets):
        return [text_by_id[int(row_id)] for row_id in chunk_ids]

    chunk_report = embed_outer_chunks(
        model, sample_indices=ids,
        out_train=os.path.join(out_dir, "embeddings"),
        receipt_dir=os.path.join(out_dir, "chunk-receipts"),
        text_dir="sealed-inventory", text_shards=[], offsets=np.array([0, len(ids)]),
        model_commit=model_revision, compute_dtype="float32", batch_size=256,
        chunk_rows=OUTER_CHUNK_ROWS, fetch_fn=fetch, embed_fn=embed_fn)
    exactly_two_new = (
        len(chunk_report["chunks"]) == 2 and chunk_report["new_chunks"] == 2 and
        chunk_report["resumed_chunks"] == 0)
    calibration_wall = float(chunk_report["chunks"][0]["wall_s"])
    heldout_wall = float(chunk_report["chunks"][1]["wall_s"])
    prediction_error = abs(calibration_wall - heldout_wall) / max(heldout_wall, 1e-9)
    zero_oom = warm_telemetry.get("oom_retries") == 0 and chunk_report["oom_retries"] == 0
    required_chunk_phases = {
        "source_fetch", "prompt_and_context_hash",
        "model_encode_including_tokenization", "validate_and_output_publish",
        "receipt_publish",
    }
    chunk_phases_complete = all(
        set(chunk.get("phase_wall_s") or {}) == required_chunk_phases
        for chunk in chunk_report["chunks"])
    output_shapes = [chunk.get("output_shape") for chunk in chunk_report["chunks"]]
    output_shapes_valid = (
        all(shape == [OUTER_CHUNK_ROWS, ROUND0005_DIMENSIONS]
            for shape in output_shapes)
        if production_contract else
        bool(output_shapes and all(
            isinstance(shape, list) and len(shape) == 2 and
            shape[0] == OUTER_CHUNK_ROWS and shape[1] == output_shapes[0][1]
            for shape in output_shapes)))
    checks = {
        "production_order_preserved": True,
        "prompted_effectively_truncated_token_profile": all(
            set(report["token_length_quantiles"]) == {"median", "p95", "p99", "maximum"}
            for report in token_report.values()),
        "median_p95_p99_maximum_regimes_bound": all(
            set(report["regimes"]) == {"median", "p95", "p99", "maximum"}
            for report in token_report.values()),
        "batch_256": all((chunk.get("embedding") or {}).get("requested_batch_size") == 256
                         for chunk in chunk_report["chunks"]),
        "exactly_two_new_atomic_25k_chunks": (
            exactly_two_new and chunk_report["chunk_rows"] == OUTER_CHUNK_ROWS),
        "output_shape_and_dimension": output_shapes_valid,
        "zero_oom": zero_oom,
        "heldout_prediction_error": prediction_error <= max_prediction_error,
        "prediction_covers_fetch_load_embed_write": chunk_phases_complete,
        "corrupt_future_preflight_before_new_work": (
            chunk_report.get("preflight_complete_before_new_work") is True),
    }
    fixed_overhead = (float(inventory_load_wall_s) + float(model_load_wall_s) +
                      token_wall + warmup_wall)
    predicted_full_node = fixed_overhead + 2.0 * calibration_wall
    observed_full_node = fixed_overhead + calibration_wall + heldout_wall
    return {
        "schema": CALIBRATION_SCHEMA,
        "production_contract": bool(production_contract),
        "testbed_seal": testbed_seal,
        "inventory_identity_sha256": inventory_identity_sha256,
        "passed": all(checks.values()),
        "training_performed": False,
        "checks": checks,
        "convention": convention,
        "phase_wall_s": {
            "inventory_load": round(float(inventory_load_wall_s), 6),
            "model_load": round(float(model_load_wall_s), 6),
            "token_profile": round(token_wall, 6),
            "warmup": round(warmup_wall, 6),
            "calibration_chunk_all_phases": calibration_wall,
            "heldout_chunk_all_phases": heldout_wall,
            "fixed_inventory_model_tokenizer_warmup": round(fixed_overhead, 6),
            "predicted_full_node": round(predicted_full_node, 6),
            "observed_full_node": round(observed_full_node, 6),
            "total_after_model_load": round(time.monotonic() - all_started, 6),
        },
        "token_lengths": token_report,
        "prediction": {
            "calibration_prediction_s": calibration_wall,
            "heldout_actual_s": heldout_wall,
            "absolute_relative_error": round(prediction_error, 6),
            "maximum_error": max_prediction_error,
            "covered_chunk_phases": [
                "source_fetch", "prompt_and_context_hash",
                "model_encode_including_tokenization", "validate_and_output_publish",
                "receipt_publish"],
            "includes_inventory_load_s": float(inventory_load_wall_s),
            "includes_model_load_s": float(model_load_wall_s),
            "predicted_full_node_s": round(predicted_full_node, 6),
            "observed_full_node_s": round(observed_full_node, 6),
        },
        "warmup": warm_telemetry,
        "chunks": chunk_report,
    }


def main(argv=None) -> int:
    from basemap.run_controller import require_round0005_child_admission
    require_round0005_child_admission("experiments/calibrate_jina_embedding.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-manifest", required=True)
    parser.add_argument("--testbed-seal", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-seal", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prediction-error", type=float, default=0.15)
    args = parser.parse_args(argv)
    if args.dtype != "float32":
        raise ValueError("Round 0005 representative calibration requires float32 output")
    require_active_lease()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Jina embedding calibration requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    inventory_started = time.monotonic()
    inventory, frame = validate_inventory(
        args.inventory_manifest, expected_testbed_seal=args.testbed_seal,
        require_round0005=True)
    inventory_wall = time.monotonic() - inventory_started
    model_seal = validate_staged_model_seal(
        args.model_seal, expected_root=args.model_path,
        expected_revision=ROUND0005_MODEL_REVISION,
        expected_model_id=ROUND0005_MODEL_ID,
        expected_testbed_seal=args.testbed_seal, require_round0005=True)
    revision = model_seal["model_revision"]
    convention = inventory.get("convention") or {}
    if convention.get("model_revision") != revision:
        raise ValueError("calibration inventory and staged model revisions differ")
    load_started = time.monotonic()
    model, commit = load_model(
        device="cuda", dtype=args.dtype, model_path=args.model_path)
    model_load_wall = time.monotonic() - load_started
    runtime_model = inspect_loaded_jina_model(model)
    if commit is not None and commit != revision:
        raise ValueError(f"loaded model revision {commit!r} != sealed revision {revision!r}")
    report = certify_with_model(
        frame=frame, model=model, model_revision=revision, convention=convention,
        out_dir=args.out_dir, batch_size=args.batch_size,
        max_prediction_error=args.max_prediction_error,
        inventory_load_wall_s=inventory_wall, model_load_wall_s=model_load_wall,
        production_contract=True, testbed_seal=inventory["testbed_seal"],
        inventory_identity_sha256=inventory["identity_sha256"])
    report["runtime_model_verification"] = {
        **runtime_model,
        "runtime_commit_hash": commit,
        "revision_proof": "exact_staged_model_closure",
        "revision_substitution_used": False,
    }
    report["inventory_manifest"] = expected_input_signature(args.inventory_manifest)
    report["model_seal"] = expected_input_signature(args.model_seal)
    report["model_path"] = expected_input_signature(args.model_path)
    report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4)
    report_path = os.path.join(args.out_dir, "report.json")
    atomic_write_new_json(report_path, report, immutable=True)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "prediction": report["prediction"],
                      "peak_gpu_gb": report["peak_gpu_gb"]}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
