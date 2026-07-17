"""Prepare two matched 25k-row Jina calibration inventories without CUDA."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       ordered_array_sha256, sha256_bytes)
from basemap.output_safety import (atomic_build_new_file, atomic_write_new_json,
                                   refuse_existing)
from basemap.round0005_staging import (
    ROUND0005_CALIBRATION_SEED, ROUND0005_DIMENSIONS, ROUND0005_MODEL_ID,
    ROUND0005_MODEL_REVISION, ROUND0005_NORMALIZATION, ROUND0005_POOLING,
    ROUND0005_SOURCE_DTYPE, ROUND0005_SOURCE_ROWS, ROUND0005_STORAGE_DTYPE,
    make_testbed_seal_reference,
)
from experiments.embed_prompted_200k import (
    MODEL_ID, OUTER_CHUNK_ROWS, PROMPT_PREFIX, fetch_texts_for_indices,
    ordered_text_sha256, verify_shard_alignment,
)


def _quantiles(values) -> dict:
    values = np.asarray(values)
    return {name: float(np.percentile(values, percentile)) for name, percentile in
            (("median", 50), ("p95", 95), ("p99", 99), ("maximum", 100))}


def select_calibration_rows(*, sample_ids: np.ndarray, source_rows: int,
                            seed: int, production_contract: bool) \
        -> tuple[np.ndarray, np.ndarray, dict]:
    needed = 2 * OUTER_CHUNK_ROWS
    if production_contract:
        pair_count = int(source_rows) // needed
        if pair_count <= 0:
            raise ValueError("source corpus is too short for two aligned calibration chunks")
        pair_index = int(seed % pair_count)
        start = pair_index * needed
        chosen_ids = np.arange(start, start + needed, dtype=np.int64)
        selection = {
            "method": "seeded_aligned_adjacent_source_chunks",
            "seed": int(seed), "pair_index": pair_index,
            "chunk_rows": OUTER_CHUNK_ROWS,
            "source_position_range": [start, start + needed],
            "production_order_preserved": True, "length_sorting": False,
        }
    else:
        if len(sample_ids) < needed:
            raise ValueError(f"calibration needs {needed} unique rows, found {len(sample_ids)}")
        start = int(seed % (len(sample_ids) - needed + 1))
        chosen_ids = np.asarray(sample_ids[start:start + needed], dtype=np.int64)
        selection = {
            "method": "seeded_contiguous_testbed_sample_window",
            "seed": int(seed), "chunk_rows": OUTER_CHUNK_ROWS,
            "source_position_range": [start, start + needed],
            "production_order_preserved": True, "length_sorting": False,
        }
    positions = np.arange(start, start + needed, dtype=np.int64)
    return positions, chosen_ids, selection


def prepare(*, testbed: str, text_dir: str, embed_dir: str, out_parquet: str,
            out_manifest: str, model_revision: str,
            seed: int = ROUND0005_CALIBRATION_SEED,
            testbed_seal_path: str | None = None,
            production_contract: bool = False) -> dict:
    sample_path = os.path.join(os.path.realpath(testbed), "sample_indices.npy")
    raw_sample_ids = np.load(sample_path)
    if raw_sample_ids.ndim != 1 or raw_sample_ids.dtype != np.dtype("int64"):
        raise ValueError("calibration sample IDs must be an exact one-dimensional int64 array")
    sample_ids = np.asarray(raw_sample_ids)
    needed = 2 * OUTER_CHUNK_ROWS
    if len(sample_ids) < needed:
        raise ValueError(f"calibration needs {needed} unique rows, found {len(sample_ids)}")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("calibration sample IDs must be unique")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision) or len(set(model_revision)) == 1:
        raise ValueError("calibration model revision must be full immutable 40-hex")
    if production_contract and (
            model_revision != ROUND0005_MODEL_REVISION or
            seed != ROUND0005_CALIBRATION_SEED):
        raise ValueError("production calibration revision/seed constants changed")
    if production_contract and testbed_seal_path is None:
        raise ValueError("production calibration inventory requires the shared testbed seal")
    seal_reference = (make_testbed_seal_reference(
        testbed_seal_path, require_round0005=production_contract)
                      if testbed_seal_path is not None else None)
    text_shards, embed_sizes, offsets, source_dim = verify_shard_alignment(
        embed_dir, text_dir)
    source_rows = int(offsets[-1])
    first_embedding_name = sorted(
        name for name in os.listdir(embed_dir) if name.endswith(".npy"))[0]
    actual_source_dtype = np.load(
        os.path.join(embed_dir, first_embedding_name), mmap_mode="r").dtype.name
    if production_contract:
        with open(seal_reference["seal_path"], encoding="utf-8") as handle:
            sealed = json.load(handle)
        if (os.path.realpath(testbed) != sealed["testbed_root"] or
                os.path.realpath(embed_dir) != sealed["source_embedding_root"] or
                os.path.realpath(text_dir) != sealed["source_text_root"]):
            raise ValueError("calibration inputs differ from the shared testbed seal")
        if (source_rows != ROUND0005_SOURCE_ROWS or source_dim != ROUND0005_DIMENSIONS or
                text_shards != [item["name"] for item in
                                sealed["matching_source_texts"]["shards"]]):
            raise ValueError("calibration source inventory differs from the exact sealed corpus")
    # Generic fixture compatibility. Certifying production always selects two
    # adjacent, 25k-aligned source-global chunks and never length-sorts them.
    positions, chosen_ids, selection = select_calibration_rows(
        sample_ids=sample_ids, source_rows=source_rows, seed=seed,
        production_contract=production_contract)
    texts = fetch_texts_for_indices(chosen_ids, text_dir, text_shards, offsets)
    char_lengths = np.asarray([len(text) for text in texts], dtype=np.int64)
    phases = np.array(["calibration"] * OUTER_CHUNK_ROWS +
                      ["heldout"] * OUTER_CHUNK_ROWS)
    frame = pd.DataFrame({
        "phase": phases,
        "source_position": positions,
        "global_id": chosen_ids,
        "text": texts,
        "character_length": char_lengths,
    })
    out_parquet = os.path.realpath(out_parquet)
    out_manifest = os.path.realpath(out_manifest)
    for path in (out_parquet, out_manifest):
        if not path.startswith("/data/"):
            raise ValueError("calibration inventory outputs must live under /data")
        refuse_existing(path, label="calibration inventory output")
    atomic_build_new_file(
        out_parquet, lambda tmp: frame.to_parquet(tmp, index=False), immutable=True)
    convention = {
        "model_id": ROUND0005_MODEL_ID,
        "model_revision": model_revision,
        "prompt_policy": "literal_prefix",
        "prompt_bytes_hex": PROMPT_PREFIX.encode("utf-8").hex(),
        "pooling": ROUND0005_POOLING,
        "source_dtype": (ROUND0005_SOURCE_DTYPE if production_contract
                         else actual_source_dtype),
        "compute_dtype": ROUND0005_STORAGE_DTYPE,
        "output_dtype": ROUND0005_STORAGE_DTYPE,
        "normalization": ROUND0005_NORMALIZATION,
        "dimensions": ROUND0005_DIMENSIONS if production_contract else int(source_dim),
    }
    phase_reports = {}
    for phase in ("calibration", "heldout"):
        subset = frame[frame.phase == phase]
        phase_reports[phase] = {
            "rows": int(len(subset)),
            "global_ids_ordered_sha256": ordered_array_sha256(
                subset.global_id.to_numpy(np.int64)),
            "texts_ordered_sha256": ordered_text_sha256(subset.text.tolist()),
            "character_length_quantiles": _quantiles(subset.character_length),
        }
    source_shards = [expected_input_signature(os.path.join(text_dir, name))
                     for name in text_shards]
    source_embedding_shards = [
        expected_input_signature(os.path.join(embed_dir, name))
        for name in sorted(name for name in os.listdir(embed_dir) if name.endswith(".npy"))
    ]
    manifest = {
        "schema": "jina_calibration_inventory.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "production_contract": bool(production_contract),
        "testbed_seal": seal_reference,
        "seed": int(seed),
        "selection": selection,
        "source_position_range": [int(positions[0]), int(positions[-1]) + 1],
        "outer_chunk_rows": OUTER_CHUNK_ROWS,
        "inventory": expected_input_signature(out_parquet),
        "source_sample_indices": expected_input_signature(sample_path),
        "source_text_shards": source_shards,
        "source_embedding_shards": source_embedding_shards,
        "source_embedding_rows": source_rows,
        "source_embedding_shard_rows": [int(value) for value in embed_sizes],
        "convention": convention,
        "phases": phase_reports,
    }
    manifest["identity_sha256"] = sha256_bytes(canonical_json({
        key: manifest[key] for key in (
            "production_contract", "testbed_seal", "seed", "selection",
            "source_position_range", "outer_chunk_rows", "inventory",
            "source_sample_indices", "source_text_shards", "source_embedding_shards",
            "source_embedding_rows", "source_embedding_shard_rows", "convention", "phases")
    }))
    atomic_write_new_json(out_manifest, manifest, immutable=True)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", required=True)
    parser.add_argument("--testbed-seal", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--embed-dir", required=True)
    parser.add_argument("--out-parquet", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--model-revision", default=ROUND0005_MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=ROUND0005_CALIBRATION_SEED)
    args = parser.parse_args(argv)
    report = prepare(**vars(args), production_contract=True)
    print(json.dumps({"identity_sha256": report["identity_sha256"],
                      "inventory": report["inventory"],
                      "phases": report["phases"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
