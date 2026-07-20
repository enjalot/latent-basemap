#!/usr/bin/env python3
"""Prepare slim queue manifests for basemap-100m universality/census rounds."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)


RUN_ROOT = "/home/enjalot/code/latent-basemap-run"
LAB_ROOT = "/home/enjalot/code/latent-labs/basemap-100m"
INPUT_PACK_MANIFEST = "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json"
R0019_COORDINATES = "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates"
MODEL_PATH = "/data/latent-basemap/runs/round-0019/queue/artifacts/train/model.pt"
MINILM_QUERIES = "/data/latent-basemap/track1/minilm_queries.npy"
GPU_LEASE_PATH = "/data/latent-basemap/.gpu_lease"
GPU_UUID = "GPU-2c4d2a68-2646-901a-e61c-fbc61f5c9072"
SENTENCE_MODEL_SNAPSHOT = (
    "/data/hf/sentence-transformers/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
R0022_PROVISIONAL_PANEL = (
    "/data/latent-basemap/runs/round-0022/queue/artifacts/panel/"
    "universality-panel-v1.json"
)


def _deadline(hours: float) -> str:
    return (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)
    ).isoformat(timespec="seconds")


def _cache_environment(queue_root: str) -> dict[str, str]:
    cache_root = os.path.join(queue_root, "cache")
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": os.path.join(cache_root, "pythonpycacheprefix"),
        "MPLCONFIGDIR": os.path.join(cache_root, "mplconfigdir"),
        "NUMBA_CACHE_DIR": os.path.join(cache_root, "numba_cache_dir"),
        "TORCH_HOME": os.path.join(cache_root, "torch_home"),
        "TRITON_CACHE_DIR": os.path.join(cache_root, "triton_cache_dir"),
        "XDG_CACHE_HOME": os.path.join(cache_root, "xdg_cache_home"),
    }


def _child_environment(queue_root: str, *, gpu: bool) -> dict[str, str]:
    env = {
        **_cache_environment(queue_root),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HF_HOME": "/data/hf",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SENTENCE_TRANSFORMERS_HOME": "/data/hf/sentence-transformers",
    }
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID if gpu else ""
    return env


def _materialized_chunk_inputs() -> list[dict[str, Any]]:
    with open(INPUT_PACK_MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    members = manifest["capability_payload"]["materialized_fp16"]["ordered_members"]
    out = []
    for item in members:
        out.append(
            {
                "canonical_path": os.path.realpath(item["path"]),
                "kind": "file",
                "bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
            }
        )
    return out


def _coordinate_inputs() -> list[dict[str, Any]]:
    receipt = expected_input_signature(os.path.join(R0019_COORDINATES, "actual-transform.json"))
    with open(receipt["canonical_path"], encoding="utf-8") as handle:
        record = json.load(handle)
    ordered = record["stream_capability"]["capability_payload"]["ordered_chunks"]
    chunks = []
    for item in ordered:
        path = os.path.join(
            R0019_COORDINATES, f"chunk-{item['chunk_index']:05d}", "coordinates.npy"
        )
        chunks.append(
            {
                "canonical_path": os.path.realpath(path),
                "kind": "file",
                "bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
            }
        )
    return [receipt, *chunks]


def _hf_snapshot_file_inputs() -> list[dict[str, Any]]:
    inputs = []
    for directory, _, files in os.walk(SENTENCE_MODEL_SNAPSHOT):
        for name in sorted(files):
            path = os.path.realpath(os.path.join(directory, name))
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            inputs.append(
                {
                    "canonical_path": path,
                    "kind": "file",
                    "bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
            )
    # Multiple snapshot symlinks can resolve to the same blob only once for preflight.
    unique = {item["canonical_path"]: item for item in inputs}
    return [unique[path] for path in sorted(unique)]


def _file_inputs(paths: list[str]) -> list[dict[str, Any]]:
    return [expected_input_signature(path) for path in paths]


def _dedupe(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in inputs:
        prior = seen.get(item["canonical_path"])
        if prior is not None and prior != item:
            raise RuntimeError(f"input signature conflict for {item['canonical_path']}")
        seen[item["canonical_path"]] = item
    return [seen[path] for path in sorted(seen)]


def _base_manifest(
    *,
    round_id: str,
    release_sha: str,
    round_file: str,
    queue_root: str,
    gpu_hours_cap: float,
    execution_authority: str,
    gpu: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "program": f"basemap-100m-round-{round_id}",
        "round_id": round_id,
        "round_sha256": sha256_file(round_file),
        "release_sha": release_sha,
        "execution_authority": execution_authority,
        "required_reviews": ["0019"],
        "gpu_hours_cap": gpu_hours_cap,
        "queue_class": "research",
        "training_performed": False,
        "deadline_utc": _deadline(12),
        "repo_root": RUN_ROOT,
        "lease_path": GPU_LEASE_PATH,
        "child_environment": _child_environment(queue_root, gpu=gpu),
        "jobs": [],
    }


def prepare_round0020(release_sha: str) -> str:
    round_root = ensure_data_directory("/data/latent-basemap/runs/round-0020")
    queue_root = create_fresh_directory(os.path.join(round_root, "queue"), label="R0020 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe(
        [
            expected_input_signature(INPUT_PACK_MANIFEST),
            *_materialized_chunk_inputs(),
            *_coordinate_inputs(),
        ]
    )
    manifest = _base_manifest(
        round_id="0020",
        release_sha=release_sha,
        round_file=os.path.join(LAB_ROOT, "round-0020-2026-07-19.md"),
        queue_root=queue_root,
        gpu_hours_cap=0.1,
        execution_authority="autonomous-cpu",
        gpu=False,
    )
    manifest["jobs"] = [
        {
            "id": "no_training_seal_canary",
            "deps": [],
            "done_marker": os.path.join(artifacts, "duplicate_census.done.json"),
            "outputs": [os.path.join(artifacts, "duplicate-census")],
            "expected_inputs": inputs,
            "p90_wall_s": 7200,
            "node_policy": {"gpu_required": False},
        }
    ]
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def _universality_inputs(*, include_r0022_panel: bool) -> list[dict[str, Any]]:
    extra = [R0022_PROVISIONAL_PANEL] if include_r0022_panel else []
    probe_inputs = _file_inputs(
        [
            MODEL_PATH,
            INPUT_PACK_MANIFEST,
            MINILM_QUERIES,
            "/data/embeddings/beir/scifact-pooled-minilm/corpus_vectors.npy",
            "/data/embeddings/beir/scifact-pooled-minilm/query_vectors.npy",
            "/data/embeddings/beir/scifact-pooled-minilm/corpus_ids.json",
            "/data/embeddings/beir/scifact-pooled-minilm/query_ids.json",
            "/data/hf/datasets/mteb___scifact/corpus/0.0.0/cf10ab6856b15b0e670ef8ae5dae4e266c12d035/scifact-corpus.arrow",
            "/data/embeddings/beir/trec-covid-pooled-minilm/corpus_vectors.npy",
            "/data/embeddings/beir/trec-covid-pooled-minilm/queries_vectors.npy",
            "/data/embeddings/beir/trec-covid-pooled-minilm/corpus_ids.json",
            "/data/embeddings/beir/trec-covid-pooled-minilm/queries_ids.json",
            "/data/embeddings/beir/trec-covid-pooled-minilm/topk_indices.npy",
            "/data/embeddings/beir/trec-covid-pooled-minilm/topk_meta.json",
            "/data/embeddings/dadabase/minilm.npy",
            "/data/embeddings/dadabase/jokes.parquet",
            "/data/latent-basemap/runs/round-0019/queue/artifacts/semantic-renders/sample-semantic-ids.npy",
            *extra,
        ]
    )
    return _dedupe(
        [
            *probe_inputs,
            *_materialized_chunk_inputs(),
            *_coordinate_inputs(),
            *_hf_snapshot_file_inputs(),
        ]
    )


def _prepare_universality_round(
    *,
    round_id: str,
    release_sha: str,
    gpu_hours_cap: float,
    include_r0022_panel: bool,
) -> str:
    round_root = ensure_data_directory(f"/data/latent-basemap/runs/round-{round_id}")
    queue_root = create_fresh_directory(os.path.join(round_root, "queue"), label=f"R{round_id} queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _universality_inputs(include_r0022_panel=include_r0022_panel)
    manifest = _base_manifest(
        round_id=round_id,
        release_sha=release_sha,
        round_file=os.path.join(LAB_ROOT, f"round-{round_id}-2026-07-19.md"),
        queue_root=queue_root,
        gpu_hours_cap=gpu_hours_cap,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    manifest["jobs"] = [
        {
            "id": "no_training_seal_canary",
            "deps": [],
            "done_marker": os.path.join(artifacts, "no_training_seal_canary.done.json"),
            "outputs": [os.path.join(artifacts, "canary")],
            "expected_inputs": inputs,
            "p90_wall_s": 300,
            "node_policy": {"gpu_required": True},
        },
        {
            "id": "registered_panel",
            "deps": ["no_training_seal_canary"],
            "done_marker": os.path.join(artifacts, "registered_panel.done.json"),
            "outputs": [os.path.join(artifacts, "panel")],
            "expected_inputs": inputs,
            "p90_wall_s": 3600,
            "node_policy": {"gpu_required": True},
        },
        {
            "id": "semantic_renders",
            "deps": ["registered_panel"],
            "done_marker": os.path.join(artifacts, "semantic_renders.done.json"),
            "outputs": [os.path.join(artifacts, "universality-renders")],
            "expected_inputs": inputs,
            "p90_wall_s": 600,
            "node_policy": {"gpu_required": True},
        },
    ]
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def prepare_round0022(release_sha: str) -> str:
    return _prepare_universality_round(
        round_id="0022",
        release_sha=release_sha,
        gpu_hours_cap=1.5,
        include_r0022_panel=False,
    )


def prepare_round0028(release_sha: str) -> str:
    return _prepare_universality_round(
        round_id="0028",
        release_sha=release_sha,
        gpu_hours_cap=0.03,
        include_r0022_panel=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--round",
        choices=("0020", "0022", "0028", "both"),
        default="both",
        help="which queue to prepare",
    )
    args = parser.parse_args(argv)
    paths = []
    if args.round in {"0020", "both"}:
        paths.append(prepare_round0020(args.release_sha))
    if args.round in {"0022", "both"}:
        paths.append(prepare_round0022(args.release_sha))
    if args.round == "0028":
        paths.append(prepare_round0028(args.release_sha))
    print(json.dumps({"queue_manifests": paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
