#!/usr/bin/env python3
"""Prepare the slim queue manifest for Round 0026."""
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
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory


RUN_ROOT = "/home/enjalot/code/latent-basemap-run"
LAB_ROOT = "/home/enjalot/code/latent-labs/basemap-100m"
QUEUE_ROOT = "/data/latent-basemap/runs/round-0026/queue"
PYTHON_DEPS = "/data/latent-basemap/runs/round-0026/python-deps"
INPUT_PACK = "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json"
QUERY_POOL = "/data/latent-basemap/track1/minilm_queries.npy"
GPU_LEASE_PATH = "/data/latent-basemap/.gpu_lease"
GPU_UUID = "GPU-2c4d2a68-2646-901a-e61c-fbc61f5c9072"


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


def _child_environment(queue_root: str) -> dict[str, str]:
    return {
        **_cache_environment(queue_root),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
        "BASEMAP_ROUND0026_GPU_UUID": GPU_UUID,
        "PYTHONPATH": PYTHON_DEPS,
    }


def _materialized_chunk_inputs() -> list[dict[str, Any]]:
    with open(INPUT_PACK, encoding="utf-8") as handle:
        manifest = json.load(handle)
    members = manifest["capability_payload"]["materialized_fp16"]["ordered_members"]
    return [
        {
            "canonical_path": os.path.realpath(item["path"]),
            "kind": "file",
            "bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in members
    ]


def _model_inputs() -> list[dict[str, Any]]:
    paths = [
        "/data/latent-basemap/runs/round-0019/queue/artifacts/train/model.pt",
        "/data/latent-basemap/runs/round-0024/queue/artifacts/h1024/train/model.pt",
        "/data/latent-basemap/runs/round-0024/queue/artifacts/h4096/train/model.pt",
    ]
    return [expected_input_signature(path) for path in paths if os.path.exists(path)]


def _dependency_file_inputs() -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for path in sorted(Path(PYTHON_DEPS).rglob("*")):
        if path.is_file():
            inputs.append(expected_input_signature(str(path)))
    return inputs


def _dedupe(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in inputs:
        prior = seen.get(item["canonical_path"])
        if prior is not None and prior != item:
            raise RuntimeError(f"input signature conflict for {item['canonical_path']}")
        seen[item["canonical_path"]] = item
    return [seen[path] for path in sorted(seen)]


def prepare_round0026(release_sha: str) -> str:
    round_root = ensure_data_directory("/data/latent-basemap/runs/round-0026")
    queue_root = create_fresh_directory(QUEUE_ROOT, label="R0026 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    inputs = _dedupe(
        [
            expected_input_signature(INPUT_PACK),
            expected_input_signature(QUERY_POOL),
            *_materialized_chunk_inputs(),
            *_model_inputs(),
            *_dependency_file_inputs(),
        ]
    )
    manifest = {
        "schema_version": 1,
        "program": "basemap-100m-round-0026",
        "round_id": "0026",
        "round_sha256": sha256_file(os.path.join(LAB_ROOT, "round-0026-2026-07-19.md")),
        "release_sha": release_sha,
        "execution_authority": "autonomous-gpu",
        "required_reviews": ["0019"],
        "gpu_hours_cap": 0.2,
        "queue_class": "research",
        "training_performed": False,
        "deadline_utc": _deadline(4),
        "repo_root": RUN_ROOT,
        "lease_path": GPU_LEASE_PATH,
        "child_environment": _child_environment(queue_root),
        "jobs": [
            {
                "id": "export_parity_canary",
                "deps": [],
                "done_marker": os.path.join(artifacts, "export_parity_canary.done.json"),
                "outputs": [os.path.join(artifacts, "canary")],
                "expected_inputs": inputs,
                "p90_wall_s": 600,
                "node_policy": {"gpu_required": False},
            },
            {
                "id": "cpu_inference_profile",
                "deps": ["export_parity_canary"],
                "done_marker": os.path.join(artifacts, "cpu_inference_profile.done.json"),
                "outputs": [os.path.join(artifacts, "cpu-profile")],
                "expected_inputs": inputs,
                "p90_wall_s": 5400,
                "node_policy": {"gpu_required": False},
            },
            {
                "id": "gpu_inference_profile",
                "deps": ["cpu_inference_profile"],
                "done_marker": os.path.join(artifacts, "gpu_inference_profile.done.json"),
                "outputs": [os.path.join(artifacts, "inference-profile")],
                "expected_inputs": inputs,
                "p90_wall_s": 900,
                "node_policy": {"gpu_required": True},
            },
        ],
    }
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"queue_manifest": prepare_round0026(args.release_sha)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
