#!/usr/bin/env python3
"""Prepare the slim queue manifest for basemap-100m Round 0023."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0023_program import R0019_REFERENCE, train_config_for_seed
from basemap.round0014_transform import build_transform_template


RUN_ROOT = "/home/enjalot/code/latent-basemap-run"
LAB_ROOT = "/home/enjalot/code/latent-labs/basemap-100m"
GPU_LEASE_PATH = "/data/latent-basemap/.gpu_lease"
GPU_UUID = "GPU-2c4d2a68-2646-901a-e61c-fbc61f5c9072"
IMPLEMENTATION_BASE_COMMIT = "df2abffb388c7c57d19ff0912f597ad12cf0f688"

R0019_QUEUE = "/data/latent-basemap/runs/round-0019/queue/queue.json"
R0019_HIGH_D_REFERENCE = "/data/latent-basemap/runs/round-0019/queue/artifacts/high-d-reference"
R0019_PANEL = "/data/latent-basemap/runs/round-0019/queue/artifacts/panel/panel.json"
R0019_TRAIN = "/data/latent-basemap/runs/round-0019/queue/artifacts/train"
R0019_COORDINATES = "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates"
R0019_RENDER = "/data/latent-basemap/runs/round-0019/queue/artifacts/semantic-renders"
R0019_SAMPLE_IDS = os.path.join(R0019_RENDER, "sample-semantic-ids.npy")


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
        "HF_HOME": "/data/hf",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SENTENCE_TRANSFORMERS_HOME": "/data/hf/sentence-transformers",
        "CUDA_VISIBLE_DEVICES": GPU_UUID,
    }


def _r0019_queue_inputs() -> list[dict[str, Any]]:
    with open(R0019_QUEUE, encoding="utf-8") as handle:
        queue = json.load(handle)
    inputs: list[dict[str, Any]] = []
    for job in queue["jobs"]:
        inputs.extend(job.get("expected_inputs", []))
    return inputs


def _coordinate_inputs(root: str) -> list[dict[str, Any]]:
    receipt = expected_input_signature(os.path.join(root, "actual-transform.json"))
    with open(receipt["canonical_path"], encoding="utf-8") as handle:
        record = json.load(handle)
    ordered = record["stream_capability"]["capability_payload"]["ordered_chunks"]
    chunks = []
    for item in ordered:
        path = os.path.join(root, f"chunk-{item['chunk_index']:05d}", "coordinates.npy")
        chunks.append(expected_input_signature(path))
    return [receipt, *chunks]


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


def _write_transform_templates(*, queue_root: str, release_sha: str) -> dict[str, str]:
    inputs_root = ensure_data_directory(os.path.join(queue_root, "inputs"))
    templates: dict[str, str] = {}
    for seed in (43, 44):
        config, digest = train_config_for_seed(seed)
        template = build_transform_template(
            release_root=RUN_ROOT,
            release_sha=release_sha,
            train_output_relative_path=f"artifacts/seed{seed}/train/model.pt",
            production_config=config,
            production_config_sha256=digest,
        )
        path = os.path.join(inputs_root, f"seed{seed}-transform-spec-template.json")
        atomic_write_new_json(path, template, immutable=True)
        templates[str(seed)] = path
    return templates


def _assert_r0019_pins() -> None:
    pins = {
        os.path.join(R0019_TRAIN, "model.pt"): R0019_REFERENCE["seed42_model_sha256"],
        os.path.join(R0019_COORDINATES, "actual-transform.json"): R0019_REFERENCE[
            "seed42_coordinate_receipt_sha256"
        ],
        R0019_PANEL: R0019_REFERENCE["seed42_panel_sha256"],
        R0019_SAMPLE_IDS: R0019_REFERENCE["semantic_sample_ids_file_sha256"],
        os.path.join(R0019_HIGH_D_REFERENCE, "reference-receipt.json"): R0019_REFERENCE[
            "high_d_reference_receipt_sha256"
        ],
        os.path.join(R0019_HIGH_D_REFERENCE, "reference.npz"): R0019_REFERENCE[
            "high_d_reference_npz_sha256"
        ],
        os.path.join(R0019_HIGH_D_REFERENCE, "recall50-truth.npy"): R0019_REFERENCE[
            "recall50_truth_sha256"
        ],
    }
    for path, digest in pins.items():
        signature = expected_input_signature(path)
        if signature["sha256"] != digest:
            raise RuntimeError(f"Round 0023 pinned R0019 input changed: {path}")


def prepare_round0023(release_sha: str) -> str:
    _assert_r0019_pins()
    round_root = ensure_data_directory("/data/latent-basemap/runs/round-0023")
    queue_root = create_fresh_directory(os.path.join(round_root, "queue"), label="R0023 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    templates = _write_transform_templates(queue_root=queue_root, release_sha=release_sha)
    reference_inputs = _file_inputs(
        [
            os.path.join(LAB_ROOT, "round-0023-2026-07-19.md"),
            os.path.join(LAB_ROOT, "review-0019-2026-07-20.md"),
            os.path.join(LAB_ROOT, "review-0021-2026-07-20.md"),
            R0019_QUEUE,
            os.path.join(R0019_TRAIN, "model.pt"),
            os.path.join(R0019_TRAIN, "train-receipt.json"),
            R0019_PANEL,
            os.path.join(R0019_RENDER, "render-manifest.json"),
            R0019_SAMPLE_IDS,
            os.path.join(R0019_HIGH_D_REFERENCE, "reference-receipt.json"),
            os.path.join(R0019_HIGH_D_REFERENCE, "reference.npz"),
            os.path.join(R0019_HIGH_D_REFERENCE, "recall50-truth.npy"),
            "/data/latent-basemap/runs/round-0019/input/duplicate-cap.npz",
            "/data/latent-basemap/runs/round-0019/input/r0018-duplicate-baseline.json",
            "/data/latent-basemap/runs/round-0018/posthoc-untrained-floor.json",
            templates["43"],
            templates["44"],
        ]
    )
    inputs = _dedupe([*_r0019_queue_inputs(), *_coordinate_inputs(R0019_COORDINATES), *reference_inputs])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "program": "basemap-100m-round-0023",
        "round_id": "0023",
        "round_sha256": sha256_file(os.path.join(LAB_ROOT, "round-0023-2026-07-19.md")),
        "release_sha": release_sha,
        "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
        "r0019_scientific_base_commit": R0019_REFERENCE["release_base_commit"],
        "execution_authority": "autonomous-gpu",
        "required_reviews": ["0019", "0021"],
        "gpu_hours_cap": 5.8,
        "queue_class": "research",
        "training_performed": True,
        "deadline_utc": _deadline(8),
        "repo_root": RUN_ROOT,
        "lease_path": GPU_LEASE_PATH,
        "child_environment": _child_environment(queue_root),
        "scientific_contract": {
            "treatment": "R0019 local duplicate cap; seeds 43/44 differ only by optimizer.seed",
            "reference_seed42": {
                "train": expected_input_signature(os.path.join(R0019_TRAIN, "train-receipt.json")),
                "model": expected_input_signature(os.path.join(R0019_TRAIN, "model.pt")),
                "coordinates": expected_input_signature(
                    os.path.join(R0019_COORDINATES, "actual-transform.json")
                ),
                "panel": expected_input_signature(R0019_PANEL),
                "render": expected_input_signature(os.path.join(R0019_RENDER, "render-manifest.json")),
            },
            "sample_semantic_ids": expected_input_signature(R0019_SAMPLE_IDS),
            "reuse_high_d_reference": expected_input_signature(
                os.path.join(R0019_HIGH_D_REFERENCE, "reference-receipt.json")
            ),
        },
        "jobs": [],
    }
    paths: dict[str, dict[str, str]] = {}
    for seed in ("43", "44"):
        root = ensure_data_directory(os.path.join(artifacts, f"seed{seed}"))
        paths[seed] = {
            "canary": os.path.join(root, "canary"),
            "train": os.path.join(root, "train"),
            "coordinates": os.path.join(root, "coordinates"),
            "panel": os.path.join(root, "panel"),
            "semantic_renders": os.path.join(root, "semantic-renders"),
        }
    jobs: list[dict[str, Any]] = [
        {
            "id": "seed43_no_training_seal_canary",
            "handler": "no_training_seal_canary",
            "seed": 43,
            "deps": [],
            "done_marker": os.path.join(artifacts, "seed43_no_training_seal_canary.done.json"),
            "outputs": [paths["43"]["canary"]],
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
        },
        {
            "id": "seed43_train_30m",
            "handler": "train_seed42_30m",
            "seed": 43,
            "deps": ["seed43_no_training_seal_canary"],
            "done_marker": os.path.join(artifacts, "seed43_train_30m.done.json"),
            "outputs": [paths["43"]["train"]],
            "expected_inputs": inputs,
            "p90_wall_s": 4800.0,
            "node_policy": {"gpu_required": True, "training_performed": True},
            "canary_output": paths["43"]["canary"],
        },
        {
            "id": "seed43_transform_30m",
            "handler": "transform_30m",
            "seed": 43,
            "deps": ["seed43_train_30m"],
            "done_marker": os.path.join(artifacts, "seed43_transform_30m.done.json"),
            "outputs": [paths["43"]["coordinates"]],
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "train_output": paths["43"]["train"],
            "transform_spec_template": templates["43"],
        },
        {
            "id": "seed43_registered_panel",
            "handler": "registered_panel",
            "seed": 43,
            "deps": ["seed43_transform_30m"],
            "done_marker": os.path.join(artifacts, "seed43_registered_panel.done.json"),
            "outputs": [paths["43"]["panel"]],
            "expected_inputs": inputs,
            "p90_wall_s": 2600.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "canary_output": paths["43"]["canary"],
            "train_output": paths["43"]["train"],
            "transform_output": paths["43"]["coordinates"],
            "reference_output": R0019_HIGH_D_REFERENCE,
        },
        {
            "id": "seed43_semantic_renders",
            "handler": "semantic_renders",
            "seed": 43,
            "deps": ["seed43_registered_panel"],
            "done_marker": os.path.join(artifacts, "seed43_semantic_renders.done.json"),
            "outputs": [paths["43"]["semantic_renders"]],
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "transform_output": paths["43"]["coordinates"],
            "panel_output": paths["43"]["panel"],
            "sample_semantic_ids": R0019_SAMPLE_IDS,
        },
        {
            "id": "seed44_train_30m",
            "handler": "train_seed42_30m",
            "seed": 44,
            "deps": ["seed43_train_30m"],
            "done_marker": os.path.join(artifacts, "seed44_train_30m.done.json"),
            "outputs": [paths["44"]["train"]],
            "expected_inputs": inputs,
            "p90_wall_s": 4800.0,
            "node_policy": {"gpu_required": True, "training_performed": True},
            "canary_output": paths["43"]["canary"],
        },
        {
            "id": "seed44_transform_30m",
            "handler": "transform_30m",
            "seed": 44,
            "deps": ["seed44_train_30m"],
            "done_marker": os.path.join(artifacts, "seed44_transform_30m.done.json"),
            "outputs": [paths["44"]["coordinates"]],
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "train_output": paths["44"]["train"],
            "transform_spec_template": templates["44"],
        },
        {
            "id": "seed44_registered_panel",
            "handler": "registered_panel",
            "seed": 44,
            "deps": ["seed44_transform_30m"],
            "done_marker": os.path.join(artifacts, "seed44_registered_panel.done.json"),
            "outputs": [paths["44"]["panel"]],
            "expected_inputs": inputs,
            "p90_wall_s": 2600.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "canary_output": paths["43"]["canary"],
            "train_output": paths["44"]["train"],
            "transform_output": paths["44"]["coordinates"],
            "reference_output": R0019_HIGH_D_REFERENCE,
        },
        {
            "id": "seed44_semantic_renders",
            "handler": "semantic_renders",
            "seed": 44,
            "deps": ["seed44_registered_panel"],
            "done_marker": os.path.join(artifacts, "seed44_semantic_renders.done.json"),
            "outputs": [paths["44"]["semantic_renders"]],
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": True, "training_performed": False},
            "transform_output": paths["44"]["coordinates"],
            "panel_output": paths["44"]["panel"],
            "sample_semantic_ids": R0019_SAMPLE_IDS,
        },
        {
            "id": "layout_disparity",
            "handler": "layout_disparity",
            "seed": 43,
            "deps": ["seed43_semantic_renders", "seed44_semantic_renders"],
            "done_marker": os.path.join(artifacts, "layout_disparity.done.json"),
            "outputs": [os.path.join(artifacts, "layout-disparity")],
            "expected_inputs": inputs,
            "p90_wall_s": 300.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
            "coordinate_outputs": {
                "43": paths["43"]["coordinates"],
                "44": paths["44"]["coordinates"],
            },
            "train_outputs": {"43": paths["43"]["train"], "44": paths["44"]["train"]},
            "panel_outputs": {"43": paths["43"]["panel"], "44": paths["44"]["panel"]},
            "render_outputs": {
                "43": paths["43"]["semantic_renders"],
                "44": paths["44"]["semantic_renders"],
            },
            "sample_semantic_ids": R0019_SAMPLE_IDS,
        },
    ]
    manifest["jobs"] = jobs
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"queue_manifests": [prepare_round0023(args.release_sha)]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
