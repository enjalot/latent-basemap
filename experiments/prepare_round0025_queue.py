#!/usr/bin/env python3
"""Prepare the slim queue manifest for basemap-100m Round 0025."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature, sha256_file
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from experiments.bench_int8_gather import build_source_plan


RUN_ROOT = "/home/enjalot/code/latent-basemap-run"
LAB_ROOT = "/home/enjalot/code/latent-labs/basemap-100m"
ROUND_ID = "0025"
ROUND_ROOT = "/data/latent-basemap/runs/round-0025"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0025-2026-07-19.md")
GPU_LEASE_PATH = "/data/latent-basemap/.gpu_lease"
ENVIRONMENT_MANIFEST = "/data/latent-basemap/envs/run-env.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--round-root", default=ROUND_ROOT)
    parser.add_argument("--environment-manifest", default=ENVIRONMENT_MANIFEST)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--out")
    return parser


def _deadline(hours: float = 9.0) -> str:
    return (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)
    ).isoformat(timespec="seconds")


def _load_environment(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    required = ("venv_path", "freeze_file", "freeze_sha256", "identity_sha256",
                "gpu_uuid", "gpu_name")
    if not isinstance(value, dict) or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in required):
        raise ValueError("Round 0025 environment manifest is incomplete")
    return value


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


def _child_environment(queue_root: str, environment: dict[str, Any]) -> dict[str, str]:
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
        "CUDA_VISIBLE_DEVICES": environment["gpu_uuid"],
    }


def _git_stdout(args: list[str], *, cwd: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True)
    return proc.stdout.strip()


def _verify_release_checkout(run_root: str, release_sha: str,
                             environment: dict[str, Any]) -> None:
    run_root = os.path.realpath(run_root)
    head = _git_stdout(["rev-parse", "HEAD"], cwd=run_root)
    if head != release_sha:
        raise RuntimeError(
            f"release checkout {run_root} is at {head}, expected {release_sha}")
    dirty = _git_stdout(["status", "--porcelain"], cwd=run_root)
    if dirty:
        raise RuntimeError(f"release checkout {run_root} is dirty:\n{dirty}")
    expected_venv = os.path.realpath(os.path.join(run_root, ".venv"))
    if os.path.realpath(environment["venv_path"]) != expected_venv:
        raise RuntimeError(
            f"environment venv {environment['venv_path']} does not match {expected_venv}")
    python = os.path.join(expected_venv, "bin", "python")
    if not os.path.isfile(python):
        raise FileNotFoundError(f"release venv interpreter missing: {python}")


def _dedupe(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in inputs:
        path = item["canonical_path"]
        prior = seen.get(path)
        if prior is not None and prior != item:
            raise RuntimeError(f"input signature conflict for {path}")
        seen[path] = item
    return [seen[path] for path in sorted(seen)]


def _job(
    *,
    job_id: str,
    handler: str,
    deps: list[str],
    output: str,
    artifacts: str,
    inputs: list[dict[str, Any]],
    p90_wall_s: float,
    gpu_required: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "handler": handler,
        "deps": deps,
        "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
        "outputs": [output],
        "expected_inputs": inputs,
        "p90_wall_s": p90_wall_s,
        "node_policy": {
            "gpu_required": gpu_required,
            "training_performed": False,
        },
        **(extra or {}),
    }


def build_queue(args: argparse.Namespace) -> dict[str, Any]:
    run_root = os.path.realpath(args.run_root)
    environment_path = os.path.realpath(args.environment_manifest)
    environment = _load_environment(environment_path)
    _verify_release_checkout(run_root, args.release_sha, environment)

    print("Round 0025: hashing MiniLM source plan", file=sys.stderr, flush=True)
    source_plan = build_source_plan(hash_files=True)
    print(
        "Round 0025: source plan "
        f"{source_plan['identity_sha256']} with "
        f"{len(source_plan['sources'])} files",
        file=sys.stderr,
        flush=True,
    )

    round_root = ensure_data_directory(args.round_root, label="Round 0025 root")
    queue_root = create_fresh_directory(
        os.path.join(round_root, "queue"), label="Round 0025 queue")
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    ensure_data_directory(os.path.join(queue_root, "logs"))
    for path in _cache_environment(queue_root).values():
        if path != "1":
            ensure_data_directory(path)

    out = os.path.realpath(args.out or os.path.join(queue_root, "queue.json"))
    if out != os.path.join(queue_root, "queue.json"):
        raise ValueError("Round 0025 queue output must be round-root/queue/queue.json")

    code_inputs = [
        expected_input_signature(os.path.join(run_root, "experiments/bench_int8_gather.py")),
        expected_input_signature(os.path.join(run_root, "experiments/run_round0014_node.py")),
        expected_input_signature(os.path.join(run_root, "experiments/prepare_round0025_queue.py")),
        expected_input_signature(os.path.join(run_root, "basemap/output_safety.py")),
        expected_input_signature(os.path.join(run_root, "basemap/artifact_identity.py")),
        expected_input_signature(ROUND_FILE),
        expected_input_signature(environment_path),
        expected_input_signature(environment["freeze_file"]),
    ]
    source_inputs = [item["signature"] for item in source_plan["sources"]]
    inputs = _dedupe([*code_inputs, *source_inputs])

    int8_output = os.path.join(artifacts, "int8-shards")
    shard_manifest = os.path.join(int8_output, "int8-shards-v1.json")
    bench_output = os.path.join(artifacts, "bench")
    manifest = {
        "schema_version": 1,
        "program": "basemap-100m-round-0025",
        "round_id": ROUND_ID,
        "round_sha256": sha256_file(ROUND_FILE),
        "release_sha": args.release_sha,
        "execution_authority": "autonomous-gpu",
        "required_reviews": [],
        "environment_freeze_sha": environment["freeze_sha256"],
        "environment_identity_sha": environment["identity_sha256"],
        "gpu_hours_cap": 0.6,
        "queue_class": "research",
        "training_performed": False,
        "deadline_utc": args.deadline_utc or _deadline(9.0),
        "repo_root": run_root,
        "lease_path": GPU_LEASE_PATH,
        "environment_manifest": environment_path,
        "child_environment": _child_environment(queue_root, environment),
        "round0025": {
            "source_plan": source_plan,
            "shard_manifest": shard_manifest,
            "registered_loop": {
                "batch_rows": 8192,
                "dimension": 384,
                "warmup_iterations": 500,
                "measured_iterations": 20_000,
                "prefetch_buffers": 2,
                "operation": (
                    "uniform-row-gather-int8-to-pinned-host-overlapped-with-current-"
                    "h2d-dequant-bf16"
                ),
            },
            "disk_free_abort_gib": 400.0,
            "ram_150m_skip_threshold_gib": 70.0,
        },
        "jobs": [
            _job(
                job_id="round0025_canary",
                handler="round0025_canary",
                deps=[],
                output=os.path.join(artifacts, "canary"),
                artifacts=artifacts,
                inputs=inputs,
                p90_wall_s=300.0,
                gpu_required=True,
                extra={"block_rows": 65_536},
            ),
            _job(
                job_id="round0025_quantize",
                handler="round0025_quantize",
                deps=["round0025_canary"],
                output=int8_output,
                artifacts=artifacts,
                inputs=inputs,
                p90_wall_s=14_400.0,
                gpu_required=False,
                extra={"block_rows": 65_536},
            ),
            _job(
                job_id="round0025_bench",
                handler="round0025_bench",
                deps=["round0025_quantize"],
                output=bench_output,
                artifacts=artifacts,
                inputs=inputs,
                p90_wall_s=3_600.0,
                gpu_required=True,
                extra={
                    "shard_manifest": shard_manifest,
                    "iterations": 20_000,
                    "warmup": 500,
                    "batch_rows": 8192,
                },
            ),
        ],
    }
    atomic_write_new_json(out, manifest, immutable=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_queue(args)
    print(json.dumps({
        "queue_manifest": os.path.join(args.round_root, "queue", "queue.json"),
        "round_id": manifest["round_id"],
        "release_sha": manifest["release_sha"],
        "source_plan_identity_sha256": manifest["round0025"]["source_plan"][
            "identity_sha256"],
        "jobs": [job["id"] for job in manifest["jobs"]],
        "gpu_hours_cap": manifest["gpu_hours_cap"],
        "source_files": len(manifest["round0025"]["source_plan"]["sources"]),
        "trailing_fragments": manifest["round0025"]["source_plan"][
            "trailing_fragments"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
