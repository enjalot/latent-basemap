#!/usr/bin/env python3
"""Prepare, but never launch, the R0171 sharded-fp32 replacement Q2 rung."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0171_prompted_8m import (
    CAPABILITY,
    GRAPH_EXECUTION,
    GRAPH_VECTOR_STORAGE,
    ROUND_ID,
    scale_train_config,
)
from experiments import prepare_round0166_queue as base
from experiments.prepare_round0020_0022_queues import LAB_ROOT


ROUND_ROOT = "/data/latent-basemap/runs/round-0171"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0171-2026-08-03.md")


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0171 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0166_cpu_smoke.py",
        "tests/test_round0166_prompted_8m.py",
        "tests/test_round0171_prompted_8m.py",
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = prompt_contract.seal({
        "schema": "round0171-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
        "path_exercised": (
            "Q2 train -> seal -> checkpoint reload -> panel plus deterministic "
            "two-shard fp32-IVF merge and dispatch/config bindings"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0171 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _configure() -> None:
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.ROUND_ROOT = ROUND_ROOT
    base.RELEASE_ROOT = RELEASE_ROOT
    base.ROUND_FILE = ROUND_FILE
    base.HANDLER_MODULE = "experiments.round0171_nodes"
    base.QUEUE_SCHEMA = "round0171-prompted-english-8m-sharded-fp32-queue-v1"
    base.QUEUE_LABEL = "R0171 sharded-fp32 GPU queue"
    base.GRAPH_VECTOR_STORAGE = GRAPH_VECTOR_STORAGE
    base.GPU_HOURS_CAP = 8.0
    base.SELECT_P90_WALL_S = 900.0
    base.GRAPH_P90_WALL_S = 14_400.0
    base.TRAIN_P90_WALL_S = 6_000.0
    base.EVALUATION_P90_WALL_S = 3_600.0
    base.scale_train_config = scale_train_config
    base._release_cpu_smoke = _release_cpu_smoke


def prepare_round0171(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    _configure()
    return base.prepare_round0166(release_sha=release_sha, queue_root=queue_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "graph_execution": GRAPH_EXECUTION,
        "queue_manifest": prepare_round0171(
            release_sha=args.release_sha, queue_root=args.queue_root
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
