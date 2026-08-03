#!/usr/bin/env python3
"""Prepare, but never launch, the R0170 fp16-IVF replacement Q2 rung."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.round0170_prompted_8m import (
    CAPABILITY,
    GRAPH_VECTOR_STORAGE,
    ROUND_ID,
    scale_train_config,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import prepare_round0166_queue as base
from experiments.prepare_round0020_0022_queues import LAB_ROOT


ROUND_ROOT = "/data/latent-basemap/runs/round-0170"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0170-2026-08-03.md")


def _release_cpu_smoke(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0170 release checkout differs from requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0166_cpu_smoke.py",
        "tests/test_round0166_prompted_8m.py",
        "tests/test_round0170_prompted_8m.py",
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
        "schema": "round0170-release-cpu-smoke-v1",
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
            "reused Q2 train -> seal -> checkpoint reload -> panel path plus "
            "R0170 fp16-IVF dispatch/config binding"
        ),
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0170 release CPU smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


def _configure() -> None:
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.ROUND_ROOT = ROUND_ROOT
    base.RELEASE_ROOT = RELEASE_ROOT
    base.ROUND_FILE = ROUND_FILE
    base.HANDLER_MODULE = "experiments.round0170_nodes"
    base.QUEUE_SCHEMA = "round0170-prompted-english-8m-fp16-ivf-queue-v1"
    base.QUEUE_LABEL = "R0170 fp16-IVF GPU queue"
    base.GRAPH_VECTOR_STORAGE = GRAPH_VECTOR_STORAGE
    base.scale_train_config = scale_train_config
    base._release_cpu_smoke = _release_cpu_smoke


def prepare_round0170(
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
        "queue_manifest": prepare_round0170(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
