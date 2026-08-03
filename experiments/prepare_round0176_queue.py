#!/usr/bin/env python3
"""Prepare, but never launch, the R0176 prompted-universality queue."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap import round0167_prompted_universality as contract_base
from basemap.round0176_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    Round0176Error,
)
from experiments import prepare_round0167_queue as base
from experiments.prepare_round0020_0022_queues import LAB_ROOT


ROUND_ROOT = "/data/latent-basemap/runs/round-0176"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0176-2026-08-03.md")
MAPS = {
    "r0115-prompted-2m-seed42": (
        "/data/latent-basemap/runs/round-0115/queue-attempt-2/artifacts/"
        "document/train/model.pt"
    ),
    "r0117-prompted-2m-seed43": (
        "/data/latent-basemap/runs/round-0117/queue/artifacts/document/train/model.pt"
    ),
    "r0171-prompted-8m-seed42": (
        "/data/latent-basemap/runs/round-0171/queue/artifacts/seed42-train/model.pt"
    ),
}


def _configure() -> None:
    contract_base.ROUND_ID = ROUND_ID
    contract_base.CAPABILITY = CAPABILITY
    contract_base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    contract_base.Round0167Error = Round0176Error
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    base.ROUND_ROOT = ROUND_ROOT
    base.RELEASE_ROOT = RELEASE_ROOT
    base.ROUND_FILE = ROUND_FILE
    base.MAPS = MAPS
    base.Q2_ROUND_ID = "0171"
    base.Q2_CAPABILITY = None
    base.Q2_MAP_ROLE = (
        "accepted R0171 negative-result artifact; mechanically valid model, "
        "not a released map capability and not evidence of Q2 quality"
    )
    # The inherited per-node p90s sum to 2.233 GPU-h.  R0172's 2.0 h cap was
    # therefore internally inconsistent even though its expected execution was
    # much shorter.  Keep those conservative node bounds and make the cap
    # cumulative-safe instead of weakening a node timeout.
    base.GPU_HOURS_MINIMUM = 0.45
    base.GPU_HOURS_EXPECTED = 0.70
    base.GPU_HOURS_MAXIMUM = 2.50
    base.HANDLER_MODULE = "experiments.round0176_nodes"
    base.QUEUE_SCHEMA = "round0176-prompted-universality-queue-v1"
    base.QUEUE_LABEL = "R0176 prompted universality queue"


def prepare_round0176(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    _configure()
    return base.prepare_round0167(release_sha=release_sha, queue_root=queue_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0176(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
