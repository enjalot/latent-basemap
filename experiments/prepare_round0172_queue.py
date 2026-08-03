#!/usr/bin/env python3
"""Prepare, but never launch, the R0172 corrected prompted-universality queue."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap import round0167_prompted_universality as contract_base
from basemap.round0171_prompted_8m import CAPABILITY as Q2_CAPABILITY
from basemap.round0172_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    Round0172Error,
)
from experiments import prepare_round0167_queue as base
from experiments.prepare_round0020_0022_queues import LAB_ROOT


ROUND_ROOT = "/data/latent-basemap/runs/round-0172"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0172-2026-08-03.md")
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
    contract_base.Round0167Error = Round0172Error
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    base.ROUND_ROOT = ROUND_ROOT
    base.RELEASE_ROOT = RELEASE_ROOT
    base.ROUND_FILE = ROUND_FILE
    base.MAPS = MAPS
    base.Q2_ROUND_ID = "0171"
    base.Q2_CAPABILITY = Q2_CAPABILITY
    base.HANDLER_MODULE = "experiments.round0172_nodes"
    base.QUEUE_SCHEMA = "round0172-prompted-universality-queue-v1"
    base.QUEUE_LABEL = "R0172 prompted universality queue"


def prepare_round0172(
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
        "queue_manifest": prepare_round0172(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
