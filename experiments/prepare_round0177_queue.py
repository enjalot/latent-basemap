#!/usr/bin/env python3
"""Prepare, but never launch, the R0177 duplicate-aware prompted panel."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap import round0167_prompted_universality as contract_base
from basemap.round0177_prompted_universality import (
    CAPABILITY,
    PROMPTED_MAP_ORDER,
    ROUND_ID,
    Round0177Error,
)
from experiments import prepare_round0167_queue as base
from experiments import prepare_round0176_queue as prior
from experiments.prepare_round0020_0022_queues import LAB_ROOT


ROUND_ROOT = "/data/latent-basemap/runs/round-0177"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0177-2026-08-03.md")


def _configure() -> None:
    prior._configure()
    contract_base.ROUND_ID = ROUND_ID
    contract_base.CAPABILITY = CAPABILITY
    contract_base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    contract_base.Round0167Error = Round0177Error
    base.ROUND_ID = ROUND_ID
    base.CAPABILITY = CAPABILITY
    base.PROMPTED_MAP_ORDER = PROMPTED_MAP_ORDER
    base.ROUND_ROOT = ROUND_ROOT
    base.RELEASE_ROOT = RELEASE_ROOT
    base.ROUND_FILE = ROUND_FILE
    base.GPU_HOURS_MINIMUM = 0.15
    base.GPU_HOURS_EXPECTED = 0.35
    base.GPU_HOURS_MAXIMUM = 2.50
    base.HANDLER_MODULE = "experiments.round0177_nodes"
    base.QUEUE_SCHEMA = "round0177-prompted-universality-queue-v1"
    base.QUEUE_LABEL = "R0177 duplicate-aware prompted universality queue"
    base.PROBE_FAMILY_POLICY = (
        "retain the exact R0142 full panel as diagnostic primary; also score "
        "a paired sensitivity excluding the union of probe/control query "
        "positions with exact stored-fp16 corpus copies"
    )


def prepare_round0177(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    _configure()
    return base.prepare_round0167(
        release_sha=release_sha, queue_root=queue_root
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument(
        "--queue-root", default=os.path.join(ROUND_ROOT, "queue")
    )
    args = parser.parse_args(argv)
    print(json.dumps({
        "queue_manifest": prepare_round0177(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
