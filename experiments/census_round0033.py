#!/usr/bin/env python3
"""Run the CPU-only R0033 exact encoded-row eligibility census."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.int8_eligibility import build_int8_eligibility_census


MANIFEST = "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/int8-shards-v1.json"
INT8 = "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/minilm-int8-150m/embeddings.i8"
SCALES = "/data/latent-basemap/runs/round-0025/queue/artifacts/int8-shards/minilm-int8-150m/scales.f16"


def run_job(_active: dict, job: dict) -> None:
    """Slim-runner entry point for the one CPU-only census job."""
    build_int8_eligibility_census(
        manifest_path=MANIFEST,
        int8_path=INT8,
        scales_path=SCALES,
        output_root=job["outputs"][0],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--int8", default=INT8)
    parser.add_argument("--scales", default=SCALES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    receipt = build_int8_eligibility_census(
        manifest_path=os.path.realpath(args.manifest),
        int8_path=os.path.realpath(args.int8),
        scales_path=os.path.realpath(args.scales),
        output_root=os.path.realpath(args.out),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
