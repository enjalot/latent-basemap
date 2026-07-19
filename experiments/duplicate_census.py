#!/usr/bin/env python3
"""Run Round 0020's exact fp16 duplicate-family census."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.duplicate_census import build_duplicate_census


DEFAULT_PACK_MANIFEST = "/data/latent-basemap/runs/round-0013/30m-input-pack-v1.json"
DEFAULT_R0019_COORDINATES = (
    "/data/latent-basemap/runs/round-0019/queue/artifacts/coordinates"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-manifest", default=DEFAULT_PACK_MANIFEST)
    parser.add_argument("--r0019-coordinates", default=DEFAULT_R0019_COORDINATES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    coordinates = None
    receipt_path = None
    if args.r0019_coordinates:
        from experiments import run_round0014_node as node

        node.configure_round0019()
        coordinates = node.StreamedCoordinateArray(os.path.realpath(args.r0019_coordinates))
        receipt_path = os.path.join(os.path.realpath(args.r0019_coordinates), "actual-transform.json")
    receipt = build_duplicate_census(
        pack_manifest=os.path.realpath(args.pack_manifest),
        output_root=os.path.realpath(args.out),
        coordinates=coordinates,
        coordinate_receipt_path=receipt_path,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

