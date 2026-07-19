#!/usr/bin/env python3
"""Persist R0018's registered baseline for the R0019 duplicate treatment."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.duplicate_diagnostics import duplicate_component_diagnostics
from basemap.duplicate_multiplicity import load_duplicate_cap
from basemap.output_safety import atomic_write_new_json


CAP_PATH = "/data/latent-basemap/runs/round-0019/input/duplicate-cap.npz"
CAP_SHA256 = "cb5617f7ef672801c59a6ecbe87af4c7c65390ec59b1305fbff77ec673aad007"
COORDINATE_ROOT = (
    "/data/latent-basemap/runs/round-0018/queue/artifacts/coordinates"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    from experiments import run_round0014_node as node

    node.configure_round0018()
    cap = load_duplicate_cap(
        CAP_PATH,
        expected_sha256=CAP_SHA256,
        row_count=30_000_000,
        fixed_edges_per_source=15,
    )
    coordinates = node.StreamedCoordinateArray(COORDINATE_ROOT)
    diagnostics = duplicate_component_diagnostics(
        coordinates,
        excluded_rows=cap["excluded_rows"],
        representative_rows=cap["representative_rows"],
    )
    body = {
        "schema": "round0019-duplicate-component-baseline.v1",
        "baseline_round": "0018",
        "diagnostics": diagnostics,
        "inputs": {
            "duplicate_cap": expected_input_signature(CAP_PATH),
            "coordinate_capability": expected_input_signature(
                os.path.join(COORDINATE_ROOT, "actual-transform.json")
            ),
        },
    }
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    atomic_write_new_json(args.out, receipt, immutable=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
